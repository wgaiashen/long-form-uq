"""Softmax-attention pooling, built as a small FAMILY on one axis, not a single model.

The mean-pool SAPLMA baseline averages the answer-token hidden states (uniform weight per token).
This learns the weights with a single attention step (a learned query scores each token, a softmax
gives the weights, the pooled vector is their weighted sum, a linear head reads it). The point is
not one architecture but a family along one axis -- how much, and how, the weights select tokens:

    mean-pool                 uniform weights (the baseline; recovered as temperature -> inf)
    softmax-attention + T     a learned query, temperature T controls how peaked the weights get
                              (McKenzie's softmax-probe knob: high T ~ uniform, low T ~ select few)
    softmax-attention + pos   the above, plus a learned positional bias on the scores, so the query
                              can prefer early/late tokens, not only by content
    softmax-attention answer  the above (content only), but pooling over the answer tokens only
                              (the last-prompt token is masked out -- a span mask, NOT Orgad's
                              important-token method, which waits for the authors' code)

Same linear head throughout, so the comparison isolates the aggregation. Two guards against fooling
ourselves, both cheap:
  - p >> n discipline: weight decay on the query and head, single low-capacity query, and the
    temperature is selected on a VALIDATION split carved from train, never on test.
  - watch the weights, not just PRR: --diag dumps the attention weights for a few examples (entropy,
    peak weight, mass on answer tokens, top-weighted token strings) so we can see whether attention
    concentrates on answer tokens or stays flat -- the most informative line either way.

The interesting comparison is long-form + OOD, where a smarter pooler can separate from the mean;
short-form ID is where every aggregator ties, so read the numbers there as a sanity check only.

Runs on the per-token cache (cache/pertok, layer 15), positional alignment (record idx is not
unique). CPU or GPU.

    python scripts/checks/attn_pool.py --datasets sciq,trivia_qa,pubmed_qa
    python scripts/checks/attn_pool.py --datasets sciq --diag        # dump attention weights
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("HF_HOME", "/vol/gpudata/gs925-msc_project/hf_cache")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402

MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"
SEED = 1
TEMP_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
WEIGHT_DECAY = 1e-2          # regularises the query + head against p >> n overfitting
VAL_FRAC = 0.2              # validation carved from train for temperature selection (never test)


# XL datasets whose per-token cache + records live in a prompt-regime namespace (not the default cache/).
# This is what lets ExpertQA load through the SAME interface as the core datasets (organic ProbeDriftXL).
PROMPT_REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}


def load_per_token(model, dataset, layer, label_field="correctness"):
    """Per-example token states aligned POSITIONALLY to records (idx is not unique).

    label_field names WHICH correctness signal to read (e.g. `correctness` for the core sets,
    `faithfulness` for ExpertQA). Do not rely on the default bare `correctness` for a reported table
    -- it holds whatever labeller ran last and differs by dataset; pass an explicit field and stamp
    the model. See scripts/checks/aggregation_table.py.

    The cache root is derived from the dataset's prompt-regime (via Config), so a regime-namespaced XL
    dataset like ExpertQA (cache/expertqa_rp12/) loads identically to a core dataset (cache/). The XL sets
    are split-less (ExpertQA all-`test`; med_quad/samsum all-`train`); carving a held-out EVAL split is the
    rung layer's job (xl_rungs.eval_split), NOT here, so their use as TRAINING SOURCES stays unchanged and
    the core-5 rung numbers do not move."""
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    path = cfg.cache_dir / "pertok" / f"{cache._slug(model)}__{dataset}__ID__L{layer}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    records = cache.load_records(cfg.cache_dir, cache.run_key(model, dataset, "ID"))
    st = z["states"]
    if len(st) != len(records):
        raise SystemExit(f"{dataset}: per-token cache {len(st)} != records {len(records)}")
    states = [np.asarray(st[k], dtype=np.float32) for k in range(len(records))]
    # RUNTIME MODEL GUARD (2026-07-27): the ONLY model is Llama-3.1-8B (hidden dim 4096). Fail loud if a
    # cache from another model slipped in (Qwen=1536, Gemma=3584) — protects the CURRENT numbers, not just
    # future sessions (the doc rule in CLAUDE.md does the latter).
    d = states[0].shape[-1] if len(states) else 4096
    if d != 4096:
        raise SystemExit(f"{dataset}: per-token cache hidden dim {d} != 4096 (Llama). Wrong-model cache?")
    split = np.array([r["split"] for r in records])
    y = np.array([r.get(label_field, np.nan) for r in records], dtype=float)
    return states, split, y, int(z["layer"]), records


def pad_prior(priors_list, tmax, device):
    """Pad per-example prior weight vectors (each length = that example's token count, G+1) to (B, tmax),
    matching pad_batch's padding so the prior aligns with X/mask. Zeros in the pad region."""
    P = torch.zeros(len(priors_list), tmax, dtype=torch.float32)
    for i, p in enumerate(priors_list):
        p = np.asarray(p, dtype=np.float32)
        P[i, :len(p)] = torch.from_numpy(p)
    return P.to(device)


def pad_batch(states_list, device):
    """Pad to (B, Tmax, d); return X, mask (1=real token), and positional features (B, Tmax, 2):
    [relative position in the response, inverse response length]."""
    d = states_list[0].shape[1]
    tmax = max(s.shape[0] for s in states_list)
    X = torch.zeros(len(states_list), tmax, d, dtype=torch.float32)
    mask = torch.zeros(len(states_list), tmax, dtype=torch.float32)
    pos = torch.zeros(len(states_list), tmax, 2, dtype=torch.float32)
    for i, s in enumerate(states_list):
        t = s.shape[0]
        X[i, :t] = torch.from_numpy(s)
        mask[i, :t] = 1.0
        rng = torch.arange(t, dtype=torch.float32)
        pos[i, :t, 0] = rng / max(t - 1, 1)     # 0 at first token, 1 at last
        pos[i, :t, 1] = 1.0 / t                  # inverse length (short vs long response)
    return X.to(device), mask.to(device), pos.to(device)


class AttnPool(nn.Module):
    """One learned-query softmax-attention step over token states, then a linear head.

    temperature  divides the scores before softmax (peakedness knob).
    use_position  adds a learned bias from the positional features to the scores.
    answer_only   handled by the caller via the mask (it zeros the last-prompt-token column), so the
                  module stays a pure pooling step.
    forward returns (logit, attention_weights) so the weights can be inspected.
    """

    def __init__(self, d, temperature=1.0, use_position=False, freeze_query=False,
                 frozen_prior=False, beta=1.0):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(d))       # learned query (init 0 => starts at mean-pool)
        if freeze_query or frozen_prior:
            self.q.requires_grad_(False)            # q stays 0 (uniform), or is unused (frozen prior)
        self.head = nn.Linear(d, 1)
        self.scale = d ** 0.5
        self.temperature = temperature
        self.use_position = use_position
        self.pos = nn.Linear(2, 1) if use_position else None
        # S3 (prior-init pooling): a per-token PRIOR weight can replace / seed the learned query.
        #   frozen_prior=True  -> attention IS the renormalised prior; head-only training (arm C, ours).
        #   frozen_prior=False + prior given -> annealed additive log-prior: scores = X@q + beta*log(prior)
        #     (arm D, Joe's "start from a distribution, then learn away"; beta scales the prior's pull).
        self.frozen_prior = frozen_prior
        self.beta = beta

    def forward(self, X, mask, posfeat, prior=None):
        # arm C (frozen prior): attention IS the renormalised prior over real tokens; the query is unused.
        if prior is not None and self.frozen_prior:
            a = prior * mask                                        # zero the padded/masked tokens
            a = a / a.sum(dim=1, keepdim=True).clamp(min=1e-9)      # renormalise over real tokens
            pooled = (a.unsqueeze(-1) * X).sum(dim=1)
            return self.head(pooled).squeeze(-1), a
        scores = (X @ self.q) / (self.scale * self.temperature)     # (B, T)
        if self.use_position:
            scores = scores + self.pos(posfeat).squeeze(-1)
        if prior is not None:                                       # arm D: annealed additive log-prior tilt
            scores = scores + self.beta * torch.log(prior.clamp(min=1e-9))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        a = torch.softmax(scores, dim=1)                            # (B, T)
        pooled = (a.unsqueeze(-1) * X).sum(dim=1)                    # (B, d)
        return self.head(pooled).squeeze(-1), a


def _mask_answer_only(mask):
    """Zero the first column (the last-prompt token), so pooling is over answer tokens only."""
    m = mask.clone()
    m[:, 0] = 0.0
    return m


def train_attn(states, y, tr_idx, device, seed=SEED, temperature=1.0, use_position=False,
               answer_only=False, weight_decay=WEIGHT_DECAY, wd_query=0.0, freeze_query=False,
               epochs=60, bs=32, lr=1e-3, shrink_lambda=0.0,
               prior_list=None, frozen_prior=False, beta=1.0):
    """`prior_list` (S3) = per-example prior weight vectors aligned to `states` (length G+1 each). With
    frozen_prior=True the attention IS the renormalised prior and only the head trains (arm C); with
    frozen_prior=False it is an annealed additive log-prior tilt on the learned query (arm D). prior_list=None
    keeps the plain learned/frozen-query pooler (arms A/B) unchanged."""
    """`shrink_lambda` > 0 = H5 (shrink-the-pooler): add a shrink-to-uniform penalty on the attention weights
    to the BCE loss, pulling the learned attention toward mean-pool (its unsupervised prior). Tests whether
    moderation-toward-the-unsupervised-prior — the mechanism that helps weighted-MSP — makes the ATTENTION
    POOLER OOD-robust too. Penalty = mean over real tokens of (a_i·n − 1)² (the same avg-1 shrink as wMSP).
    Default 0.0 = the plain pooler (unchanged), so existing callers are untouched."""
    torch.manual_seed(seed)
    d = states[0].shape[1]
    model = AttnPool(d, temperature=temperature, use_position=use_position,
                     freeze_query=freeze_query or frozen_prior,
                     frozen_prior=frozen_prior, beta=beta).to(device)
    # Separate param groups: the head (and positional bias) get weight decay against p >> n, but the
    # query is left UN-decayed (wd_query=0) so it can actually move off zero -- with decay on it, the
    # query collapsed to 0 and the attention stayed uniform (the first run's diagnostic showed this).
    head_params = [p for n, p in model.named_parameters() if not n.startswith("q")]
    groups = [{"params": head_params, "weight_decay": weight_decay}]
    if not (freeze_query or frozen_prior):
        groups.append({"params": [model.q], "weight_decay": wd_query})
    opt = torch.optim.Adam(groups, lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(tr_idx), generator=g).tolist()
        for b in range(0, len(perm), bs):
            idx = [tr_idx[perm[j]] for j in range(b, min(b + bs, len(perm)))]
            X, mask, pos = pad_batch([states[i] for i in idx], device)
            if answer_only:
                mask = _mask_answer_only(mask)
            prior_b = (pad_prior([prior_list[i] for i in idx], X.shape[1], device)
                       if prior_list is not None else None)
            opt.zero_grad()
            logit, a = model(X, mask, pos, prior=prior_b)
            loss = lossf(logit, yt[idx])
            if shrink_lambda > 0:                       # H5: shrink attention toward mean-pool (avg-1 MSE)
                n_real = mask.sum(1, keepdim=True).clamp(min=1.0)      # (B,1) real-token count
                dev = ((a * n_real - 1.0) ** 2) * mask                 # only real tokens contribute
                loss = loss + shrink_lambda * (dev.sum(1) / n_real.squeeze(1)).mean()
            loss.backward()
            opt.step()
    return model


def attn_prr(model, states, y, te_idx, device, answer_only=False, bs=64):
    model.eval()
    preds = np.zeros(len(te_idx))
    with torch.no_grad():
        for b in range(0, len(te_idx), bs):
            idx = te_idx[b: b + bs]
            X, mask, pos = pad_batch([states[i] for i in idx], device)
            if answer_only:
                mask = _mask_answer_only(mask)
            logit, _ = model(X, mask, pos)
            preds[b: b + len(idx)] = torch.sigmoid(logit).cpu().numpy()
    unc = 1.0 - preds
    return results.prr([y[i] for i in te_idx], unc)


def select_temperature(states, y, tr_idx, device, seed, use_position, answer_only):
    """Carve a validation split from train (never test), train at each temperature, pick the one with
    the best validation PRR. Returns (best_T, [(T, val_prr), ...])."""
    g = np.random.RandomState(seed)
    order = list(tr_idx)
    g.shuffle(order)
    n_val = int(len(order) * VAL_FRAC)
    val_idx, sub_tr = order[:n_val], order[n_val:]
    curve = []
    for T in TEMP_GRID:
        m = train_attn(states, y, sub_tr, device, seed=seed, temperature=T,
                       use_position=use_position, answer_only=answer_only, epochs=40)
        curve.append((T, attn_prr(m, states, y, val_idx, device, answer_only=answer_only)))
    best_T = max(curve, key=lambda c: c[1])[0]
    return best_T, curve


def run_variant(states, y, tr_idx, te_idx, device, seeds, use_position, answer_only):
    """Select temperature on val (seed 1), then train at that T over all seeds and report test PRR."""
    best_T, curve = select_temperature(states, y, tr_idx, device, SEED, use_position, answer_only)
    prrs = []
    for sd in seeds:
        m = train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T,
                       use_position=use_position, answer_only=answer_only)
        prrs.append(attn_prr(m, states, y, te_idx, device, answer_only=answer_only))
    return np.array(prrs), best_T, curve


def diagnose_weights(states, y, tr_idx, records, device, model_name, n=4):
    """Train the plain softmax-attention variant, then dump the attention weights for a few test
    examples: entropy, peak weight, mass on answer tokens (rows >= 1), and the top-weighted token
    strings. This shows whether attention concentrates on the answer or stays near-uniform."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    best_T, _ = select_temperature(states, y, tr_idx, device, SEED, False, False)
    model = train_attn(states, y, tr_idx, device, seed=SEED, temperature=best_T)
    model.eval()
    print(f"  [diag] plain softmax-attention, selected T={best_T}")
    shown = 0
    for k in range(len(states)):
        if records[k]["split"] != "test":
            continue
        X, mask, pos = pad_batch([states[k]], device)
        with torch.no_grad():
            _, a = model(X, mask, pos)
        a = a[0, : states[k].shape[0]].cpu().numpy()
        ent = float(-(a * np.log(a + 1e-12)).sum())
        unif = float(np.log(len(a)))           # entropy of the uniform weights (the mean-pool point)
        ans_mass = float(a[1:].sum())          # mass on answer tokens (row 0 = last-prompt token)
        ids = window_ids(records[k])
        pieces = tok.convert_ids_to_tokens(ids)
        top = np.argsort(a)[::-1][:3]
        toptok = ", ".join(f"{pieces[i]!r}={a[i]:.2f}" for i in top)
        print(f"    ex{k} y={records[k].get('correctness'):.2f}  entropy {ent:.2f}/{unif:.2f}(unif)  "
              f"peak {a.max():.2f}  answer-mass {ans_mass:.2f}  top: {toptok}")
        shown += 1
        if shown >= n:
            break


def window_ids(record):
    """Token ids over the SAPLMA window: last-prompt token then all answer tokens (matches rows)."""
    return [record["prompt_token_ids"][-1]] + list(record["gen_token_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--datasets", default="sciq,trivia_qa,pubmed_qa")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--label-field", default="correctness",
                    help="which correctness signal to train/eval on (explicit, not the bare "
                         "'correctness' default for a reported table -- see load_per_token)")
    ap.add_argument("--seeds", default="1,2,3",
                    help="seeds for the final probe; reports mean +/- std so a small margin can be "
                         "told from seed noise")
    ap.add_argument("--diag", action="store_true", help="also dump attention weights for a few examples")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  label-field: {args.label_field}")

    for dataset in args.datasets.split(","):
        loaded = load_per_token(args.model, dataset, args.layer, args.label_field)
        if loaded is None:
            print(f"\n==== {dataset}: no per-token cache, skip ====")
            continue
        states, split, y, layer, records = loaded
        if np.isnan(y).any():
            print(f"\n==== {dataset}: missing correctness labels, skip ====")
            continue
        tr_idx = [i for i in range(len(states)) if split[i] == "train"]
        te_idx = [i for i in range(len(states)) if split[i] == "test"]
        print(f"\n==== {dataset} (layer {layer}, train {len(tr_idx)}, test {len(te_idx)}) ====")

        Xmean = np.stack([s.mean(axis=0) for s in states])
        ytr = y[tr_idx]
        # Baselines. (1) mean-pool + SAPLMA MLP, the project's main baseline. (2) mean-pool + sklearn
        # linear. (3) the CONTROLLED isolation: the attention model with the query FROZEN at zero, so
        # the weights are uniform (= mean-pool) but the head is the SAME torch linear head as the
        # learned-attention rows. The attention variants must be compared against THIS, not the
        # sklearn linear, or a torch-vs-sklearn head difference masquerades as an aggregation win.
        mlp = probe.train_probe_mlp(Xmean[tr_idx], ytr)
        prr_mlp = results.prr([y[i] for i in te_idx], probe.uncertainty(mlp, Xmean[te_idx]))
        lin = probe.train_probe(Xmean[tr_idx], ytr)
        prr_lin = results.prr([y[i] for i in te_idx], probe.uncertainty(lin, Xmean[te_idx]))
        uni = train_attn(states, y, tr_idx, device, seed=SEED, freeze_query=True)
        prr_uni = attn_prr(uni, states, y, te_idx, device)
        print(f"  {'mean-pool + SAPLMA MLP':30s} {prr_mlp:+.3f}   (main baseline)")
        print(f"  {'mean-pool + sklearn linear':30s} {prr_lin:+.3f}")
        print(f"  {'uniform (frozen-q, torch head)':30s} {prr_uni:+.3f}   (the controlled baseline)")

        variants = [("softmax-attention", False, False),
                    ("  + position", True, False),
                    ("  answer-only", False, True)]
        for name, use_pos, ans_only in variants:
            prrs, best_T, curve = run_variant(states, y, tr_idx, te_idx, device, seeds,
                                              use_pos, ans_only)
            spread = f" +/- {prrs.std():.3f}" if len(seeds) > 1 else ""
            valstr = " ".join(f"{T}:{v:+.2f}" for T, v in curve)
            # Compare against the controlled uniform baseline (same torch head), so the delta is the
            # aggregation, not the head optimizer.
            print(f"  {name:30s} {prrs.mean():+.3f}{spread}   (T*={best_T}, vs uniform "
                  f"{prrs.mean() - prr_uni:+.3f}; val {valstr})")

        if args.diag:
            diagnose_weights(states, y, tr_idx, records, device, args.model)


if __name__ == "__main__":
    main()
