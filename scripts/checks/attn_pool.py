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

# HF_HOME fallback for callers that did not source an env script. On RCS, pbs/_env.sh already
# exports HF_HOME, so this whole block is a no-op there.
# ⚠️ It must NOT hard-code DoC's /vol/gpudata: that path is inside the hard 50 GB quota and already
# holds Qwen-14B (27.5 GB), so a 32B checkpoint (~65 GB in bf16) cannot fit. Prefer the un-quota'd
# /vol/bitbucket root and only fall back to gpudata when bitbucket is absent.
if "HF_HOME" not in os.environ:
    for _cand in ("/vol/bitbucket/gs925/hf_cache", "/vol/gpudata/gs925-msc_project/hf_cache"):
        if Path(_cand).parent.is_dir():
            os.environ["HF_HOME"] = _cand
            break

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402

MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"

# Expected residual-stream width per model. The per-token cache guard below checks against THIS,
# not against a bare 128000-style constant, so a wrong-model cache still fails loudly while a
# legitimately different model is simply a new entry here.
# ⚠️ The guard exists because a model-agnostic glob once loaded the dropped Qwen-1.5B cache into
# PART A's headline rows (the project conventions). Keeping it FAIL-LOUD is the point; only its constant was
# ever wrong.
# ⚠️ Values are read from each checkpoint's own config.json (`hidden_size`), never assumed from the
# family: gemma-2-9b-it is 3584, NOT the 4096 an 8-9B model invites you to guess.
EXPECTED_HIDDEN_DIM = {
    # development populations (the two the hypothesis was formed on)
    "meta-llama/Meta-Llama-3.1-8B": 4096,
    "Qwen/Qwen2.5-14B": 5120,
    # replication populations, W-Models (2026-08-15)
    "meta-llama/Llama-3.1-8B-Instruct": 4096,
    "google/gemma-2-9b": 3584,        # base -- the population actually run (see prereg M5 §8)
    "google/gemma-2-9b-it": 3584,     # instruct -- WITHDRAWN 2026-08-15, kept so old caches fail loudly
    "Qwen/Qwen2.5-32B": 5120,
}
SEED = 1
TEMP_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
WEIGHT_DECAY = 1e-2          # regularises the query + head against p >> n overfitting
VAL_FRAC = 0.2              # validation carved from train for temperature selection (never test)


# XL datasets whose per-token cache + records live in a prompt-regime namespace (not the default cache/).
# This is what lets ExpertQA load through the SAME interface as the core datasets (organic ProbeDriftXL).
_BASE_PROMPT_REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}

# v2 INDIRECTION (2026-07-31). Every ladder driver resolves its cache root through this dict, so with the
# mapping hardcoded there was no way to point the ladder at a regenerated (v2) cache -- and editing the
# dict in place would silently redirect the v1 reads too, which is exactly the "never mix v1 and v2" rule
# it would be breaking. Overrides are supplied per dataset through the environment instead:
#
#     LUQ_REGIME="samsum=v2pilot_samsum_plain,xsum=v2_xsum" python scripts/checks/<driver>.py ...
#
# Follows the LUQ_EXPERTQA_LABEL pattern in xl_rungs.py. Unset = the v1 mapping, byte-identical to before.
# An empty value ("samsum=") pins a dataset to the DEFAULT cache root explicitly, which is not the same as
# omitting it -- omitting means "whatever the base map says", pinning means "the v1 root, on purpose".
def _parse_regime_override(raw):
    out = {}
    for item in (s.strip() for s in raw.split(",")):
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"LUQ_REGIME entry {item!r} is not dataset=regime; refusing to guess which "
                             "cache root was meant.")
        ds, rg = item.split("=", 1)
        out[ds.strip()] = rg.strip()
    return out


PROMPT_REGIME = dict(_BASE_PROMPT_REGIME)
_REGIME_OVERRIDE = _parse_regime_override(os.environ.get("LUQ_REGIME", ""))
if _REGIME_OVERRIDE:
    PROMPT_REGIME.update(_REGIME_OVERRIDE)
    # LOUD: a run reading regenerated caches must say so in its own log, or a v2 number could later be
    # mistaken for a v1 one purely because nothing recorded which cache it came from.
    print(f"[LUQ_REGIME] cache-root overrides active: {_REGIME_OVERRIDE}", flush=True)


def regime_tag():
    """Short, filename-safe tag naming the active regime override ('' when none). Used by the ladder
    drivers to keep v1 and v2 result CSVs on separate paths -- see the note on --out in probedriftlong."""
    if not _REGIME_OVERRIDE:
        return ""
    return "__" + "_".join(f"{k}-{v}" for k, v in sorted(_REGIME_OVERRIDE.items()))


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
    # RUNTIME MODEL GUARD (2026-07-27; made model-aware 2026-08-08). Fail loud if the cache on disk is
    # not the width THIS model should produce — that is how the dropped Qwen-1.5B cache once got loaded
    # into PART A's headline rows via a model-agnostic glob (the project conventions).
    # ⚠️ The guard is still FAIL-LOUD and still per-model; only the hard-coded 4096 was wrong. A model
    # absent from EXPECTED_HIDDEN_DIM is itself an error, NOT a pass — an unknown model must not skip
    # the check, or the guard quietly stops guarding exactly when a new model is introduced.
    if model not in EXPECTED_HIDDEN_DIM:
        raise SystemExit(f"{dataset}: no expected hidden dim registered for {model!r}. Add it to "
                         f"EXPECTED_HIDDEN_DIM — an unregistered model must not bypass the cache guard.")
    want = EXPECTED_HIDDEN_DIM[model]
    d = states[0].shape[-1] if len(states) else want
    if d != want:
        raise SystemExit(f"{dataset}: per-token cache hidden dim {d} != {want} expected for {model}. "
                         "Wrong-model cache?")
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


def pad_head_priors(head_priors, idx, tmax, device):
    """S7 — stack per-HEAD recipes into the (B, T, Q) tensor the multi-head forward expects.

    `head_priors` is a list of length Q, one entry per query head: either a per-example prior list (a
    fixed recipe) or None (a free, learned head). A free head's column is filled with zeros and is never
    read, because only the indices in `frozen_heads` are substituted -- but the column must still exist so
    the tensor's head axis lines up with the attention's, rather than being silently re-indexed.
    """
    cols = [(pad_prior([hp[i] for i in idx], tmax, device) if hp is not None
             else torch.zeros(len(idx), tmax, device=device))
            for hp in head_priors]
    return torch.stack(cols, dim=2)                                  # (B, T, Q)


def normalise_target(D, mask):
    """Row-normalise a padded target to a DISTRIBUTION over real tokens, so it is on the same footing as
    the attention `a` (a softmax, which already sums to 1 over real tokens).

    Convention note: `prior_builders` returns UNNORMALISED weights (content-mass is 0/1, NLL is clipped
    surprisal). Supervising `a` toward an unnormalised target would penalise the total mass rather than
    the shape, which is not what the paper's loss means.

    A row that sums to zero RAISES rather than falling back to uniform. The builders already guarantee a
    non-zero row and count their own fallbacks, so a zero here is a genuine bug -- and silently swapping
    in uniform would turn the supervised arm into its own control, which is exactly the failure mode that
    manufactures a null."""
    D = D * mask
    s = D.sum(1, keepdim=True)
    if bool((s <= 0).any()):
        bad = int((s <= 0).sum())
        raise SystemExit(f"aux target: {bad} example(s) have zero mass over their real tokens. The prior "
                         "builders guarantee a non-zero row, so this is a bug -- refusing to substitute "
                         "uniform, which would silently make the supervised arm identical to its control.")
    return D / s


def aux_penalty(a, D, mask, normalise=False):
    """The paper's squared-error attention penalty, optionally referenced to UNIFORM attention.

    Returns (penalty, n_dropped). `normalise=False` is the paper's raw sum and is BYTE-IDENTICAL to what
    this file did before, so every existing caller is unchanged.

    WHY THE NORMALISED FORM EXISTS. For `a` uniform and a flat k-sparse `D` over n real tokens the raw sum
    has a closed form:

        Σᵢ (aᵢ − Dᵢ)²  =  k(1/k − 1/n)² + (n−k)(1/n)²  =  1/k − 1/n

    It splits on **k**, not n. A SPARSE target (k fixed) is therefore already length-invariant; only a
    DENSE target (k = αn) scales as 1/n. So a plain `.mean()` would be the wrong fix -- it squares the
    dense problem AND injects length-dependence into the sparse targets. Dividing by the same quantity
    evaluated at uniform attention fixes both at once:

        aux_ref = Σᵢ (1/n − Dᵢ)²  ==  ΣD² − 1/n        (the same identity)

    The ratio is 1.0 when attention is uniform, 0.0 when it matches the target, and dimensionless -- so one
    lambda means the same thing on a 56-token xsum answer and a 768-token med_quad one, and lambda=1.0 is
    genuine equal weighting against a BCE of ~0.7, which is the paper's operating regime. Under the raw
    sum it was not: at a fixed lambda the supervision strength varied with output length, confounding it
    with the very variable the ID headline rests on.

    ⚠️ NO clamp_min ON THE DENOMINATOR. `aux_ref` is exactly 0 when D is uniform, which is the documented
    degenerate case the content-mass builder falls back to. Clamping would turn 0/0 into a large finite
    penalty, handing rows that carry NO target the largest gradient in the batch -- the "silent default
    that returns a plausible number" class this project bans. Those rows are DROPPED from the penalty and
    COUNTED, so the caller can report them; a dropped row is visibly not-measured, never quietly weighted.
    """
    raw = ((a - D) ** 2 * mask).sum(1)
    if not normalise:
        return raw.mean(), 0
    n_real = mask.sum(1, keepdim=True).clamp(min=1.0)
    uni = mask / n_real
    ref = ((uni - D) ** 2 * mask).sum(1)
    keep = ref > 1e-12                       # uniform target -> no shape to supervise toward
    n_dropped = int((~keep).sum())
    if not bool(keep.any()):
        return raw.sum() * 0.0, n_dropped    # whole batch degenerate: contribute nothing, still counted
    return (raw[keep] / ref[keep]).mean(), n_dropped


def shuffle_target(D, mask, generator):
    """Permute each row's target WITHIN its real tokens -- the paper's randomised control.

    In the paper this shuffled version performed WORSE than no supervision at all, which is what makes it
    the right control: it holds the target's marginal distribution fixed and destroys only its ALIGNMENT
    to the tokens, so any gain that survives cannot be explained as generic regularisation from
    constraining the attention."""
    out = torch.zeros_like(D)
    n = mask.sum(1).long()
    for i in range(D.shape[0]):
        t = int(n[i])
        if t > 0:
            perm = torch.randperm(t, generator=generator).to(D.device)
            out[i, :t] = D[i, :t][perm]
    return out


def normalised_entropy(a, mask, eps=1e-9):
    """Per-example attention entropy, normalised to [0,1] by log(T_real).

    Normalised because raw H scales with length, and these datasets span ~2 tokens (sciq) to ~384
    (expertqa); an unnormalised threshold would bind on long examples and never on short ones purely
    as a length artefact. 1.0 = uniform, 0 = all mass on one token. Same convention as
    pool_attention_ood_diag, so the numbers here are comparable to the dissolution work.
    """
    a = a * mask
    H = -(a * torch.log(a + eps)).sum(dim=1)
    T = mask.sum(dim=1).clamp(min=2.0)                 # log(1)=0 would divide by zero
    return H / torch.log(T)


def attention_entropies(model, states, idx, device, bs=64, answer_only=False, prior_list=None):
    """PER-EXAMPLE normalised attention entropy over `idx`, as an array.

    B.3 needs the DISTRIBUTION, not just the mean. A one-sided sharpness penalty can only act on the
    LEFT TAIL, so if a dataset has no meaningful mass below the threshold the penalty has nothing to
    act on -- and "there is no over-sharp subpopulation to fix" is a cleaner and more informative
    answer than "the penalty acted and did not help"."""
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            X, mask, pos = pad_batch([states[i] for i in idx[b:b + bs]], device)
            if answer_only:
                mask = _mask_answer_only(mask)
            # The PRIOR ARMS need their prior here. Arms C and D only produce their real attention when
            # the prior is supplied; calling the model without it measures the entropy of a DIFFERENT
            # (untilted) distribution and reports it as the arm's. Same defect that was fixed in
            # head_attention_correlation. None = arms A/B, unchanged.
            prior_b = (pad_prior([prior_list[i] for i in idx[b:b + bs]], X.shape[1], device)
                       if prior_list is not None else None)
            _l, a = model(X, mask, pos, prior=prior_b)
            if a.dim() == 3:                            # multi-head: average over heads
                a = a.mean(dim=2)
            out.append(normalised_entropy(a, mask).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def mean_attention_entropy(model, states, idx, device, bs=64, answer_only=False, prior_list=None):
    """Mean normalised attention entropy over `idx`. B.3 reports this PER ARM, so we can tell whether
    the penalty actually did what it claims INDEPENDENTLY of whether PRR moved -- a null is only
    interpretable if we know the constraint bound."""
    e = attention_entropies(model, states, idx, device, bs=bs, answer_only=answer_only,
                            prior_list=prior_list)
    return float(e.mean()) if len(e) else 0.0


def head_attention_correlation(model, states, idx, device, bs=64, answer_only=False, head_priors=None):
    """B.2's PRIMARY diagnostic: do the K attention heads actually differ after training?

    ⚠️ This, not PRR, is the question. We already measured the heads collapsing to pairwise correlation
    0.996–1.000, which made multi-head indistinguishable from a control using K classifiers on ONE
    attention pattern. Selective supervision is supposed to prevent that collapse STRUCTURALLY. **If it
    does not, the PRR is uninformative and the whole multi-head line closes with it** — a PRR difference
    between arms that have identical attention is a difference in the classifier, not the aggregation.

    Returns (mean_offdiag_corr, full KxK matrix). Correlation is computed per example over its REAL
    tokens, then averaged over examples, so padding cannot inflate agreement and a long example does not
    dominate a short one.
    """
    model.eval()
    K = model.n_query
    if K < 2:
        return float("nan"), np.full((K, K), np.nan)
    acc, n = np.zeros((K, K)), 0
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            sub = [states[i] for i in idx[b:b + bs]]
            X, mask, pos = pad_batch(sub, device)
            if answer_only:
                mask = _mask_answer_only(mask)
            # S7: a pooler with frozen recipe heads REQUIRES its per-head prior, so the diagnostic must
            # pass the same recipes the model was trained with. Calling it without them raised, which
            # would have taken out the one measurement that decides whether the heads really differ.
            sub_idx = idx[b:b + bs]
            prior_b = (pad_head_priors(head_priors, sub_idx, X.shape[1], device)
                       if head_priors is not None else None)
            _logit, a = model(X, mask, pos, prior=prior_b)      # (B, T, K)
            a = a.detach().cpu().numpy(); m = mask.detach().cpu().numpy().astype(bool)
            for j in range(a.shape[0]):
                real = m[j]
                if real.sum() < 3:                              # too short for a meaningful correlation
                    continue
                W = a[j][real].T                                # (K, T_real)
                if np.allclose(W.std(axis=1), 0):               # a degenerate (uniform) head
                    continue
                acc += np.corrcoef(W); n += 1
    if n == 0:
        return float("nan"), np.full((K, K), np.nan)
    C = acc / n
    off = C[~np.eye(K, dtype=bool)]
    return float(off.mean()), C


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


def _make_head(d, head_hidden):
    """Classifier head on the pooled vector.

    head_hidden=None -> a single nn.Linear(d, 1), BYTE-IDENTICAL to the original pooler head (same
      module, same RNG draw), so every existing caller (arms A/B/D, multi-head) is unchanged.
    head_hidden=(256,128,64) -> the SAPLMA MLP stack Linear->ReLU->...->Linear(.,1), mirroring
      luq.probe.train_probe_mlp's build (probe.py). This lets the head axis of the 2x2 be a pure head
      swap trained by the SAME train_attn recipe as the linear-head cells (head_aggregation_2x2.py).
    All params are UN-prefixed by 'q', so train_attn's param-group split routes them to the decayed
      (wd) group exactly like the linear head -- the query stays the only un-decayed param.
    """
    if head_hidden is None:
        return nn.Linear(d, 1)
    layers = []
    prev = d
    for h in head_hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, 1)]     # final bare logit (sigmoid applied at scoring time)
    return nn.Sequential(*layers)


class AttnPool(nn.Module):
    """One learned-query softmax-attention step over token states, then a linear head.

    temperature  divides the scores before softmax (peakedness knob).
    use_position  adds a learned bias from the positional features to the scores.
    answer_only   handled by the caller via the mask (it zeros the last-prompt-token column), so the
                  module stays a pure pooling step.
    forward returns (logit, attention_weights) so the weights can be inspected.
    """

    def __init__(self, d, temperature=1.0, use_position=False, freeze_query=False,
                 frozen_prior=False, beta=1.0, n_query=1, n_head=1, head_hidden=None,
                 frozen_heads=()):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(d))       # primary learned query (init 0 => starts at mean-pool)
        if freeze_query or frozen_prior:
            self.q.requires_grad_(False)            # q stays 0 (uniform), or is unused (frozen prior)
        self.head = _make_head(d, head_hidden)      # primary head (None -> Linear(d,1); tuple -> SAPLMA MLP)
        self.scale = d ** 0.5
        self.temperature = temperature
        self.use_position = use_position
        self.pos = nn.Linear(2, 1) if use_position else None
        # S3 (prior-init pooling): a per-token PRIOR weight can replace / seed the learned query.
        #   frozen_prior=True  -> attention IS the renormalised prior; head-only training (arm C, ours).
        #   frozen_prior=False + prior given -> CONSTANT additive log-prior: scores = X@q + beta*log(prior)
        #     (arm D, Joe's "start from a distribution, then learn away"; beta scales the prior's pull).
        #     ⚠️ beta is a CONSTANT, not a schedule -- this used to be described as "annealed", which is
        #     wrong and was corrected 2026-08-06. `q` is initialised to ZEROS, so at step 0 the attention is
        #     exactly softmax(beta*log(prior)) = the renormalised prior at beta=1, i.e. arm D starts exactly
        #     where arm C sits. It "learns away" because `q` moves and X@q grows to overwhelm a FIXED tilt,
        #     not because beta decays. (The only real schedule in this file is `aux_drop_epoch`, on the
        #     auxiliary loss, which is a different mechanism.)
        self.frozen_prior = frozen_prior
        self.beta = beta
        # S6 (Joe idea 2 — multi-head): ADDITIONAL queries/heads beyond the primary, created ONLY when K>1, so
        # the default single-head pooler (n_query=n_head=1) is BYTE-IDENTICAL to before — same self.q/self.head,
        # same forward path, same RNG draw order -> arm A reproduced <1e-6. Extra queries init small-random (not
        # 0) to break symmetry so the heads CAN diverge (else all-zero queries share a gradient and collapse by
        # construction, which would fake Joe's "do they converge" test).
        self.n_query = n_query
        self.n_head = n_head
        self.q_rest = nn.Parameter(torch.randn(n_query - 1, d) * 0.02) if n_query > 1 else None
        self.heads_rest = nn.ModuleList([_make_head(d, head_hidden) for _ in range(n_head - 1)]) if n_head > 1 else None
        # S7 (Joe ideas 3+4 — heads with DIFFERENT FIXED recipes): indices of query heads whose attention
        # IS a supplied recipe rather than a learned query. This is the whole point of the arm: B.2 measured
        # heads that share a condition collapsing to pairwise correlation 1.0000, and recipes CANNOT collapse
        # into each other because what makes them differ is not learned. Empty tuple => nothing changes and
        # every pre-S7 caller (arms A/B/C/D, the B.2 multi-head sweep) takes exactly its old path.
        self.frozen_heads = tuple(sorted(set(int(h) for h in frozen_heads)))
        if self.frozen_heads and (min(self.frozen_heads) < 0 or max(self.frozen_heads) >= n_query):
            raise ValueError(f"frozen_heads={self.frozen_heads} outside 0..{n_query - 1}")

    def forward(self, X, mask, posfeat, prior=None):
        # arm C (frozen prior): attention IS the renormalised prior over real tokens; the query is unused.
        if prior is not None and self.frozen_prior:
            a = prior * mask                                        # zero the padded/masked tokens
            a = a / a.sum(dim=1, keepdim=True).clamp(min=1e-9)      # renormalise over real tokens
            pooled = (a.unsqueeze(-1) * X).sum(dim=1)
            return self.head(pooled).squeeze(-1), a
        # SINGLE-HEAD fast path (arms A/B/D, K=1) — unchanged, bit-identical to the pre-S6 pooler.
        if self.n_query == 1 and self.n_head == 1:
            scores = (X @ self.q) / (self.scale * self.temperature)     # (B, T)
            if self.use_position:
                scores = scores + self.pos(posfeat).squeeze(-1)
            if prior is not None:                                       # arm D: CONSTANT additive log-prior tilt
                scores = scores + self.beta * torch.log(prior.clamp(min=1e-9))
            scores = scores.masked_fill(mask == 0, float("-inf"))
            a = torch.softmax(scores, dim=1)                            # (B, T)
            pooled = (a.unsqueeze(-1) * X).sum(dim=1)                    # (B, d)
            return self.head(pooled).squeeze(-1), a
        # MULTI-HEAD path (S6): Q queries -> Q attention distributions -> Q pooled vectors -> H classifier heads,
        # each ensembled. MH: Q=H=K, head k reads query k's pooled. ABLATION: Q=1, H=K, all heads share one
        # attention/pooled (isolates "more classifiers" from "attention diversity" — Joe's mandatory control).
        q_all = self.q.unsqueeze(0) if self.n_query == 1 else torch.cat([self.q.unsqueeze(0), self.q_rest], 0)
        scores = (X @ q_all.t()) / (self.scale * self.temperature)     # (B, T, Q)
        if self.use_position:
            scores = scores + self.pos(posfeat)                         # (B, T, 1) broadcast over Q
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        a = torch.softmax(scores, dim=1)                                # (B, T, Q)
        # S7 — substitute the FIXED RECIPE for the learned attention on the frozen heads. `prior` is
        # (B, T, Q) here, one recipe per query head, and only the columns named in `frozen_heads` are
        # replaced; the rest keep their learned query, so a mixed "3 recipes + 1 free head" pooler is
        # expressible. The replaced columns never enter the loss through `scores`, so their rows of
        # `q_rest` receive no gradient -- frozen without needing per-row requires_grad.
        # `normalise_target` is reused deliberately: it RAISES on a zero-mass row instead of clamping to a
        # near-zero denominator. A clamp here would hand a recipe that selected nothing an all-but-zero
        # attention row and a zero pooled vector, i.e. a plausible number in place of an absence.
        if self.frozen_heads:
            if prior is None:
                raise ValueError("frozen_heads set but no prior supplied — the recipe heads have no recipe")
            if prior.dim() != 3 or prior.shape[2] != a.shape[2]:
                raise ValueError(f"per-head prior must be (B, T, Q={a.shape[2]}), got {tuple(prior.shape)}")
            cols = [normalise_target(prior[:, :, q], mask) if q in self.frozen_heads else a[:, :, q]
                    for q in range(a.shape[2])]
            a = torch.stack(cols, dim=2)
        pooled = torch.einsum("btq,btd->bqd", a, X)                     # (B, Q, d)
        heads_all = [self.head] + (list(self.heads_rest) if self.heads_rest is not None else [])
        logits = []
        for k in range(self.n_head):
            src = pooled[:, k] if self.n_query > 1 else pooled[:, 0]     # MH: own query; ABLATION: shared pooled
            logits.append(heads_all[k](src).squeeze(-1))
        return torch.stack(logits, dim=1), a                           # (B, H), (B, T, Q)


def _mask_answer_only(mask):
    """Zero the first column (the last-prompt token), so pooling is over answer tokens only."""
    m = mask.clone()
    m[:, 0] = 0.0
    return m


def train_attn(states, y, tr_idx, device, seed=SEED, temperature=1.0, use_position=False,
               answer_only=False, weight_decay=WEIGHT_DECAY, wd_query=0.0, freeze_query=False,
               epochs=60, bs=32, lr=1e-3, shrink_lambda=0.0,
               prior_list=None, frozen_prior=False, beta=1.0, n_query=1, n_head=1,
               aux_target=None, aux_lambda=0.0, aux_drop_epoch=None, aux_shuffle=False,
               aux_heads=None, aux_normalise=False, aux_dropped=None,
               ent_lambda=0.0, ent_threshold=0.7, head_hidden=None,
               head_priors=None, frozen_heads=()):
    """`n_query`/`n_head` (S6 multi-head, both default 1 = the single-head pooler, unchanged): MH = n_query=n_head=K
    (K queries, K heads, ensembled by mean-of-sigmoids); ABLATION = n_query=1, n_head=K (one attention, K heads)."""
    """`prior_list` (S3) = per-example prior weight vectors aligned to `states` (length G+1 each). With
    frozen_prior=True the attention IS the renormalised prior and only the head trains (arm C); with
    frozen_prior=False it is a CONSTANT additive log-prior tilt on the learned query (arm D; beta does
    NOT decay -- see the note in __init__). prior_list=None
    keeps the plain learned/frozen-query pooler (arms A/B) unchanged."""
    """`shrink_lambda` > 0 = H5 (shrink-the-pooler): add a shrink-to-uniform penalty on the attention weights
    to the BCE loss, pulling the learned attention toward mean-pool (its unsupervised prior). Tests whether
    moderation-toward-the-unsupervised-prior — the mechanism that helps weighted-MSP — makes the ATTENTION
    POOLER OOD-robust too. Penalty = mean over real tokens of (a_i·n − 1)² (the same avg-1 shrink as wMSP).
    Default 0.0 = the plain pooler (unchanged), so existing callers are untouched."""
    """`aux_target` / `aux_lambda` / `aux_drop_epoch` / `aux_shuffle` (B.1 — auxiliary-loss attention
    supervision, Stacey/Belinkov/Rei AAAI-22, adapted). Adds the paper's term to the LOSS:

        L_total = L_task + (λ/H) · Σ_h Σ_i (a_hi − d_i)²

    ⚠️ This is a different mechanism from anything already here. `prior_list` (S3) modifies the attention
    SCORE — the attention is pushed, and when the push is removed it jumps back. This trains the attention
    to MATCH the target, so it moves away smoothly when the constraint lifts.

    MSE, not KL: the paper tested both and found MSE better for supervising attention (their §related
    work, vs Pruthi et al. 2020).

    `aux_target` = per-example target vectors aligned to `states` (length G+1 each, same convention as
    `prior_list`), normalised here to a distribution over real tokens. `aux_lambda` is the paper's λ (they
    swept [0.2, 1.8] step 0.2 and found 1.0 best for BERT, 0.8 for DeBERTa). `aux_drop_epoch=N` runs the
    'strong at the start, removed after N epochs' schedule; None keeps λ on throughout (the 'low weight
    ~0.1 all the way' variant). `aux_shuffle=True` is the MANDATORY control — the same target permuted
    within each example. Default aux_lambda=0.0 leaves every existing caller byte-identical."""
    torch.manual_seed(seed)
    d = states[0].shape[1]
    model = AttnPool(d, temperature=temperature, use_position=use_position,
                     freeze_query=freeze_query or frozen_prior,
                     frozen_prior=frozen_prior, beta=beta, n_query=n_query, n_head=n_head,
                     head_hidden=head_hidden, frozen_heads=frozen_heads).to(device)
    # S7: `head_priors` (one recipe per query head, None = free head) is the multi-head sibling of
    # `prior_list`. Passing both is a wiring mistake, not a valid configuration -- `prior_list` is the
    # single-head (B, T) convention and would be silently ignored on the multi-head path.
    if head_priors is not None:
        if prior_list is not None:
            raise SystemExit("train_attn: pass head_priors OR prior_list, not both")
        if len(head_priors) != n_query:
            raise SystemExit(f"head_priors has {len(head_priors)} entries for n_query={n_query}")
        for h in frozen_heads:
            if head_priors[int(h)] is None:
                raise SystemExit(f"head {h} is in frozen_heads but its recipe is None")
    # Separate param groups: the heads (and positional bias) get weight decay against p >> n, but the
    # queries are left UN-decayed (wd_query=0) so they can actually move off zero -- with decay on them, the
    # query collapsed to 0 and the attention stayed uniform (the first run's diagnostic showed this).
    # `q`-prefixed = the query params (q + the S6 extra queries q_rest); everything else = heads + pos.
    head_params = [p for n, p in model.named_parameters() if not n.startswith("q")]
    query_params = [p for n, p in model.named_parameters() if n.startswith("q")]
    groups = [{"params": head_params, "weight_decay": weight_decay}]
    if not (freeze_query or frozen_prior):
        groups.append({"params": query_params, "weight_decay": wd_query})
    opt = torch.optim.Adam(groups, lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    g = torch.Generator().manual_seed(seed)
    # separate generator for the shuffled-target control, so turning the control on cannot perturb the
    # batch order and thereby change the run for a reason unrelated to the shuffling
    g_shuf = torch.Generator().manual_seed(seed + 10_000)
    # Mutable one-slot counter so the degenerate-target rows the penalty DROPS are visible to the caller
    # (a dropped row must read as not-measured, never as measured-and-zero). Caller may pass its own list.
    if aux_dropped is None:
        aux_dropped = [0]
    if aux_lambda > 0 and aux_target is None:
        raise SystemExit("aux_lambda > 0 with aux_target=None: refusing to run a 'supervised' arm with no "
                         "target, which would silently be the unsupervised baseline under another name.")
    for ep in range(epochs):
        perm = torch.randperm(len(tr_idx), generator=g).tolist()
        for b in range(0, len(perm), bs):
            idx = [tr_idx[perm[j]] for j in range(b, min(b + bs, len(perm)))]
            X, mask, pos = pad_batch([states[i] for i in idx], device)
            if answer_only:
                mask = _mask_answer_only(mask)
            prior_b = (pad_prior([prior_list[i] for i in idx], X.shape[1], device)
                       if prior_list is not None else None)
            if head_priors is not None:                 # S7: (B, T, Q) per-head recipes
                prior_b = pad_head_priors(head_priors, idx, X.shape[1], device)
            opt.zero_grad()
            logit, a = model(X, mask, pos, prior=prior_b)
            if logit.dim() == 2:                        # S6 multi-head (B,H): mean of the per-head BCE losses
                loss = sum(lossf(logit[:, k], yt[idx]) for k in range(logit.shape[1])) / logit.shape[1]
                # B.2 — SELECTIVE head supervision. The paper supervised 3 of 12 heads and found that
                # supervising ALL of them was WORSE than a subset: "supervising all of them in the same
                # direction can potentially have adverse effects... allowing for diversity between the
                # roles of the supervised and unsupervised heads." So the diversity is meant to be
                # STRUCTURAL -- a supervised minority and a free majority -- rather than hoped for.
                # a is (B, T, Q) here; supervise only the first J query heads and leave the rest free.
                if aux_lambda > 0 and (aux_drop_epoch is None or ep < aux_drop_epoch):
                    J = a.shape[2] if aux_heads is None else int(aux_heads)
                    if not 0 <= J <= a.shape[2]:
                        raise SystemExit(f"aux_heads={J} outside 0..{a.shape[2]} query heads")
                    if J > 0:
                        D = pad_prior([aux_target[i] for i in idx], X.shape[1], device)
                        D = normalise_target(D, mask)
                        if aux_shuffle:
                            D = shuffle_target(D, mask, g_shuf)
                        # (λ/J)·Σ_h Σ_i (a_hi − d_i)², i.e. the paper's form with H = heads SUPERVISED
                        terms = [aux_penalty(a[:, :, h], D, mask, normalise=aux_normalise) for h in range(J)]
                        aux_dropped[0] += sum(t[1] for t in terms)
                        pen = sum(t[0] for t in terms) / J
                        loss = loss + aux_lambda * pen
            else:
                loss = lossf(logit, yt[idx])
                if shrink_lambda > 0:                   # H5: shrink attention toward mean-pool (avg-1 MSE; single-head)
                    n_real = mask.sum(1, keepdim=True).clamp(min=1.0)      # (B,1) real-token count
                    dev = ((a * n_real - 1.0) ** 2) * mask                 # only real tokens contribute
                    loss = loss + shrink_lambda * (dev.sum(1) / n_real.squeeze(1)).mean()
                # B.3: ONE-SIDED entropy penalty. Penalise ONLY when the attention is too SHARP, and
                # leave already-broad distributions completely untouched.
                #
                # ⚠️ WHY THIS IS A DIFFERENT TEST, NOT A THIRD VARIANT OF B.1/B.2. Those supervised the
                # attention toward a TARGET, and both found the target carries no information (real ≈
                # shuffled on every cell). This constrains a PROPERTY of the distribution and uses no
                # target at all, so the "a shuffled target does just as well" failure mode structurally
                # cannot arise here.
                #
                # The hinge is the whole point: penalty = relu(τ − H_norm)², which is EXACTLY ZERO for
                # any example already at or above the threshold. A penalty that binds everywhere is just
                # the two-sided entropy control we already ran and found PRR-neutral, so `ent_frac_bound`
                # is reported to prove the one-sidedness is real rather than nominal.
                if ent_lambda > 0:
                    Hn = normalised_entropy(a, mask)
                    viol = torch.relu(ent_threshold - Hn)      # 0 wherever the attention is broad enough
                    loss = loss + ent_lambda * (viol ** 2).mean()
                # B.1: auxiliary supervision toward a target attention distribution. Active only while the
                # schedule says so -- aux_drop_epoch=N is the paper-adjacent "strong then removed" variant
                # (Joe's suggestion at the 31 July meeting; the paper itself has no annealing schedule).
                if aux_lambda > 0 and (aux_drop_epoch is None or ep < aux_drop_epoch):
                    D = pad_prior([aux_target[i] for i in idx], X.shape[1], device)
                    D = normalise_target(D, mask)
                    if aux_shuffle:
                        D = shuffle_target(D, mask, g_shuf)
                    # (λ/H)·Σ_i (a_i − d_i)², H=1 on this single-head path; summed over tokens, mean over
                    # the batch. Masked so padding contributes nothing. aux_normalise=True divides by the
                    # same quantity at uniform attention (see aux_penalty) so λ is comparable across
                    # dataset length and target density.
                    pen, nd = aux_penalty(a, D, mask, normalise=aux_normalise)
                    aux_dropped[0] += nd
                    loss = loss + aux_lambda * pen
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
            p = torch.sigmoid(logit)
            if p.dim() == 2:                     # S6 multi-head: ensemble by mean-of-sigmoids
                p = p.mean(dim=1)
            preds[b: b + len(idx)] = p.cpu().numpy()
    unc = 1.0 - preds
    return results.prr([y[i] for i in te_idx], unc)


def select_temperature(states, y, tr_idx, device, seed, use_position, answer_only, head_hidden=None):
    """Carve a validation split from train (never test), train at each temperature, pick the one with
    the best validation PRR. Returns (best_T, [(T, val_prr), ...]).

    head_hidden is passed straight to train_attn so T is selected FOR THE HEAD ARCHITECTURE IN USE
    (armA's protocol is "select T for this architecture", not "reuse the linear-head T"). Default None
    keeps the existing linear-head callers byte-identical."""
    g = np.random.RandomState(seed)
    order = list(tr_idx)
    g.shuffle(order)
    n_val = int(len(order) * VAL_FRAC)
    val_idx, sub_tr = order[:n_val], order[n_val:]
    curve = []
    for T in TEMP_GRID:
        m = train_attn(states, y, sub_tr, device, seed=seed, temperature=T,
                       use_position=use_position, answer_only=answer_only, epochs=40,
                       head_hidden=head_hidden)
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
