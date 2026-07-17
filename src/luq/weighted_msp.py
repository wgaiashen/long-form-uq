"""Weighted MSP: learn a per-token weight on the model's own logprob signal.

WHY THIS METHOD (the motivation, in plain words)
------------------------------------------------
Our OOD result (scripts/checks/msp_floor.py) is that plain MSP -- the model's own token
logprobs -- is the strong, shift-invariant baseline on SHORT-form QA (it even beats every
trained probe once you leave the training task), while the supervised hidden-state probes
(SAPLMA / the attention pooler) win in-distribution but collapse out-of-distribution. The
hidden state is a rich but FRAGILE feature under shift; the logprob is a weak but ROBUST one.

This method sits in between: it keeps the robust quantity being scored -- the per-token
negative log-likelihood (NLL = -logprob, the MSP building block) -- but LEARNS how much each
token should count, via a small probe on the token's hidden state. The sequence score is a
weighted sum of the per-token NLLs:

    q_i = sum_t  w_t * nll_t          (higher q = more uncertain)

with w_t = g(hidden_state_t) a learned per-token weight. Plain MSP is the special case
w_t = 1 (see `weight_mode="constant"`), so the method can only help relative to that floor;
the bet is that a learned weighting extracts signal plain MSP misses while staying closer to
MSP's OOD robustness than a full hidden-state probe.

This is adapted from Joe's `temp_idea_1_msp_probe` (his `MLP_NN` / `MLP` in `msp_probe_uq.py`):
same 4-layer weight probe, same soft-rank training loss, same weight modes. We reimplement it
in our own pipeline (our per-token cache + our records) rather than driving his lm-polygraph
estimator, per Joe's "write it into your own codebase" rule.

ALIGNMENT (the classic bug this guards against)
-----------------------------------------------
Our per-token cache stores the SAPLMA window `[last_prompt_token] + gen_tokens` -- that is
G+1 rows (row 0 is the last PROMPT token). `record["token_logprobs"]` has one value per
GENERATED token -- G values. So we DROP row 0 of the states (`answer_states`) to line the G
weights up 1:1 with the G NLL values. This is the same de-alignment the visualiser does
(`w[1:]`); getting it wrong silently mis-weights every token.

TRAINING TARGET / LOSS
----------------------
The target is incorrectness = 1 - correctness (soft judge label). The loss is a differentiable
soft-rank (Spearman) MSE, ported verbatim from Joe: within each minibatch, rank the predicted
q scores softly and match them to the true incorrectness ranks. PRR is itself a ranking metric,
so a ranking loss is the natural objective (this is NOT torchsort -- it is Joe's hand-rolled
sigmoid pairwise soft rank).
"""
import numpy as np
import torch
import torch.nn as nn

from . import msp
from .weighting import smooth_raw  # shared transform (P1.1 neighbour smoothing)


# --------------------------------------------------------------------------------------
# Per-example inputs, aligned
# --------------------------------------------------------------------------------------

def per_token_nll(record):
    """NLL (-logprob) for each GENERATED token. Length G (one per generated token)."""
    return np.asarray([-lp for lp in record["token_logprobs"]], dtype=np.float32)


def answer_states(state):
    """Drop row 0 (the last-prompt token) so the remaining G rows align 1:1 with the G NLL
    values. The per-token cache window is [last_prompt_token] + gen_tokens (G+1 rows); NLL is
    one value per generated token (G). See the module docstring's ALIGNMENT note."""
    return np.asarray(state[1:], dtype=np.float32)


def build_answer_masks(tok, records):
    """The Orgad exact-answer overlay: per-record 0/1 mask over the G generated tokens, True on the
    exact-answer span (where the gold answer appears in the generation), all-ones fallback when the
    span is not located. This is Joe's idea (his overlay only drew it; here we USE it to restrict the
    weighted-MSP sum). Cheap, CPU-only, no GPU/API -- it reuses our gold-substring locator
    `luq.features.orgad.locate_answer_rows`, which returns per-token-cache ROW indices over the window
    [P-1 : P+G]; since `answer_states`/`per_token_nll` drop the row-0 anchor, cache row r -> token index
    r-1. SHORT-FORM only (single locatable answer span); on long-form / unlocated rows it falls back to
    all tokens, i.e. plain weighted-MSP for that example. Returns (masks, n_located)."""
    from .features import orgad
    masks, n_located = [], 0
    for r in records:
        g = len(r["gen_token_ids"])
        m = np.zeros(g, dtype=np.float32)
        rows, found = orgad.locate_answer_rows(tok, r["prompt_token_ids"], r["gen_token_ids"], r["target"])
        if found:
            for row in rows:
                idx = row - 1                        # cache row (row 0 = anchor) -> token index
                if 0 <= idx < g:
                    m[idx] = 1.0
        if m.sum() == 0:                             # not located / empty -> all tokens (plain weighted-MSP)
            m[:] = 1.0
        else:
            n_located += 1
        masks.append(m)
    return masks, n_located


# --------------------------------------------------------------------------------------
# The learned per-token weighter + the soft-rank loss (both from Joe's msp_probe_uq.py)
# --------------------------------------------------------------------------------------

class TokenWeightMLP(nn.Module):
    """A 4-layer MLP that maps one token's hidden state to one scalar (the SAPLMA architecture
    with a 1-unit head). The scalar is the raw weight before any normalisation."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):                 # x: (T, d) -> (T,)
        return self.net(x).squeeze(-1)


def _soft_rank(q, temperature: float = 1.0):
    """Differentiable soft rank of q within the batch (higher q -> higher rank). Verbatim from
    Joe: rank_i = 1 + sum_j sigmoid((q_i - q_j)/T), excluding j == i."""
    n = q.shape[0]
    diff = q.unsqueeze(1) - q.unsqueeze(0)          # diff[i,j] = q_i - q_j
    mask = 1.0 - torch.eye(n, device=q.device)
    return 1.0 + (torch.sigmoid(diff / temperature) * mask).sum(dim=1)


def _true_rank(incorrectness):
    """Hard 1-indexed ranks from the incorrectness targets (higher incorrectness -> higher rank)."""
    n = incorrectness.shape[0]
    order = torch.argsort(incorrectness)
    ranks = torch.empty(n, dtype=torch.float32, device=incorrectness.device)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float32, device=incorrectness.device)
    return ranks


# --- Blondel et al. 2020 differentiable soft rank (arXiv:2002.08871), the Phase-4 loss upgrade ------
# Joe's `_soft_rank` above is a hand-rolled O(n^2) sigmoid pairwise rank. Blondel's operator is EXACT,
# O(n log n), order-preserving, and has better-behaved gradients -- Joe pointed at it as the lever to
# improve the underwhelming pairwise version. It ships as torchsort.soft_rank. Imported lazily+guarded
# so this module (and the pairwise loss) still work on a node where torchsort is not built.
try:
    import torchsort  # noqa: E402
    _HAVE_TORCHSORT = True
except Exception:  # not installed / extension not built
    torchsort = None
    _HAVE_TORCHSORT = False


def _blondel_soft_rank(q, eps):
    """Blondel differentiable soft rank of a 1-D score vector q (higher q -> higher rank). `eps` is
    torchsort's regularization_strength: smaller -> closer to the true (hard) rank but flatter
    gradients; larger -> smoother but more biased. torchsort ranks along the last dim of a 2-D input,
    so we add and drop a batch axis."""
    if not _HAVE_TORCHSORT:
        raise RuntimeError("torchsort not installed/built -> loss='blondel' unavailable. "
                           "pip install torchsort on a compute node, or use loss='pairwise'.")
    # Run on CPU regardless of q's device: torchsort's CUDA op only exists if the extension was built
    # with nvcc (we force a CPU-only build to avoid RCS's system-CUDA conflicts). The batch is ~32, so
    # the copy is negligible; .cpu()/.to() are differentiable, so gradients still flow back to q.
    r = torchsort.soft_rank(q.cpu().unsqueeze(0), regularization_strength=eps).squeeze(0)
    return r.to(q.device)


def _spearman_loss(soft_r, target_rank):
    """Negative differentiable Spearman = -Pearson(soft_r, target_rank) (Blondel section 6.3).
    Minimising it maximises the rank correlation between the predicted scores and the true
    incorrectness ranks -- exactly the ranking PRR rewards. Both are standardised, so the scale of the
    soft ranks (which `eps` changes) does not matter."""
    sr = soft_r - soft_r.mean()
    tr = target_rank - target_rank.mean()
    return -(sr * tr).sum() / (sr.norm() * tr.norm() + 1e-8)


# --------------------------------------------------------------------------------------
# Weights and the sequence score
# --------------------------------------------------------------------------------------

# Llama-3 reserved/special tokens are ids >= 128000 (all_special_ids = {128000 <|begin_of_text|>,
# 128001 <|end_of_text|>}). They must NOT carry weight: ~70% of xsum/cnn generations END in the EOS
# token, and the learned weighter otherwise concentrates its softmax mass there (a content-free "I'm
# done" token -- clearly visible in the token-weight visualiser). SAR already zeroes special tokens;
# the weighted-MSP weighting now does too. Only the LEARNED modes exclude them; `constant` (= the plain
# MSP floor) is left untouched so the constant==MSP grounding test and the floor stay invariant.
_SPECIAL_ID_MIN = 128000


def content_keep(record):
    """1.0 for content tokens, 0.0 for special/reserved tokens (Llama-3 ids >= 128000), over the G
    generated tokens -- used to drop the trailing <|end_of_text|> from the learned weighting."""
    return np.array([0.0 if int(t) >= _SPECIAL_ID_MIN else 1.0 for t in record["gen_token_ids"]],
                    dtype=np.float32)


def _weights_from_raw(raw, weight_mode: str, keep=None):
    """Turn the MLP's raw per-token scalars into the weights actually used.

    normalised   : softmax over the sequence, times n -> positive, average 1 (a re-weighting
                   of a length-n mean; recovers plain MSP if the softmax is uniform).
    unconstrained: use the raw scalars directly (can be negative).
    constant     : all ones -> plain MSP (the built-in ablation / floor).

    `keep` (optional 0/1 tensor over the G tokens): special-token exclusion. Excluded positions are set
    to -inf BEFORE the softmax (so they get weight 0 AND do not steal softmax mass from content tokens),
    and the average-1 scaling uses the KEPT count. Not applied to `constant` (the floor is invariant).
    """
    n = raw.shape[0]
    if weight_mode == "normalised":
        if keep is None:
            return torch.softmax(raw, dim=0) * n
        n_kept = torch.clamp(keep.sum(), min=1.0)
        masked = raw.masked_fill(keep < 0.5, float("-inf"))
        return torch.softmax(masked, dim=0) * n_kept            # special tokens -> 0; kept average 1
    if weight_mode == "unconstrained":
        return raw if keep is None else raw * keep              # special tokens -> weight 0
    if weight_mode == "constant":
        return torch.ones_like(raw)                             # floor: unchanged (all tokens)
    raise ValueError(f"weight_mode must be normalised|unconstrained|constant, got {weight_mode!r}")


def _seq_q(raw, nll, weight_mode: str, length_normalise: bool, mask=None, smooth_n=0, keep=None,
           return_w=False):
    """One sequence's score q = sum_t w_t * nll_t, divided by length if length_normalise.

    `mask` (optional, the Orgad exact-answer overlay): a 0/1 tensor over the G tokens. When given, the
    sum is restricted to the answer-bearing tokens (non-answer tokens contribute 0), and length
    normalisation divides by the number of answer tokens, not the full length.

    `keep` (optional 0/1 tensor): special-token exclusion (see `_weights_from_raw`). Excluded tokens get
    weight 0 (pre-softmax), and length normalisation divides by the KEPT (content) count.

    `smooth_n` (P1.1 neighbour smoothing): if >1, moving-average the raw per-token logits over a window of
    smooth_n tokens BEFORE the softmax, so the learned weights vary gently (span property, not lone
    spikes). No effect on constant mode. `return_w` also returns the weight vector (for a penalty term).
    Both default to the no-op, so the existing path is unchanged."""
    if smooth_n and smooth_n > 1 and weight_mode != "constant":
        raw = smooth_raw(raw, smooth_n)
    w = _weights_from_raw(raw, weight_mode, keep=keep)
    wn = w * nll
    if mask is not None:
        wn = wn * mask
    q = wn.sum()
    if length_normalise:
        if mask is not None:
            denom = (mask * keep).sum() if keep is not None else mask.sum()
        elif keep is not None:
            denom = keep.sum()
        else:
            denom = torch.as_tensor(float(nll.shape[0]), device=wn.device)
        q = q / torch.clamp(denom, min=1.0)
    return (q, w) if return_w else q


# --------------------------------------------------------------------------------------
# Train / predict
# --------------------------------------------------------------------------------------

def train_weighted_msp(states, records, y, tr_idx, device, *, weight_mode="normalised",
                       length_normalise=True, seed=1, n_epochs=5, batch_size=32, lr=1e-3,
                       loss="pairwise", blondel_eps=0.1, masks=None, smooth_n=0, reg=None, reg_lambda=0.0,
                       exclude_special=True):
    """Learn the token weighter by a ranking loss. `y` is correctness (higher = better); the target is
    incorrectness = 1 - y. Returns the trained model (unused for constant mode).

    `loss` selects the ranking surrogate:
      "pairwise"  Joe's hand-rolled O(n^2) sigmoid soft-rank MSE (the original; the fallback baseline).
      "blondel"   Blondel 2020 differentiable Spearman via torchsort.soft_rank (exact, O(n log n)); the
                  Phase-4 upgrade. `blondel_eps` is torchsort's regularization_strength (start small).

    Defaults follow Joe (AdamW, 5 epochs, batch 32, lr 1e-3, softmax `normalised`, length_normalise
    True). NOTE: which of sum vs length-normalised is better is TASK-DEPENDENT, not a settled win either
    way -- against the judge label plain msp_sum beats perplexity on pubmed (+0.20 vs -0.17) while the two
    tie on short-form (see msp_floor.py). So length_normalise is a knob to sweep, not a fixed truth."""
    torch.manual_seed(seed)
    d = answer_states(states[tr_idx[0]]).shape[1]
    model = TokenWeightMLP(d).to(device)
    # constant mode has no learnable effect on q (weights are all ones), so skip training.
    if weight_mode == "constant":
        return model

    emb = [torch.from_numpy(answer_states(states[i])).to(device) for i in tr_idx]
    nll = [torch.from_numpy(per_token_nll(records[i])).to(device) for i in tr_idx]
    msk = ([torch.from_numpy(masks[i]).to(device) for i in tr_idx] if masks is not None else None)
    kep = ([torch.from_numpy(content_keep(records[i])).to(device) for i in tr_idx]
           if exclude_special else None)
    incorrect = torch.tensor([1.0 - float(y[i]) for i in tr_idx], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    n_seq = len(tr_idx)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(n_seq, generator=g).tolist()
        for b in range(0, n_seq, batch_size):
            batch = perm[b:b + batch_size]
            if len(batch) < 2:
                continue  # a rank loss needs >=2 items to compare (skip a size-1 tail batch)
            use_reg = reg is not None and reg_lambda > 0 and weight_mode != "constant"
            if use_reg:
                # collect the weight vectors too, so a moderation penalty (P1.1a) can be added
                qs, ws = [], []
                for j in batch:
                    qj, wj = _seq_q(model(emb[j]), nll[j], weight_mode, length_normalise,
                                    mask=(msk[j] if msk is not None else None), smooth_n=smooth_n,
                                    keep=(kep[j] if kep is not None else None), return_w=True)
                    qs.append(qj); ws.append(wj)
                q = torch.stack(qs)
                penalty = torch.stack([reg(w) for w in ws]).mean()
            else:
                q = torch.stack([_seq_q(model(emb[j]), nll[j], weight_mode, length_normalise,
                                        mask=(msk[j] if msk is not None else None), smooth_n=smooth_n,
                                        keep=(kep[j] if kep is not None else None))
                                 for j in batch])
                penalty = None
            target = _true_rank(incorrect[batch])
            if loss == "blondel":
                loss_val = _spearman_loss(_blondel_soft_rank(q, blondel_eps), target)
            else:  # "pairwise": Joe's hand-rolled sigmoid soft-rank MSE
                loss_val = ((_soft_rank(q) - target) ** 2).mean()
            if penalty is not None:
                loss_val = loss_val + reg_lambda * penalty      # P1.1a moderation toward uniform/MSP
            opt.zero_grad()
            loss_val.backward()
            opt.step()
    return model


def predict_weighted_msp(model, states, records, idx, device, *, weight_mode="normalised",
                         length_normalise=True, masks=None, smooth_n=0, exclude_special=True):
    """Sequence uncertainty q for each example in `idx` (higher = more uncertain). `smooth_n` (P1.1c) can
    smooth the weights at scoring time even for a model trained without it (post-hoc smoothing)."""
    model.eval()
    out = np.zeros(len(idx))
    with torch.no_grad():
        for k, i in enumerate(idx):
            nll = torch.from_numpy(per_token_nll(records[i])).to(device)
            if weight_mode == "constant":
                raw = torch.zeros_like(nll)          # weights are all ones; raw is ignored
            else:
                raw = model(torch.from_numpy(answer_states(states[i])).to(device))
            m = torch.from_numpy(masks[i]).to(device) if masks is not None else None
            # constant mode = the plain-MSP floor: never exclude tokens (keeps constant==MSP invariant).
            keep = (torch.from_numpy(content_keep(records[i])).to(device)
                    if (exclude_special and weight_mode != "constant") else None)
            out[k] = float(_seq_q(raw, nll, weight_mode, length_normalise, mask=m, smooth_n=smooth_n,
                                  keep=keep).item())
    return out


def weighted_msp_unc(states, records, y, tr_idx, te_idx, device, *, weight_mode="normalised",
                     length_normalise=True, seed=1, loss="pairwise", blondel_eps=0.1, masks=None,
                     smooth_n=0, reg=None, reg_lambda=0.0, exclude_special=True):
    """Train on tr_idx, return test-set uncertainties for te_idx. Ladder-compatible drop-in
    (same shape as attn_pool.attn_unc): higher = more uncertain. `loss` picks the ranking surrogate
    ('pairwise' = Joe's original, 'blondel' = the torchsort soft-rank upgrade). `masks` (optional) is
    the Orgad exact-answer overlay: a per-record 0/1 array over the G tokens restricting the score to
    answer-bearing tokens (build with build_answer_masks).

    P1.1 smoothing knobs (all default to the no-op): `smooth_n` neighbour-smooths the weights (applied at
    both train and score time); `reg` (a penalty(w) from luq.weighting, e.g. shrink_to_uniform) with
    `reg_lambda` moderates the weight distribution toward uniform/MSP during training."""
    model = train_weighted_msp(states, records, y, tr_idx, device, weight_mode=weight_mode,
                               length_normalise=length_normalise, seed=seed, loss=loss,
                               blondel_eps=blondel_eps, masks=masks, smooth_n=smooth_n,
                               reg=reg, reg_lambda=reg_lambda, exclude_special=exclude_special)
    return predict_weighted_msp(model, states, records, te_idx, device,
                                weight_mode=weight_mode, length_normalise=length_normalise, masks=masks,
                                smooth_n=smooth_n, exclude_special=exclude_special)


# --------------------------------------------------------------------------------------
# The grounding check: constant mode MUST equal plain MSP
# --------------------------------------------------------------------------------------

def constant_equals_msp_maxdiff(records, idx, length_normalise: bool) -> float:
    """The `constant` weight mode (all weights 1) must reduce to plain MSP exactly:
    q = sum(nll)  ==  msp 'sum'      (length_normalise False)
    q = mean(nll) ==  msp 'perplexity' (length_normalise True)
    Returns the max abs difference over `idx` -- assert it is ~0 (both derive from the same
    logprobs, so it should be numerically exact)."""
    agg = "perplexity" if length_normalise else "sum"
    diffs = []
    for i in idx:
        nll = per_token_nll(records[i])
        q = float(nll.mean() if length_normalise else nll.sum())
        ref = msp.msp_uncertainty(records[i]["token_logprobs"], agg)
        diffs.append(abs(q - ref))
    return max(diffs) if diffs else 0.0
