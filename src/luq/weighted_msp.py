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

This is adapted from the `temp_idea_1_msp_probe` (his `MLP_NN` / `MLP` in `msp_probe_uq.py`):
same 4-layer weight probe, same soft-rank training loss, same weight modes. We reimplement it
in our own pipeline (our per-token cache + our records) rather than driving his lm-polygraph
estimator, per the "write it into your own codebase" rule.

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
soft-rank (Spearman) MSE, ported verbatim from the reference implementation: within each minibatch, rank the predicted
q scores softly and match them to the true incorrectness ranks. PRR is itself a ranking metric,
so a ranking loss is the natural objective (this is NOT torchsort -- it is the hand-rolled
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
    span is not located. This is the idea (his overlay only drew it; here we USE it to restrict the
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
# The learned per-token weighter + the soft-rank loss (both from the msp_probe_uq.py)
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
    reference: rank_i = 1 + sum_j sigmoid((q_i - q_j)/T), excluding j == i."""
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
# the `_soft_rank` above is a hand-rolled O(n^2) sigmoid pairwise rank. Blondel's operator is EXACT,
# O(n log n), order-preserving, and has better-behaved gradients -- It was flagged as the lever to
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

# THE >=128000 RULE IS LLAMA-3 ONLY, AND IT IS A SILENT BUG ON ANY OTHER MODEL (2026-08-08).
# Qwen2.5's vocabulary is 152,064 with its specials at 151,643+, so `id >= 128000` would classify a
# large band of ORDINARY CONTENT TOKENS as special and zero their weight -- no crash, no warning,
# just a quietly different method. This feeds `content_keep`, which feeds weighted MSP, which is a
# reported result, so it has to be model-aware.
#
# The mechanism is a module-level set that DEFAULTS TO EXACTLY THE OLD BEHAVIOUR, so no existing
# caller changes and no committed number moves. A driver running a non-Llama model calls
# `set_special_ids(...)` once at startup.
#
# VERIFIED byte-identical on Llama before this landed: scanning all 11 cached record files, the
# ONLY id >= 128000 that ever occurs in a generation is 128001 (<|end_of_text|>, 10,384 occurrences),
# and it IS in `all_special_ids`. So set-membership and the >= test agree on every row we have.
_SPECIAL_IDS = None          # None => fall back to the Llama-3 reserved-range test below


def set_special_ids(ids):
    """Register the tokenizer's special ids for the model being run. Pass `tok.all_special_ids`.

    Call this ONCE at driver startup for any non-Llama model. Passing None restores the Llama-3
    reserved-range default. Kept as module state rather than a new argument because `content_keep`
    is called from several places (including another workstream's `sharpening_wmsp.py`), and
    changing its signature would break them.
    """
    global _SPECIAL_IDS
    _SPECIAL_IDS = None if ids is None else {int(i) for i in ids}


def content_keep(record, special_ids=None):
    """1.0 for content tokens, 0.0 for special tokens, over the G generated tokens.

    Used to drop the trailing end-of-text from the learned weighting: ~70% of xsum/cnn generations
    end in EOS, and the weighter otherwise concentrates its softmax mass on that content-free "I'm
    done" token. Only the LEARNED modes exclude them; `constant` (= the plain MSP floor) is left
    untouched so the constant==MSP grounding test and the floor stay invariant.

    Resolution order: the explicit `special_ids` argument, then whatever `set_special_ids` registered,
    then the Llama-3 `id >= 128000` reserved-range test.
    """
    ids = special_ids if special_ids is not None else _SPECIAL_IDS
    toks = record["gen_token_ids"]
    if ids is None:
        return np.array([0.0 if int(t) >= _SPECIAL_ID_MIN else 1.0 for t in toks], dtype=np.float32)
    ids = {int(i) for i in ids}
    return np.array([0.0 if int(t) in ids else 1.0 for t in toks], dtype=np.float32)


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
        # EVERY-TOKEN-EXCLUDED IS THE NaN CASE (found 2026-08-03 via the asqa wMSP outlier).
        # If keep is all-zero, masked_fill sets EVERY position to -inf and softmax(all -inf) = NaN, so
        # the whole example scores NaN. The clamp above protects the SCALE but not the softmax. Those
        # NaNs then flowed into prr(), which used to rank them arbitrarily and return a plausible number
        # -- asqa's wMSP read -0.0439 for all 8 variants and 3 seeds purely because of this.
        # Falling back to UNIFORM weights over all tokens is the honest choice: with no token judged
        # keepable there is no basis to prefer any, and uniform reduces wMSP to plain MSP for that
        # example rather than inventing a ranking.
        if float(keep.sum()) < 0.5:
            return torch.ones_like(raw)
        masked = raw.masked_fill(keep < 0.5, float("-inf"))
        return torch.softmax(masked, dim=0) * n_kept            # special tokens -> 0; kept average 1
    if weight_mode == "unconstrained":
        return raw if keep is None else raw * keep              # special tokens -> weight 0
    if weight_mode == "constant":
        return torch.ones_like(raw)                             # floor: unchanged (all tokens)
    raise ValueError(f"weight_mode must be normalised|unconstrained|constant, got {weight_mode!r}")


def _segment_mean_raw(raw, sid):
    """Replace each token's raw weight with the MEAN raw of its segment (sentence). All tokens in a
    segment then share one weight after the softmax -> "one learned weight per sentence, broadcast to its
    tokens" (design note 7). `sid` is a long tensor of segment ids over the G tokens. Differentiable (scatter-mean),
    so gradients still flow to the MLP. Reduces to plain per-token weighting when every token is its own
    segment, and to uniform when all tokens share one segment."""
    n_seg = int(sid.max().item()) + 1
    sums = torch.zeros(n_seg, dtype=raw.dtype, device=raw.device).index_add_(0, sid, raw)
    counts = torch.zeros(n_seg, dtype=raw.dtype, device=raw.device).index_add_(0, sid, torch.ones_like(raw))
    seg_mean = sums / torch.clamp(counts, min=1.0)
    return seg_mean[sid]


def _segment_softmax_weights(raw, sid, keep=None):
    """SENTENCE-level softmax weighting (segment_mode='softmax'), the length-confound-free alternative to
    `_segment_mean_raw`.

    THE CONFOUND IT FIXES. The flat segment weight (`_segment_mean_raw` + token softmax) gives sentence s a
    TOTAL weight proportional to len(s) x exp(score_s): softmax is over TOKENS, so a longer sentence with the
    same learned score gets more total mass. In a project where length is the dominant lever on this family
    (PART XV), that baked-in length term is a confound.

    THIS instead: one score per sentence (mean of its tokens' raw) -> softmax OVER SENTENCES -> distribute
    each sentence's mass uniformly across its tokens, length-normalised. So sentence s contributes exactly
    p_s to the (length-normalised) score regardless of how many tokens it has.

        seg_mean_s = mean_{t in s} raw_t ;  p = softmax(seg_mean) over the S sentences (sums to 1)
        w_t = p_s / len(s) * n            (so sum_t w_t = n -> average 1, same convention as the token path)

    TWO exact reduction limits (unit-tested):
      * ONE sentence (whole response)      -> p=[1], w_t = 1/n * n = 1  => plain MSP.
      * EVERY token its own sentence       -> seg_mean = raw, len=1, w = softmax(raw)*n => plain per-token wMSP.
    Under UNIFORM raw it reduces to the sentence-BALANCED mean NLL (each sentence weighted equally), which is
    the intended length-free baseline, not plain MSP.
    """
    n_seg = int(sid.max().item()) + 1
    # `keep` (1 on kept tokens, 0 on excluded specials/EOS) makes special tokens contribute NOTHING to the
    # per-sentence mean, the length count, or the output weight -- the same exclude-special behaviour the flat
    # path gets from _weights_from_raw's -inf masking. Default (all-ones) = every token kept.
    if keep is None:
        keep = torch.ones_like(raw)
    # THE SAME EVERY-TOKEN-EXCLUDED NaN AS THE TOKEN PATH (fixed 2026-08-05). When `keep` is all-zero
    # every segment gets `counts == 0`, so the `torch.where` below sets EVERY seg_mean to -inf and
    # softmax(all -inf) = NaN. That was left unchased when the token-level case was fixed on 2026-08-03,
    # and it is why `wmsp_seg_softmax` alone came back NaN on 17 of its 42 long cells while the other
    # seven wMSP variants were clean. Same honest fallback as the token path: with no token judged
    # keepable there is no basis to prefer any, so weight uniformly (wMSP reduces to plain MSP for that
    # example) rather than emit a NaN that prr() must then refuse.
    if float(keep.sum()) < 0.5:
        return torch.ones_like(raw)
    kraw = raw * keep
    sums = torch.zeros(n_seg, dtype=raw.dtype, device=raw.device).index_add_(0, sid, kraw)
    counts = torch.zeros(n_seg, dtype=raw.dtype, device=raw.device).index_add_(0, sid, keep)  # KEPT count/segment
    seg_mean = sums / torch.clamp(counts, min=1.0)
    # a sentence with zero kept tokens must not steal softmax mass -> push it to -inf before the softmax.
    seg_mean = torch.where(counts > 0, seg_mean, torch.full_like(seg_mean, float("-inf")))
    p = torch.softmax(seg_mean, dim=0)                       # over sentences with >=1 kept token, sums to 1
    n_kept = keep.sum()
    w = (p[sid] / torch.clamp(counts[sid], min=1.0)) * n_kept
    return w * keep                                         # excluded tokens -> weight 0; sum(w) = n_kept


def _seq_q(raw, nll, weight_mode: str, length_normalise: bool, mask=None, smooth_n=0, keep=None,
           segment_ids=None, return_w=False, smooth_causal=False, segment_mode="mean"):
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
        raw = smooth_raw(raw, smooth_n, causal=smooth_causal)   # causal=the literal "previous n tokens"
    if segment_ids is not None and weight_mode != "constant" and segment_mode == "softmax":
        # sentence-softmax path produces FINAL avg-1 weights directly (skip _weights_from_raw). It handles
        # `keep` itself (excludes specials from the per-sentence mean, length count, and output weight).
        w = _segment_softmax_weights(raw, segment_ids, keep=keep)
    else:
        if segment_ids is not None and weight_mode != "constant":
            raw = _segment_mean_raw(raw, segment_ids)      # one weight per sentence, flat (design note 7)
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
                       exclude_special=True, keep=None, segment_ids=None, smooth_causal=False,
                       segment_mode="mean"):
    """Learn the token weighter by a ranking loss. `y` is correctness (higher = better); the target is
    incorrectness = 1 - y. Returns the trained model (unused for constant mode).

    `loss` selects the ranking surrogate:
      "pairwise"  the hand-rolled O(n^2) sigmoid soft-rank MSE (the original; the fallback baseline).
      "blondel"   Blondel 2020 differentiable Spearman via torchsort.soft_rank (exact, O(n log n)); the
                  Phase-4 upgrade. `blondel_eps` is torchsort's regularization_strength (start small).

    Defaults follow the reference implementation (AdamW, 5 epochs, batch 32, lr 1e-3, softmax `normalised`, length_normalise
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
    # keep-mask channel: a caller-supplied per-record subset (design note 4: content/punct-restricted weighting)
    # overrides the default special-token exclusion. Aligned to `records`, indexed by tr_idx.
    if keep is not None:
        kep = [torch.from_numpy(np.asarray(keep[i], dtype=np.float32)).to(device) for i in tr_idx]
    elif exclude_special:
        kep = [torch.from_numpy(content_keep(records[i])).to(device) for i in tr_idx]
    else:
        kep = None
    seg = ([torch.from_numpy(np.asarray(segment_ids[i])).long().to(device) for i in tr_idx]
           if segment_ids is not None else None)
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
                                    keep=(kep[j] if kep is not None else None),
                                    segment_ids=(seg[j] if seg is not None else None), return_w=True,
                                    smooth_causal=smooth_causal, segment_mode=segment_mode)
                    qs.append(qj); ws.append(wj)
                q = torch.stack(qs)
                penalty = torch.stack([reg(w) for w in ws]).mean()
            else:
                q = torch.stack([_seq_q(model(emb[j]), nll[j], weight_mode, length_normalise,
                                        mask=(msk[j] if msk is not None else None), smooth_n=smooth_n,
                                        keep=(kep[j] if kep is not None else None), segment_mode=segment_mode,
                                        segment_ids=(seg[j] if seg is not None else None),
                                        smooth_causal=smooth_causal)
                                 for j in batch])
                penalty = None
            target = _true_rank(incorrect[batch])
            if loss == "blondel":
                loss_val = _spearman_loss(_blondel_soft_rank(q, blondel_eps), target)
            else:  # "pairwise": the hand-rolled sigmoid soft-rank MSE
                loss_val = ((_soft_rank(q) - target) ** 2).mean()
            if penalty is not None:
                loss_val = loss_val + reg_lambda * penalty      # P1.1a moderation toward uniform/MSP
            opt.zero_grad()
            loss_val.backward()
            opt.step()
    return model


def predict_weighted_msp(model, states, records, idx, device, *, weight_mode="normalised",
                         length_normalise=True, masks=None, smooth_n=0, exclude_special=True, keep=None,
                         segment_ids=None, smooth_causal=False, segment_mode="mean"):
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
            if weight_mode == "constant":
                kmask = None
            elif keep is not None:
                kmask = torch.from_numpy(np.asarray(keep[i], dtype=np.float32)).to(device)
            elif exclude_special:
                kmask = torch.from_numpy(content_keep(records[i])).to(device)
            else:
                kmask = None
            smask = (torch.from_numpy(np.asarray(segment_ids[i])).long().to(device)
                     if (segment_ids is not None and weight_mode != "constant") else None)
            out[k] = float(_seq_q(raw, nll, weight_mode, length_normalise, mask=m, smooth_n=smooth_n,
                                  keep=kmask, segment_ids=smask, smooth_causal=smooth_causal,
                                  segment_mode=segment_mode).item())
    return out


def weighted_msp_unc(states, records, y, tr_idx, te_idx, device, *, weight_mode="normalised",
                     length_normalise=True, seed=1, loss="pairwise", blondel_eps=0.1, masks=None,
                     smooth_n=0, reg=None, reg_lambda=0.0, exclude_special=True, keep=None,
                     segment_ids=None, smooth_causal=False, segment_mode="mean"):
    """Train on tr_idx, return test-set uncertainties for te_idx. Ladder-compatible drop-in
    (same shape as attn_pool.attn_unc): higher = more uncertain. `loss` picks the ranking surrogate
    ('pairwise' = the original, 'blondel' = the torchsort soft-rank upgrade). `masks` (optional) is
    the Orgad exact-answer overlay: a per-record 0/1 array over the G tokens restricting the score to
    answer-bearing tokens (build with build_answer_masks).

    P1.1 smoothing knobs (all default to the no-op): `smooth_n` neighbour-smooths the weights (applied at
    both train and score time); `reg` (a penalty(w) from luq.weighting, e.g. shrink_to_uniform) with
    `reg_lambda` moderates the weight distribution toward uniform/MSP during training."""
    model = train_weighted_msp(states, records, y, tr_idx, device, weight_mode=weight_mode,
                               length_normalise=length_normalise, seed=seed, loss=loss,
                               blondel_eps=blondel_eps, masks=masks, smooth_n=smooth_n,
                               reg=reg, reg_lambda=reg_lambda, exclude_special=exclude_special, keep=keep,
                               segment_ids=segment_ids, smooth_causal=smooth_causal,
                               segment_mode=segment_mode)
    return predict_weighted_msp(model, states, records, te_idx, device,
                                weight_mode=weight_mode, length_normalise=length_normalise, masks=masks,
                                smooth_n=smooth_n, exclude_special=exclude_special, keep=keep,
                                segment_ids=segment_ids, smooth_causal=smooth_causal,
                                segment_mode=segment_mode)


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
