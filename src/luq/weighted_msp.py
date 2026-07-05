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


# --------------------------------------------------------------------------------------
# Weights and the sequence score
# --------------------------------------------------------------------------------------

def _weights_from_raw(raw, weight_mode: str):
    """Turn the MLP's raw per-token scalars into the weights actually used.

    normalised   : softmax over the sequence, times n -> positive, average 1 (a re-weighting
                   of a length-n mean; recovers plain MSP if the softmax is uniform).
    unconstrained: use the raw scalars directly (can be negative).
    constant     : all ones -> plain MSP (the built-in ablation / floor).
    """
    n = raw.shape[0]
    if weight_mode == "normalised":
        return torch.softmax(raw, dim=0) * n
    if weight_mode == "unconstrained":
        return raw
    if weight_mode == "constant":
        return torch.ones_like(raw)
    raise ValueError(f"weight_mode must be normalised|unconstrained|constant, got {weight_mode!r}")


def _seq_q(raw, nll, weight_mode: str, length_normalise: bool):
    """One sequence's score q = sum_t w_t * nll_t, divided by length if length_normalise."""
    w = _weights_from_raw(raw, weight_mode)
    q = (w * nll).sum()
    if length_normalise:
        q = q / nll.shape[0]
    return q


# --------------------------------------------------------------------------------------
# Train / predict
# --------------------------------------------------------------------------------------

def train_weighted_msp(states, records, y, tr_idx, device, *, weight_mode="normalised",
                       length_normalise=True, seed=1, n_epochs=5, batch_size=32, lr=1e-3):
    """Learn the token weighter by the soft-rank loss. `y` is correctness (higher = better);
    the target is incorrectness = 1 - y. Returns the trained model (unused for constant mode).

    Defaults follow Joe (AdamW, 5 epochs, batch 32, lr 1e-3, softmax `normalised`, length_normalise
    required -> we default True as the standard convention). NOTE: which of sum vs length-normalised
    is better is TASK-DEPENDENT, not a settled win either way -- against the judge label plain msp_sum
    beats perplexity on pubmed (+0.20 vs -0.17) while the two tie on short-form (see msp_floor.py). So
    length_normalise is a knob to sweep, not a fixed truth; do not assume it helps."""
    torch.manual_seed(seed)
    d = answer_states(states[tr_idx[0]]).shape[1]
    model = TokenWeightMLP(d).to(device)
    # constant mode has no learnable effect on q (weights are all ones), so skip training.
    if weight_mode == "constant":
        return model

    emb = [torch.from_numpy(answer_states(states[i])).to(device) for i in tr_idx]
    nll = [torch.from_numpy(per_token_nll(records[i])).to(device) for i in tr_idx]
    incorrect = torch.tensor([1.0 - float(y[i]) for i in tr_idx], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    n_seq = len(tr_idx)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(n_seq, generator=g).tolist()
        for b in range(0, n_seq, batch_size):
            batch = perm[b:b + batch_size]
            q = torch.stack([_seq_q(model(emb[j]), nll[j], weight_mode, length_normalise)
                             for j in batch])
            loss = ((_soft_rank(q) - _true_rank(incorrect[batch])) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def predict_weighted_msp(model, states, records, idx, device, *, weight_mode="normalised",
                         length_normalise=True):
    """Sequence uncertainty q for each example in `idx` (higher = more uncertain)."""
    model.eval()
    out = np.zeros(len(idx))
    with torch.no_grad():
        for k, i in enumerate(idx):
            nll = torch.from_numpy(per_token_nll(records[i])).to(device)
            if weight_mode == "constant":
                raw = torch.zeros_like(nll)          # weights are all ones; raw is ignored
            else:
                raw = model(torch.from_numpy(answer_states(states[i])).to(device))
            out[k] = float(_seq_q(raw, nll, weight_mode, length_normalise).item())
    return out


def weighted_msp_unc(states, records, y, tr_idx, te_idx, device, *, weight_mode="normalised",
                     length_normalise=True, seed=1):
    """Train on tr_idx, return test-set uncertainties for te_idx. Ladder-compatible drop-in
    (same shape as attn_pool.attn_unc): higher = more uncertain."""
    model = train_weighted_msp(states, records, y, tr_idx, device, weight_mode=weight_mode,
                               length_normalise=length_normalise, seed=seed)
    return predict_weighted_msp(model, states, records, te_idx, device,
                                weight_mode=weight_mode, length_normalise=length_normalise)


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
