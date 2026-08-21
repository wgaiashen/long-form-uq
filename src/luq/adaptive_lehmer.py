"""Per-response adaptive Lehmer aggregation — W8 (prereg/adaptive_lehmer_aggregation.md).

THE IDEA IN ONE LINE: perplexity (mean NLL) and msp_min (max NLL) are the two ends of one
concentration axis; instead of fixing the concentration, a tiny gate predicts ONE scalar
beta_i >= 0 per response, and the score is always the Lehmer mean of that response's token NLLs:

    U_i(beta_i) = sum_t l_it^(beta_i + 1) / sum_t l_it^beta_i        l_it = token NLLs

    beta = 0  -> mean NLL (perplexity's ranking)        beta -> inf -> max NLL (msp_min's ranking)

The constraint that makes this a probability method rather than another probe: the gate's inputs
change ONLY beta. If all token NLLs were equal, no beta could change the score — hidden states
cannot inject a correctness estimate directly, they can only choose how concentrated the
probability evidence is aggregated.

FOUR GATES, sharing the identical scoring function (B2 of the plan):
    GLOBAL     one learned scalar g for the whole training cell (the honest alternative to
               test-selecting a beta);
    NLLSHAPE   g = linear(predeclared 10-dim NLL-shape vector), standardised on TRAIN stats only;
    HS         g = linear(mean-pooled response hidden state, canonical middle layer);
    HYBRID     g = g_h(hidden) + g_n(nll-shape) + b, the two contributions logged separately.
Everywhere beta_i = BETA_MAX * sigmoid(g_i), BETA_MAX = 16 (the top of the canonical finite
Lehmer grid; fixed in advance, never tuned on results).

TRAINING mirrors the canonical weighted-MSP recipe verbatim (weighted_msp.train_weighted_msp:
AdamW lr=1e-3, 5 epochs, batch 32, the pairwise sigmoid soft-rank MSE against 1-y, batches < 2
skipped, torch.manual_seed(seed)). Only the gate parameters train; the LLM is frozen; there is
no validation sweep and no new hyperparameter.

NUMERICS: log-space. With m = max(l), log U = m + logsumexp((b+1)(log l - relative)) -
logsumexp(b(log l - relative)) computed via torch.logsumexp on b*log(l) and (b+1)*log(l) shifted
by their own maxima. Zero NLLs (prob-1.0 tokens) are floored at NLL_FLOOR before the power —
recorded here once: l^b with l = 0 and b = 0 is ill-defined exactly at the perplexity endpoint,
and the floor (1e-12) changes no ranking at any beta (a token the model was certain of carries
~0 mass at every beta).

The shuffled-hidden control (B6.4) lives in the driver: it permutes hidden vectors WITHIN each
source dataset with a seed derived from the training seed, once per run (never per epoch).
"""
from __future__ import annotations

import numpy as np
import torch

BETA_MAX = 16.0
NLL_FLOOR = 1e-12

# The predeclared NLL-shape feature vector (B2.2). Order is FIXED — the standardiser and every
# trained gate depend on it; do not reorder, insert, or append after any PRR exists.
NLL_SHAPE_FEATURES = [
    "log_T", "mean_nll", "std_nll", "max_nll", "max_z",
    "top1_mass_share", "top5_mass_share", "top10pct_mass_share",
    "nll_entropy_norm", "max_minus_second",
]
EPS = 1e-12


def nll_shape_vector(nll: np.ndarray) -> np.ndarray:
    """The 10 predeclared features for one response. Mirrors aggregation_regime_rows.features
    (same definitions, same edge policy) but returns a dense vector with NaN-free entries:
    undefined values (T == 1 second-max, zero-sum shares) become 0.0 HERE ONLY because a linear
    gate cannot consume NaN — the flag for those cases is log_T = 0 / the zero std, and the A2
    audit table (where absence must stay visible) is the reporting surface, not this vector."""
    T = len(nll)
    s = float(nll.sum())
    srt = np.sort(nll)[::-1]
    mx = float(srt[0])
    mean, std = float(nll.mean()), float(nll.std())
    if s > 0:
        p = nll / s
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        top1, top5 = mx / s, float(srt[:min(5, T)].sum()) / s
        top10 = float(srt[:max(1, int(np.ceil(0.10 * T)))].sum()) / s
        ent_norm = ent / np.log(T) if T > 1 else 0.0
    else:
        top1 = top5 = top10 = ent_norm = 0.0
    return np.array([
        np.log(T), mean, std, mx, (mx - mean) / (std + EPS),
        top1, top5, top10, ent_norm,
        (mx - float(srt[1])) if T > 1 else 0.0,
    ], dtype=np.float32)


def lehmer_score_torch(log_l: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """log-space Lehmer mean for ONE response. log_l: (T,) log of floored NLLs; beta: scalar
    tensor (differentiable). Returns U (not log U) as a tensor."""
    a = beta * log_l
    b = (beta + 1.0) * log_l
    return torch.exp(torch.logsumexp(b, dim=0) - torch.logsumexp(a, dim=0))


def lehmer_score_np(nll: np.ndarray, beta: float) -> float:
    """Non-differentiable convenience twin (same floor, same log-space path)."""
    log_l = np.log(np.maximum(np.asarray(nll, dtype=np.float64), NLL_FLOOR))
    if np.isinf(beta):
        return float(np.max(nll))
    a, b = beta * log_l, (beta + 1.0) * log_l
    am, bm = a.max(), b.max()
    return float(np.exp(bm + np.log(np.exp(b - bm).sum()) - am - np.log(np.exp(a - am).sum())))


class AdaptiveLehmerGate(torch.nn.Module):
    """One of the four B2 gates. `mode` in {global, nllshape, hs, hybrid}.

    hs / hybrid consume the canonical mean-pooled hidden state (d_hidden); nllshape / hybrid
    consume the standardised 10-dim shape vector. Single linear layers to one scalar each —
    no hidden layers, by design (the gate must stay too small to become a probe)."""

    def __init__(self, mode: str, d_hidden: int = 0):
        super().__init__()
        assert mode in ("global", "nllshape", "hs", "hybrid"), mode
        self.mode = mode
        self.bias = torch.nn.Parameter(torch.zeros(1))
        if mode in ("nllshape", "hybrid"):
            self.lin_n = torch.nn.Linear(len(NLL_SHAPE_FEATURES), 1, bias=False)
            torch.nn.init.zeros_(self.lin_n.weight)      # start at the constant-gate control
        if mode in ("hs", "hybrid"):
            assert d_hidden > 0, "hs/hybrid need the hidden-state dimension"
            self.lin_h = torch.nn.Linear(d_hidden, 1, bias=False)
            torch.nn.init.zeros_(self.lin_h.weight)

    def g_parts(self, h: torch.Tensor | None, x: torch.Tensor | None):
        """(g_h, g_n, bias) — the separately-logged contributions (B7 diagnostics)."""
        g_h = self.lin_h(h).squeeze(-1) if self.mode in ("hs", "hybrid") else None
        g_n = self.lin_n(x).squeeze(-1) if self.mode in ("nllshape", "hybrid") else None
        return g_h, g_n, self.bias

    def beta(self, h: torch.Tensor | None, x: torch.Tensor | None,
             n: int | None = None) -> torch.Tensor:
        """`n` sets the batch size when neither input tensor is given (the GLOBAL gate)."""
        g_h, g_n, b = self.g_parts(h, x)
        size = (h.shape[0] if h is not None else
                (x.shape[0] if x is not None else (n if n is not None else 1)))
        g = b.expand(size).clone()
        if g_h is not None:
            g = g + g_h
        if g_n is not None:
            g = g + g_n
        return BETA_MAX * torch.sigmoid(g)


def constant_beta_bias(beta: float) -> float:
    """The bias that makes a zero-weight gate produce exactly `beta` (B6.3 control)."""
    p = beta / BETA_MAX
    return float(np.log(p / (1.0 - p)))


class ShapeStandardiser:
    """Feature-wise (x - mean) / std fit on the TRAINING POOL ONLY (B2.2); applied unchanged to
    test rows. std floors at 1e-6 so a constant training feature contributes 0, loudly not NaN."""

    def __init__(self, X_train: np.ndarray):
        self.mean = X_train.mean(axis=0)
        self.std = np.maximum(X_train.std(axis=0), 1e-6)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std
