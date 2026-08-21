"""Shared token-weighting machinery for BOTH tracks (the plan's §Shared code requirement).

Track A (weighted-MSP) and Track B (the attention pooler / Idea 2) are the same operation
`score = f(Σ_t w_t · x_t)`, differing only in what `w` multiplies (Track A: the scalar NLL; Track B: the
4096-dim hidden state) and what reads the result. So the *shape* of a good weight distribution is
track-agnostic and lives here, imported by both training loops — no regulariser or pool primitive
implemented twice.

CONVENTION: weights are in the **average-1** form (a length-n vector that averages to 1, i.e. sums to n).
Track A's `normalised` mode already produces this (`softmax(raw)·n`); Track B's softmax attention (sums to
1) is converted by ×n at the boundary. Every regulariser/transform here assumes average-1, so the same
`penalty(w)` means the same thing on both tracks. The floor is `w = 1` (uniform) for both: Track A → plain
MSP, Track B → mean-pool.

All torch, so penalties can be added directly to either training loss and gradients flow.
"""
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# The one pooling primitive (both tracks call this; if they were two functions they'd drift)
# --------------------------------------------------------------------------------------

def pool(x, w):
    """Σ_t w_t · x_t, broadcasting over the feature dim. x: [T] (NLLs) or [T, d] (hidden states);
    w: [T]. Returns [] or [d]."""
    if x.dim() == 1:
        return (w * x).sum()
    return (w.unsqueeze(-1) * x).sum(dim=0)


def normalize_avg1(raw):
    """Softmax over the sequence, ×n → positive weights that average to 1 (sum to n). This is exactly
    weighted_msp's `normalised` mode and the average-1 canonical form used by every penalty here."""
    n = raw.shape[0]
    return torch.softmax(raw, dim=0) * n


def _prob(w):
    """The probability distribution p = w / n (sums to 1) implied by an average-1 weight vector."""
    return w / w.sum().clamp_min(1e-12)


# --------------------------------------------------------------------------------------
# Regularisers: penalty(w) of an average-1 distribution, ≥ 0, and 0 at the uniform floor
# --------------------------------------------------------------------------------------

def kl_to_uniform(w):
    """KL(p ‖ uniform) = Σ p log(p / (1/n)) = log n − H(p). ≥ 0, exactly 0 at uniform. Penalising this
    pulls the weight distribution toward uniform (= toward plain MSP / mean-pool)."""
    p = _prob(w).clamp_min(1e-12)
    n = w.shape[0]
    return (p * torch.log(p * n)).sum()


# entropy_penalty is KL-to-uniform up to the constant log n (identical gradient); kept as a named alias so
# a plan reference to "entropy penalty" resolves.
entropy_penalty = kl_to_uniform


def shrink_to_uniform(w):
    """Mean squared deviation from the uniform floor, (1/n) Σ (w_t − 1)². 0 at uniform. This is the
    'shrink toward MSP (Track A) / mean-pool (Track B)' penalty — one function, both floors are w = 1."""
    return ((w - 1.0) ** 2).mean()


def tv_penalty(w):
    """Total variation (1/n) Σ_t |w_t − w_{t−1}|. 0 when the weights are flat; penalises choppy,
    lone-token spikes → gentle peaks/troughs (the neighbour-smoothing idea, as a loss term)."""
    if w.shape[0] < 2:
        return torch.zeros((), device=w.device, dtype=w.dtype)
    return (w[1:] - w[:-1]).abs().sum() / w.shape[0]


def entropy_hinge(w, threshold=0.7):
    """the SPECIFIC idea (item 1): a penalty that fires ONLY when the weight distribution gets too peaked
    — i.e. only when its NORMALISED entropy H(p)/log n drops BELOW `threshold` (1 = uniform floor, 0 = one
    token gets everything). Unlike `kl_to_uniform`/`entropy_penalty` (which push toward uniform on EVERY
    batch), this is a hinge: **0 while the weights stay smooth**, and only pushes back once they cross the
    threshold. Returns relu(threshold − H_norm)², ≥ 0, exactly 0 when H_norm ≥ threshold. Tune the threshold
    with functools.partial(entropy_hinge, threshold=…) when passing as `reg` to train_weighted_msp."""
    n = w.shape[0]
    if n <= 1:
        return torch.zeros((), device=w.device, dtype=w.dtype)
    p = _prob(w).clamp_min(1e-12)
    H = -(p * torch.log(p)).sum()
    log_n = torch.log(torch.tensor(float(n), device=w.device, dtype=w.dtype))
    H_norm = H / log_n                                   # 1 = uniform, 0 = one-hot
    return F.relu(threshold - H_norm) ** 2


REGULARISERS = {"kl_uniform": kl_to_uniform, "entropy": entropy_penalty,
                "shrink": shrink_to_uniform, "tv": tv_penalty, "entropy_hinge": entropy_hinge}


# --------------------------------------------------------------------------------------
# Transforms on raw scores / source weights (shared by P1.1 smoothing and P1.4 β-sharpen)
# --------------------------------------------------------------------------------------

def smooth_raw(raw, k, causal=False):
    """Moving-average of the raw per-token logits over a window of k tokens (the 'smooth over the
    previous n tokens → gentle peaks/troughs'). Replicate-pads the ends so length is preserved. k ≤ 1 is a
    no-op. Applied BEFORE the softmax so it shapes the weight distribution.
    `causal=False` (default): SYMMETRIC/centered window (uses both neighbours). `causal=True`: BACKWARD-only
    window (each token averaged with the k−1 tokens BEFORE it) — the literal 'previous n tokens'."""
    if k <= 1 or raw.numel() <= 1:
        return raw
    T = raw.shape[0]
    pad = k - 1
    x = raw.view(1, 1, -1)
    xp = F.pad(x, (pad, 0) if causal else (pad // 2, pad - pad // 2), mode="replicate")
    ker = torch.ones(1, 1, k, device=raw.device, dtype=raw.dtype) / k
    return F.conv1d(xp, ker).view(-1)[:T]


def beta_sharpen(s, beta):
    """Turn a non-negative SOURCE score vector s (Orgad mask / SAR relevance / CSL attention) into an
    average-1 weight w ∝ s^β. β = 0 → uniform (the floor); β = 1 → proportional; β > 1 → sharper. Used by
    the P1.4 pooling hook so all unsupervised sources share one sharpening knob."""
    s = s.clamp_min(0.0)
    if beta == 0:
        return torch.ones_like(s)
    p = s ** beta
    tot = p.sum()
    if tot <= 0:
        return torch.ones_like(s)
    return p / tot * s.shape[0]


# --------------------------------------------------------------------------------------
# The shared floor-reduction check (one helper, both tracks' limit cases)
# --------------------------------------------------------------------------------------

def reduces_to_uniform(w, tol=1e-6):
    """True if the average-1 weight vector is (numerically) the uniform floor w = 1. The limit both
    tracks must hit as the moderation strength → ∞ (Track A → plain MSP, Track B → mean-pool)."""
    return bool((w - 1.0).abs().max().item() < tol)
