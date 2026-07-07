"""Phase-4 sanity: does the Blondel soft-rank loss actually train the token weighter, and does it hold
up against Joe's pairwise fallback? Runs on a SYNTHETIC set with a KNOWN answer, so we can check
recovery, not just that the loss decreases.

Construction: each sequence has T tokens with random hidden states h_t and per-token NLLs. A FIXED
hidden direction v defines the TRUE per-token weight w*_t = softplus(v . h_t); the true sequence score
is q* = mean_t(w*_t * nll_t), and correctness is a monotone-decreasing function of q* plus label noise.
So a weighter that recovers v ranks sequences by q* -- and Spearman(predicted q, true q*) on a held-out
split measures recovery. We train the REAL weighted_msp.TokenWeightMLP with loss='pairwise' vs
'blondel' and compare that Spearman to the untrained baseline.

PASS = both losses beat the untrained model, and blondel is at least on par with pairwise (the upgrade
should not be worse on a clean synthetic signal). Needs torchsort for the blondel arm.

    python scripts/checks/weighted_msp_loss_sanity.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from luq import weighted_msp as W  # noqa: E402


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def make_synthetic(n=400, d=16, seed=0):
    """Return (states, records, y, q_true) with a recoverable weighting signal."""
    rng = np.random.RandomState(seed)
    v = rng.randn(d).astype(np.float32)                       # the true weight direction
    states, records, q_true = [], [], []
    for _ in range(n):
        T = int(rng.randint(5, 30))
        answer_h = rng.randn(T, d).astype(np.float32)
        h = np.vstack([rng.randn(1, d).astype(np.float32), answer_h])   # row 0 = prompt anchor (dropped)
        nll = np.abs(rng.randn(T)).astype(np.float32) + 0.5            # positive per-token NLLs
        w_star = np.log1p(np.exp(answer_h @ v))                        # softplus(v . h) > 0
        q_true.append(float((w_star * nll).mean()))
        states.append(h)
        records.append({"token_logprobs": [float(-x) for x in nll]})   # logprob = -nll
    q_true = np.array(q_true)
    # correctness decreases with q_true (higher weighted-NLL -> more uncertain -> less correct) + noise
    z = (q_true - q_true.mean()) / (q_true.std() + 1e-8)
    correct = 1.0 / (1.0 + np.exp(z)) + 0.05 * rng.randn(n)
    y = np.clip(correct, 0.0, 1.0).astype(np.float32)
    return states, records, y, q_true


def recovery(states, records, y, q_true, tr, te, device, loss):
    model = W.train_weighted_msp(states, records, y, tr, device, weight_mode="normalised",
                                 length_normalise=True, seed=1, loss=loss)
    q_pred = W.predict_weighted_msp(model, states, records, te, device,
                                    weight_mode="normalised", length_normalise=True)
    return spearman(q_pred, q_true[te])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | torchsort available: {W._HAVE_TORCHSORT}", flush=True)
    states, records, y, q_true = make_synthetic()
    n = len(states)
    tr, te = list(range(0, int(0.75 * n))), list(range(int(0.75 * n), n))

    # untrained baseline: a fresh random weighter should have ~0 correlation with q_true
    torch.manual_seed(0)
    m0 = W.TokenWeightMLP(states[0].shape[1]).to(device)
    q0 = W.predict_weighted_msp(m0, states, records, te, device,
                                weight_mode="normalised", length_normalise=True)
    base = spearman(q0, q_true[te])
    print(f"untrained baseline  Spearman(q_pred, q_true) = {base:+.3f}", flush=True)

    s_pair = recovery(states, records, y, q_true, tr, te, device, "pairwise")
    print(f"pairwise (Joe)      Spearman = {s_pair:+.3f}", flush=True)

    if W._HAVE_TORCHSORT:
        s_bl = recovery(states, records, y, q_true, tr, te, device, "blondel")
        print(f"blondel (torchsort) Spearman = {s_bl:+.3f}", flush=True)
        ok = (s_pair > base + 0.1) and (s_bl > base + 0.1) and (s_bl >= s_pair - 0.05)
        print(f"\nPASS = {ok}  (both beat untrained; blondel >= pairwise - 0.05)", flush=True)
        sys.exit(0 if ok else 1)
    else:
        print("\ntorchsort NOT available -> blondel arm skipped; install it then re-run.", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
