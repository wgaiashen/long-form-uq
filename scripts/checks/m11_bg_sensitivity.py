"""How much does a hidden-state difference of the size seen between accelerators change a DISTANCE?

WHY THIS EXISTS. The background verification compares two fitted statistics by their largest
elementwise relative difference, on the centroid and on the inverse covariance. That was the wrong
choice and this measures how wrong.

An inverse covariance is not consumed elementwise. It is consumed as the quadratic form
d^T S d summed over sixteen million terms, and inverting a near-singular covariance is exactly the
operation that lets a small change in the input produce a large change in an individual entry while
the quadratic form barely moves. So a large elementwise difference is consistent with identical
distances, and a gate built on it can fail for reasons that have no bearing on any result.

WHAT THIS DOES. It perturbs the background states by a controlled relative amount, of the order of
the difference between two accelerator generations, refits the statistics, and reports BOTH numbers
side by side: what the elementwise gate says, and what the per-token distances actually do. The ratio
between them is the amplification the gate introduces.

It settles a question that must not be settled by argument: whether a failing elementwise number
means the distances have moved.

    python scripts/checks/m11_bg_sensitivity.py --states <bg states npz> --budget 384
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import mahalanobis as MD  # noqa: E402


def elementwise_gate(a, b):
    """The metric the background verification uses, reproduced exactly."""
    dc = float(np.max(np.abs(a.centroid - b.centroid) / np.maximum(np.abs(a.centroid), 1e-12)))
    ds = float(np.max(np.abs(a.sigma_inv - b.sigma_inv) / np.maximum(np.abs(a.sigma_inv), 1e-12)))
    return dc, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--budget", type=int, default=384)
    ap.add_argument("--scales", default="1e-7,1e-6,1e-5",
                    help="relative perturbation sizes to try. The project's documented "
                         "cross-accelerator hidden-state difference is about 1e-6 relative.")
    ap.add_argument("--n-score", type=int, default=200,
                    help="rows to score distances on")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(ROOT / args.states, allow_pickle=True)
    rows = []
    for s in z["states"]:
        s = np.asarray(s, dtype=np.float32)
        if len(s):
            rows.append(s[:args.budget + 1])
    print(f"{len(rows)} background rows | budget {args.budget} | "
          f"{sum(len(r) for r in rows)} tokens", flush=True)

    ids = [("__c4__", i) for i in range(len(rows))]
    base = MD.fit_md(rows, layer=15, row_ids=ids, kind="base")
    score_rows = rows[:args.n_score]
    d_base = MD.md_mean(score_rows, base)
    print(f"reference statistic fitted: {base.n_tokens} tokens, jitter {base.jitter:g}")
    print(f"reference distances: mean {np.nanmean(d_base):.4f}, "
          f"sd {np.nanstd(d_base):.4f}\n", flush=True)

    rs = np.random.RandomState(args.seed)
    print(f"{'perturbation':>13} | {'centroid rel':>13} {'sigma_inv rel':>14} | "
          f"{'distance rel':>13} {'rank corr':>10}")
    print("-" * 76)
    for scale in [float(x) for x in args.scales.split(",")]:
        pert = [r * (1.0 + rs.randn(*r.shape).astype(np.float32) * scale) for r in rows]
        st = MD.fit_md(pert, layer=15, row_ids=[("__p__", i) for i in range(len(pert))], kind="p")
        dc, ds = elementwise_gate(base, st)
        d_new = MD.md_mean(score_rows[:args.n_score], st)
        ok = np.isfinite(d_base) & np.isfinite(d_new)
        rel = float(np.max(np.abs(d_new[ok] - d_base[ok]) / np.maximum(np.abs(d_base[ok]), 1e-12)))
        # Only the ORDER matters to a prediction-rejection ratio, so report it too.
        from scipy.stats import spearmanr
        rho = float(spearmanr(d_base[ok], d_new[ok]).statistic)
        print(f"{scale:>13.0e} | {dc:>13.3e} {ds:>14.3e} | {rel:>13.3e} {rho:>10.6f}")

    print("\nREADING: compare the two middle columns against the two on the right. If a perturbation")
    print("that leaves the distances essentially unchanged still produces a large elementwise")
    print("difference, then the elementwise gate is measuring conditioning of the matrix inverse,")
    print("not fidelity of the method, and it cannot be used to accept or reject a background.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
