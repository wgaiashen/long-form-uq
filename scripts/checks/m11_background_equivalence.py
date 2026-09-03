"""Does persisting the background statistic change it?

The full-layer reproduction fits the background once per layer and per generation budget and writes
the result to a file, where the single-layer driver fitted it in memory on every run. That is a new
code path, and a new code path between a set of hidden states and a distance has to be shown to
change nothing before it is trusted at thirty-one layers that have nothing to check against.

This compares, on the SAME states file and the SAME window, the statistic the single-layer driver
fits in memory against the one written to disk by scripts/01q_background_stats.py. Equality here is
not a tolerance question: both fit the identical array with the identical call, so anything short of
exact agreement means the persistence path is doing something.

This deliberately says nothing about whether a recomputed hidden state matches a cached one. That is
a separate question about where a tensor was computed, and it is recorded separately.

    python scripts/checks/m11_background_equivalence.py \\
        --states cache/background_c4/meta-llama_Meta-Llama-3.1-8B__L15__b384.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache  # noqa: E402
from luq import mahalanobis as MD  # noqa: E402
import md_hybrids as MH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--states", required=True)
    ap.add_argument("--budgets", default="56,128,256,384")
    ap.add_argument("--stats-dir", default="cache/background_c4")
    args = ap.parse_args()

    slug = cache._slug(args.model)
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    status = 0

    for b in budgets:
        # The single-layer driver's own path, unchanged.
        rows, layer, _ = MH.load_background(ROOT / args.states, b)
        want = MD.fit_md(rows, layer=layer,
                         row_ids=[("__c4__", i) for i in range(len(rows))], kind=f"bg{b}")

        f = ROOT / args.stats_dir / f"{slug}__bgstats__L{layer}__b{b}.npz"
        if not f.exists():
            print(f"b{b}: no persisted statistic at {f.name} -> NOT CHECKED")
            status = 1
            continue
        got = MD.load_stats(f)

        dc = bool(np.array_equal(want.centroid, got.centroid))
        ds = bool(np.array_equal(want.sigma_inv, got.sigma_inv))
        dn = (want.n_tokens, want.n_rows, want.jitter) == (got.n_tokens, got.n_rows, got.jitter)
        print(f"b{b} L{layer}: centroid exact {dc} | sigma_inv exact {ds} | counts and jitter {dn} "
              f"| {want.n_tokens} tokens over {len(rows)} rows")
        if not (dc and ds and dn):
            # Report the size of the disagreement, because "not equal" alone does not say whether
            # this is a dtype problem or a different corpus.
            print(f"    max |d centroid| = {np.max(np.abs(want.centroid - got.centroid)):.3e}")
            print(f"    max |d sigma_inv| = {np.max(np.abs(want.sigma_inv - got.sigma_inv)):.3e}")
            status = 1

    print("\nBACKGROUND PERSISTENCE CHECK: " + ("PASS" if status == 0 else "FAIL"))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
