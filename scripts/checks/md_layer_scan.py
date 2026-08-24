#!/usr/bin/env python
"""Pass one of the layer sensitivity: per-example Mahalanobis distances at ONE layer.

Registered in prereg/M9_layer_distance_sensitivity.md.

WHY TWO PASSES
--------------
The supervised distance family needs one distance per layer per example, then a regression across
layers. Holding eleven layers of per-token states in memory at once is not possible: one layer of the
eight-dataset population is about 19 GB. So this pass loads ONE layer, reduces it to the only thing
the regression needs -- a scalar distance per example -- and persists that. Eleven such files are a
few megabytes in total, and pass two reads them all.

The reduction is exactly the reference's: statistics fitted on the first half of the training pool
produce the development distances, and the statistics are then RE-ESTIMATED ON THE WHOLE POOL before
evaluation distances are produced. Both are kept because the regression is fitted on the first and
applied to the second, and collapsing them would change the method.

    python scripts/checks/md_layer_scan.py --model meta-llama/Meta-Llama-3.1-8B --layer 3
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from sklearn.model_selection import train_test_split                            # noqa: E402

from luq import cache, msp as msp_mod                                           # noqa: E402
from luq import mahalanobis as MD                                               # noqa: E402
import md_hybrids as MH                                                         # noqa: E402
from attn_pool import PROMPT_REGIME                                             # noqa: E402
from xl_rungs import build_rows                                                 # noqa: E402
from probe_drift_long import LONG_SRC, cells_long, sampled_train_idx            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--evals", default=",".join(LONG_SRC))
    ap.add_argument("--rungs", default="")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--metric-thr", type=float, default=MD.DEFAULT_METRIC_THR)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--debug-dim", type=int, default=0,
                    help="truncate hidden states to N dims: SMOKE TEST ONLY, never a measurement")
    args = ap.parse_args()

    slug = cache._slug(args.model)
    seeds = [int(s) for s in args.seeds.split(",")]
    evals = [e for e in args.evals.split(",") if e]
    want_rungs = None
    if args.rungs:
        want_rungs = {MH.RUNG_ALIASES.get(r.strip(), r.strip()) for r in args.rungs.split(",")}
    MH._DEBUG_DIM = args.debug_dim
    if args.debug_dim:
        print("=" * 90)
        print(f"SMOKE TEST -- hidden states truncated to {args.debug_dim} dims. NOT a measurement.")
        print("=" * 90)

    out_dir = ROOT / (args.out_dir or f"results/hybrids/mdscan__{slug}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cells are resolved against the FULL source pool first, as in the hybrid driver, so a restricted
    # --evals cannot silently shrink a training spec that still wears the same rung name.
    all_cells = [c for c in cells_long(set(LONG_SRC), evals) if c[0] != "Long->Short"]
    cells = all_cells if want_rungs is None else [c for c in all_cells if c[0] in want_rungs]
    cells = sorted(cells, key=lambda c: (c[0], tuple(sorted(d for d, _ in c[2])), c[1]))
    needed = {X for _, X, _ in cells} | {d for _, _, spec in cells for d, _ in spec}
    print(f"MODEL {args.model} | LAYER {args.layer} | {len(cells)} cells | seeds {seeds}")
    print(f"DATASETS: {sorted(needed)}", flush=True)

    PT = MH.load_population(args.model, args.layer, needed)
    stats_cache = MD.BoundedStatsCache(max_entries=12)
    written = 0

    for rung, X, spec in cells:
        if X not in PT or any(d not in PT for d, _ in spec):
            print(f"  [{rung}/{X}] inputs missing -> cell SKIPPED, left absent (never zero)", flush=True)
            continue
        t0 = time.time()
        store = {}
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            tr_states = [PT[d][0][i] for d, i in train_rows]
            te_states = [PT[d][0][i] for d, i in test_rows]
            ytr = np.array([PT[d][2][i] for d, i in train_rows], float)
            yte = np.array([PT[d][2][i] for d, i in test_rows], float)
            tr_recs = [PT[d][3][i] for d, i in train_rows]
            te_recs = [PT[d][3][i] for d, i in test_rows]

            half_a, half_dev = train_test_split(list(range(len(tr_states))), test_size=MD.DEV_SIZE,
                                                shuffle=True, random_state=MD.DEV_RANDOM_STATE)
            ids_a = [(d, int(i)) for (d, i) in [train_rows[k] for k in half_a]]
            ids_all = [(d, int(i)) for (d, i) in train_rows]
            # First half -> development distances. Whole pool -> evaluation distances. The reference
            # resets its fitted flag between the two, which is what this pair of fits reproduces.
            st_a = MD.fit_md([tr_states[k] for k in half_a], ytr[half_a], args.metric_thr,
                             layer=args.layer, row_ids=ids_a, kind="fg", cache=stats_cache)
            st_all = MD.fit_md(tr_states, ytr, args.metric_thr,
                               layer=args.layer, row_ids=ids_all, kind="fg", cache=stats_cache)
            store[f"dev_md__{sd}"] = MD.md_mean([tr_states[k] for k in half_dev], st_a)
            store[f"test_md__{sd}"] = MD.md_mean(te_states, st_all)
            store[f"y_dev__{sd}"] = ytr[half_dev]
            store[f"y_test__{sd}"] = yte
            # The probability score travels with the distances so pass two needs no record access.
            store[f"msp_dev__{sd}"] = np.array(
                [msp_mod.msp_uncertainty(r["token_logprobs"], "sum")
                 for r in [tr_recs[k] for k in half_dev]])
            store[f"msp_test__{sd}"] = np.array(
                [msp_mod.msp_uncertainty(r["token_logprobs"], "sum") for r in te_recs])
            store[f"n_tokens__{sd}"] = np.array([st_all.n_tokens])
            store[f"jitter__{sd}"] = np.array([st_all.jitter])

        if not store:
            continue
        if args.debug_dim:
            print(f"  [{rung:14s}/{X:14s}] {len(store)//8} seeds ({time.time()-t0:.0f}s)", flush=True)
            continue
        np.savez_compressed(out_dir / f"L{args.layer}__{X}__{rung}__{slug}.npz",
                            seeds=np.array(seeds), layer=np.array([args.layer]), **store)
        written += 1
        print(f"  [{rung:14s}/{X:14s}] written ({time.time()-t0:.0f}s)", flush=True)

    if args.debug_dim:
        print("\nSMOKE TEST COMPLETE -- nothing written.")
        return
    print(f"\nwrote {written} cell files to {out_dir.relative_to(ROOT)} "
          f"(peak RSS {MH.peak_rss_gb():.1f} GB)")


if __name__ == "__main__":
    main()
