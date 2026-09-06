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

WHAT ONE CELL FILE HOLDS
------------------------
Per seed: the development and evaluation mean distances, the relative versions of both where a
background statistic exists for this layer, the labels, the sequence probability score, and the mean
token entropy where that input exists. Everything the layer-combining pass needs, and nothing that
would make it open a record again.

An input that is absent stays absent. A missing background leaves the relative arrays out of the file
entirely, and a missing entropy cache writes an array of nan, so such a cell reads as not measured
rather than as measured and zero.

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
from luq.data import MAX_NEW_TOKENS                                             # noqa: E402


def _accelerator_name():
    """The card this scan ran on, or 'cpu'. Never fails: provenance must not stop a job."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                       stderr=subprocess.DEVNULL).decode().strip()[:12]
    except Exception:
        return "unknown"


def apply_window(arr, window):
    """The per-token window this run is measured on.

    The cached array runs from the last prompt position through the final generated position. The
    reference implementation concatenates the last prompt position with one hidden state per
    generation step, which is one row shorter, so dropping the final row turns one into the other
    exactly. No re-extraction is involved and no other row is touched.
    """
    if window == "project":
        return arr
    return arr[:-1] if len(arr) else arr


def load_bg_stats(root, bg_dir, slug, layer, window, budgets):
    """Fitted background statistics for this layer, one per generation budget.

    Returns a dict keyed by budget. A budget with no file is simply absent, so a caller can tell
    'this budget was never fitted' from 'this budget was fitted and gave nothing'.
    """
    from luq import mahalanobis as _MD
    tag = "" if window == "project" else "__refwin"
    out = {}
    for b in sorted(budgets):
        f = root / bg_dir / f"{slug}__bgstats__L{layer}__b{b}{tag}.npz"
        if f.exists():
            out[b] = _MD.load_stats(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--evals", default=",".join(LONG_SRC))
    ap.add_argument("--rungs", default="")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--metric-thr", type=float, default=MD.DEFAULT_METRIC_THR)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--window", default="project", choices=["project", "ref"],
                    help="'project' keeps the cached window, the last prompt position through the "
                         "final generated position, which every existing result here was measured "
                         "on. 'ref' drops the final row, which is exactly the reference "
                         "implementation's window and needs no re-extraction.")
    ap.add_argument("--background-dir", default="cache/background_c4",
                    help="where scripts/01q_background_stats.py wrote its fitted statistics")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute cells whose output already exists. Off by default so a job that "
                         "ran out of walltime continues rather than starting again.")
    ap.add_argument("--require-background", action="store_true",
                    help="fail rather than silently omitting the relative distance. Use this in any "
                         "job whose output is meant to be complete.")
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

    # The window is part of the directory name, not just of the file contents. Two scans of the same
    # layer under different windows are different measurements and must never land on each other.
    _wtag = "" if args.window == "project" else "_refwin"
    out_dir = ROOT / (args.out_dir or f"results/hybrids/mdscan{_wtag}__{slug}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Recorded per cell file because the layers of one combined result are not all produced on the
    # same machine. Two layers computed on different accelerator generations differ in their last
    # bits, which is a fact worth being able to look up rather than reconstruct from job logs.
    CARD = _accelerator_name()
    COMMIT = _git_commit()
    print(f"WINDOW {args.window} | accelerator {CARD} | commit {COMMIT}", flush=True)

    # Cells are resolved against the FULL source pool first, as in the hybrid driver, so a restricted
    # --evals cannot silently shrink a training spec that still wears the same rung name.
    all_cells = [c for c in cells_long(set(LONG_SRC), evals) if c[0] != "Long->Short"]
    cells = all_cells if want_rungs is None else [c for c in all_cells if c[0] in want_rungs]
    cells = sorted(cells, key=lambda c: (c[0], tuple(sorted(d for d, _ in c[2])), c[1]))
    needed = {X for _, X, _ in cells} | {d for _, _, spec in cells for d, _ in spec}
    print(f"MODEL {args.model} | LAYER {args.layer} | {len(cells)} cells | seeds {seeds}")
    print(f"DATASETS: {sorted(needed)}", flush=True)

    PT = MH.load_population(args.model, args.layer, needed)

    # The background is fitted per layer and per generation budget, and an evaluation dataset is
    # always scored against the background at its own budget, which is the partition the
    # single-layer runs used.
    budgets = sorted({MAX_NEW_TOKENS[X] for _, X, _ in cells})
    BG = load_bg_stats(ROOT, args.background_dir, slug, args.layer, args.window, budgets)
    missing_bg = [b for b in budgets if b not in BG]
    if missing_bg:
        msg = (f"background statistics absent at layer {args.layer} for budget(s) "
               f"{missing_bg} (window '{args.window}')")
        if args.require_background:
            sys.exit(f"FATAL: {msg}. Run scripts/01q_background_stats.py for this layer first.")
        print(f"  {msg} -> the relative distance is OMITTED for those cells, never substituted",
              flush=True)
    for b in sorted(BG):
        print(f"  background b{b}: {BG[b].n_tokens} tokens, jitter {BG[b].jitter:g}", flush=True)

    # Mean token entropy, the third feature of the probability-augmented variants. Absent for a
    # dataset means nan for its rows, never a filled-in value.
    ENT = MH.load_entropy(args.model, sorted(needed))
    n_ent = sum(1 for d in needed if ENT.get(d) is not None)
    print(f"  entropy present for {n_ent}/{len(needed)} datasets", flush=True)

    stats_cache = MD.BoundedStatsCache(max_entries=12)
    written = 0

    for rung, X, spec in cells:
        # RESUME. A scan of one layer takes several hours and a walltime overrun previously threw
        # away every completed cell. A cell whose file is already on disk is skipped, but only after
        # it has been opened and found to contain what the combining pass needs: a truncated file
        # from a job killed mid-write would otherwise be treated as done. Pass --overwrite to redo.
        _out = out_dir / f"L{args.layer}__{X}__{rung}__{slug}.npz"
        if _out.exists() and not args.overwrite:
            try:
                _z = np.load(_out, allow_pickle=True)
                _seeds = [int(v) for v in np.asarray(_z["seeds"]).ravel()]
                if all(f"test_md__{sd}" in _z.files for sd in _seeds):
                    print(f"  [{rung:14s}/{X:14s}] already done, skipping", flush=True)
                    written += 1
                    continue
                print(f"  [{rung}/{X}] existing file is incomplete -> recomputing", flush=True)
            except Exception as e:
                print(f"  [{rung}/{X}] existing file unreadable ({type(e).__name__}) -> recomputing",
                      flush=True)
        if X not in PT or any(d not in PT for d, _ in spec):
            print(f"  [{rung}/{X}] inputs missing -> cell SKIPPED, left absent (never zero)", flush=True)
            continue
        t0 = time.time()
        store = {}
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            tr_states = [apply_window(PT[d][0][i], args.window) for d, i in train_rows]
            te_states = [apply_window(PT[d][0][i], args.window) for d, i in test_rows]
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
            dev_states = [tr_states[k] for k in half_dev]
            # One pass over the foreground distances serves both aggregations. With a background
            # present this is the relative family's only extra cost beyond its own einsum.
            bg = BG.get(MAX_NEW_TOKENS[X])
            if bg is not None:
                store[f"dev_md__{sd}"], store[f"dev_rmd__{sd}"] = MD.md_and_rmd_mean(
                    dev_states, st_a, bg)
                store[f"test_md__{sd}"], store[f"test_rmd__{sd}"] = MD.md_and_rmd_mean(
                    te_states, st_all, bg)
            else:
                store[f"dev_md__{sd}"] = MD.md_mean(dev_states, st_a)
                store[f"test_md__{sd}"] = MD.md_mean(te_states, st_all)
            store[f"y_dev__{sd}"] = ytr[half_dev]
            store[f"y_test__{sd}"] = yte
            # The probability score travels with the distances so pass two needs no record access.
            store[f"msp_dev__{sd}"] = np.array(
                [msp_mod.msp_uncertainty(r["token_logprobs"], "sum")
                 for r in [tr_recs[k] for k in half_dev]])
            store[f"msp_test__{sd}"] = np.array(
                [msp_mod.msp_uncertainty(r["token_logprobs"], "sum") for r in te_recs])
            # The mean token entropy travels with the distances for the same reason the probability
            # score does. A dataset with no entropy cache contributes nan, so a cell that mixes
            # sources is visibly partial rather than quietly averaged over what happens to exist.
            store[f"ent_dev__{sd}"] = np.array(
                [(ENT[d][i] if ENT.get(d) is not None else np.nan)
                 for d, i in [train_rows[k] for k in half_dev]], dtype=float)
            store[f"ent_test__{sd}"] = np.array(
                [(ENT[d][i] if ENT.get(d) is not None else np.nan)
                 for d, i in test_rows], dtype=float)
            store[f"n_tokens__{sd}"] = np.array([st_all.n_tokens])
            store[f"jitter__{sd}"] = np.array([st_all.jitter])

        if not store:
            continue
        if args.debug_dim:
            print(f"  [{rung:14s}/{X:14s}] {len(store)//8} seeds ({time.time()-t0:.0f}s)", flush=True)
            continue
        np.savez_compressed(out_dir / f"L{args.layer}__{X}__{rung}__{slug}.npz",
                            seeds=np.array(seeds), layer=np.array([args.layer]),
                            window=str(args.window), bg_budget=np.int64(MAX_NEW_TOKENS[X]),
                            has_background=np.int64(MAX_NEW_TOKENS[X] in BG),
                            card=str(CARD), commit=str(COMMIT), **store)
        written += 1
        print(f"  [{rung:14s}/{X:14s}] written ({time.time()-t0:.0f}s)", flush=True)

    if args.debug_dim:
        print("\nSMOKE TEST COMPLETE -- nothing written.")
        return
    print(f"\nwrote {written} cell files to {out_dir.relative_to(ROOT)} "
          f"(peak RSS {MH.peak_rss_gb():.1f} GB)")


if __name__ == "__main__":
    main()
