#!/usr/bin/env python
"""Substitute a learned token weighting for the probability branch of the published hybrid back-off.

Registered in prereg/M8_cawsa_hbo_substitution.md before any number here was computed.

WHAT THIS PRODUCES
------------------
Per cell of the long-form grid, on exactly the rows the master used, six comparable scores:

  msp          the published sequence-probability score, -sum log p
  floor_min    minimum token probability, the strongest simple aggregate on long form
  cawsa        the frozen learned token weighting at shrinkage 2, read from the sidecar
  saplma       the probe, read from the sidecar
  hbo          the published back-off, probability branch = msp
  cawsa_hbo    the same estimator with the probability branch replaced by cawsa

Nothing else about the back-off changes: the distance, its percentile against the held-out half of
the training pool, the weight rule and the rank transform are computed ONCE per cell and seed and
fed to both estimators, so the two differ in exactly one input.

WHY A SIDECAR IS WRITTEN
------------------------
The distance fit and scoring are 99 percent of the runtime and are the only part that needs the
per-token cache. Persisting the gate quantities per example means any further substitution into the
probability branch costs seconds instead of hours, and lets the identity checks in section 6 of the
registration be re-run later without a GPU-sized cache present.

    python scripts/checks/cawsa_hbo.py --model meta-llama/Meta-Llama-3.1-8B --layer 15 \
        --evals pubmed_qa --rungs 1ds-Diff --debug-dim 64
"""
import argparse
import csv as _csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from sklearn.model_selection import train_test_split                            # noqa: E402

from luq import cache, msp as msp_mod, results                                  # noqa: E402
from luq import mahalanobis as MD                                               # noqa: E402
import md_hybrids as MH                                                         # noqa: E402
from attn_pool import PROMPT_REGIME                                             # noqa: E402
from xl_rungs import build_rows                                                 # noqa: E402
from probe_drift_long import LONG_SRC, cells_long, sampled_train_idx            # noqa: E402

# The sidecar keys for the two frozen report-facing scores. Named here so a rename upstream fails
# loudly at the assertion below rather than silently selecting a different method.
CAWSA_KEY = "unc__wmsp_shrink2"
FLOORMIN_KEY = "unc__floor_min"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--evals", default=",".join(LONG_SRC))
    ap.add_argument("--rungs", default="")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--metric-thr", type=float, default=MD.DEFAULT_METRIC_THR)
    ap.add_argument("--perex-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--debug-dim", type=int, default=0,
                    help="truncate hidden states to N dims: SMOKE TEST ONLY, never a measurement")
    args = ap.parse_args()

    model, layer = args.model, args.layer
    slug = cache._slug(model)
    seeds = [int(s) for s in args.seeds.split(",")]
    evals = [e for e in args.evals.split(",") if e]
    want_rungs = None
    if args.rungs:
        want_rungs = {MH.RUNG_ALIASES.get(r.strip(), r.strip()) for r in args.rungs.split(",")}

    smoke = bool(args.debug_dim)
    if smoke:
        print("=" * 90)
        print(f"SMOKE TEST -- hidden states truncated to {args.debug_dim} dimensions. The numbers "
              f"below are NOT a measurement and must never enter a results table.")
        print("=" * 90)
    MH._DEBUG_DIM = args.debug_dim

    print(f"MODEL {model} | layer {layer} | seeds {seeds} | metric_thr {args.metric_thr}")
    print(f"REGIME MAP: {json.dumps({d: PROMPT_REGIME.get(d, '') for d in LONG_SRC}, sort_keys=True)}")

    # Cells resolved against the FULL source pool first, exactly as the hybrid driver does, so a
    # restricted --evals cannot silently shrink a training spec that wears the same rung name.
    all_cells = [c for c in cells_long(set(LONG_SRC), evals) if c[0] != "Long->Short"]
    cells = all_cells
    if want_rungs is not None:
        cells = [c for c in cells if c[0] in want_rungs]
    if args.max_cells:
        cells = cells[:args.max_cells]
    cells = sorted(cells, key=lambda c: (c[0], tuple(sorted(d for d, _ in c[2])), c[1]))
    needed = {X for _, X, _ in cells} | {d for _, _, spec in cells for d, _ in spec}
    print(f"\nCELLS SELECTED: {len(cells)} of {len(all_cells)} in the full grid")
    print(f"DATASETS THOSE CELLS TOUCH: {sorted(needed)}")

    print("\nLOADING POPULATION", flush=True)
    PT = MH.load_population(model, layer, needed)

    perex_dir = ROOT / (args.perex_dir or f"results/perex_clean__{slug}")
    if not perex_dir.is_dir():
        sys.exit(f"FATAL: per-example sidecars absent at {perex_dir}. The frozen learned weighting "
                 f"and the probe must be TAKEN from the master's own vectors, not reproduced, or the "
                 f"substitution is not against the published estimator on the published rows.")
    print(f"per-example sidecars: {perex_dir.relative_to(ROOT)}")

    gate_dir = ROOT / f"results/hybrids/hbo_gate__{slug}"
    if not smoke:
        gate_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGRID: {len(cells)} cells to compute\n", flush=True)
    stats_cache = MD.BoundedStatsCache(max_entries=12)
    rows_out, diag_out = [], []
    zero_w_rungs = {}

    for rung, X, spec in cells:
        if X not in PT:
            print(f"  [{rung}/{X}] eval target has no cache -> cell left BLANK", flush=True)
            continue
        srcs = [d for d, _ in spec]
        if any(d not in PT for d in srcs):
            print(f"  [{rung}/{X}] sources missing -> BLANK", flush=True)
            continue

        sp = perex_dir / f"{X}__{rung}__{slug}.npz"
        if not sp.exists():
            print(f"  [{rung}/{X}] no sidecar -> BLANK", flush=True)
            continue
        side = np.load(sp, allow_pickle=True)
        for k in (CAWSA_KEY, FLOORMIN_KEY, "unc__saplma", "unc__floor_sum"):
            if k not in side.files:
                sys.exit(f"FATAL [{rung}/{X}]: sidecar lacks '{k}'. Refusing to guess a substitute.")

        t_cell = time.time()
        per, degenerate, gate_store = {}, {}, {}
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            tr_states = [PT[d][0][i] for d, i in train_rows]
            te_states = [PT[d][0][i] for d, i in test_rows]
            ytr = np.array([PT[d][2][i] for d, i in train_rows], float)
            yte = np.array([PT[d][2][i] for d, i in test_rows], float)
            te_recs = [PT[d][3][i] for d, i in test_rows]

            # Alignment PROVED, not assumed: the sidecar's stored labels must equal this cell's
            # evaluation labels exactly, which fails if the rows or their order differ by one place.
            sy = np.asarray(side["y"], float)
            if sy.shape != yte.shape or not np.array_equal(sy, yte):
                sys.exit(f"FATAL [{rung}/{X}]: sidecar labels do not match this cell's evaluation "
                         f"labels ({sy.shape} vs {yte.shape}). Refusing to mix two populations.")
            sseeds = [int(x) for x in np.asarray(side["seeds"]).ravel()]
            if sd not in sseeds:
                sys.exit(f"FATAL [{rung}/{X}]: sidecar has seeds {sseeds}, asked for {sd}.")
            si = sseeds.index(sd)

            msp_te = np.array([msp_mod.msp_uncertainty(r["token_logprobs"], "sum") for r in te_recs])
            fs = np.asarray(side["unc__floor_sum"][si], float)
            dev = float(np.max(np.abs(fs - msp_te))) if len(fs) == len(msp_te) else float("inf")
            if dev > 1e-9:
                sys.exit(f"FATAL [{rung}/{X}] seed {sd}: recomputed sequence probability differs from "
                         f"the master's floor_sum by {dev:.3e}. Refusing to call either published MSP.")

            cawsa_te = np.asarray(side[CAWSA_KEY][si], float)
            saplma_te = np.asarray(side["unc__saplma"][si], float)
            floormin_te = np.asarray(side[FLOORMIN_KEY][si], float)

            # Registration section 6.1: the branch must actually have changed.
            if np.allclose(cawsa_te, msp_te):
                sys.exit(f"FATAL [{rung}/{X}] seed {sd}: the learned weighting is numerically equal to "
                         f"the published probability score. The substitution would be a no-op.")

            # --- the gate, computed ONCE and shared by both estimators -------------------------
            half_a, half_dev = train_test_split(list(range(len(tr_states))), test_size=MD.DEV_SIZE,
                                                shuffle=True, random_state=MD.DEV_RANDOM_STATE)
            ids_a = [(d, int(i)) for (d, i) in [train_rows[k] for k in half_a]]
            ids_all = [(d, int(i)) for (d, i) in train_rows]
            st_a = MD.fit_md([tr_states[k] for k in half_a], ytr[half_a], args.metric_thr,
                             layer=layer, row_ids=ids_a, kind="fg", cache=stats_cache)
            st_all = MD.fit_md(tr_states, ytr, args.metric_thr,
                               layer=layer, row_ids=ids_all, kind="fg", cache=stats_cache)
            dev_md = MD.md_mean([tr_states[k] for k in half_dev], st_a)
            test_md = MD.md_mean(te_states, st_all)

            hbo_pub, pct, w_sup = MD.hbo(msp_te, saplma_te, test_md, dev_md)
            hbo_cawsa, pct2, w_sup2 = MD.hbo(cawsa_te, saplma_te, test_md, dev_md)

            # Registration section 6.3: the gate must be identical between the two estimators.
            if not (np.array_equal(pct, pct2) and np.array_equal(w_sup, w_sup2)):
                sys.exit(f"FATAL [{rung}/{X}] seed {sd}: the distance gate differs between the two "
                         f"estimators. They are supposed to differ in the probability branch alone.")

            # Registration section 6.4: where the supervised weight is zero everywhere, the modified
            # estimator IS the learned weighting, so its rank vector must match exactly. This is the
            # designed behaviour of the published back-off, not a result -- see registration 0.
            if float(np.max(w_sup)) == 0.0:
                exp = rankdata(cawsa_te)
                d0 = float(np.max(np.abs(hbo_cawsa - exp)))
                if d0 != 0.0:
                    sys.exit(f"FATAL [{rung}/{X}] seed {sd}: supervised weight is zero for every "
                             f"example, so the modified score must equal the rank of the learned "
                             f"weighting exactly, but differs by {d0:.3e}.")
                zero_w_rungs.setdefault(rung, 0)
                zero_w_rungs[rung] += 1

            v = {"msp": msp_te, "floor_min": floormin_te, "cawsa": cawsa_te, "saplma": saplma_te,
                 "hbo": hbo_pub, "cawsa_hbo": hbo_cawsa}
            # A constant score must not become a number: np.argsort breaks ties by row position, so a
            # flat vector is ranked arbitrarily and scores whatever that permutation happens to give.
            for m, vec in v.items():
                vec = np.asarray(vec, float)
                if np.isfinite(vec).all() and float(np.ptp(vec)) == 0.0:
                    degenerate[m] = degenerate.get(m, 0) + 1
                    per.setdefault(m, []).append(float("nan"))
                    print(f"    [{rung}/{X}] seed {sd}: '{m}' is CONSTANT -> blank, not a number",
                          flush=True)
                    continue
                per.setdefault(m, []).append(results.prr(yte, vec))

            print(f"    [{rung}/{X}] seed {sd}: gate mean_w_sv={np.mean(w_sup):.4f} "
                  f"max_w_sv={np.max(w_sup):.4f} frac(R>0.5)={np.mean(pct > 0.5):.3f} | "
                  f"PRR cawsa={results.prr(yte, cawsa_te):.6f} cawsa_hbo="
                  f"{results.prr(yte, hbo_cawsa):.6f} hbo={results.prr(yte, hbo_pub):.6f} "
                  f"msp={results.prr(yte, msp_te):.6f}", flush=True)
            gate_store[sd] = dict(test_md=test_md, dev_md=dev_md, pct=pct, w_sup=w_sup,
                                  msp=msp_te, cawsa=cawsa_te, saplma=saplma_te,
                                  hbo=hbo_pub, cawsa_hbo=hbo_cawsa, y=yte)
            diag_out.append(dict(model=model, eval=X, rung=rung, seed=sd,
                                 n_train=len(train_rows), n_test=len(test_rows),
                                 mean_R=float(np.mean(pct)),
                                 frac_R_gt_half=float(np.mean(pct > 0.5)),
                                 mean_w_sv=float(np.mean(w_sup)),
                                 max_w_sv=float(np.max(w_sup))))

        if not per:
            continue
        train_lab = "+".join(d for d, _ in spec)
        for m, vals in sorted(per.items()):
            a = np.array(vals, float)
            ok = a[np.isfinite(a)]
            rows_out.append(dict(model=model, eval=X, rung=rung, train=train_lab, method=m,
                                 prr_mean=(round(float(ok.mean()), 6) if len(ok) else ""),
                                 prr_std=(round(float(ok.std()), 6) if len(ok) else ""),
                                 n_seeds=len(ok), degenerate_seeds=degenerate.get(m, 0),
                                 layer=layer, metric_thr=args.metric_thr))
        line = "  ".join(
            f"{m} {np.nanmean(vals):+.3f}" if np.isfinite(np.nanmean(vals)) else f"{m} ----"
            for m, vals in sorted(per.items()))
        print(f"  [{rung:14s}/{X:14s}] {line}   ({time.time() - t_cell:.0f}s)", flush=True)

        if not smoke and gate_store:
            np.savez_compressed(
                gate_dir / f"{X}__{rung}__{slug}.npz",
                seeds=np.array(sorted(gate_store)),
                **{f"{k}__{sd}": np.asarray(gate_store[sd][k])
                   for sd in sorted(gate_store) for k in gate_store[sd]})

    if smoke:
        print("\nSMOKE TEST COMPLETE -- nothing written.")
        return
    out = ROOT / (args.out or f"results/hybrids/pdl_cawsa_hbo__{slug}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    dpath = out.with_name(out.stem + "__diagnostics.csv")
    with open(dpath, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(diag_out[0])); w.writeheader(); w.writerows(diag_out)
    print(f"\nwrote {out.relative_to(ROOT)} ({len(rows_out)} rows)")
    print(f"wrote {dpath.relative_to(ROOT)} ({len(diag_out)} rows)")
    print(f"wrote gate sidecars -> {gate_dir.relative_to(ROOT)}")
    if zero_w_rungs:
        print("\nrungs where the supervised weight was zero for every example, by cell-seed count:")
        for r, n in sorted(zero_w_rungs.items()):
            print(f"  {r:16s} {n}")
        print("On those the modified estimator IS the learned weighting by construction. That is the "
              "designed behaviour of the back-off, verified exactly above, and is not a result.")


if __name__ == "__main__":
    main()
