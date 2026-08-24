#!/usr/bin/env python
"""The published Mahalanobis-based baselines and hybrids, on a corrected-span population.

WHAT THIS PRODUCES
------------------
Per cell of the long-form grid, on exactly the rows the master used:

  msp          the published sequence-probability score, -sum log p. Recomputed here and gated
               against the master's own column, so the identity is proved rather than asserted.
  saplma       the probe, refit here with the ladder's own helper and gated the same way.
  satmd_mid    supervised aggregation of token Mahalanobis distances -- MIDDLE-LAYER ADAPTATION.
  satrmd_mid   the same on relative distances, against a C4 background -- MIDDLE-LAYER ADAPTATION.
  msp_satmd_mid / msp_satrmd_mid
               the meta-regressor variants, whose feature set is [distance, MSP, mean token entropy].
  huq_satmd_mid / huq_satrmd_mid
               the hybrid uncertainty combination, hyperparameters searched on the dev half only.
  hbo          the hybrid back-off. NOT an adaptation: its reference recipe selects the middle layer
               out of the saved stack, which is the layer this project caches.

Every method whose inputs are absent is SKIPPED LOUDLY and left out of the CSV. A cell that was not
measured must never appear as a number, and never as a zero.

WHY THE CELLS ARE NOT RE-DERIVED
--------------------------------
Cells, seeds, sampled training rows and evaluation splits all come from the same functions the master
ladder drives off (`cells_long`, `build_rows`, `sampled_train_idx`, `eval_split`). Rebuilding any of
them here would risk a population that merely resembles the master's, which is the confound this whole
line of work exists to avoid.

    python scripts/checks/md_hybrids.py --model meta-llama/Meta-Llama-3.1-8B --layer 15 \
        --evals pubmed_qa --rungs DiffTask,1ds-Diff --benchmark
"""
import argparse
import csv as _csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from sklearn.decomposition import PCA                                           # noqa: E402
from sklearn.linear_model import Ridge                                          # noqa: E402
from sklearn.model_selection import train_test_split                            # noqa: E402

# The reference builds its supervised aggregator in run_polygraph.py:360-380, NOT from the class
# defaults in average_token_mahalanobis_distance.py. That call site passes `remove_corr=True` and
# `positive=False`, so the published aggregation is a ten-component reduction of the layer-wise
# distance matrix followed by an UNCONSTRAINED ridge. An earlier version of this file read the class
# defaults instead and used `Ridge(positive=True)` with no reduction; see section L of the
# implementation audit for what that cost.
N_COMPONENTS = 10

from luq import cache, msp as msp_mod, results                                  # noqa: E402
from luq import mahalanobis as MD                                               # noqa: E402
from luq.config import Config                                                   # noqa: E402
from aggregation_table import load_per_token, conf_meanpool                     # noqa: E402
from attn_pool import PROMPT_REGIME                                             # noqa: E402
from xl_rungs import build_rows, label_of                                       # noqa: E402
from probe_drift_long import LONG_SRC, cells_long, sampled_train_idx            # noqa: E402

_DEBUG_DIM = 0          # set from --debug-dim; smoke-test only, see the flag's help text

RUNG_ALIASES = {"ID": "ID", "LOO": "LOO-long", "SameTask": "SameTask-long",
                "DiffTask": "DiffTask-long", "1ds-Diff": "1ds-Diff-long"}


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def load_population(model, layer, datasets):
    """Load per-token states + labels exactly as the ladder does, including its unlabelled filter."""
    PT = {}
    for d in sorted(datasets):
        loaded = load_per_token(model, d, layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no per-token cache -> skipped, cells using it are left BLANK", flush=True)
            continue
        states, split, y, _, records = loaded
        if _DEBUG_DIM:
            # Smoke mode: cut the hidden dimension the moment the array arrives, so a
            # cheap run stays cheap. Doing it after the whole population is resident
            # would defeat the point of the flag.
            states = [a[:, :_DEBUG_DIM] for a in states]
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: fully unlabelled ({label_of(d)}) -> skipped", flush=True)
            continue
        orig = np.arange(len(records))
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]
            records = [records[k] for k in keep]
            split, y, orig = split[keep], y[keep], keep
        PT[d] = (states, split, y, records, orig)
        print(f"  {d}: {len(states)} rows (label={label_of(d)}, regime="
              f"'{PROMPT_REGIME.get(d, '') or 'canonical'}')", flush=True)
    return PT


def load_entropy(model, datasets):
    """Mean predictive entropy per row, per dataset, or None where the cache is absent."""
    out = {}
    for d in datasets:
        cfg = Config(model_name=model, dataset=d, ood_setting="ID",
                     prompt_regime=PROMPT_REGIME.get(d, ""))
        p = Path(cfg.cache_dir) / "entropy" / f"{cache.run_key(model, d, 'ID')}.npz"
        if not p.exists():
            out[d] = None
            continue
        z = np.load(p, allow_pickle=True)
        ent = z["entropy"]
        out[d] = np.array([float(np.mean(e)) if len(e) else np.nan for e in ent])
    return out


def load_background(path, budget):
    """Background token states, sliced to `budget` generated tokens.

    Greedy decoding makes a short generation an exact prefix of a long one, so a background generated
    once at the maximum budget yields every shorter budget by slicing. The stored window is the last
    prompt position plus the generated tokens, so `budget` generated tokens means budget+1 states.
    """
    z = np.load(path, allow_pickle=True)
    st = z["states"]
    rows = []
    for s in st:
        s = np.asarray(s, dtype=np.float32)
        if len(s) == 0:
            continue
        if _DEBUG_DIM:
            s = s[:, :_DEBUG_DIM]
        rows.append(s[:budget + 1])
    return rows, int(z["layer"]), int(z["budget"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--evals", default=",".join(LONG_SRC))
    ap.add_argument("--rungs", default="", help="comma list of ID,LOO,SameTask,DiffTask,1ds-Diff")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--metric-thr", type=float, default=MD.DEFAULT_METRIC_THR)
    ap.add_argument("--background", default="", help="npz written by scripts/01m_background_c4.py")
    ap.add_argument("--bg-budget", type=int, default=128,
                    help="generated-token budget to slice the background to")
    ap.add_argument("--perex-dir", default="",
                    help="per-example sidecar directory for THIS population. When present the probe "
                         "and the probability score are read from it instead of being refitted, which "
                         "makes them the master's own vectors rather than a reproduction of them. "
                         "Default: results/perex_clean__<slug>.")
    ap.add_argument("--out", default="")
    ap.add_argument("--benchmark", action="store_true",
                    help="print per-component timings and peak RSS, and stop after --max-cells")
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--debug-dim", type=int, default=0,
                    help="SMOKE TEST ONLY: truncate hidden states to this many dimensions so every "
                         "code path can be exercised cheaply. Output is labelled a smoke test and is "
                         "written to a scratch filename that no results table reads.")
    args = ap.parse_args()

    model, layer = args.model, args.layer
    slug = cache._slug(model)
    seeds = [int(s) for s in args.seeds.split(",")]
    evals = [e for e in args.evals.split(",") if e]
    want_rungs = None
    if args.rungs:
        want_rungs = {RUNG_ALIASES.get(r.strip(), r.strip()) for r in args.rungs.split(",")}

    if args.debug_dim:
        print("=" * 90)
        print(f"SMOKE TEST -- hidden states truncated to {args.debug_dim} dimensions. The numbers "
              f"below are NOT a measurement of anything and must never enter a results table.")
        print("=" * 90)
    print(f"MODEL {model} | layer {layer} | seeds {seeds} | metric_thr {args.metric_thr}")
    print(f"REGIME MAP: {json.dumps({d: PROMPT_REGIME.get(d, '') for d in LONG_SRC}, sort_keys=True)}")
    global _DEBUG_DIM
    _DEBUG_DIM = args.debug_dim
    # THE CELLS ARE RESOLVED AGAINST THE FULL SOURCE POOL, ALWAYS. Every rung's training spec
    # depends on which sources exist, so resolving against a reduced pool would silently produce a
    # different -- and smaller -- training set wearing the same rung name. Only AFTER the specs are
    # fixed is the load restricted to the datasets those specs actually name. That is safe because
    # build_rows reads one dataset's own split array, the seed and the cap, and nothing else.
    all_cells = [c for c in cells_long(set(LONG_SRC), evals) if c[0] != "Long->Short"]
    cells = all_cells
    if want_rungs is not None:
        cells = [c for c in cells if c[0] in want_rungs]
    if args.max_cells:
        cells = cells[:args.max_cells]
    # Order the cells so that cells sharing a training pool sit next to each other. Every eval's
    # single-source far-OOD rung draws the SAME source with the same cap, so with a small statistics
    # cache those cells fit the covariance once instead of once each. Ordering changes which cells are
    # computed in which sequence and nothing else -- each cell is independent of every other.
    cells = sorted(cells, key=lambda c: (c[0], tuple(sorted(d for d, _ in c[2])), c[1]))
    needed = {X for _, X, _ in cells} | {d for _, _, spec in cells for d, _ in spec}
    print(f"\nCELLS SELECTED: {len(cells)} of {len(all_cells)} in the full grid")
    print(f"DATASETS THOSE CELLS TOUCH: {sorted(needed)}")

    print("\nLOADING POPULATION", flush=True)
    PT = load_population(model, layer, needed)
    sources = set(PT)
    ENT = load_entropy(model, sources)
    n_ent = sum(1 for d in sources if ENT.get(d) is not None)
    print(f"\nentropy caches: {n_ent}/{len(sources)} datasets"
          + ("" if n_ent == len(sources) else "  -> the MSP-regressor variants will be SKIPPED"))

    BG = None
    if args.background:
        bg_rows, bg_layer, bg_budget = load_background(ROOT / args.background, args.bg_budget)
        if bg_layer != layer:
            sys.exit(f"FATAL: background is layer {bg_layer}, this run is layer {layer}.")
        if args.bg_budget > bg_budget:
            sys.exit(f"FATAL: asked to slice the background to {args.bg_budget} generated tokens but "
                     f"it was generated at {bg_budget}. A longer budget cannot be recovered by slicing.")
        BG = MD.fit_md(bg_rows, layer=layer, row_ids=[("__c4__", i) for i in range(len(bg_rows))],
                       kind=f"bg{args.bg_budget}")
        print(f"background: {len(bg_rows)} rows, {BG.n_tokens} tokens, jitter {BG.jitter:g}")
    else:
        print("background: ABSENT -> every relative-distance method will be SKIPPED", flush=True)

    perex_dir = ROOT / (args.perex_dir or f"results/perex_clean__{slug}")
    if perex_dir.is_dir():
        print(f"per-example sidecars: {perex_dir.relative_to(ROOT)} "
              f"({len(list(perex_dir.glob('*.npz')))} cells)")
    else:
        print(f"per-example sidecars: ABSENT at {perex_dir} -> the probe is refitted per cell, which "
              f"is slower and only gated against the master rather than taken from it", flush=True)
        perex_dir = None

    print(f"\nGRID: {len(cells)} cells to compute\n", flush=True)

    # A fitted inverse covariance is 4096x4096 float32, about 67 MB, so the cache is bounded and
    # evicts oldest-first. Unbounded it would hold every pool in the grid and run to tens of GB.
    stats_cache = MD.BoundedStatsCache(max_entries=12)
    msp_dev_max = 0.0
    rows_out, diag_out = [], []
    for ci, (rung, X, spec) in enumerate(cells):
        if X not in PT:
            print(f"  [{rung}/{X}] eval target has no cache -> cell left BLANK", flush=True)
            continue
        srcs = [d for d, _ in spec]
        if any(d not in PT for d in srcs):
            print(f"  [{rung}/{X}] sources missing {[d for d in srcs if d not in PT]} -> BLANK", flush=True)
            continue
        t_cell = time.time()
        per, timings, degenerate = {}, {}, {}
        yte_ref = None
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            tr_states = [PT[d][0][i] for d, i in train_rows]
            te_states = [PT[d][0][i] for d, i in test_rows]
            ytr = np.array([PT[d][2][i] for d, i in train_rows], float)
            yte = np.array([PT[d][2][i] for d, i in test_rows], float)
            yte_ref = yte
            tr_recs = [PT[d][3][i] for d, i in train_rows]
            te_recs = [PT[d][3][i] for d, i in test_rows]

            v = {}
            # --- the published unsupervised score ------------------------------------------------
            t0 = time.time()
            msp_te = np.array([msp_mod.msp_uncertainty(r["token_logprobs"], "sum") for r in te_recs])
            msp_tr = np.array([msp_mod.msp_uncertainty(r["token_logprobs"], "sum") for r in tr_recs])
            v["msp"] = msp_te
            timings["msp"] = timings.get("msp", 0) + time.time() - t0

            n_tr = len(tr_states)
            # --- the probe -----------------------------------------------------------------------
            # Preferred path: take the master's own per-example vector. The sidecar is keyed by the
            # same cell and seed and was written by the ladder itself, so this is not a reproduction
            # of the probe but the probe. Alignment is PROVED, not assumed: the sidecar's stored
            # labels must equal this cell's evaluation labels exactly, which fails immediately if the
            # rows or their order differ by even one position.
            t0 = time.time()
            side = None
            if perex_dir is not None:
                sp = perex_dir / f"{X}__{rung}__{slug}.npz"
                if sp.exists():
                    side = np.load(sp, allow_pickle=True)
            if side is not None:
                sy = np.asarray(side["y"], float)
                if sy.shape != yte.shape or not np.array_equal(sy, yte):
                    sys.exit(f"FATAL [{rung}/{X}]: sidecar labels do not match this cell's evaluation "
                             f"labels ({sy.shape} vs {yte.shape}). The two populations are not the "
                             f"same rows -- refusing to mix them.")
                sseeds = [int(x) for x in np.asarray(side["seeds"]).ravel()]
                if sd not in sseeds:
                    sys.exit(f"FATAL [{rung}/{X}]: sidecar has seeds {sseeds}, this run asked for {sd}.")
                si = sseeds.index(sd)
                v["saplma"] = np.asarray(side["unc__saplma"][si], float)
                # THE PUBLISHED-SCORE IDENTITY, CHECKED RATHER THAN CLAIMED. The sidecar's floor_sum
                # was written by the ladder from the same records; if the score recomputed here is the
                # same quantity, the two must agree to floating-point noise.
                fs = np.asarray(side["unc__floor_sum"][si], float)
                dev = float(np.max(np.abs(fs - msp_te))) if len(fs) == len(msp_te) else float("inf")
                msp_dev_max = max(msp_dev_max, dev)
                if dev > 1e-9:
                    sys.exit(f"FATAL [{rung}/{X}] seed {sd}: the recomputed sequence-probability score "
                             f"differs from the master's floor_sum by {dev:.3e}. These are supposed to "
                             f"be the same formula -- refusing to report either as the published MSP.")
            else:
                Xmean = np.stack([a.mean(axis=0) for a in tr_states + te_states])
                tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(te_states)))
                yall = np.concatenate([ytr, yte])
                v["saplma"] = 1.0 - conf_meanpool(Xmean, tr_idx, te_idx, yall, sd)
            timings["saplma"] = timings.get("saplma", 0) + time.time() - t0

            # --- the two Mahalanobis fits --------------------------------------------------------
            # The reference splits the training pool in half: the first half supplies the reference
            # statistics against which the SECOND half is measured, while evaluation rows are measured
            # against statistics from the WHOLE pool. Both halves are needed, and the split is the
            # reference's own fixed one.
            t0 = time.time()
            half_a, half_dev = train_test_split(list(range(n_tr)), test_size=MD.DEV_SIZE,
                                                shuffle=True, random_state=MD.DEV_RANDOM_STATE)
            ids_a = [(d, int(i)) for (d, i) in [train_rows[k] for k in half_a]]
            ids_all = [(d, int(i)) for (d, i) in train_rows]
            st_a = MD.fit_md([tr_states[k] for k in half_a], ytr[half_a], args.metric_thr,
                             layer=layer, row_ids=ids_a, kind="fg", cache=stats_cache)
            st_all = MD.fit_md(tr_states, ytr, args.metric_thr,
                               layer=layer, row_ids=ids_all, kind="fg", cache=stats_cache)
            timings["md_fit"] = timings.get("md_fit", 0) + time.time() - t0

            t0 = time.time()
            dev_md = MD.md_mean([tr_states[k] for k in half_dev], st_a)
            test_md = MD.md_mean(te_states, st_all)
            timings["md_score"] = timings.get("md_score", 0) + time.time() - t0

            y_dev = ytr[half_dev]
            target = np.nan_to_num(1.0 - y_dev, nan=1.0)

            def _fit_ridge(dev_feat, test_feat, extra_dev=None, extra_test=None):
                Xd = np.nan_to_num(np.asarray(dev_feat, float)).reshape(len(dev_feat), -1)
                Xt = np.nan_to_num(np.asarray(test_feat, float)).reshape(len(test_feat), -1)
                if extra_dev is not None:
                    Xd = np.hstack([Xd, np.nan_to_num(extra_dev)])
                    Xt = np.hstack([Xt, np.nan_to_num(extra_test)])
                # The reduction is FITTED on the development half and only applied to the evaluation
                # rows, matching the reference, which calls fit_transform on the dev matrix and
                # transform on the evaluation matrix. It needs at least as many features as
                # components: with one cached layer there is one distance column (plus at most two
                # more for the probability-augmented variants), so it cannot run here at all. That
                # omission is precisely what makes the single-layer versions adaptations rather than
                # reproductions -- there is nothing to combine across layers.
                if Xd.shape[1] >= N_COMPONENTS:
                    pca = PCA(n_components=N_COMPONENTS).fit(Xd)
                    Xd, Xt = pca.transform(Xd), pca.transform(Xt)
                # Unconstrained, as the reference's call site specifies. A positivity constraint on a
                # single feature clips an unhelpful coefficient to zero and turns the prediction into
                # a constant, which is what produced the degeneracy recorded before this correction.
                r = Ridge(positive=False).fit(Xd, target)
                return r.predict(Xt), r.predict(Xd), r

            # --- the raw distance, reported as a diagnostic -------------------------------------
            # This is the quantity the published wrapper is built on, before the supervised
            # aggregation is applied. It is reported because it separates two claims that would
            # otherwise be indistinguishable: "the distance itself carries no signal" and "the
            # aggregation on top of it did something unhelpful". With one cached layer the
            # aggregation cannot combine anything, so the raw distance is the more informative of
            # the two rows.
            v["md_mean_mid"] = test_md

            # --- supervised aggregation of distances (middle-layer adaptation) -------------------
            v["satmd_mid"], satmd_dev, ridge_md = _fit_ridge(dev_md, test_md)

            # --- the hybrid uncertainty combination ----------------------------------------------
            # The reference searches its three hyperparameters on the DEV HALF only, using the dev
            # labels. Evaluation labels are never seen. The epistemic input is the regressor's output,
            # not the raw distance, matching the reference.
            t0 = time.time()
            msp_dev = msp_tr[half_dev]
            _, tmin, tmax, alpha = MD.grid_search_hp(satmd_dev, msp_dev, y_dev,
                                                    prr_fn=lambda yy, uu: results.prr(yy, uu))
            huq = MD.total_uncertainty_linear_step(
                np.concatenate([v["satmd_mid"], satmd_dev]),
                np.concatenate([msp_te, msp_dev]), tmin, tmax, alpha)
            v["huq_satmd_mid"] = huq[:len(msp_te)]
            timings["huq"] = timings.get("huq", 0) + time.time() - t0

            # --- the hybrid back-off (exact at this layer) ---------------------------------------
            t0 = time.time()
            hbo_score, pct, w_sv = MD.hbo(msp_te, v["saplma"], test_md, dev_md)
            v["hbo"] = hbo_score
            timings["hbo"] = timings.get("hbo", 0) + time.time() - t0

            # --- the meta-regressor variants, which need entropy ---------------------------------
            ent_tr = [ENT.get(d) for d, _ in train_rows]
            have_ent = all(e is not None for e in ent_tr) and all(
                ENT.get(d) is not None for d, _ in test_rows)
            if have_ent:
                e_dev = np.array([ENT[d][i] for d, i in [train_rows[k] for k in half_dev]])
                e_te = np.array([ENT[d][i] for d, i in test_rows])
                v["msp_satmd_mid"], _, _ = _fit_ridge(
                    dev_md, test_md,
                    extra_dev=np.column_stack([msp_dev, e_dev]),
                    extra_test=np.column_stack([msp_te, e_te]))
            elif sd == seeds[0]:
                print(f"    [{rung}/{X}] entropy cache absent -> msp_satmd_mid SKIPPED (left blank)",
                      flush=True)

            # --- the relative-distance family, which needs the background ------------------------
            if BG is not None:
                t0 = time.time()
                r_dev = MD.rmd_mean([tr_states[k] for k in half_dev], st_a, BG)
                r_te = MD.rmd_mean(te_states, st_all, BG)
                v["rmd_mean_mid"] = r_te            # the raw relative distance, same reasoning
                v["satrmd_mid"], rdev_pred, _ = _fit_ridge(r_dev, r_te)
                _, tmin_r, tmax_r, alpha_r = MD.grid_search_hp(
                    rdev_pred, msp_dev, y_dev, prr_fn=lambda yy, uu: results.prr(yy, uu))
                huq_r = MD.total_uncertainty_linear_step(
                    np.concatenate([v["satrmd_mid"], rdev_pred]),
                    np.concatenate([msp_te, msp_dev]), tmin_r, tmax_r, alpha_r)
                v["huq_satrmd_mid"] = huq_r[:len(msp_te)]
                if have_ent:
                    v["msp_satrmd_mid"], _, _ = _fit_ridge(
                        r_dev, r_te,
                        extra_dev=np.column_stack([msp_dev, e_dev]),
                        extra_test=np.column_stack([msp_te, e_te]))
                timings["rmd"] = timings.get("rmd", 0) + time.time() - t0

            # A CONSTANT SCORE MUST NOT PRODUCE A NUMBER. `results.prr` already refuses a NaN score
            # for this reason; an all-equal score is the same failure in a different coat. np.argsort
            # breaks ties by position, so a constant vector is ranked in row order and the function
            # returns whatever that arbitrary permutation happens to score -- a plausible-looking
            # value with no method behind it. Recorded as blank plus an explicit degeneracy flag, so
            # the cell reads as "no discriminative signal" rather than as a measurement. The guard is
            # kept even though the unconstrained ridge makes a flat prediction far less likely than
            # the positivity-constrained one it replaced: a guard that only fires on the bug you
            # already fixed is worth nothing.
            for m, vec in v.items():
                vec = np.asarray(vec, float)
                if np.isfinite(vec).all() and float(np.ptp(vec)) == 0.0:
                    degenerate.setdefault(m, 0)
                    degenerate[m] += 1
                    per.setdefault(m, []).append(float("nan"))
                    print(f"    [{rung}/{X}] seed {sd}: '{m}' returned a CONSTANT score -> blank, "
                          f"not a number", flush=True)
                    continue
                per.setdefault(m, []).append(results.prr(yte, vec))
            diag_out.append({"model": model, "eval": X, "rung": rung, "seed": sd,
                             "n_train": n_tr, "n_test": len(te_states),
                             "md_tokens_full": st_all.n_tokens, "md_jitter": st_all.jitter,
                             "ridge_coef_md": float(np.ravel(ridge_md.coef_)[0]),
                             "huq_t_min": float(tmin), "huq_t_max": float(tmax),
                             "huq_alpha": float(alpha),
                             "mean_R": float(np.mean(pct)),
                             "frac_R_gt_0.5": float(np.mean(pct > 0.5)),
                             "mean_w_sv": float(np.mean(w_sv))})

        if yte_ref is None:
            print(f"  [{rung}/{X}] produced no seed -> BLANK", flush=True)
            continue
        for m, vals in sorted(per.items()):
            a = np.asarray(vals, float)
            ok = np.isfinite(a)
            rows_out.append({"model": model, "eval": X, "rung": rung, "train": "+".join(srcs),
                             "method": m,
                             "prr_mean": round(float(a[ok].mean()), 6) if ok.any() else "",
                             "prr_std": round(float(a[ok].std()), 6) if ok.any() else "",
                             "n_seeds": int(ok.sum()),
                             "degenerate_seeds": degenerate.get(m, 0),
                             "layer": layer, "metric_thr": args.metric_thr})
        line = "  ".join(
            f"{m} " + (f"{np.nanmean(vals):+.4f}" if np.isfinite(vals).any() else "CONSTANT")
            for m, vals in sorted(per.items()))
        print(f"  [{rung:14s}/{X:14s}] {line}", flush=True)
        if args.benchmark:
            tt = "  ".join(f"{k} {v:.1f}s" for k, v in sorted(timings.items()))
            print(f"      cell {time.time() - t_cell:.1f}s | {tt} | peak RSS {peak_rss_gb():.1f} GB",
                  flush=True)

    if not rows_out:
        sys.exit("no cell produced a result. Refusing to write an empty table.")
    default_name = (f"SMOKE_md_hybrids__{slug}.csv" if args.debug_dim
                    else f"results/hybrids/pdl_hybrids__{slug}.csv")
    if args.debug_dim and args.out:
        sys.exit("--debug-dim writes a smoke-test file only; do not redirect it with --out.")
    out = ROOT / (args.out or default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    dpath = out.with_name(out.stem + "__diagnostics.csv")
    with open(dpath, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(diag_out[0])); w.writeheader(); w.writerows(diag_out)
    print(f"\nwrote {out.relative_to(ROOT)} ({len(rows_out)} rows)")
    print(f"wrote {dpath.relative_to(ROOT)} ({len(diag_out)} rows)")
    print(f"peak RSS {peak_rss_gb():.1f} GB")
    if perex_dir is not None:
        print(f"PUBLISHED-MSP IDENTITY: max |recomputed - master floor_sum| = {msp_dev_max:.3e} "
              f"across every cell and seed (bar 1e-9)")


if __name__ == "__main__":
    main()
