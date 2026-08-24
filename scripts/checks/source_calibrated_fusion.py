#!/usr/bin/env python
"""Source-calibrated fusion of the learned token weighting and the hidden-state probe.

Registered in prereg/M10_source_calibrated_cawsa_saplma_fusion.md, with amendment 1 of the same date.

WHAT THIS IS FOR
----------------
The existing fixed 50:50 rank ensemble of the two methods gives replicated evidence that their
signals are complementary, but it ranks each response against the whole evaluation cohort, so it
cannot score one response on its own. This estimator asks whether the complementary signal survives
when every target dependence is removed: each score is mapped through an empirical distribution
function built ONLY from the source rows the two models were fitted on, and the two percentiles are
averaged with a fixed, unsearched weight.

    q_m(x) = ( #{s_i < s} + 0.5 * #{s_i = s} + 0.5 ) / (n + 1)
    fusion = 0.5 * q_learned + 0.5 * q_probe

TARGET-INDEPENDENT, NOT PRODUCTION-READY
----------------------------------------
The property tested here is that one response's score does not depend on any other target example
(gates A and B below). That is target independence. It is not a claim about latency, robustness or
monitoring, and the words "production ready" do not belong anywhere near this estimator.

WHERE THE NUMBERS COME FROM
---------------------------
The TARGET-side component scores are taken from the canonical ladder's own per-example vectors, so
the fusion is built on the master's numbers rather than on a reproduction of them. The models are
retrained here only to score the SOURCE rows, which nobody has ever computed, and the retrained
target scores exist solely to prove row for row that the retrained model IS the frozen one.

    python scripts/checks/source_calibrated_fusion.py --model meta-llama/Meta-Llama-3.1-8B \
        --evals pubmed_qa --rungs 1ds-Diff --seeds 1 --gates-only
"""
import argparse
import csv as _csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from scipy.stats import rankdata, spearmanr                                     # noqa: E402

# torch, transformers and the project modules are imported INSIDE main(), after the structural gates.
# Gates A to D are pure arithmetic on the percentile map and need no model stack at all, so keeping
# the heavy imports out of module scope means they can be run in a second on any machine. That is the
# point of having them: a gate nobody can afford to run is not a gate.

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B"
W_FUSION = 0.5
# Row-level fidelity bar for gate E. The ladder's own equivalence gates use this order; a difference
# above it means the retrained model is not the frozen one and the run stops.
GATE_E_TOL = 1e-5


def ecdf_percentile(reference, x):
    """Mid-rank empirical distribution function, frozen by the registration.

    Deterministic, tie-aware, and never returns exactly 0 or 1. `reference` is the SOURCE score
    distribution; `x` are the scores to map. Nothing about `x` influences the mapping of any other
    element of `x` -- that independence is the whole point and is checked by gates A and B.
    """
    ref = np.sort(np.asarray(reference, float))
    x = np.asarray(x, float)
    n = len(ref)
    lo = np.searchsorted(ref, x, side="left")         # count of reference strictly below
    hi = np.searchsorted(ref, x, side="right")        # count at or below
    ties = hi - lo
    return (lo + 0.5 * ties + 0.5) / (n + 1)


def run_gates(verbose=True):
    """Gates A-D on synthetic data. Gate E needs a real cell and runs inside the grid."""
    rng = np.random.RandomState(0)
    ref_c, ref_s = rng.randn(500), rng.randn(500) * 3 + 1
    tgt_c, tgt_s = rng.randn(40), rng.randn(40) * 3 + 1
    fused = W_FUSION * ecdf_percentile(ref_c, tgt_c) + (1 - W_FUSION) * ecdf_percentile(ref_s, tgt_s)
    ok = {}

    # A: one example alone must equal the same example inside the batch, exactly.
    solo = np.array([W_FUSION * ecdf_percentile(ref_c, [tgt_c[i]])[0]
                     + (1 - W_FUSION) * ecdf_percentile(ref_s, [tgt_s[i]])[0]
                     for i in range(len(tgt_c))])
    ok["A no target dependence"] = (float(np.max(np.abs(solo - fused))) == 0.0,
                                    f"max|d| = {np.max(np.abs(solo - fused)):.3e} (must be exactly 0)")

    # B: permuting the target must permute the scores and change nothing else.
    p = rng.permutation(len(tgt_c))
    perm = W_FUSION * ecdf_percentile(ref_c, tgt_c[p]) + (1 - W_FUSION) * ecdf_percentile(ref_s, tgt_s[p])
    ok["B frozen reference"] = (float(np.max(np.abs(perm - fused[p]))) == 0.0,
                                f"max|d| = {np.max(np.abs(perm - fused[p])):.3e} (must be exactly 0)")

    # C: each percentile map must be non-decreasing in its own score.
    grid = np.linspace(-6, 6, 400)
    qc, qs = ecdf_percentile(ref_c, grid), ecdf_percentile(ref_s, grid)
    ok["C monotonicity"] = (bool((np.diff(qc) >= 0).all() and (np.diff(qs) >= 0).all()),
                            "both percentile maps non-decreasing")

    # D: worse on both components must fuse to more uncertain.
    lowc, lows = float(np.min(ref_c)) - 1, float(np.min(ref_s)) - 1
    hic, his = float(np.max(ref_c)) + 1, float(np.max(ref_s)) + 1
    f_low = W_FUSION * ecdf_percentile(ref_c, [lowc])[0] + (1 - W_FUSION) * ecdf_percentile(ref_s, [lows])[0]
    f_hi = W_FUSION * ecdf_percentile(ref_c, [hic])[0] + (1 - W_FUSION) * ecdf_percentile(ref_s, [his])[0]
    ok["D orientation"] = (f_hi > f_low, f"more-uncertain case fuses to {f_hi:.4f} vs {f_low:.4f}")

    if verbose:
        w = max(len(k) for k in ok)
        for k, (passed, detail) in ok.items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {k:<{w}}  {detail}")
    return all(v[0] for v in ok.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--evals", default="", help="default: every long-form source")
    ap.add_argument("--rungs", default="")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--perex-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--gates-only", action="store_true", help="run gates A-D and stop")
    ap.add_argument("--debug-dim", type=int, default=0,
                    help="truncate hidden states to N dims: SMOKE TEST ONLY, never a measurement")
    args = ap.parse_args()

    print("GATES A-D (synthetic, no data needed)")
    if not run_gates():
        sys.exit("FATAL: a structural gate failed. The estimator is not target-independent.")
    if args.gates_only:
        print("\ngates-only: stopping before the grid.")
        return

    # Heavy imports deferred to here so the structural gates above stay instant.
    global torch, AutoTokenizer, cache, probe, results, weighted_msp, MH, build_rows
    global LONG_SRC, cells_long, sampled_train_idx, CAWSA_KW
    import torch                                                                # noqa: E402
    from transformers import AutoTokenizer                                      # noqa: E402
    from luq import cache, probe, results, weighted_msp                         # noqa: E402
    from luq.weighting import shrink_to_uniform                                 # noqa: E402
    import md_hybrids as MH                                                     # noqa: E402
    from xl_rungs import build_rows                                             # noqa: E402
    from probe_drift_long import LONG_SRC, cells_long, sampled_train_idx        # noqa: E402
    # The frozen report-facing configuration. Not a tunable: the registration fixes shrinkage at 2
    # and the fusion weight at one half, and forbids searching either.
    CAWSA_KW = dict(weight_mode="normalised", reg=shrink_to_uniform, reg_lambda=2.0)

    slug = cache._slug(args.model)
    seeds = [int(s) for s in args.seeds.split(",")]
    evals = [e for e in (args.evals or ",".join(LONG_SRC)).split(",") if e]
    want_rungs = None
    if args.rungs:
        want_rungs = {MH.RUNG_ALIASES.get(r.strip(), r.strip()) for r in args.rungs.split(",")}
    MH._DEBUG_DIM = args.debug_dim
    smoke = bool(args.debug_dim)
    if smoke:
        print("=" * 90)
        print(f"SMOKE TEST -- hidden states truncated to {args.debug_dim} dims. NOT a measurement, and "
              f"gate E cannot pass here because the frozen vectors were made at full width.")
        print("=" * 90)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    # Same guard the ladder applies: without the real special-token ids a non-Llama model silently
    # falls back to a range test that zeroes ordinary content tokens. Reproducing the frozen scores
    # requires reproducing this too.
    if args.model != DEFAULT_MODEL:
        from luq import token_subsets
        weighted_msp.set_special_ids(tok.all_special_ids)
        token_subsets.set_special_ids(tok.all_special_ids)
        print(f"special-token ids registered from the {args.model} tokenizer "
              f"({len(tok.all_special_ids)} ids)")

    perex_dir = ROOT / (args.perex_dir or f"results/perex_clean__{slug}")
    if not perex_dir.is_dir():
        sys.exit(f"FATAL: frozen per-example vectors absent at {perex_dir}. The fusion must be built "
                 f"on the master's own target scores, not on a reproduction of them.")

    all_cells = [c for c in cells_long(set(LONG_SRC), evals) if c[0] != "Long->Short"]
    cells = all_cells if want_rungs is None else [c for c in all_cells if c[0] in want_rungs]
    if args.max_cells:
        cells = cells[:args.max_cells]
    cells = sorted(cells, key=lambda c: (c[0], tuple(sorted(d for d, _ in c[2])), c[1]))
    needed = {X for _, X, _ in cells} | {d for _, _, spec in cells for d, _ in spec}
    print(f"\nMODEL {args.model} | layer {args.layer} | {len(cells)} cells | seeds {seeds}")
    print(f"DATASETS: {sorted(needed)}", flush=True)

    PT = MH.load_population(args.model, args.layer, needed)
    rows_out, diag_out = [], []
    gateE_worst = 0.0

    for rung, X, spec in cells:
        if X not in PT or any(d not in PT for d, _ in spec):
            print(f"  [{rung}/{X}] inputs missing -> cell SKIPPED, left absent (never zero)", flush=True)
            continue
        sp = perex_dir / f"{X}__{rung}__{slug}.npz"
        if not sp.exists():
            print(f"  [{rung}/{X}] no frozen sidecar -> SKIPPED", flush=True)
            continue
        side = np.load(sp, allow_pickle=True)
        sseeds = [int(s) for s in np.asarray(side["seeds"]).ravel()]
        t_cell = time.time()
        per, degen = {}, {}

        for sd in seeds:
            if sd not in sseeds:
                print(f"  [{rung}/{X}] seed {sd} absent from the sidecar -> SKIPPED", flush=True)
                continue
            si = sseeds.index(sd)
            train_rows, test_rows = build_rows(X, spec, PT, sd, sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            # Exactly the ladder's own per-cell construction, so tr_idx/te_idx mean the same thing.
            n_tr = len(train_rows)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float)
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]

            sy = np.asarray(side["y"], float)
            if sy.shape != yte.shape or not np.array_equal(sy, yte):
                sys.exit(f"FATAL [{rung}/{X}]: sidecar labels do not match this cell's evaluation "
                         f"labels. Refusing to mix two populations.")

            # --- train ONCE per cell and seed, then score source and target -----------------------
            t0 = time.time()
            wm = weighted_msp.train_weighted_msp(states, records, y, tr_idx, device,
                                                 length_normalise=True, seed=sd, **CAWSA_KW)
            c_src = np.asarray(weighted_msp.predict_weighted_msp(
                wm, states, records, tr_idx, device, length_normalise=True,
                weight_mode=CAWSA_KW["weight_mode"]), float)
            c_tgt_re = np.asarray(weighted_msp.predict_weighted_msp(
                wm, states, records, te_idx, device, length_normalise=True,
                weight_mode=CAWSA_KW["weight_mode"]), float)
            t_cawsa = time.time() - t0

            t0 = time.time()
            Xmean = np.stack([s.mean(axis=0) for s in states])
            clf = probe.train_probe_mlp(Xmean[tr_idx], y[tr_idx], seed=sd)
            s_src = 1.0 - np.asarray(clf.p_correct(Xmean[tr_idx]), float)
            s_tgt_re = 1.0 - np.asarray(clf.p_correct(Xmean[te_idx]), float)
            t_probe = time.time() - t0

            # --- GATE E: row-level raw-score fidelity against the frozen vectors ------------------
            # Amendment 1.3. Agreement of prediction-rejection ratio is NOT accepted in place of this:
            # two different vectors can score the same value. If the retrained model does not
            # reproduce the frozen one row for row, it is not the frozen model, and the source
            # reference distribution it produces is not the frozen model's either.
            c_tgt = np.asarray(side["unc__wmsp_shrink2"][si], float)
            s_tgt = np.asarray(side["unc__saplma"][si], float)
            eg = {}
            for nm, a, b in (("cawsa", c_tgt_re, c_tgt), ("saplma", s_tgt_re, s_tgt)):
                if a.shape != b.shape:
                    sys.exit(f"FATAL [{rung}/{X}] seed {sd}: {nm} recomputed {a.shape} vs frozen "
                             f"{b.shape} -- different row sets.")
                d = np.abs(a - b)
                rho = float(spearmanr(a, b).statistic) if len(a) > 2 else float("nan")
                disc = float(np.mean(np.sign(np.subtract.outer(a, a))
                                     != np.sign(np.subtract.outer(b, b))))
                eg[nm] = (float(d.max()), float(d.mean()), rho, disc)
                gateE_worst = max(gateE_worst, float(d.max()))
            if not smoke:
                for nm, (mx, mn, rho, disc) in eg.items():
                    if mx > GATE_E_TOL:
                        sys.exit(
                            f"FATAL [{rung}/{X}] seed {sd}: GATE E FAILED for '{nm}'.\n"
                            f"  max|d| = {mx:.3e}  mean|d| = {mn:.3e}  spearman = {rho:.6f}  "
                            f"discordant pairs = {disc:.4f}  (bar {GATE_E_TOL:g})\n"
                            f"  The retrained model does not reproduce the frozen one row for row, so "
                            f"it is not the frozen model and its SOURCE reference distribution cannot "
                            f"be trusted either. Stopping rather than widening the tolerance or "
                            f"falling back to agreement of the rejection ratio.")

            # --- the fusion, on the FROZEN target vectors ----------------------------------------
            qc = ecdf_percentile(c_src, c_tgt)
            qs = ecdf_percentile(s_src, s_tgt)
            fusion = W_FUSION * qc + (1 - W_FUSION) * qs
            # The comparator: the same 50:50 combination taken over the TARGET cohort's own ranks,
            # which is what makes it a diagnostic rather than an estimator.
            cohort = W_FUSION * rankdata(c_tgt) + (1 - W_FUSION) * rankdata(s_tgt)

            v = {"saplma": s_tgt, "cawsa": c_tgt, "cawsa_saplma_src50": fusion,
                 "cawsa_saplma_cohort50": cohort}
            for m, vec in v.items():
                vec = np.asarray(vec, float)
                if np.isfinite(vec).all() and float(np.ptp(vec)) == 0.0:
                    degen[m] = degen.get(m, 0) + 1
                    per.setdefault(m, []).append(float("nan"))
                    continue
                per.setdefault(m, []).append(results.prr(yte, vec))

            diag_out.append(dict(
                model=args.model, eval=X, rung=rung, seed=sd,
                n_source=len(tr_idx), n_target=len(te_idx),
                # Descriptive only (amendment 1.4). A fixed half-and-half on the percentile scale is
                # equal weighting only if the two spreads are comparable. Reported, never acted on.
                q_cawsa_sd=float(np.std(qc)), q_saplma_sd=float(np.std(qs)),
                q_cawsa_iqr=float(np.subtract(*np.percentile(qc, [75, 25]))),
                q_saplma_iqr=float(np.subtract(*np.percentile(qs, [75, 25]))),
                gateE_cawsa_max=eg["cawsa"][0], gateE_cawsa_mean=eg["cawsa"][1],
                gateE_cawsa_spearman=eg["cawsa"][2], gateE_cawsa_discordant=eg["cawsa"][3],
                gateE_saplma_max=eg["saplma"][0], gateE_saplma_mean=eg["saplma"][1],
                gateE_saplma_spearman=eg["saplma"][2], gateE_saplma_discordant=eg["saplma"][3],
                t_cawsa_s=round(t_cawsa, 1), t_probe_s=round(t_probe, 1)))

        if not per:
            continue
        train_lab = "+".join(d for d, _ in spec)
        for m, vals in sorted(per.items()):
            a = np.array(vals, float); okv = a[np.isfinite(a)]
            rows_out.append(dict(model=args.model, eval=X, rung=rung, train=train_lab, method=m,
                                 prr_mean=(round(float(okv.mean()), 6) if len(okv) else ""),
                                 prr_std=(round(float(okv.std()), 6) if len(okv) else ""),
                                 n_seeds=len(okv), degenerate_seeds=degen.get(m, 0)))
        line = "  ".join(f"{m} {np.nanmean(per[m]):+.3f}" for m in sorted(per))
        print(f"  [{rung:14s}/{X:14s}] {line}   ({time.time() - t_cell:.0f}s)", flush=True)

    if smoke:
        print("\nSMOKE TEST COMPLETE -- nothing written.")
        return
    if not rows_out:
        sys.exit("FATAL: no cell produced a result.")
    out = ROOT / (args.out or f"results/hybrids/pdl_fusion__{slug}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    dpath = out.with_name(out.stem + "__diagnostics.csv")
    with open(dpath, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(diag_out[0])); w.writeheader(); w.writerows(diag_out)
    print(f"\nwrote {out.relative_to(ROOT)} ({len(rows_out)} rows)")
    print(f"wrote {dpath.relative_to(ROOT)} ({len(diag_out)} rows)")
    print(f"GATE E: worst row-level |d| over every cell and seed = {gateE_worst:.3e} "
          f"(bar {GATE_E_TOL:g})")
    print(f"peak RSS {MH.peak_rss_gb():.1f} GB")


if __name__ == "__main__":
    main()
