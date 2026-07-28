"""Extend the headline result (PART XV.6/XV.8): does the PROBE share MSP's length decay?

XV established: MSP's PRR decays monotonically with generation length across regimes (corr −0.896), while
the hidden-state PROBE does not (sciq Q4: MSP +0.38 vs probe +0.91; cnn/xsum: probe beats MSP at every band).
That asymmetry is the strongest single argument for the white-box bet. It was shown on sciq + cnn + xsum
only. This generalises it to every dataset with a pooled feature cache, ID, by length quartile.

Per dataset: train the mean-pool SAPLMA probe on the train split, then WITHIN each length quartile of the
test split score BOTH the probe and the best MSP-family floor (sum/perplexity/min), and report the gap.

GUARD (the check that cleared a false alarm in XV.2): also report the LABEL distribution per quartile
(mean correctness, std, frac@0, frac@1). If a quartile's PRR looks anomalous it must be checked against the
labels before it is believed -- a degenerate (all-correct or no-variance) subgroup produces a meaningless PRR.

CPU-only: reads the cached pooled features (all layers, we use L15) + records. No GPU, no new extraction.
    python scripts/checks/probe_vs_msp_length.py
"""
import csv as _csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import msp, probe, results  # noqa: E402

MODEL_SLUG = "meta-llama_Meta-Llama-3.1-8B"
LAYER = 15
N_BOOT = 5000        # A3 (2026-07-27): raised 500 -> 5000. At 500 each 95% tail was ~12 points; two
                     # significant cells sat at CI-lo +0.070 / +0.011, so the 16/27 count was budget-unstable.
FDR_ALPHA = 0.05     # A3: Benjamini-Hochberg FDR across the 27 cells (no per-cell correction otherwise).
# Core-6 (real train/test) + the XL sets (eval-only / train-only -> carved train/test below). 2026-07-27:
# extended per the V-round to cover the long/factuality sets (expertqa = longest-output, the best case).
DATASETS = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail",
            "med_quad", "samsum", "expertqa", "asqa", "factscore"]
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}  # namespaced caches
LABEL = {"expertqa": "factuality", "factscore": "factuality"}                                # rest: correctness


def load(dataset):
    # PIN THE MODEL (bugfix 2026-07-27): the old model-agnostic globs `*{dataset}*` matched the DROPPED
    # Qwen-2.5-1.5B dev-model cache too, and glob[0] returned it (alphabetically first) for datasets that
    # still have a Qwen file (pubmed, xsum) -> PART A's pubmed/xsum were computed on Qwen, not Llama.
    base = ROOT / "cache" / REGIME[dataset] if dataset in REGIME else ROOT / "cache"
    fc = glob.glob(str(base / "features" / f"{MODEL_SLUG}__{dataset}__ID__saplma.npz"))
    rc = glob.glob(str(base / "records" / f"{MODEL_SLUG}__{dataset}__ID.jsonl"))
    if not fc or not rc:
        return None
    feats = np.load(fc[0])["feats"][:, LAYER, :]                 # (n, 4096); guard: Llama hidden dim = 4096
    if feats.shape[-1] != 4096:
        raise SystemExit(f"{dataset}: feature hidden dim {feats.shape[-1]} != 4096 (Llama). Wrong-model cache?")
    recs = [json.loads(l) for l in open(rc[0])]
    if len(recs) != len(feats):
        print(f"  {dataset}: feats {len(feats)} != records {len(recs)} -> skip", flush=True)
        return None
    y = np.array([r.get(LABEL.get(dataset, "correctness"), np.nan) for r in recs], float)
    split = np.array([r.get("split") for r in recs])
    length = np.array([len(r["token_logprobs"]) for r in recs], float)
    return feats, recs, y, split, length


def main():
    out_rows = []
    print(f"{'dataset':14s} {'quartile':>8s} {'n':>5s} {'medlen':>7s} {'meanY':>6s} "
          f"{'MSP':>7s} {'PROBE':>7s} {'gap':>7s}", flush=True)
    print("-" * 72)
    for d in DATASETS:
        loaded = load(d)
        if loaded is None:
            print(f"  {d}: no cache -> skip", flush=True)
            continue
        feats, recs, y, split, length = loaded
        ok = np.isfinite(y)
        okv = np.where(ok)[0]
        # Core sets have a real train/test split -> use it. Eval-only (expertqa/asqa/factscore) and train-only
        # (med_quad/samsum) sets have ONE split value -> carve a DETERMINISTIC 75/25 (seed 0), so the probe has
        # both. This mirrors xl_rungs.eval_split's "one stable test set" for split-less XL evals.
        if len(np.unique(split[ok])) >= 2:
            tr = np.where((split == "train") & ok)[0]
            te = np.where((split == "test") & ok)[0]
        else:
            perm = np.random.RandomState(0).permutation(okv)
            n_te = int(round(len(okv) * 0.25))
            te, tr = perm[:n_te], perm[n_te:]
        if len(tr) < 150 or len(te) < 120:
            print(f"  {d}: too few labelled rows (tr={len(tr)}, te={len(te)}) -> skip", flush=True)
            continue
        clf = probe.train_probe_mlp(feats[tr], y[tr], seed=1)
        unc = probe.uncertainty(clf, feats[te])                 # probe uncertainty on test
        qs = np.percentile(length[te], [25, 50, 75])
        bands = [(-1, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 1e18)]
        for qi, (lo, hi) in enumerate(bands):
            m = (length[te] > lo) & (length[te] <= hi)
            if m.sum() < 60:
                continue
            yy = y[te][m]
            if yy.std() < 1e-6:                                 # degenerate subgroup -> PRR meaningless
                print(f"  {d} Q{qi+1}: no label variance (mean {yy.mean():.2f}) -> PRR undefined, skipped",
                      flush=True)
                continue
            sub = [recs[i] for i in te[m]]
            msp_vecs = {a: np.array([msp.msp_uncertainty(r["token_logprobs"], a) for r in sub])
                        for a in ("sum", "perplexity", "min")}
            prr_a = {a: results.prr(yy, v) for a, v in msp_vecs.items()}
            msp_bar = prr_a["min"]                          # the PRE-REGISTERED msp_min bar
            mspv = max(prr_a.values())                      # best-of-three = the baseline's BEST shot
            pu = unc[m]; prbv = results.prr(yy, pu)
            # A1: the error set is PRR's real denominator, not n. Two measures:
            #   n_err   = LITERAL hard-zero count (label==0) -- faithful for the BINARY string-match sets.
            #   err_mass= n*(1-meanY) -- the effective error count. For binary Y this EQUALS n_err exactly
            #             (n*(1-#ones/n) = #zeros); for the GRADED judge sets the judge rarely returns exactly
            #             0.0, so n_err badly undercounts and err_mass is the honest denominator. Flag thinness
            #             on err_mass (label-type-agnostic); cnn has n_err~4 but err_mass~300 = NOT thin.
            n_err = int((yy == 0).sum())
            err_mass = float(len(yy) * (1.0 - yy.mean()))
            # Bootstrap CI + one-sided p on (probe - best-of-three): best-of-three recomputed per resample
            # (the winning aggregate may vary). N_BOOT resamples, seed 1 -> the noise floor on the gap.
            rs = np.random.RandomState(1); diffs = np.empty(N_BOOT)
            for b in range(N_BOOT):
                ix = rs.randint(0, len(yy), len(yy)); yb = yy[ix]
                diffs[b] = results.prr(yb, pu[ix]) - max(results.prr(yb, v[ix]) for v in msp_vecs.values())
            ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
            p_boot = float((diffs <= 0).mean())     # one-sided bootstrap p (H1: probe > best-of-3)
            medlen = float(np.median(length[te][m]))
            flag = "  <THIN err_mass<30" if err_mass < 30 else ""
            print(f"{d:14s} {'Q'+str(qi+1):>8s} {int(m.sum()):5d} z={n_err:4d} m={err_mass:6.0f} {medlen:7.0f} "
                  f"{yy.mean():6.2f} {mspv:+7.3f} {prbv:+7.3f} {prbv-mspv:+7.3f} [{ci_lo:+.3f},{ci_hi:+.3f}] "
                  f"p={p_boot:.4f}{flag}", flush=True)
            out_rows.append({"dataset": d, "quartile": qi + 1, "n": int(m.sum()), "n_err": n_err,
                             "err_mass": round(err_mass, 1),
                             "median_len": round(medlen, 1), "mean_correctness": round(float(yy.mean()), 3),
                             "label_std": round(float(yy.std()), 3),
                             "msp_sum": round(prr_a["sum"], 4), "msp_perplexity": round(prr_a["perplexity"], 4),
                             "msp_min": round(msp_bar, 4), "msp_bestof3": round(mspv, 4),
                             "probe_prr": round(prbv, 4),
                             "gap_probe_minus_best3": round(prbv - mspv, 4),
                             "gap_probe_minus_mspmin": round(prbv - msp_bar, 4),
                             "gap_ci_lo": round(ci_lo, 4), "gap_ci_hi": round(ci_hi, 4),
                             "p_boot": round(p_boot, 4), "n_boot": N_BOOT})

    # INVARIANT (V5 guard, 2026-07-27): best-of-three is BY CONSTRUCTION >= each of {sum,perplexity,min}.
    # Assert it on the EMITTED rows so any future refactor of the emit/merge step fails loud — the same class
    # of bug that (in hand-transcription) once put sciq's msp_min into trivia's row.
    for r in out_rows:
        assert r["msp_bestof3"] >= max(r["msp_sum"], r["msp_perplexity"], r["msp_min"]) - 1e-9, \
            f"INVARIANT VIOLATED {r['dataset']} Q{r['quartile']}: best3 {r['msp_bestof3']} < a variant"
    # A3: Benjamini-Hochberg FDR across ALL cells on the one-sided bootstrap p-values. Reject H0 for the
    # largest k with p_(k) <= (k/m)*alpha, then flag every cell with p <= that threshold.
    if out_rows:
        m_tests = len(out_rows)
        ranked = sorted(range(m_tests), key=lambda i: out_rows[i]["p_boot"])
        bh_thresh = -1.0
        for rank, i in enumerate(ranked, start=1):
            if out_rows[i]["p_boot"] <= (rank / m_tests) * FDR_ALPHA:
                bh_thresh = max(bh_thresh, out_rows[i]["p_boot"])
        for r in out_rows:
            r["bh_significant"] = int(bh_thresh >= 0 and r["p_boot"] <= bh_thresh)
    out = ROOT / "results" / f"probe_vs_msp_length__{MODEL_SLUG}.csv"
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["dataset", "quartile", "n", "n_err", "err_mass", "median_len",
                                           "mean_correctness", "label_std", "msp_sum", "msp_perplexity",
                                           "msp_min", "msp_bestof3", "probe_prr", "gap_probe_minus_best3",
                                           "gap_probe_minus_mspmin", "gap_ci_lo", "gap_ci_hi", "p_boot",
                                           "n_boot", "bh_significant"])
        w.writeheader(); w.writerows(out_rows)
    # summary: headline = probe vs best-of-three (the baseline's BEST shot). Report the raw CI count, the
    # BH-adjusted count, and how many cells are noise-dominated (n_err<30).
    if out_rows:
        g3 = [r["gap_probe_minus_best3"] for r in out_rows]
        sig = [r for r in out_rows if r["gap_ci_lo"] > 0]                 # CI excludes 0 -> significant win
        bh = [r for r in out_rows if r["bh_significant"]]
        thin = [r for r in out_rows if r["err_mass"] < 30]
        print(f"\nprobe beats best-of-three MSP: {sum(1 for g in g3 if g > 0)}/{len(g3)} cells (>0), "
              f"{len(sig)}/{len(out_rows)} SIGNIFICANT (raw 95% CI > 0), {len(bh)}/{len(out_rows)} survive "
              f"BH-FDR@{FDR_ALPHA}; mean gap {np.mean(g3):+.3f} ({N_BOOT} resamples)", flush=True)
        print(f"  thin cells (err_mass<30, noise-dominated): {len(thin)} -> "
              f"{', '.join(r['dataset']+' Q'+str(r['quartile']) for r in thin)}", flush=True)
        print(f"  (vs the msp_min bar: {sum(1 for r in out_rows if r['gap_probe_minus_mspmin']>0)}/{len(out_rows)}; "
              f"mean {np.mean([r['gap_probe_minus_mspmin'] for r in out_rows]):+.3f})", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
