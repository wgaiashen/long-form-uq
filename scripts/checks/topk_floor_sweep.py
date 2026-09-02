#!/usr/bin/env python
"""Top-k floor sweep -- WHERE DOES THE ERROR SIGNAL LIVE?  (§D.6; no training, no GPU.)

For each dataset, PRR of the *mean of the k lowest token logprobs* as k sweeps 1..all.
The endpoints are already-known floor variants:
    k=1    -> the single lowest logprob  == msp_min's RANKING  (1-exp(lp_min) is monotone in -lp_min)
    k=all  -> the mean logprob           == perplexity          (identical, -mean(lp))
Everything in between is unexplored. Uses cached record["token_logprobs"] ONLY.

CORRECTNESS GATE (printed BEFORE the curve is read): my k=1 PRR must equal the published §B.2
msp_min and my k=all PRR must equal published perplexity, on the SAME eval population the floor
table uses (xl_rungs.eval_split, the driver's own test split). If an endpoint misses §B.2 the
sweep is on the wrong population and nothing else is trustworthy.
"""
import argparse
import os
import sys, csv as _csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "checks"))
from luq import cache, msp, results          # noqa: E402
from luq.config import Config                 # noqa: E402
from xl_rungs import eval_split, label_of     # noqa: E402
from attn_pool import PROMPT_REGIME           # noqa: E402

import matplotlib                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt               # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
DATASETS = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail",
            "med_quad", "samsum", "expertqa", "asqa"]

# Published §B.2 floor PRR (from the probedriftlong ID CSVs; sciq/trivia from the §B.2 motivation table).
# These are the EXTERNAL gate targets -- my computed msp_min / perplexity must reproduce them.
PUBLISHED = {
    "sciq":          {"min": 0.755,   "ppl": 0.541},
    "trivia_qa":     {"min": 0.747,   "ppl": 0.733},
    "pubmed_qa":     {"min": 0.371,   "ppl": -0.1736},
    "xsum":          {"min": -0.0149, "ppl": -0.1719},
    "cnn_dailymail": {"min": 0.1198,  "ppl": 0.4096},
    "med_quad":      {"min": 0.1492,  "ppl": 0.0771},
    "samsum":        {"min": -0.0243, "ppl": 0.1128},
    "expertqa":      {"min": 0.2054,  "ppl": 0.0393},
    "asqa":          {"min": 0.2498,  "ppl": 0.3161},
    # Added 2026-09-01 so the sweep can cover the full eight-dataset long-form panel. FActScore was
    # absent from the historical run, which is why the original result is a seven-dataset one; the
    # two panels are labelled and must never be averaged together. Values read from the matched
    # setting of results/pdl_master__meta-llama_Meta-Llama-3.1-8B.csv, where they are identical to
    # the corrected-span master because this dataset carries no span correction.
    "factscore":     {"min": 0.4283,  "ppl": 0.3260},
}
K_ABS = [1, 2, 3, 5, 10, 25, 50, 100]      # + "all"
FRACS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
# REGIME-AWARE GATE REFERENCE. PUBLISHED above is the ORIGINAL-span reference. A dataset redirected
# to a corrected-span cache root (LUQ_REGIME="med_quad=cleanv2") legitimately has different floors,
# so checking it against the original values would be a false failure. Only redirected datasets get
# the corrected reference; every dataset still reading its original cache is still checked against
# PUBLISHED, and that residual check is the invariance test for the unchanged datasets. Values are
# the matched-setting floor cells of results/cleanv2/pdl_cleanv2_master__meta-llama_Meta-Llama-3.1-8B.csv.
PUBLISHED_CORRECTED_SPAN = {
    "med_quad": {"min": -0.0246, "ppl": +0.1118},
}


def published_for(dataset):
    """Expected floors for `dataset`, accounting for an active cache-root override."""
    active = dict(item.split("=", 1) for item in os.environ.get("LUQ_REGIME", "").split(",")
                  if "=" in item)
    if active.get(dataset, "").strip() == "cleanv2":
        if dataset not in PUBLISHED_CORRECTED_SPAN:
            raise SystemExit(
                f"{dataset} is redirected to the corrected-span cache but no corrected-span floor "
                "reference is registered for it. Refusing to gate corrected data against the "
                "original reference, and refusing to skip the check.")
        return PUBLISHED_CORRECTED_SPAN[dataset]
    return PUBLISHED[dataset]


GATE_EXT_TOL = 0.01      # my floor vs published §B.2 (CSV is rounded to 4dp; seed-stable so should be ~exact)
GATE_INT_TOL = 1e-9      # k=1 vs msp_min ranking, k=all vs perplexity (identical by construction)


def load_light(dataset):
    """records + split + y ONLY (no per-token states) -- the floor needs only token_logprobs/split/y.
    Mirrors attn_pool.load_per_token's record/split/y logic + probedriftlong's finite filter, minus states."""
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    records = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, dataset, "ID"))
    lf = label_of(dataset)
    split = np.array([r["split"] for r in records])
    y = np.array([r.get(lf, np.nan) for r in records], dtype=float)
    finite = np.isfinite(y)                 # IDENTICAL to probedriftlong.py:169-175
    if not finite.all():
        keep = np.where(finite)[0]
        records = [records[k] for k in keep]; split = split[keep]; y = y[keep]
    return records, split, y, lf


def score_lowk(lp, k):
    """-(mean of the k lowest logprobs). k>=T -> -mean(all) == perplexity. Higher = more uncertain."""
    lp = np.asarray(lp, dtype=float)
    kk = min(int(k), len(lp))
    return float(-np.sort(lp)[:kk].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help="comma list to sweep. The default is the historical panel; pass an "
                         "explicit list to run a different coverage panel, and label it with --panel.")
    ap.add_argument("--panel", default="historical",
                    help="label written into every row, so two coverage panels can share one file "
                         "without their macros ever being averaged together by accident.")
    ap.add_argument("--out", default=None,
                    help="output CSV path. Give one when running under a cache-root override, so a "
                         "corrected-span table can never overwrite the original-span one.")
    ap.add_argument("--append", action="store_true",
                    help="append to --out instead of replacing it, for writing a second panel.")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in PUBLISHED and d not in PUBLISHED_CORRECTED_SPAN]
    if unknown:
        raise SystemExit(f"no floor reference registered for {unknown}; the endpoint gate cannot run")

    out_rows = []          # (panel, dataset, sweep, k, prr, n_eval, med_len)
    curves = {}            # dataset -> (xs, prrs, med_len, prr_k1, prr_kall)
    feats = {}             # dataset -> label-free features + oracle-k
    gate_all_ok = True

    print("=" * 78)
    print("ENDPOINT GATE (reproduce §B.2 BEFORE reading the curve)")
    print("=" * 78)
    for d in datasets:
        records, split, y, lf = load_light(d)
        _, te = eval_split(split)                       # default seed=0 -> matches probedriftlong exactly
        yte = y[te]
        lps = [np.asarray(records[i]["token_logprobs"], dtype=float) for i in te]
        Ts = np.array([len(l) for l in lps])
        med_len = float(np.median(Ts))

        # ---- endpoint gate ----
        v_min = np.array([msp.msp_uncertainty(l, "min") for l in lps])
        v_ppl = np.array([msp.msp_uncertainty(l, "perplexity") for l in lps])
        prr_min = results.prr(yte, v_min); prr_ppl = results.prr(yte, v_ppl)
        v_k1 = np.array([score_lowk(l, 1) for l in lps])
        v_kall = np.array([score_lowk(l, 10**9) for l in lps])
        prr_k1 = results.prr(yte, v_k1); prr_kall = results.prr(yte, v_kall)
        pub = published_for(d)
        d_ext_min = abs(prr_min - pub["min"]); d_ext_ppl = abs(prr_ppl - pub["ppl"])
        d_int_1 = abs(prr_k1 - prr_min); d_int_all = abs(prr_kall - prr_ppl)
        ok = (d_ext_min < GATE_EXT_TOL and d_ext_ppl < GATE_EXT_TOL
              and d_int_1 < GATE_INT_TOL and d_int_all < GATE_INT_TOL)
        gate_all_ok = gate_all_ok and ok
        print(f"[{d:14s}] n={len(te):4d} medT={med_len:6.1f} label={lf:12s} | "
              f"msp_min me {prr_min:+.4f} vs §B.2 {pub['min']:+.4f} Δ{d_ext_min:.4f} | "
              f"ppl me {prr_ppl:+.4f} vs §B.2 {pub['ppl']:+.4f} Δ{d_ext_ppl:.4f} | "
              f"k1==min Δ{d_int_1:.1e} kall==ppl Δ{d_int_all:.1e}  {'PASS' if ok else 'FAIL <=='}",
              flush=True)

        # ---- absolute-k sweep ----
        xs, prrs = [], []
        for k in K_ABS:
            v = np.array([score_lowk(l, k) for l in lps])
            p = results.prr(yte, v)
            out_rows.append((d, "abs", k, p, len(te), med_len)); xs.append(k); prrs.append(p)
        out_rows.append((d, "abs", "all", prr_kall, len(te), med_len))
        xs.append("all"); prrs.append(prr_kall)
        curves[d] = (xs, prrs, med_len, prr_k1, prr_kall)

        # ---- fractional-k sweep (k as a fraction of each example's T) ----
        frac_prrs = []
        for f in FRACS:
            v = np.array([score_lowk(l, max(1, int(round(f * len(l))))) for l in lps])
            p = results.prr(yte, v)
            out_rows.append((d, "frac", f, p, len(te), med_len)); frac_prrs.append(p)

        # ---- oracle-k (DIAGNOSTIC only) + label-free features for the predictability test ----
        grid_prr = prrs[:]                              # abs-k grid incl 'all'
        best_i = int(np.argmax(grid_prr))
        best_k = xs[best_i]; best_prr = grid_prr[best_i]
        pooled = np.concatenate(lps)
        # label-free shape features (per dataset): concentration/spread descriptors of the logprob dist
        per_ex_gap = np.array([float(np.mean(l) - np.min(l)) for l in lps])   # min-mean gap (concentration)
        feats[d] = {
            "med_len": med_len,
            "skew": float(((pooled - pooled.mean()) ** 3).mean() / (pooled.std() ** 3 + 1e-12)),
            "tail_mass": float(np.mean([np.sort(l)[:max(1, len(l) // 10)].sum() / (l.sum() + 1e-12) for l in lps])),
            "gap": float(np.mean(per_ex_gap)),          # mean over examples of (mean-min) logprob gap
            "best_k": best_k, "best_prr": best_prr,
            "prr_min": prr_min, "prr_ppl": prr_ppl,
            "frac_prrs": frac_prrs,
        }

    print("\nGATE OVERALL:", "PASS -- curve is trustworthy" if gate_all_ok
          else "FAIL -- population wrong, DO NOT read the curve")

    # ---- write the sweep CSV ----
    if args.out:
        csv_path = Path(args.out)
        outdir = csv_path.parent
    else:
        outdir = Path(os.environ.get("EPHEMERAL", str(Path.home() / "ephemeral"))
                      + "/luq_overnight_results")
        csv_path = outdir / "topk_floor_sweep__meta-llama_Meta-Llama-3.1-8B.csv"
    outdir.mkdir(parents=True, exist_ok=True)
    append = args.append and csv_path.exists()
    with open(csv_path, "a" if append else "w", newline="") as fh:
        w = _csv.writer(fh)
        if not append:
            w.writerow(["panel", "dataset", "sweep", "k", "prr", "n_eval", "med_len"])
        for r in out_rows:
            w.writerow([args.panel] + list(r))
    print(f"\nwrote {csv_path}  ({len(out_rows)} rows)")

    # ---- figure: PRR vs k, one curve per dataset, log-x, k=1 + k=all marked ----
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    cmap = plt.get_cmap("tab10")
    for j, d in enumerate(datasets):
        xs, prrs, med_len, prr_k1, prr_kall = curves[d]
        xnum = [x for x in xs if x != "all"]
        pnum = prrs[:len(xnum)]
        c = cmap(j % 10)
        ax.plot(xnum, pnum, "-o", color=c, ms=4, lw=1.6, label=f"{d} (medT={med_len:.0f})")
        ax.plot([med_len], [prr_kall], marker="*", color=c, ms=13, mec="k", mew=0.5, zorder=5)  # k=all @ medT
        ax.plot([xnum[0]], [pnum[0]], marker="s", color=c, ms=6, mec="k", mew=0.5, zorder=5)    # k=1
    ax.set_xscale("log")
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("k  (mean of the k lowest token logprobs;  = k=1 IS msp_min,  = k=all IS perplexity @ medT)")
    ax.set_ylabel("PRR")
    ax.set_title("Where does the error signal live? top-k floor sweep (§D.6)")
    ax.legend(fontsize=7.5, ncol=2, loc="best")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    png = csv_path.with_suffix(".png")
    fig.savefig(png, dpi=130); print(f"wrote {png}")

    # ---- oracle-k table (DIAGNOSTIC -- flagged, never a bar) ----
    print("\n" + "=" * 78)
    print("ORACLE-k per dataset (DIAGNOSTIC ONLY -- label-selected, NOT a baseline the methods beat)")
    print("=" * 78)
    print(f"{'dataset':14s}{'best_k':>8s}{'best_prr':>10s}{'msp_min':>10s}{'ppl':>9s}{'gain_vs_best_pub':>18s}")
    for d in datasets:
        f = feats[d]
        best_pub = max(f["prr_min"], f["prr_ppl"])
        print(f"{d:14s}{str(f['best_k']):>8s}{f['best_prr']:>+10.4f}{f['prr_min']:>+10.4f}"
              f"{f['prr_ppl']:>+9.4f}{f['best_prr'] - best_pub:>+18.4f}")

    # ---- predictability of k, LEAVE-ONE-DATASET-OUT (the interesting outcome; n=9 -> exploratory) ----
    # Map each dataset to a target on the FRACTIONAL grid (best fractional-k), predict it label-free from a
    # single shape feature via leave-one-out, and score PRR at the predicted frac-k vs the fixed endpoints.
    print("\n" + "=" * 78)
    print("PREDICTABILITY of k, LODO (label-free) -- n=9, EXPLORATORY")
    print("=" * 78)
    ds = DATASETS
    best_frac_i = {d: int(np.argmax(feats[d]["frac_prrs"])) for d in ds}   # index into FRACS
    for fname in ["gap", "med_len", "skew", "tail_mass"]:
        xf = np.array([feats[d][fname] for d in ds], float)
        yf = np.array([best_frac_i[d] for d in ds], float)
        # correlation of the feature with the oracle frac-k index
        r = float(np.corrcoef(xf, yf)[0, 1]) if np.std(xf) > 0 and np.std(yf) > 0 else float("nan")
        # LODO 1-feature linear predictor of the frac-k index, clamped to the grid
        pred_prr, orc_prr = [], []
        for i, d in enumerate(ds):
            tr = [k for k in range(len(ds)) if k != i]
            A = np.polyfit(xf[tr], yf[tr], 1) if np.std(xf[tr]) > 0 else [0.0, yf[tr].mean()]
            j = int(np.clip(round(np.polyval(A, xf[i])), 0, len(FRACS) - 1))
            pred_prr.append(feats[d]["frac_prrs"][j])
            orc_prr.append(feats[d]["frac_prrs"][best_frac_i[d]])
        print(f"  feature={fname:10s} corr(feat, oracle-frac-k)={r:+.3f} | "
              f"LODO mean PRR @predicted-k {np.mean(pred_prr):+.4f} "
              f"(oracle-k {np.mean(orc_prr):+.4f}, always-msp_min {np.mean([feats[d]['prr_min'] for d in ds]):+.4f}, "
              f"always-ppl {np.mean([feats[d]['prr_ppl'] for d in ds]):+.4f})", flush=True)
    print("\nInterpretation: a LODO predicted-k that beats BOTH fixed endpoints cross-dataset = a new "
          "label-free method; otherwise oracle-k stays a diagnostic only.")


if __name__ == "__main__":
    main()
