"""The TRUE wMSP content-token anchor on Qwen -- a cross-model check of the Llama mechanism finding
that shrinkage benefit is predicted by ANCHOR QUALITY (mu_C, the mean NLL over the exact content-token
mask learned wMSP scores against), not by the size of the unregularised learned correction.

NO MODEL FITTING, GENERATION, JUDGING, OR WEIGHT RECONSTRUCTION. mu_C is read straight off the
cached Tier-1 records' `token_logprobs` (already on disk from the run that produced every Qwen
number to date). No per-token hidden-state pool is touched, no GPU is used.

WHAT THIS COMPUTES, PER DATASET (the RAW-SPAN report-facing population, LUQ_REGIME unset -- the
same population `results/pdl_master__Qwen_Qwen2.5-14B.csv` and the primary
`results/wmsp_lambda_qwen__*.csv` shrinkage numbers were scored on):

  mu_C   = mean(NLL over the CONTENT-TOKEN mask, i.e. weighted_msp.content_keep -- the EXACT mask
           learned wMSP scores against, special tokens excluded)
  mu_all = mean(NLL over ALL G tokens) -- msp.msp_uncertainty(..., "perplexity"), the canonical floor
           already published per-dataset in the project's working notes

Both are scored with the SAME `results.prr` used everywhere else, on the SAME eval-target test rows
(`xl_rungs.eval_split`) every rung in the ladder tests against -- these unsupervised floors are
rung-invariant (they never depend on the training pool), so ONE PRR per dataset is the whole story.

THE PRE-SPECIFIED TEST (n=8 datasets, Spearman, two-sided):
    x = PRR(mu_C) per dataset
    y = PRR(wmsp_shrink2) - PRR(wmsp_norm), macro-OOD (mean of the 4 OOD rungs), per dataset,
        read from the existing gated master (results/pdl_master__Qwen_Qwen2.5-14B.csv) --
        no lambda is fit or selected here.

Usage:
    python scripts/checks/qwen_content_anchor.py --model Qwen/Qwen2.5-14B
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache, msp, results, weighted_msp                # noqa: E402
from luq.config import Config                                    # noqa: E402
from transformers import AutoTokenizer                            # noqa: E402
from attn_pool import PROMPT_REGIME                                # noqa: E402
from xl_rungs import eval_split, label_of                          # noqa: E402

EVALS = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa", "asqa", "factscore"]
OOD_RUNGS = ["SameTask-long", "LOO-long", "DiffTask-long", "1ds-Diff-long"]


def macro_ood_from_master(master_csv, method, ev):
    """Mean PRR over the 4 OOD rungs for one (method, eval) cell, read from the gated master --
    the SAME numbers already published in the project's working notes. No fitting here."""
    import csv as _csv
    vals = []
    with open(master_csv) as f:
        for r in _csv.DictReader(f):
            if r["eval"] == ev and r["method"] == method and r["rung"] in OOD_RUNGS:
                vals.append(float(r["prr_mean"]))
    if len(vals) != 4:
        raise SystemExit(f"{ev}/{method}: expected 4 OOD rungs in {master_csv}, found {len(vals)}")
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B")
    ap.add_argument("--master-csv", default=str(ROOT / "results" / "pdl_master__Qwen_Qwen2.5-14B.csv"))
    args = ap.parse_args()

    # The Qwen special-token trap, carried over verbatim from wmsp_lambda_qwen.py / probedriftlong.py:
    # content_keep's fallback is the Llama-3 id>=128000 range test. Qwen2.5's specials start at 151,643
    # in a 152,064 vocab, so without registering the real ids, content_keep would zero-weight nothing
    # (or the wrong tokens) and mu_C would silently be a different, wrong mask.
    tok = AutoTokenizer.from_pretrained(args.model)
    weighted_msp.set_special_ids(tok.all_special_ids)
    print(f"special-token ids registered from {args.model} tokenizer ({len(tok.all_special_ids)} ids)")

    slug = cache._slug(args.model)
    rows = []
    for d in EVALS:
        cfg = Config(model_name=args.model, dataset=d, ood_setting="ID",
                     prompt_regime=PROMPT_REGIME.get(d, ""))
        key = cache.run_key(args.model, d, "ID")
        records = cache.load_records(cfg.cache_dir, key)
        lf = label_of(d)
        y_all = np.array([r.get(lf, np.nan) for r in records], dtype=float)
        finite = np.isfinite(y_all)
        if not finite.all():
            keep = np.where(finite)[0]
            records = [records[k] for k in keep]
            y_all = y_all[keep]
        split = np.array([r["split"] for r in records])
        _, te_idx = eval_split(split)
        te_idx = list(te_idx)
        if not te_idx:
            raise SystemExit(f"{d}: empty eval-split test set -- cannot score")

        muC, muAll, ratios = [], [], []
        for i in te_idx:
            rec = records[i]
            nll = weighted_msp.per_token_nll(rec)
            mask = weighted_msp.content_keep(rec)
            n_kept = float(mask.sum())
            g = len(mask)
            if n_kept < 0.5:
                raise SystemExit(f"{d} row {i}: content mask is all-zero (G={g}) -- cannot form mu_C; "
                                 "stop, do not silently fall back to all-token mean")
            muC.append(float(nll[mask > 0.5].mean()))
            muAll.append(msp.msp_uncertainty(rec["token_logprobs"], "perplexity"))
            ratios.append(n_kept / g)
        yte = y_all[te_idx]
        muC = np.asarray(muC); muAll = np.asarray(muAll); ratios = np.asarray(ratios)

        prr_muC = results.prr(yte, muC)
        prr_all = results.prr(yte, muAll)
        rows.append({
            "dataset": d, "n_test": len(te_idx), "label_field": lf,
            "prr_muC": prr_muC, "prr_canonical_perplexity": prr_all, "delta": prr_muC - prr_all,
            "ratio_mean": float(ratios.mean()), "ratio_median": float(np.median(ratios)),
            "ratio_p5": float(np.percentile(ratios, 5)), "ratio_p95": float(np.percentile(ratios, 95)),
            "ratio_min": float(ratios.min()), "ratio_max": float(ratios.max()),
        })
        print(f"  {d:14s} n={len(te_idx):4d}  PRR(mu_C)={prr_muC:+.4f}  "
              f"PRR(canonical mean-NLL)={prr_all:+.4f}  delta={prr_muC - prr_all:+.4f}  "
              f"n/G mean={ratios.mean():.4f} [{ratios.min():.4f},{ratios.max():.4f}]")

    # ---- macro-OOD shrink2-norm gain per dataset, read from the gated master (no fitting) ----
    gains = []
    for r in rows:
        g_shrink2 = macro_ood_from_master(args.master_csv, "wmsp_shrink2", r["dataset"])
        g_norm = macro_ood_from_master(args.master_csv, "wmsp_norm", r["dataset"])
        r["ood_gain_shrink2_minus_norm"] = g_shrink2 - g_norm
        gains.append(r["ood_gain_shrink2_minus_norm"])

    x_muC = np.array([r["prr_muC"] for r in rows])
    x_all = np.array([r["prr_canonical_perplexity"] for r in rows])
    y_gain = np.array(gains)

    rho_primary, p_primary = spearmanr(x_muC, y_gain)
    rho_secondary, p_secondary = spearmanr(x_all, y_gain)

    print("\n" + "=" * 100)
    print("PRE-SPECIFIED PRIMARY TEST -- x=PRR(mu_C), y=PRR(shrink2)-PRR(norm) macro-OOD, n=8, Spearman")
    print("=" * 100)
    print(f"rho = {rho_primary:+.4f}   two-sided p = {p_primary:.4f}")
    for r in rows:
        print(f"  {r['dataset']:14s} x={r['prr_muC']:+.4f}  y={r['ood_gain_shrink2_minus_norm']:+.4f}")

    print("\n" + "=" * 100)
    print("SECONDARY DESCRIPTIVE ONLY -- x=canonical all-token mean-NLL PRR, same y")
    print("=" * 100)
    print(f"rho = {rho_secondary:+.4f}   two-sided p = {p_secondary:.4f}")

    out_csv = ROOT / "results" / "analysis" / f"qwen_content_anchor__{slug}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
