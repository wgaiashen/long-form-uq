#!/usr/bin/env python
"""Step 2b — FROZEN-SCORE truncation sensitivity, token-based methods. Read-only, no training, no £.

WHAT THIS ANSWERS. The trailing next-template continuation is inside the token stream every
token-probability method aggregates over. This measures what that junk is doing to the SCORES, with
the LABELS HELD FIXED at raw — isolating score-side contamination from label-side contamination.

WHAT IT MUST NOT BE USED FOR. Nothing here decides whether a truncation boundary is valid. That
was decided in `luq.template_restart` from the dataset's own prompt template, written and committed
BEFORE this ran. Choosing a boundary because it improves a PRR is selection on the outcome, and this
project has already measured methods latching onto junk signal. **A correction remains a correction
even if our preferred method gets worse — and here the expected direction is worse.**

METHOD. For each record: take the cached generated `token_logprobs`, re-tokenise the retained prefix
to find how many generated tokens survive the frozen boundary, and slice. Same scorer, same labels,
same test split as the ladder — only the token span differs. No model is loaded, nothing is trained,
no probe is touched.

Scored (all training-free, hence RUNG-INVARIANT — one number per dataset, n = 8, never 40 cells):
    perplexity · msp_min · msp_sum · Lehmer beta=1 · softmax tau=1

NOT scored here, and why: SAPLMA / uniform / attention / wMSP-norm / wMSP-shrink@2 are supervised,
and Step 2b requires EXISTING trained probes rather than retraining. **No Qwen probe or
weighter exists on disk** (`cache/probes/` holds Llama attention poolers only), so that half cannot
be run under that constraint. It is reported as blocked, not silently skipped or quietly retrained.

NUISANCE DIAGNOSTICS, clearly not UQ methods: PRR(severe_indicator) and PRR(bleed_indicator) — how
much ranking power a bare degeneracy flag has against the label. These bound how much of any
method's PRR could be degeneracy detection.

    python scripts/checks/truncation_sensitivity.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache, degeneracy, msp, results, template_restart as TR   # noqa: E402
from luq.config import Config                                             # noqa: E402
from attn_pool import PROMPT_REGIME                                       # noqa: E402
from xl_rungs import eval_split, label_of                                 # noqa: E402
from sharpening_family import score_lehmer, score_softmax                 # noqa: E402
from transformers import AutoTokenizer                                    # noqa: E402

QWEN = "Qwen/Qwen2.5-14B"
LLAMA = "meta-llama/Meta-Llama-3.1-8B"
DATASETS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
MIN_KEEP = 2          # a 0- or 1-token span has no meaningful spread; keep the raw span and say so


def load(model, dataset):
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    p = Path(cfg.cache_dir) / "records" / f"{cache.run_key(model, dataset, 'ID')}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def scores(nll):
    """The five training-free scores, all 'higher = more uncertain'."""
    lp = -nll                                             # back to logprobs for luq.msp
    return {"perplexity": msp.msp_uncertainty(lp, "perplexity"),
            "msp_min": msp.msp_uncertainty(lp, "min"),
            "msp_sum": msp.msp_uncertainty(lp, "sum"),
            "lehmer_b1": score_lehmer(nll, 1.0),
            "softmax_tau1": score_softmax(nll, 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    slug = cache._slug(args.model)
    out = ROOT / (args.out or f"results/analysis/truncation_sensitivity__{slug}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    print("=" * 118)
    print(f"STEP 2b — FROZEN-SCORE TRUNCATION SENSITIVITY   model={args.model}")
    print("Labels held at RAW. Only the scored token span changes. Nothing trained, no probe touched.")
    print("Boundary: luq.template_restart, frozen from the prompt templates BEFORE this ran.")
    print("These numbers do NOT decide whether the boundary is valid. That was decided from the")
    print("   template. A correction stays a correction even if a method gets worse.")
    print("=" * 118)

    rows = []
    hdr = (f"{'dataset':14s}{'standing':13s}{'%cut':>6s}{'medTokDrop':>11s}" +
           "".join(f"{m:>22s}" for m in ["perplexity", "msp_min", "msp_sum", "lehmer_b1", "softmax_tau1"]))
    print("\n" + hdr)
    print("-" * len(hdr))
    for d in DATASETS:
        recs = load(args.model, d)
        if recs is None:
            print(f"{d:14s}  RECORDS MISSING — reported absent, not zero")
            continue
        lf = label_of(d)
        y_all = np.array([r.get(lf, np.nan) if r.get(lf) is not None else np.nan for r in recs], float)
        split = np.array([r["split"] for r in recs])
        _, te = eval_split(split)
        raw, trn, drops, standing, n_cut = [], [], [], None, 0
        sev, bld = [], []
        for r in recs:
            t = r.get("gen_text", "") or ""
            lp = np.asarray(r["token_logprobs"], dtype=float)
            nll = -lp
            keep, cut, _, st = TR.restart_cut(t, d)
            sev.append(degeneracy.is_severe(t))
            bld.append(cut is not None)
            if cut is None or not keep:
                raw.append(nll); trn.append(nll); drops.append(0.0)
                continue
            standing = standing or st
            n_keep = len(tok(keep, add_special_tokens=False).input_ids)
            n_keep = max(MIN_KEEP, min(n_keep, len(nll)))
            n_cut += 1
            raw.append(nll); trn.append(nll[:n_keep]); drops.append(1 - n_keep / max(len(nll), 1))
        sev = np.array(sev); bld = np.array(bld)

        # score on the TEST split only, exactly as the ladder does
        yt = y_all[te]
        fin = np.isfinite(yt)
        if fin.sum() < 20:
            print(f"{d:14s}  fewer than 20 labelled test rows — skipped LOUDLY, no number invented")
            continue
        idx = np.array(te)[fin]
        yy = y_all[idx]
        cells = []
        for m in ["perplexity", "msp_min", "msp_sum", "lehmer_b1", "softmax_tau1"]:
            a = results.prr(yy, np.array([scores(raw[i])[m] for i in idx]))
            b = results.prr(yy, np.array([scores(trn[i])[m] for i in idx]))
            cells.append(f"{a:+.4f}->{b:+.4f} ({b-a:+.4f})")
            rows.append({"model": args.model, "dataset": d, "method": m,
                         "standing": standing or "-", "pct_cut": round(100 * n_cut / len(recs), 2),
                         "median_frac_tokens_dropped": round(100 * float(np.median([x for x in drops if x > 0])) if n_cut else 0.0, 2),
                         "prr_raw_score_raw_label": round(a, 6),
                         "prr_trunc_score_raw_label": round(b, 6),
                         "delta_score_side": round(b - a, 6), "n_test_labelled": int(len(idx)),
                         "provenance": "frozen-score-sensitivity"})
        md = 100 * float(np.median([x for x in drops if x > 0])) if n_cut else 0.0
        print(f"{d:14s}{(standing or '-')[:12]:13s}{100*n_cut/len(recs):>5.1f}%{md:>10.1f}%"
              + "".join(f"{c:>22s}" for c in cells))

        # nuisance diagnostics -- NOT UQ methods
        for nm, ind in [("NUISANCE:severe_indicator", sev), ("NUISANCE:bleed_indicator", bld)]:
            p = results.prr(yy, ind[idx].astype(float))
            rows.append({"model": args.model, "dataset": d, "method": nm, "standing": "diagnostic",
                         "pct_cut": round(100 * n_cut / len(recs), 2), "median_frac_tokens_dropped": md,
                         "prr_raw_score_raw_label": round(p, 6), "prr_trunc_score_raw_label": "",
                         "delta_score_side": "", "n_test_labelled": int(len(idx)),
                         "provenance": "benchmark-validity-diagnostic"})

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    print("\nNUISANCE DIAGNOSTICS (a bare flag as a 'score' — bounds how much of any method's PRR")
    print("could be degeneracy detection). NOT UQ methods:")
    for r in rows:
        if r["method"].startswith("NUISANCE"):
            print(f"  {r['dataset']:14s} {r['method'][9:]:18s} PRR {r['prr_raw_score_raw_label']:+.4f}")
    print("\nBLOCKED, not skipped: SAPLMA / uniform / attention / wMSP-norm / wMSP-shrink@2 need")
    print("   EXISTING trained probes, and none exist for this model (cache/probes/ has Llama")
    print("   attention poolers only). Running them would require either persisting probes from a")
    print("   ladder run, or training once and scoring both arms with the same model — the latter")
    print("   is not 'retraining on truncated data' but it is training, so it needs approval.")


if __name__ == "__main__":
    main()
