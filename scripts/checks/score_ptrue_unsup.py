#!/usr/bin/env python
"""Score unsupervised P(True) onto the ladder grids — WITHOUT re-running any ladder.

WHY THIS CAN BE A CHEAP FOLLOW-UP
---------------------------------
Unsupervised P(True) needs no training: the score is already on each record (`ptrue_unsup`, written by
01g). So scoring it is just PRR over the eval's test split — CPU, seconds, no pertok cache, no probe.
That is why adding it does not justify restarting a 20-job ladder run (author's decision 2026-08-03).

⚠️ IT IS A FLOOR, SO IT IS SHIFT-INVARIANT. Like `msp_min`/`perplexity`, the score depends only on the
EVAL set, never on what was trained on. Its value is therefore identical across all 5 rungs, and it is
emitted for each rung so the grid has no holes. That is the same convention the existing floors follow —
it is not five independent measurements, and it should not be read as robustness to shift.

OUTPUT NAMING matches the per-eval glob the assemblers already read (`pdl_fam_*`, `xlcontrib_fam_*`), so
the rows merge into both master tables with no assembler change beyond the ALIAS entry.

    python scripts/checks/score_ptrue_unsup.py
"""
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq.config import Config  # noqa: E402
from luq import cache, results  # noqa: E402
from xl_rungs import eval_split, label_of  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = cache._slug(MODEL)
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}

LONG = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa", "asqa", "factscore"]
SHORT = ["sciq", "trivia_qa"]
LONG_RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
XL_RUNGS = ["ID", "SameTask", "LOO", "DiffTask", "OneDatasetDiffTask"]

# ⚠️ Below this, the model is not really answering yes/no at the verdict slot, so the score is measuring
# something else. Reported, never silently dropped -- framing.md notes raw prompting can whitespace-front
# the answer, which is a property of the prompting regime rather than of the method.
MASS_WARN = 0.60


def score_one(ds):
    cfg = Config(model_name=MODEL, dataset=ds, ood_setting="ID", prompt_regime=REGIME.get(ds, ""))
    recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, ds, "ID"))
    field = label_of(ds)
    unc = np.array([r.get("ptrue_unsup", np.nan) for r in recs], dtype=float)
    if not np.isfinite(unc).any():
        return None, "no ptrue_unsup on the records — run pbs/ptrue_unsup_fill.pbs first"
    y = np.array([r.get(field, np.nan) for r in recs], dtype=float)
    split = np.array([r["split"] for r in recs])
    _tr, te = eval_split(split)
    yte, ute = y[te], unc[te]
    ok = np.isfinite(yte) & np.isfinite(ute)
    if ok.sum() < 30:
        return None, f"only {int(ok.sum())} scorable test rows — refusing to report"
    # prr() itself now refuses non-finite input; filtering here keeps the reported n honest.
    prr = results.prr(yte[ok], ute[ok])
    mass = np.array([r.get("ptrue_unsup_mass", np.nan) for r in recs], dtype=float)
    return {"prr": prr, "n": int(ok.sum()), "dropped": int((~ok).sum()), "field": field,
            "mass_p50": float(np.nanmedian(mass)) if np.isfinite(mass).any() else float("nan")}, None


def main():
    print(f"Unsupervised P(True) — floor scoring (no training, no ladder re-run)\n")
    rows_long, rows_xl = [], []
    for ds in LONG + SHORT:
        res, err = score_one(ds)
        if res is None:
            print(f"  {ds:14s} SKIPPED LOUDLY: {err}")
            continue
        warn = "  ⚠️ LOW VERDICT MASS" if res["mass_p50"] < MASS_WARN else ""
        print(f"  {ds:14s} PRR {res['prr']:+.4f}  n={res['n']} (dropped {res['dropped']})  "
              f"label={res['field']}  verdict-mass p50={res['mass_p50']:.3f}{warn}")
        for rg in (LONG_RUNGS if ds in LONG else ["Long->Short"]):
            rows_long.append({"rung": rg, "eval": ds, "train": "(floor: eval set only)",
                              "method": "ptrue_unsup", "prr_mean": round(res["prr"], 4),
                              "prr_std": 0.0, "n_seeds": 1})
        for rg in XL_RUNGS:
            rows_xl.append({"rung": rg, "eval": ds, "train": "(floor: eval set only)",
                            "method": "ptrue_unsup", "prr_mean": round(res["prr"], 4),
                            "prr_std": 0.0, "n_seeds": 1})

    if not rows_long:
        sys.exit("nothing scored — refusing to write empty files.")
    for rows, stem in ((rows_long, "pdl_fam_ptrueunsup"), (rows_xl, "xlcontrib_fam_ptrueunsup")):
        out = ROOT / "results" / f"{stem}__{SLUG}.csv"
        with open(out, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {out}")
    print("\n⚠️ Same value repeats across rungs BY CONSTRUCTION (a floor depends only on the eval set). "
          "Do not read the flat row as evidence of OOD robustness.")


if __name__ == "__main__":
    main()
