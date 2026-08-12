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
import argparse
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
from provenance import provenance  # noqa: E402  stamp, so these rows are traceable like every other
from attn_pool import PROMPT_REGIME  # noqa: E402  merges LUQ_REGIME, same mechanism every other
                                      # driver uses -- so a clean-population run reads the clean
                                      # records here too, instead of this script's own frozen dict.

MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"
# EXPLICIT model pin. Default is the original string, so `python score_ptrue_unsup.py` with no args
# stays byte-identical to before. REGIME is gone -- PROMPT_REGIME (imported above) already carries
# the same three base entries plus whatever LUQ_REGIME overrides, so a second, unsynced copy of that
# mapping is not needed and cannot drift from it.
MODEL = MODEL_DEFAULT
SLUG = cache._slug(MODEL)

LONG = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa", "asqa", "factscore"]
SHORT = ["sciq", "trivia_qa"]
LONG_RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
XL_RUNGS = ["ID", "SameTask", "LOO", "DiffTask", "OneDatasetDiffTask"]

# ⚠️ Below this, the model is not really answering yes/no at the verdict slot, so the score is measuring
# something else. Reported, never silently dropped -- framing.md notes raw prompting can whitespace-front
# the answer, which is a property of the prompting regime rather than of the method.
MASS_WARN = 0.60


SIDECAR_DIR = ROOT / "results" / "sidecar_ptrue_unsup"


def _from_sidecar(ds, n_rows, recs):
    """Read scores from a synced sidecar CSV instead of the records.

    ⚠️ THIS IS WHY THE SCORER DOES NOT NEED THE MERGE TO HAVE HAPPENED. 01g writes into the Tier-1
    records, but merging on RCS would (a) touch the canonical files while 20 ladder jobs are reading
    them, and (b) bump the records mtime — which 04_eval.py treats as the signature of a RELABEL and
    uses to invalidate cached probes. Adding an unrelated field is not a relabel, so that would be a
    false staleness signal on every cached probe. Reading the sidecar avoids both; the merge becomes a
    deliberate later step rather than a prerequisite.

    Same alignment guard as the merge path: positional keying is only valid if both sides hold the same
    file, so the row count and the gold-target fingerprint must both match.
    """
    p = SIDECAR_DIR / f"ptrue_unsup__{ds}.csv"
    if not p.exists():
        return None
    import csv as _c
    import hashlib
    rows = list(_c.reader(open(p)))
    if not rows or rows[0][0] != "#n_rows":
        return None
    if int(rows[0][1]) != n_rows:
        sys.exit(f"{ds}: sidecar claims {rows[0][1]} rows, records have {n_rows}. Refusing to align.")
    h = hashlib.sha256()
    for r in recs:
        h.update(repr(r.get("target", "")).encode("utf-8", "replace"))
    if rows[0][3] != h.hexdigest()[:16]:
        sys.exit(f"{ds}: sidecar target fingerprint != local records. Refusing to align.")
    out = np.full(n_rows, np.nan)
    for r in rows[2:]:
        out[int(r[0])] = float(r[1])
    return out


def score_one(ds):
    cfg = Config(model_name=MODEL, dataset=ds, ood_setting="ID", prompt_regime=PROMPT_REGIME.get(ds, ""))
    try:
        recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, ds, "ID"))
    except FileNotFoundError:
        # A genuinely absent population (e.g. Qwen has no sciq/trivia_qa records at all -- the
        # Long->Short rung was never generated for it) is a known, out-of-scope gap, not a crash.
        return None, "no records for this (model, dataset) at all -- population was never generated"
    field = label_of(ds)
    unc = np.array([r.get("ptrue_unsup", np.nan) for r in recs], dtype=float)
    if not np.isfinite(unc).any():
        side = _from_sidecar(ds, len(recs), recs)     # not merged yet? read the sidecar directly
        if side is not None and np.isfinite(side).any():
            unc = side
    if not np.isfinite(unc).any():
        return None, "no ptrue_unsup on the records and no sidecar — run slurm/ptrue_unsup_doc.sbatch"
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
    global MODEL, SLUG
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT,
                    help="EXPLICIT model pin. Default is the original string, so a no-args "
                         "invocation stays byte-identical to before.")
    args = ap.parse_args()
    MODEL = args.model
    SLUG = cache._slug(MODEL)

    print(f"Unsupervised P(True) — floor scoring (no training, no ladder re-run)\n")
    print(f"model={MODEL}")
    PROV = provenance(strict=False)
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
            rows_long.append({**PROV, "rung": rg, "eval": ds, "train": "(floor: eval set only)",
                              "method": "ptrue_unsup", "prr_mean": round(res["prr"], 4),
                              "prr_std": 0.0, "n_seeds": 1})
        for rg in XL_RUNGS:
            rows_xl.append({**PROV, "rung": rg, "eval": ds, "train": "(floor: eval set only)",
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
