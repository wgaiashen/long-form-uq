#!/usr/bin/env python
"""Build results/sharpening_master__<slug>.csv -- ONE consolidated long-format table of every
post-7-August sharpening-axis result, regenerated from the per-run source CSVs.

THE SINGLE SOURCE TO READ. Re-run this after any new run lands; never hand-edit the output.
Schema: workstream, method, param, rung, eval, prr, n_seeds, mode, source_csv
  * FREE methods are rung-invariant: emitted ONCE with rung=ALL-RUNGS-FREE (do not multiply by 4).
  * mode distinguishes arms/controls (anchor/random/wsonly/combo, masked/unmasked, blend/shufL/const).
Verdicts, bars and tiering live in STOCKTAKE_sharpening_axis.md, not here.

    python scripts/checks/build_sharpening_master.py
"""
import csv as _csv
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
OUT = RES / f"sharpening_master__{SLUG}.csv"
rows = []


def add(ws, method, param, rung, ev, prr, n, mode, src):
    rows.append((ws, method, param, rung, ev, prr, n, mode, Path(src).name))


def rd(path):
    return _csv.DictReader(open(path))


# W1/W4 free families + length-tau + diagnostics
for src in [RES / f"sharpening_family__{SLUG}__round2.csv", RES / f"sharpening_family__{SLUG}__lengthtau.csv"]:
    if not src.exists():
        continue
    for r in rd(src):
        add("W1/W4-free", r["family"], r["param"], "ALL-RUNGS-FREE", r["dataset"],
            r["prr"], r.get("n_eval", ""), "sweep", src)

# Track 2 temperature (partial 5/8)
for src in glob.glob(str(RES / f"sharpening_wmsp_*__{SLUG}.csv")):
    if "SMOKE" in src or "verdict" in src:
        continue
    for r in rd(src):
        if r.get("kind", "").startswith("floor"):
            add("T2-temp", "floor", r.get("gamma", ""), r["rung"], r["eval"], r["prr"],
                r["n_seeds"], r.get("kind", ""), src)
        else:
            add("T2-temp", "wmsp_T_gamma", f"T{r['T0']}_g{r['gamma']}", r["rung"], r["eval"],
                r["prr"], r["n_seeds"], "grid", src)

# W5 lambda + NLL tilts
for src in glob.glob(str(RES / f"sharpening_lambda_*__{SLUG}.csv")):
    if "SMOKE" in src or "verdict" in src:
        continue
    for r in rd(src):
        add("W5", r["kind"], r["param"], r["rung"], r["eval"], r["prr"], r["n_seeds"],
            r["kind"], src)

# mask ablation
for src in glob.glob(str(RES / f"mask_ablation_*__{SLUG}.csv")):
    if "SMOKE" in src or "verdict" in src:
        continue
    for r in rd(src):
        add("mask", f"wMSP-{r['variant']}", "", r["rung"], r["eval"], r["prr"], r["n_seeds"],
            r["mode"], src)

# F5 anchor: linear, log, and (when it lands) the warm-started __logws grid
for pat, tag in [(f"anchor_msp_min_*__{SLUG}.csv", "F5-linear"),
                 (f"anchor_msp_min_*__logpen__{SLUG}.csv", "F5b-log"),
                 (f"anchor_msp_min_*__logws__{SLUG}.csv", "F5c-warmstart")]:
    for src in glob.glob(str(RES / pat)):
        if "SMOKE" in src:
            continue
        if tag == "F5-linear" and ("logpen" in src or "logws" in src):
            continue
        for r in rd(src):
            add(tag, "anchored-wMSP", r["lambda"], r["rung"], r["eval"], r["prr"],
                r["n_seeds"], r["mode"], src)

# W7b same-family blend (+ controls); W7 SAPLMA arm has no per-cell CSV (script printed + verdict)
for src in glob.glob(str(RES / f"blend_msp_wmsp_*__{SLUG}.csv")):
    for r in rd(src):
        add("W7b-blend", r["kind"], r["param"], r["rung"], r["eval"], r["prr"], r["n_seeds"],
            r["kind"], src)

with open(OUT, "w", newline="") as fh:
    w = _csv.writer(fh)
    w.writerow(["workstream", "method", "param", "rung", "eval", "prr", "n_seeds", "mode",
                "source_csv"])
    for r in sorted(rows):
        w.writerow(r)
from collections import Counter
c = Counter(r[0] for r in rows)
print(f"wrote {OUT}  ({len(rows)} rows)")
for k, v in sorted(c.items()):
    print(f"  {k:14s} {v:5d} rows")
