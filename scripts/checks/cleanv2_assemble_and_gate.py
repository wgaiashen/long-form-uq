#!/usr/bin/env python
"""Assemble the clean-v2 Llama per-eval result set and run gates A and C.

WHAT THIS BUILDS. 40 cells made of two provably different things:

  * 19 cells RECOMPUTED, because MedQuAD is either their evaluation target or one of their training
    sources -- from results/cleanv2/probedriftlong_cleanv2_<eval>__<slug>.csv.
  * 21 cells INHERITED verbatim, because MedQuAD appears nowhere in them.

WHERE THE INHERITED CELLS COME FROM, AND WHY NOT THE MASTER. They are taken from
`results/pdl_fam_<eval>__<slug>.csv`, the canonical per-eval output of this same driver -- NOT from
`results/pdl_master__<slug>.csv`. The master is an ASSEMBLED table: assemble_pdl_table.py merges a
dozen source globs, maps raw driver keys onto display names, dedupes by priority and cross-checks
overlaps. Mixing raw clean-v2 rows (`wmsp_shrink2`) with assembled master rows (`wMSP-shrink@2`)
would produce a file carrying two naming conventions for one method, and would also drag in arms
from experiments this correction never reran (multi-head, top-k, ensembles, the router). Inheriting
from the per-eval source keeps one convention, one driver and one schema.

Rendering to display names is a SEPARATE later step: point assemble_pdl_table.py at these outputs.

TWO ASYMMETRIES THIS PRINTS RATHER THAN HIDES.
  * `wmsp_shrink1_5` exists in the clean-v2 runs but not in the canonical per-eval sources, because
    the lambda = 1.5 arm was added after they were produced. It will therefore be present on the 19
    recomputed cells and absent on the 21 inherited ones.
  * `ptrue` / `lookback` come from --baselines. If a clean-v2 run was made without that flag its
    cells will lack them while the inherited cells have them. Both are reported as coverage gaps.

GATE A -- identity before inheritance: MedQuAD absent structurally, every dataset the cell touches
resolving to the SAME cache path the canonical run used (never a cleanv2 path), and that path's
content hash recorded. Any failure is a STOP, never repaired by recomputing the cell.

GATE C -- training-pool audit: eval target, rung, sources, MedQuAD row count, and the exact feature
and label namespace MedQuAD resolved to, so a silent fallback to raw is visible.

This script does not interpret any PRR. It moves rows and checks provenance.

    python scripts/checks/cleanv2_assemble_and_gate.py            # gates + coverage, no write
    python scripts/checks/cleanv2_assemble_and_gate.py --write    # also write the clean-v2 set
"""
import argparse
import hashlib
import os
import pathlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("LUQ_REGIME", "med_quad=cleanv2")

from luq import cache                                            # noqa: E402
from luq.config import Config                                    # noqa: E402
from attn_pool import PROMPT_REGIME                              # noqa: E402
from probe_drift_long.ood_settings import cells_long             # noqa: E402
from probe_drift_long.dataset_configs import LONG_SRC            # noqa: E402

# PER-MODEL SOURCE LAYOUT. The two populations were produced by different drivers and so use
# different filenames; hardcoding one model's pattern would make the other silently assemble nothing,
# which reads as "no affected cells" rather than as an error. `lambda_fill` says whether the
# lambda = 1.5 arm has to be recovered from the sharpening sweep: on Llama it postdates the canonical
# per-eval files, while the eight-dataset ladder computes it directly and needs no fill.
PROFILES = {
    "meta-llama/Meta-Llama-3.1-8B": {
        "layer": 15,
        "recomputed": "probedriftlong_cleanv2_{eval}__{slug}.csv",
        "inherited_glob": "pdl_fam_*__{slug}.csv",
        "inherited_skip": ("ptrueunsup",),
        "lambda_fill": True,
        "out_default": "pdl_cleanv2_alleval",
    },
    "google/gemma-2-9b": {
        "layer": 20,
        "recomputed": "wmodels_sens8_cleanv2__{slug}__{eval}.csv",
        "inherited_glob": "wmodels_sens8__{slug}__*.csv",
        "inherited_skip": ("master",),
        "lambda_fill": False,
        "out_default": "wmodels_sens8_cleanv2_master",
    },
}

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B"
MODEL = DEFAULT_MODEL
SLUG = cache._slug(MODEL)
PROF = PROFILES[MODEL]
CANON = ROOT / "results" / f"pdl_master__{SLUG}.csv"
OUTDIR = ROOT / "results" / "cleanv2"
LAYER = 15


def _hash(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def split_cells():
    """(affected, control) cell lists, derived structurally -- never hard-coded."""
    aff, ctl = [], []
    for rung, X, spec in cells_long(LONG_SRC, LONG_SRC):
        touches = (X == "med_quad") or any(s == "med_quad" for s, _ in spec)
        (aff if touches else ctl).append((rung, X, spec))
    return aff, ctl


def gate_a(control):
    print("\n" + "=" * 96)
    print("GATE A -- input identity for the cells that will be INHERITED")
    print("=" * 96)
    ok = True
    seen = {}
    for rung, X, spec in control:
        datasets = [X] + [s for s, _ in spec]
        if "med_quad" in datasets:
            print(f"  STOP  {X}/{rung}: med_quad present in a cell classed as control"); ok = False; continue
        for d in datasets:
            if d in seen:
                continue
            cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID",
                         prompt_regime=PROMPT_REGIME.get(d, ""))
            p = Path(cfg.cache_dir) / "pertok" / f"{cache._slug(MODEL)}__{d}__ID__L{LAYER}.npz"
            if "cleanv2" in str(p):
                print(f"  STOP  {d}: resolves into a cleanv2 path under the override -> {p}"); ok = False; continue
            if not p.exists():
                print(f"  STOP  {d}: cache missing -> {p}"); ok = False; continue
            seen[d] = _hash(p)
    print(f"  datasets touched by the {len(control)} control cells: {sorted(seen)}")
    for d, h in sorted(seen.items()):
        print(f"     {d:15} {h}  (canonical path, unchanged)")
    print(f"\n  GATE A: {'PASS -- the control cells may be inherited' if ok else '**FAIL -- STOP**'}")
    return ok


def gate_c(affected):
    print("\n" + "=" * 96)
    print("GATE C -- training-pool audit for every recomputed cell")
    print("=" * 96)
    mq_cfg = Config(model_name=MODEL, dataset="med_quad", ood_setting="ID",
                    prompt_regime=PROMPT_REGIME.get("med_quad", ""))
    mq_ns = PROMPT_REGIME.get("med_quad", "") or "<default>"
    print(f"  med_quad feature namespace : {mq_ns}   ({mq_cfg.cache_dir})")
    print(f"  med_quad label field       : correctness (in the same namespace)")
    print(f"\n  {'rung':16}{'eval':15}{'med_quad rows':>14}  sources")
    for rung, X, spec in sorted(affected):
        # cap is None on the ID cell, meaning "all of the eval's own train rows" -- not a number.
        # Summing it blind is the cap-vs-target trap the eight-dataset control already tripped over.
        caps = [c for s, c in spec if s == "med_quad"]
        if X == "med_quad":
            tag = "TARGET"
        elif not caps:
            tag = "-"
        elif any(c is None for c in caps):
            tag = "all"
        else:
            tag = str(sum(caps))
        print(f"  {rung:16}{X:15}{tag:>14}  {', '.join(s for s, _ in spec)}")
    bad = [c for c in affected if c[1] != "med_quad" and not any(s == "med_quad" for s, _ in c[2])]
    print(f"\n  GATE C: {'PASS -- every recomputed cell genuinely touches med_quad' if not bad else '**FAIL**'}")
    return not bad


def assemble(affected, control, write, in_dir=None, out_name="pdl_cleanv2_alleval"):
    src_dir = pathlib.Path(in_dir) if in_dir else OUTDIR
    print("\n" + "=" * 96)
    print("ASSEMBLE the clean-v2 per-eval result set")
    print("=" * 96)
    parts, missing = [], []
    for rung, X, _ in affected:
        f = src_dir / PROF["recomputed"].format(eval=X, slug=SLUG)
        if not f.exists():
            missing.append(("recomputed", rung, X)); continue
        d = pd.read_csv(f)
        d = d[(d["eval"] == X) & (d["rung"] == rung)]
        if d.empty:
            missing.append(("recomputed", rung, X)); continue
        d = d.copy(); d["provenance"] = "recomputed_cleanv2"
        parts.append(d)
    # The canonical per-eval sources are a GLOB, not one file per eval: some cells were filled by a
    # later per-rung job, e.g. xsum/1ds-Diff-long lives in pdl_fam_xsum_1ds-Diff__<slug>.csv rather
    # than pdl_fam_xsum__<slug>.csv. Assuming one file per eval silently loses those cells, so
    # gather every pdl_fam_* source and select the matching (eval, rung).
    fam = []
    for f in sorted((ROOT / "results").glob(PROF["inherited_glob"].format(slug=SLUG))):
        if any(s in f.name for s in PROF["inherited_skip"]):
            continue                       # a single-method side file, or an assembled master
        d = pd.read_csv(f)
        if {"eval", "rung", "method"} <= set(d.columns):
            d = d.copy(); d["_src"] = f.name
            fam.append(d)
    fam = pd.concat(fam, ignore_index=True) if fam else pd.DataFrame()

    # LAMBDA = 1.5 FOR THE INHERITED CELLS.
    # The lambda = 1.5 arm postdates the pdl_fam sources, so those cells have no `wmsp_shrink1_5`
    # row and the assembled file would carry the method on 19 cells and not the other 21. The values
    # DO exist: results/sharpening_lambda_<eval>__<slug>.csv holds kind='lambda', param='lam1.5' on
    # the complete 40-cell grid at 3 seeds. That source is the raw population, which is exactly right
    # here -- these 21 cells touch med_quad nowhere, so raw and clean-v2 are the same population for
    # them, the same reason every other method is inherited rather than recomputed.
    # VERIFIED SAME QUANTITY, not assumed: the lambda = 0 arm ('norm' there, 'wmsp_norm' here) agrees
    # on all 40 cells at max |d| = 4.8e-05, a rounding difference only -- pdl_fam stores 4 dp while
    # sharpening_lambda keeps full precision.
    lam = []
    for f in (sorted((ROOT / "results").glob(f"sharpening_lambda_*__{SLUG}.csv"))
              if PROF["lambda_fill"] else []):
        if "verdict" in f.name:
            continue
        d = pd.read_csv(f)
        if "kind" not in d.columns or "param" not in d.columns:
            continue
        d = d[(d["kind"] == "lambda") & (d["param"].astype(str) == "lam1.5")]
        if len(d):
            lam.append(d)
    lam = pd.concat(lam, ignore_index=True) if lam else pd.DataFrame()
    inh = []
    for rung, X, _ in control:
        d = fam[(fam["eval"] == X) & (fam["rung"] == rung)] if len(fam) else fam
        if len(d) == 0:
            missing.append(("inherited", rung, X)); continue
        # a method appearing in two sources must agree; disagreement is flagged, never averaged
        dup = d[d.duplicated("method", keep=False)]
        for meth, g in dup.groupby("method"):
            if g["prr_mean"].nunique() > 1:
                print(f"     DISAGREEMENT {X}/{rung}/{meth} across {sorted(set(g['_src']))} -- not averaged")
        # PREFER A MEASURED VALUE OVER A BLANK. Some methods were filled by a later per-method job:
        # asqa/ID/wmsp_seg_softmax is NaN in pdl_fam_asqa but 0.3618 in pdl_fam_asqa_segsm. Taking
        # whichever row happened to come first shipped the NaN, turning "measured" into "not
        # measured" -- the exact confusion this project bans. Sort non-null first, then dedupe.
        d = (d.assign(_isnull=d["prr_mean"].isna())
               .sort_values("_isnull", kind="stable")
               .drop_duplicates("method", keep="first")
               .drop(columns=["_src", "_isnull"]).copy())
        d["provenance"] = "inherited_unchanged"
        if "wmsp_shrink1_5" not in set(d["method"]) and len(lam):
            L = lam[(lam["eval"] == X) & (lam["rung"] == rung)]
            if len(L):
                row = d.iloc[[0]].copy()
                row["method"] = "wmsp_shrink1_5"
                row["prr_mean"] = float(L["prr"].iloc[0])
                if "prr_std" in row.columns:
                    row["prr_std"] = float(L["prr_std"].iloc[0]) if "prr_std" in L.columns else float("nan")
                row["n_seeds"] = int(L["n_seeds"].iloc[0]) if "n_seeds" in L.columns else 3
                row["provenance"] = "inherited_lambda_source"
                d = pd.concat([d, row], ignore_index=True)
        inh.append(d)
    print(f"  recomputed cells found : {len(parts)}/{len(affected)}")
    print(f"  inherited  cells found : {len(inh)}/{len(control)}   "
          f"(source: results/{PROF['inherited_glob'].format(slug=SLUG)})")
    if missing:
        for m in missing[:12]:
            print(f"     MISSING {m}")
        print("  -> NOT writing while any cell is missing: a partial file reads as a complete one.")
        return False
    master = pd.concat(parts + inh, ignore_index=True)
    rec = set(pd.concat(parts)["method"]); ihm = set(pd.concat(inh)["method"])
    if rec - ihm:
        print(f"  NOTE methods only on the 19 recomputed cells : {sorted(rec - ihm)}")
    if ihm - rec:
        print(f"  NOTE methods only on the 21 inherited cells  : {sorted(ihm - rec)}")
    print(f"  assembled {len(master)} rows over {master.groupby(['eval','rung']).ngroups} cells")
    if write:
        out = OUTDIR / f"{out_name}__{SLUG}.csv"
        master.to_csv(out, index=False)
        print(f"  wrote {out}")
    else:
        print("  (dry run: pass --write to persist)")
    return True


def main():
    global MODEL, SLUG, PROF, CANON, LAYER
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--in-dir", default=None,
                    help="directory holding the per-eval clean-v2 CSVs. Default results/cleanv2. "
                         "Point it at results/cleanv2/_firstpass_noBaselines to assemble from the "
                         "immutable backup while a later pass is still writing the live files -- "
                         "reading a CSV mid-write is how a half-populated cell becomes a number.")
    ap.add_argument("--out-name", default=None,
                    help="stem for the assembled file, so a partial-method assembly cannot be "
                         "mistaken for the final one. Defaults to the model's profile.")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(PROFILES),
                    help="which population to assemble; selects the source-file layout")
    args = ap.parse_args()

    # Rebind the module-level constants the helpers read, so one --model switches every path at once
    # rather than leaving some functions on the default model.
    MODEL = args.model
    SLUG = cache._slug(MODEL)
    PROF = PROFILES[MODEL]
    LAYER = PROF["layer"]
    CANON = ROOT / "results" / f"pdl_master__{SLUG}.csv"
    out_name = args.out_name or PROF["out_default"]
    print(f"model {MODEL}   layer {LAYER}")
    affected, control = split_cells()
    print(f"cells: {len(affected)} affected (recompute), {len(control)} control (inherit), "
          f"{len(affected) + len(control)} total")
    a = gate_a(control)
    c = gate_c(affected)
    if not (a and c):
        sys.exit("\nSTOP: a gate failed; nothing assembled.")
    assemble(affected, control, args.write, args.in_dir, out_name)


if __name__ == "__main__":
    main()
