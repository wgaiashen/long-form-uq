#!/usr/bin/env python
"""wMSP shrinkage lambda = 1.5 on the Qwen2.5-14B grid -- a POST-HOC cross-model transfer,
with a full paired lambda = 2 control.

WHERE lambda = 1.5 COMES FROM, STATED PRECISELY. It is Llama's W5 pre-committed primary
(prereg/shrinkage_lambda_and_nll_prior.md, `sharpening_lambda.py:70` LAMBDA_PRIMARY = 1.5).

DO NOT DESCRIBE IT AS "THE STRONGEST SHRINK SETTING ON LLAMA" WITHOUT THE OTHER HALF:
  * W5's REGISTERED CLAIM FAILED. lambda = 1.5 vs the incumbent shrink@2 was +0.0019 against a
    > +0.010 bar, signs 6/8 (the project's working notes). That is a TIE, not a win.
  * What justifies transferring it is the honest-selection result: it IS the best fixed value on
    the Llama cross-dataset mean (+0.2306), and the 5-arm leave-one-dataset-out selection picks it
    on 7 of 8 folds, which is why §18 re-baselined the incumbent to shrink@1.5. LODO is label-free
    selection, so this is a defensible fixed transfer -- but only the second bullet supports it.

THIS IS NOT A PRE-REGISTERED QWEN TEST. The Qwen master was already visible when it was run.
No parameter is tuned on Qwen: lambda is fixed at 1.5 and 2.0 and nothing else is tried. Output
carries provenance = post-hoc-transfer and sits OUTSIDE the M2 scorecard (the project's working notes).

WHY A SEPARATE FILE, WITH ZERO SHARED-FILE EDITS. Adding a `wmsp_shrink1p5` entry to
`probedriftlong.py`'s WMSP list would touch a file on BOTH workstreams' collision lists, obliging
coordination with RCS plus a no-op-control re-run -- for a change that is avoidable. Llama's own
`sharpening_lambda.py:39` set this precedent ("ZERO SHARED-FILE EDITS. Imports from
luq.weighted_msp and runs its own scoring loop"), and the training call below is the ladder's own
`weighted_msp.weighted_msp_unc(...)` with the ladder's own kwargs, not a re-implementation.

THE OUTSIDE CONTROL, WHICH GATES EVERYTHING. lambda = 2.0 is run as a FULL arm over the same
8 x 5 x 3 grid, not spot-checked, for two reasons: it controls on all 40 cells rather than a
sample, and it makes lambda1.5 - lambda2 a PAIRED comparison on identical draws. All 40 lambda=2
cells must reproduce the master's `wmsp_shrink2` rows to 4 dp. If they do not, the loop is not the
ladder's loop and lambda = 1.5 is not interpretable -- stop and diagnose. (Llama precedent: W5's
equivalent `norm`-anchor check passed on all 40 cells.)

    python scripts/checks/wmsp_lambda_qwen.py --evals pubmed_qa       # one eval (array task)
    python scripts/checks/wmsp_lambda_qwen.py --gate-only             # the 40-cell verdict
"""
import argparse
import csv as _csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import msp, results, weighted_msp                        # noqa: E402
from luq.weighting import shrink_to_uniform                       # noqa: E402
from transformers import AutoTokenizer                            # noqa: E402
from aggregation_table import load_per_token                      # noqa: E402
from xl_rungs import eval_split, label_of, build_rows             # noqa: E402
import probedriftlong as pdl                                      # noqa: E402

MODEL_DEFAULT = "Qwen/Qwen2.5-14B"
LLAMA = "meta-llama/Meta-Llama-3.1-8B"
LAYER_DEFAULT = 23                       # the fixed rule ceil(N/2)-1 on Qwen; not selected

# The two arms. shrink2 FIRST so the control is computed before the transfer in every cell -- if the
# control is going to fail, it should fail before any lambda=1.5 number exists to look at.
ARMS = [("wmsp_shrink2", 2.0),          # the OUTSIDE CONTROL: must match the master to 4 dp
        ("wmsp_shrink1p5", 1.5)]        # the transfer
GATE_TOL = 1e-4                          # "to 4 dp", as the plan and W5's precedent specify

FIELDS = ["model", "rung", "eval", "method", "reg_lambda", "prr_mean", "prr_std", "n_seeds",
          "carve", "git_sha", "cluster", "env_hash", "provenance"]


OUT_TAG = ""   # set from --tag in main(); "" reproduces the original fixed path exactly


def out_path(slug, ev):
    tag = f"_{OUT_TAG}" if OUT_TAG else ""
    return ROOT / "results" / f"wmsp_lambda_qwen{tag}__{slug}__{ev}.csv"


def master_path(slug):
    """The reference master for the outside control. Defaults to the canonical raw-span master;
    --master-csv overrides it so a clean-population run can gate against the clean-population
    master instead of one built on a different population it was never meant to match."""
    return Path(MASTER_CSV) if MASTER_CSV else (ROOT / "results" / f"pdl_master__{slug}.csv")


MASTER_CSV = ""   # set from --master-csv in main(); "" reproduces the original path exactly


def _flush(path, rows):
    """Atomic per-cell write. Same reasoning as probedriftlong._flush_rows: a run killed at hour N
    must leave the cells it finished on disk, and a crash mid-write must not leave a truncated CSV
    that reads as a short-but-valid grid."""
    tmp = Path(str(path) + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def master_shrink2(slug):
    """(rung, eval) -> the published wmsp_shrink2 PRR, for the outside control."""
    p = master_path(slug)
    if not p.exists():
        raise SystemExit(f"no master at {p} -- the outside control cannot run, so nothing here is "
                         f"interpretable. Build it with qwen_pdl_master.py --strict first.")
    out = {}
    for r in _csv.DictReader(open(p)):
        if r["method"] == "wmsp_shrink2":
            try:
                out[(r["rung"], r["eval"])] = float(r["prr_mean"])
            except (TypeError, ValueError):
                pass
    return out


def gate_only(slug):
    """Re-read every produced CSV and deliver the 40-cell control verdict."""
    ref = master_shrink2(slug)
    got, missing = {}, []
    for ev in pdl.LONG:
        p = out_path(slug, ev)
        if not p.exists():
            missing.append(ev); continue
        for r in _csv.DictReader(open(p)):
            if r["method"] == "wmsp_shrink2":
                got[(r["rung"], r["eval"])] = float(r["prr_mean"])
    print("=" * 100)
    print(f"OUTSIDE CONTROL -- lambda = 2.0 arm vs the published wmsp_shrink2, tolerance {GATE_TOL}")
    print("=" * 100)
    if missing:
        print(f"no CSV yet for: {', '.join(missing)} -- the verdict below is PARTIAL")
    worst, fails = 0.0, []
    for k, v in sorted(ref.items()):
        if k not in got:
            continue
        d = abs(v - got[k])
        worst = max(worst, d)
        if d > GATE_TOL:
            fails.append((k, v, got[k], d))
    print(f"cells compared: {len([k for k in ref if k in got])}/40   max |delta| {worst:.6f}")
    if fails:
        print(f"GATE FAIL on {len(fails)} cell(s) -- lambda = 1.5 is NOT interpretable. Diagnose:")
        for (rg, ev), a, b, d in fails[:10]:
            print(f"    {ev:16s} {rg:16s} master {a:+.4f}  here {b:+.4f}  d {d:.5f}")
        return 1
    if len([k for k in ref if k in got]) < 40:
        print("every compared cell matches, but coverage is partial -- not yet a pass.")
        return 1
    print("GATE PASS on all 40 cells. This loop IS the ladder's loop; the paired lambda1.5 -")
    print("   lambda2 comparison below is on identical draws.")
    # the transfer, only now
    rows = []
    for ev in pdl.LONG:
        for r in _csv.DictReader(open(out_path(slug, ev))):
            rows.append(r)
    print("\n" + "=" * 100)
    print("lambda = 1.5 vs lambda = 2.0, PAIRED per cell -- DESCRIPTIVE, no bar, no verdict.")
    print("   Llama for reference: +0.0019 mean, 6/8 signs, which FAILED its > +0.010 bar.")
    print("=" * 100)
    by = {}
    for r in rows:
        by.setdefault((r["rung"], r["eval"]), {})[r["method"]] = float(r["prr_mean"])
    alld = []
    print(f"{'rung':18s}{'mean d(1.5-2)':>15s}{'wins':>8s}")
    rungs = sorted({k[0] for k in by})
    for rg in rungs:
        ds = [by[k]["wmsp_shrink1p5"] - by[k]["wmsp_shrink2"]
              for k in by if k[0] == rg and "wmsp_shrink1p5" in by[k]]
        alld += ds
        print(f"{rg:18s}{np.mean(ds):>+15.4f}{sum(d > 0 for d in ds):>5d}/{len(ds)}")
    print(f"{'ALL 40 cells':18s}{np.mean(alld):>+15.4f}{sum(d > 0 for d in alld):>5d}/{len(alld)}")
    print("\nCells are NOT independent observations -- the unit of analysis for any claim is the")
    print("   DATASET (n = 8). This per-rung view is descriptive shape, not a test.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--evals", default="pubmed_qa")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=LAYER_DEFAULT)
    ap.add_argument("--gate-only", action="store_true",
                    help="skip training; re-read the produced CSVs and deliver the 40-cell verdict")
    ap.add_argument("--tag", default="",
                    help="EXPLICIT output-path tag. Default '' reproduces the original fixed path "
                         "(results/wmsp_lambda_qwen__<slug>__<eval>.csv) exactly, so the existing "
                         "raw-span run is never at risk of being overwritten by a later one.")
    ap.add_argument("--master-csv", default="",
                    help="EXPLICIT reference master for the outside control. Default '' reproduces "
                         "the original path (results/pdl_master__<slug>.csv). Pass a different "
                         "master (e.g. the clean-population one) so the control checks against the "
                         "population this run actually used, not a population it was never meant "
                         "to match.")
    args = ap.parse_args()

    global OUT_TAG, MASTER_CSV
    OUT_TAG = args.tag
    MASTER_CSV = args.master_csv

    from luq import cache
    slug = cache._slug(args.model)
    if args.gate_only:
        return gate_only(slug)

    evals = [e for e in args.evals.split(",") if e]
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prov = pdl._provenance()              # aborts on a dirty TRACKED tree, same as the ladder
    carve = os.environ.get("LUQ_CARVE", "legacy")

    print("=" * 100)
    print(f"wMSP LAMBDA TRANSFER   model={args.model}  layer={args.layer}  seeds={seeds}")
    print(f"evals={evals}  device={device}  LUQ_CARVE={carve}")
    print("POST-HOC CROSS-MODEL TRANSFER -- the Qwen master was already visible. Nothing is tuned")
    print("   here: lambda is fixed at 2.0 (control) and 1.5 (transfer, Llama's W5 primary), and")
    print("   W5's registered claim FAILED on Llama as a tie with shrink@2. See the docstring.")
    print("=" * 100, flush=True)

    # THE QWEN SPECIAL-TOKEN TRAP, carried over verbatim from probedriftlong.py:386-397.
    # weighted_msp.content_keep falls back to the Llama-3 `id >= 128000` range test. Qwen2.5's vocab
    # runs to 152,064 with specials at 151,643+, so under that rule 23,643 ORDINARY content tokens
    # would be zero-weighted: no crash, just a quietly different method. Registering the real ids is
    # not optional here -- the whole arm reads content_keep.
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.model != LLAMA:
        from luq import token_subsets
        weighted_msp.set_special_ids(tok.all_special_ids)
        token_subsets.set_special_ids(tok.all_special_ids)
        print(f"special-token ids registered from the {args.model} tokenizer "
              f"({len(tok.all_special_ids)} ids; Llama range-test fallback OFF)", flush=True)

    # ---------------- pools ----------------
    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        loaded = load_per_token(args.model, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> SKIPPED LOUDLY (cells needing it stay ABSENT, not 0)")
            continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            continue
        if not finite.all():                       # identical to the ladder's finite-label filter
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} rows (label={label_of(d)})", flush=True)
    sources = set(PT)

    ref = master_shrink2(slug)                     # the control target, loaded before any training
    rows = []
    for ev in evals:
        path = out_path(slug, ev)
        for rung, X, spec in pdl.cells_long(sources, [ev]):
            if X not in PT:
                continue
            if len(eval_split(PT[X][1])[1]) == 0:
                continue
            per = {n: [] for n, _ in ARMS}
            floors = []
            for sd in seeds:
                train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
                if not train_rows or not test_rows:
                    continue
                n_tr = len(train_rows)
                tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
                allrows = train_rows + test_rows
                y = np.array([PT[d][2][i] for d, i in allrows], float)
                yte = np.array([y[i] for i in te_idx], float)
                states = [PT[d][0][i] for d, i in allrows]
                records = [PT[d][3][i] for d, i in allrows]
                floors.append(results.prr(yte, np.array(
                    [msp.msp_uncertainty(records[i]["token_logprobs"], "min") for i in te_idx])))
                for name, lam in ARMS:
                    # THE LADDER'S OWN CALL AND KWARGS -- probedriftlong.py:533-534 passes exactly
                    # weighted_msp_unc(..., length_normalise=True, seed=sd, **kw) with
                    # kw = {"weight_mode": "normalised", "reg": shrink_to_uniform, "reg_lambda": L}.
                    # Default loss is "pairwise" (NOT Blondel), which is what the transfer specifies.
                    v = weighted_msp.weighted_msp_unc(
                        states, records, y, tr_idx, te_idx, device,
                        weight_mode="normalised", reg=shrink_to_uniform, reg_lambda=lam,
                        length_normalise=True, seed=sd)
                    per[name].append(results.prr(yte, np.asarray(v, float)))
            if not per[ARMS[0][0]]:
                continue
            for name, lam in ARMS:
                rows.append({"model": args.model, "rung": rung, "eval": X, "method": name,
                             "reg_lambda": lam, "prr_mean": f"{np.mean(per[name]):.6f}",
                             "prr_std": f"{np.std(per[name]):.6f}", "n_seeds": len(per[name]),
                             "carve": carve, "provenance": "post-hoc-transfer", **prov})
            # The control, printed AS THE CELL LANDS rather than at the end -- a drift shows up on
            # the first cell, not after eight hours.
            m2 = np.mean(per["wmsp_shrink2"])
            tgt = ref.get((rung, X))
            ctl = (f"ctl vs master {tgt:+.4f} d{abs(m2 - tgt):.5f} "
                   f"{'OK' if abs(m2 - tgt) <= GATE_TOL else '<<< DRIFT'}") if tgt is not None \
                else "ctl: no master row"
            print(f"  [{rung:16s} {X:14s}] msp_min {np.mean(floors):+.4f}  "
                  f"lam2 {m2:+.4f}  lam1.5 {np.mean(per['wmsp_shrink1p5']):+.4f}  {ctl}", flush=True)
            _flush(path, rows)
        print(f"  wrote {path} ({len(rows)} rows)", flush=True)

    print("\nRun `--gate-only` once all 8 evals are present for the 40-cell control verdict.")
    print("Until that passes, lambda = 1.5 is not interpretable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
