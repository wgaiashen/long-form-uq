#!/usr/bin/env python
"""Assemble the Qwen POST-HOC cross-model transfers into one table.

WHAT THIS IS FOR. Two fixed Llama configurations were transferred onto Qwen2.5-14B after the M2
grid was already scored and visible: `wmsp_shrink1p5` (Llama's W5 pre-committed lambda) and
`softmax_tau1` (Llama's W1 a-priori tau). Neither is pre-registered on this population and neither
belongs in `pdl_master__Qwen_Qwen2.5-14B.csv`, which holds the registered 600-row ladder and must
stay exactly as scored -- so they get their own table, every row stamped
`provenance = post-hoc-transfer`.

⛔ THIS REFUSES TO EMIT wmsp_shrink1p5 UNLESS ITS OUTSIDE CONTROL PASSED. The lambda = 2.0 arm has
to reproduce the master's `wmsp_shrink2` on all 40 cells to 4 dp; if it does not, the training loop
is not the ladder's loop and lambda = 1.5 is not comparable to anything in the master. Emitting the
number anyway -- with the failure noted somewhere else -- is exactly how an uninterpretable figure
ends up quoted, so the gate is enforced here rather than described.

⚠️ THE TWO METHODS HAVE DIFFERENT UNITS AND THE TABLE SAYS SO.
  * `wmsp_shrink1p5` is TRAINED: 40 cells (8 evals x 5 rungs), 3 seeds each.
  * `softmax_tau1` is TRAINING-FREE and therefore RUNG-INVARIANT: 8 numbers, one per dataset. Its
    rung column reads `rung_invariant`. Repeating it across four OOD rungs for visual tidiness
    would turn 8 observations into 32, which is the 4x pseudo-replication M2 §2 exists to forbid.
For either method the unit of analysis for any claim is the DATASET (n = 8), never the cell.

    python scripts/checks/qwen_transfers.py
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache  # noqa: E402

MODEL_DEFAULT = "Qwen/Qwen2.5-14B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
GATE_TOL = 1e-4

FIELDS = ["model", "method", "eval", "rung", "prr", "prr_std", "n_seeds", "param",
          "diff_vs_reference", "reference", "carve", "provenance", "source"]


def load_wmsp(slug):
    """(rung, eval, method) -> row, from the per-eval lambda CSVs."""
    out, files = {}, []
    for ev in LONG:
        p = ROOT / "results" / f"wmsp_lambda_qwen__{slug}__{ev}.csv"
        if not p.exists():
            continue
        files.append(p.name)
        for r in csv.DictReader(open(p)):
            out[(r["rung"], r["eval"], r["method"])] = r
    return out, files


def load_master_shrink2(slug):
    p = ROOT / "results" / f"pdl_master__{slug}.csv"
    if not p.exists():
        return {}
    return {(r["rung"], r["eval"]): float(r["prr_mean"])
            for r in csv.DictReader(open(p))
            if r["method"] == "wmsp_shrink2" and r["prr_mean"] not in ("", "nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    args = ap.parse_args()
    slug = cache._slug(args.model)
    out_csv = ROOT / "results" / f"qwen_transfers__{slug}.csv"

    print("=" * 100)
    print(f"QWEN POST-HOC CROSS-MODEL TRANSFERS   model={args.model}")
    print("Population: ProbeDriftLong, 8 long evals, layer 23, judge gpt-5-mini, carve=legacy.")
    print("⚠️ NOT pre-registered on this population. Both parameters are FIXED from Llama and")
    print("   nothing was tuned on Qwen, but the Qwen master was already visible when they ran.")
    print("   These rows sit OUTSIDE the M2 scorecard (STOCKTAKE_qwen.md §2) and never pool with it.")
    print("=" * 100)

    rows = []

    # ---------------- softmax_tau1 (training-free, rung-invariant) ----------------
    p_tau = ROOT / "results" / f"softmax_tau_qwen__{slug}.csv"
    n_tau = 0
    if p_tau.exists():
        for r in csv.DictReader(open(p_tau)):
            if r.get("method") != "softmax_tau1":
                continue
            rows.append({"model": r["model"], "method": "softmax_tau1", "eval": r["eval"],
                         "rung": "rung_invariant", "prr": r["prr"], "prr_std": "",
                         "n_seeds": "", "param": "tau=1.0",
                         "diff_vs_reference": r["diff_vs_msp_min"], "reference": "msp_min",
                         "carve": "legacy", "provenance": "post-hoc-transfer",
                         "source": p_tau.name})
            n_tau += 1
        print(f"\nsoftmax_tau1: {n_tau}/8 datasets  (training-free, ONE value each, rung_invariant)")
        if n_tau < 8:
            print(f"  ⚠️ {8 - n_tau} dataset(s) absent -- named, not silently dropped: "
                  f"{sorted(set(LONG) - {r['eval'] for r in rows})}")
    else:
        print(f"\nsoftmax_tau1: ABSENT -- no {p_tau.name}. Run softmax_tau_qwen.py.")

    # ---------------- wmsp_shrink1p5, gated on its control ----------------
    wm, files = load_wmsp(slug)
    ref = load_master_shrink2(slug)
    print(f"\nwmsp lambda sources ({len(files)}/8): {', '.join(files) if files else '(none)'}")

    ctl = [(k[0], k[1], float(v["prr_mean"]), ref[(k[0], k[1])])
           for k, v in wm.items() if k[2] == "wmsp_shrink2" and (k[0], k[1]) in ref]
    fails = [(rg, ev, a, b) for rg, ev, a, b in ctl if abs(a - b) > GATE_TOL]
    print(f"\nOUTSIDE CONTROL (lambda = 2.0 vs the master's wmsp_shrink2, tol {GATE_TOL}):")
    print(f"  cells compared {len(ctl)}/40   "
          f"max |delta| {max((abs(a - b) for _, _, a, b in ctl), default=float('nan')):.6f}")
    gate_ok = (len(ctl) == 40) and not fails
    if fails:
        print(f"  ⛔ FAIL on {len(fails)} cell(s):")
        for rg, ev, a, b in fails[:8]:
            print(f"     {ev:16s} {rg:16s} here {a:+.4f} master {b:+.4f} d {abs(a - b):.5f}")
    elif len(ctl) < 40:
        print(f"  🟡 PARTIAL -- {40 - len(ctl)} cells not yet produced; not a pass.")
    else:
        print("  ✅ PASS on all 40 cells -- this loop is the ladder's loop.")

    n_l = 0
    if gate_ok:
        for (rg, ev, m), r in sorted(wm.items()):
            if m != "wmsp_shrink1p5":
                continue
            base = wm.get((rg, ev, "wmsp_shrink2"))
            d = (float(r["prr_mean"]) - float(base["prr_mean"])) if base else None
            rows.append({"model": r["model"], "method": "wmsp_shrink1p5", "eval": ev, "rung": rg,
                         "prr": r["prr_mean"], "prr_std": r["prr_std"], "n_seeds": r["n_seeds"],
                         "param": "lambda=1.5",
                         "diff_vs_reference": f"{d:.6f}" if d is not None else "",
                         "reference": "wmsp_shrink2 (lambda=2.0, paired, same draws)",
                         "carve": r.get("carve", ""), "provenance": "post-hoc-transfer",
                         "source": f"wmsp_lambda_qwen__{slug}__{ev}.csv"})
            n_l += 1
        print(f"\nwmsp_shrink1p5: {n_l}/40 cells emitted")
    else:
        print("\nwmsp_shrink1p5: ⛔ WITHHELD. The control did not pass, so these numbers are not")
        print("   comparable to the master and are not written. This is deliberate -- a withheld")
        print("   number cannot be quoted by accident; a written one with a caveat elsewhere can.")

    # ---------------- write ----------------
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out_csv}  ({len(rows)} rows)")

    # ---------------- the dataset-level summary, which is the only valid unit ----------------
    if n_l:
        print("\n" + "=" * 100)
        print("lambda 1.5 - lambda 2.0, aggregated to the DATASET (OOD rungs averaged first, n = 8)")
        print("   Cells are NOT independent. Llama reference: +0.0019 mean, 6/8 -- which FAILED its")
        print("   > +0.010 bar. DESCRIPTIVE, no bar attached here and none may be invented.")
        print("=" * 100)
        per = []
        for ev in LONG:
            ds = [float(wm[(rg, ev, "wmsp_shrink1p5")]["prr_mean"])
                  - float(wm[(rg, ev, "wmsp_shrink2")]["prr_mean"])
                  for rg in RUNGS[1:] if (rg, ev, "wmsp_shrink1p5") in wm]
            if ds:
                per.append((ev, float(np.mean(ds))))
                print(f"  {ev:16s}{np.mean(ds):>+10.4f}")
        if per:
            v = np.array([d for _, d in per])
            print(f"  {'MEAN (n=8)':16s}{v.mean():>+10.4f}   wins {int((v > 0).sum())}/{len(v)}")

    print("\nEvery row carries provenance = post-hoc-transfer. Do NOT merge this into pdl_master and")
    print("do NOT pool it with Llama.")
    return 0 if (n_tau or n_l) else 1


if __name__ == "__main__":
    sys.exit(main())
