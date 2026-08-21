#!/usr/bin/env python
"""Q4/Q5 cross-model DESCRIPTIVE panels: Llama-3.1-8B vs Qwen2.5-14B, side by side, NEVER pooled.

Two jobs, both descriptive (no verdicts here -- the verdicts live in replication_claims.py):

  1. PANELS: per model, per method, the mean PRR over the 8 long evals at each rung plus the
     macro OOD mean (mean of the 4 OOD rung-means). Leave-one-dataset-out (LODO) sensitivity
     for the deltas the replication memo quotes, because Qwen expertqa carries a severe length
     confound (the project's working notes paragraph 10.2) and no Qwen aggregate should be quoted without
     showing what happens when expertqa is dropped.
  2. LEHMER: the per-dataset finite-beta curves from the two populations' existing CSVs
     (Llama: sharpening_family __round2, the quoted source; Qwen: lehmer_qwen). The
     "best beta" column is an argmax over the curve on TEST labels -- printed only with an
     ORACLE/DESCRIPTIVE tag, never a method claim.

Method-name mapping between the two masters (display names vs driver arm names):
Llama 'msp_min/msp_sum/perplexity/SAPLMA/armA(attention)/armB(mean-pool)/wMSP-norm/wMSP-shrink@2'
= Qwen 'floor_min/floor_sum/floor_ppl/saplma/attention/uniform/wmsp_norm/wmsp_shrink2'.
The Llama panel reads only seed_regime == 3seed rows, exactly like replication_claims.py.

Populations stay separate end to end: two panels, two curve tables, no pooled statistic.

    python scripts/checks/crossmodel_panels.py
    python scripts/checks/crossmodel_panels.py --out results/analysis/crossmodel_panels.md
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from replication_claims import CSV_DEFAULT, EVALS, OOD_RUNGS, load  # noqa: E402

RUNGS = ["ID"] + OOD_RUNGS

# canonical method -> its name in each master. Superset of replication_claims.NAMES (the panels
# also carry msp_sum / the uniform pooler / un-shrunk wMSP, which no registered claim reads).
PANEL_METHODS = [
    ("perplexity",   {"llama": "perplexity",       "qwen": "floor_ppl"}),
    ("msp_min",      {"llama": "msp_min",          "qwen": "floor_min"}),
    ("msp_sum",      {"llama": "msp_sum",          "qwen": "floor_sum"}),
    ("SAPLMA",       {"llama": "SAPLMA",           "qwen": "saplma"}),
    ("uniform-pool", {"llama": "armB(mean-pool)",  "qwen": "uniform"}),
    ("attention",    {"llama": "armA(attention)",  "qwen": "attention"}),
    ("wMSP-norm",    {"llama": "wMSP-norm",        "qwen": "wmsp_norm"}),
    ("wMSP-shrink@2", {"llama": "wMSP-shrink@2",   "qwen": "wmsp_shrink2"}),
    ("wMSP-shrink@10", {"llama": "wMSP-shrink@10", "qwen": "wmsp_shrink10"}),
]

# The deltas the replication memo quotes, LODO-checked per model. (a, b) -> delta = a - b on the
# per-eval OOD mean (or on one rung where stated).
LODO_DELTAS = [
    ("msp_min - msp_sum   (OOD mean)", "msp_min", "msp_sum", None),
    ("msp_min - perplexity (OOD mean)", "msp_min", "perplexity", None),
    ("wMSP-shrink@2 - SAPLMA (OOD mean)", "wMSP-shrink@2", "SAPLMA", None),
    ("wMSP-shrink@2 - SAPLMA (DiffTask-long)", "wMSP-shrink@2", "SAPLMA", "DiffTask-long"),
    ("wMSP-shrink@2 - SAPLMA (1ds-Diff-long)", "wMSP-shrink@2", "SAPLMA", "1ds-Diff-long"),
]

LEHMER_SRC = {
    "llama": RES / "sharpening_family__meta-llama_Meta-Llama-3.1-8B__round2.csv",
    "qwen": RES / "lehmer_qwen__Qwen_Qwen2.5-14B.csv",
}
BETA_ORDER = ["0.0", "0.5", "1.0", "2.0", "4.0", "8.0", "16.0", "inf"]


def cell(g, names, method, rung, ev):
    """One (method, rung, eval) PRR from a loaded master; KeyError = genuinely missing, fail loud."""
    return g[(names[method], rung, ev)]


def ood_mean(g, names, method, ev):
    return float(np.mean([cell(g, names, method, r, ev) for r in OOD_RUNGS]))


def panel_lines(pop, g, names):
    out = [f"### {pop} panel — mean PRR over the 8 long evals (population: its own master, "
           f"carve legacy; no pooling with the other model)", "",
           "| method | " + " | ".join(RUNGS) + " | macro OOD |",
           "|---|" + "---|" * (len(RUNGS) + 1)]
    for disp, nm in PANEL_METHODS:
        names_pop = {disp: nm[pop]}
        try:
            vals = [np.mean([cell(g, names_pop, disp, r, e) for e in EVALS]) for r in RUNGS]
            ood = np.mean([ood_mean(g, names_pop, disp, e) for e in EVALS])
        except KeyError:
            out.append(f"| {disp} | " + " | ".join(["absent"] * (len(RUNGS) + 1)) + " |")
            continue
        out.append(f"| {disp} | " + " | ".join(f"{v:+.4f}" for v in vals) + f" | {ood:+.4f} |")
    return out


def lodo_lines(pop, g, names_all):
    out = [f"### {pop} LODO — macro delta with each eval dropped (n=7 each; ALL = n=8)", ""]
    for label, a, b, rung in LODO_DELTAS:
        nm = {a: names_all[a][pop], b: names_all[b][pop]}
        def d(ev):
            if rung is None:
                return ood_mean(g, nm, a, ev) - ood_mean(g, nm, b, ev)
            return cell(g, nm, a, rung, ev) - cell(g, nm, b, rung, ev)
        try:
            per = {ev: d(ev) for ev in EVALS}
        except KeyError:
            out.append(f"- {label}: absent on this population")
            continue
        allm = float(np.mean(list(per.values())))
        drops = ", ".join(f"-{ev} {np.mean([v for e2, v in per.items() if e2 != ev]):+.4f}"
                          for ev in EVALS)
        out.append(f"- **{label}**: ALL {allm:+.4f} | {drops}")
    out.append("")
    return out


def load_lehmer(pop):
    """dataset -> {beta_str: prr} from the population's existing curve CSV. No recomputation."""
    curves = {}
    with open(LEHMER_SRC[pop]) as f:
        for r in csv.DictReader(f):
            if pop == "llama":
                if r["family"] != "lehmer_beta":
                    continue
                ds, param, prr = r["dataset"], r["param"], float(r["prr"])
            else:
                # the Qwen CSV also carries Q1/Q2/Q3 verdict rows whose fields are not numbers --
                # only rows with a parseable (beta, prr) pair belong to the curve
                ds, param = r["dataset"], r["beta"]
                try:
                    prr = float(r["prr"])
                    float(param)  # accepts 'inf' too
                except ValueError:
                    continue
            curves.setdefault(ds, {})[param] = prr
    return curves


def lehmer_lines():
    out = ["## Q5 — Lehmer finite-beta curves, per dataset, two populations side by side",
           "", "Best-beta is an argmax on TEST labels: **ORACLE / DESCRIPTIVE ONLY**, never a "
           "method result. Sources: " + ", ".join(str(p.name) for p in LEHMER_SRC.values()), ""]
    cl, cq = load_lehmer("llama"), load_lehmer("qwen")
    hdr = " | ".join(f"b={b}" for b in BETA_ORDER)
    out += [f"| dataset | model | {hdr} | best b (ORACLE) |",
            "|---|---|" + "---|" * (len(BETA_ORDER) + 1)]
    for ds in EVALS:
        for pop, cc in (("Llama", cl), ("Qwen", cq)):
            cur = cc.get(ds)
            if not cur:
                out.append(f"| {ds} | {pop} | " + "absent |" * (len(BETA_ORDER) + 1))
                continue
            # tolerate float-format drift in the param column ('2.0' vs '2')
            def get(b):
                for k, v in cur.items():
                    if k == b or (b != "inf" and k not in ("inf",) and float(k) == float(b)):
                        return v
                raise KeyError(f"{ds} {pop} beta={b}")
            vals = [get(b) for b in BETA_ORDER]
            best = BETA_ORDER[int(np.argmax(vals))]
            out.append(f"| {ds} | {pop} | " + " | ".join(f"{v:+.3f}" for v in vals)
                       + f" | {best} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="also write the markdown here")
    args = ap.parse_args()

    names_all = dict(PANEL_METHODS)
    lines = ["# Cross-model descriptive panels — Llama-3.1-8B vs Qwen2.5-14B (SEPARATE, not pooled)",
             "", "Population caption: each panel is its own model's complete ProbeDriftLong master "
             "(8 long evals x 5 rungs, 3 seeds, carve legacy). Verdicts live in "
             "replication_claims.py; everything here is descriptive.", ""]
    for pop in ("llama", "qwen"):
        g = load(CSV_DEFAULT[pop], pop)
        lines += panel_lines(pop, g, names_all) + [""]
        lines += lodo_lines(pop, g, names_all)
    lines += lehmer_lines()

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
