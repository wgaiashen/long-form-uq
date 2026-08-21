#!/usr/bin/env python
"""W8 verdict — compute the PREREGISTERED evidence package for the adaptive-Lehmer run.

Runs ONCE, on the complete grid, after prereg/adaptive_lehmer_aggregation.md (committed a82a62b, before
any PRR). It computes exactly the package §2/§3 of the prereg promise — no additional slicing,
no new subgroups, no promotion logic beyond the prereg's report-promotion rule. Interpretation
(the narrowest Decision-Gate-B pattern) is written by the researcher from this output; the
script only refuses to hide anything.

Coverage is checked FIRST (40 cells × 3 seeds × 6 methods); a partial grid aborts.

    python scripts/checks/adaptive_lehmer_verdict.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

CSV = ROOT / "results" / "adaptive_lehmer__meta-llama_Meta-Llama-3.1-8B.csv"
MASTER = ROOT / "results" / "pdl_master__meta-llama_Meta-Llama-3.1-8B.csv"
ROUND2 = ROOT / "results" / "sharpening_family__meta-llama_Meta-Llama-3.1-8B__round2.csv"
OUT = ROOT / "results" / "analysis" / "W8_ADAPTIVE_LEHMER_VERDICT.md"

EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
OOD_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
RUNGS = ["ID"] + OOD_RUNGS
METHODS = ["lehmer_global_beta", "lehmer_nllshape_beta", "lehmer_hs_beta", "lehmer_hybrid_beta",
           "lehmer_hs_beta_shuffled_h", "lehmer_hybrid_beta_shuffled_h"]
PRIMARY = "lehmer_nllshape_beta"
SEEDS = {1, 2, 3}
BOOT_B, BOOT_SEED = 10000, 12345


def load_new():
    df = pd.read_csv(CSV)
    missing = []
    for m in METHODS:
        for e in EVALS:
            for r in RUNGS:
                # NB df["eval"], never df.eval — .eval is a pandas METHOD and shadows the column
                g = df[(df["method"] == m) & (df["eval"] == e) & (df["rung"] == r)]
                if set(g["seed"]) != SEEDS:
                    missing.append((m, e, r, sorted(g["seed"])))
    if missing:
        raise SystemExit(f"GRID INCOMPLETE — refusing a partial verdict. First gaps: {missing[:6]} "
                         f"({len(missing)} total)")
    cell = df.groupby(["method", "eval", "rung"], as_index=False)["prr"].mean()
    return {(r.method, r.eval, r.rung): r.prr for r in cell.itertuples()}


def load_master():
    out = {}
    for r in pd.read_csv(MASTER).itertuples():
        if getattr(r, "seed_regime", "") == "3seed":
            out[(r.method, r.eval, r.rung)] = float(r.prr)
    return out


def load_lehmer_b1():
    d = {}
    for r in pd.read_csv(ROUND2).itertuples():
        if r.family == "lehmer_beta" and str(r.param) == "1.0":
            d[r.dataset] = float(r.prr)
    return d


def ood_mean(g, m, e, rungs=OOD_RUNGS):
    return float(np.mean([g[(m, e, r)] for r in rungs]))


def package(deltas, label, lines):
    d = np.array([deltas[e] for e in EVALS])
    _, p = wilcoxon(d, alternative="two-sided") if np.any(d != 0) else (None, 1.0)
    rng = np.random.RandomState(BOOT_SEED)
    boots = [np.mean(d[rng.randint(0, 8, 8)]) for _ in range(BOOT_B)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    lodo = {e: float(np.mean([v for k, v in deltas.items() if k != e])) for e in EVALS}
    lines.append(f"\n### {label}")
    lines.append("| dataset | delta |")
    lines.append("|---|---|")
    for e in EVALS:
        lines.append(f"| {e} | {deltas[e]:+.4f} |")
    lines.append(f"\n- macro mean delta **{d.mean():+.4f}**, median {np.median(d):+.4f}, "
                 f"signs {(d > 0).sum()}/8, exact two-sided Wilcoxon p = {p:.4f}")
    lines.append(f"- dataset-level bootstrap 95% CI for the macro mean: [{lo:+.4f}, {hi:+.4f}]")
    lines.append("- leave-one-dataset-out macro deltas: "
                 + ", ".join(f"-{e} {v:+.4f}" for e, v in lodo.items()))
    promo = (d.mean() > 0 and all(v > 0 for v in lodo.values()))
    lines.append(f"- report-promotion precondition (positive macro AND every LODO positive): "
                 f"{'met' if promo else 'NOT met'} (necessary, not sufficient — see prereg §2)")
    return d.mean(), p


def main():
    g = load_new()
    M = load_master()
    b1 = load_lehmer_b1()
    lines = ["# W8 ADAPTIVE LEHMER — preregistered evidence package (computed once)", "",
             f"Grid: 40 cells × 3 seeds × 6 methods, complete (checked). Source: {CSV.name}. "
             f"Population: canonical Llama ProbeDriftLong, carve legacy, layer 15. "
             f"Prereg: prereg/adaptive_lehmer_aggregation.md (a82a62b)."]

    # ---- primary: NLL-SHAPE vs msp_min, and vs perplexity ----
    d_min = {e: ood_mean(g, PRIMARY, e) - M[("msp_min", e, "LOO-long")] for e in EVALS}
    d_ppl = {e: ood_mean(g, PRIMARY, e) - M[("perplexity", e, "LOO-long")] for e in EVALS}
    m1, _ = package(d_min, f"PRIMARY: {PRIMARY} − msp_min (per-dataset OOD means, n=8)", lines)
    package(d_ppl, f"Endpoint robustness: {PRIMARY} − perplexity", lines)

    # ---- secondary 1: adaptation itself (vs GLOBAL) ----
    d_gl = {e: ood_mean(g, PRIMARY, e) - ood_mean(g, "lehmer_global_beta", e) for e in EVALS}
    package(d_gl, "Adaptation itself: NLL-SHAPE − GLOBAL", lines)
    d_hyb_gl = {e: ood_mean(g, "lehmer_hybrid_beta", e) - ood_mean(g, "lehmer_global_beta", e)
                for e in EVALS}
    package(d_hyb_gl, "HYBRID − GLOBAL", lines)

    # ---- secondary 2: signal ablation + shuffled controls ----
    lines.append("\n### Signal ablation & controls — macro OOD mean (per-dataset OOD means averaged)")
    lines.append("| method | macro OOD | vs shuffled-h |")
    lines.append("|---|---|---|")
    for m in METHODS:
        mo = float(np.mean([ood_mean(g, m, e) for e in EVALS]))
        ctrl = ""
        if m in ("lehmer_hs_beta", "lehmer_hybrid_beta"):
            sh = float(np.mean([ood_mean(g, m + "_shuffled_h", e) for e in EVALS]))
            ctrl = f"{mo - sh:+.4f}"
        lines.append(f"| {m} | {mo:+.4f} | {ctrl} |")

    # ---- secondary 3: descriptive comparisons ----
    lines.append("\n### Descriptive comparisons — macro OOD means (same population)")
    lines.append("| method | macro OOD |")
    lines.append("|---|---|")
    for name, val in [
        ("perplexity", np.mean([M[("perplexity", e, "LOO-long")] for e in EVALS])),
        ("msp_min", np.mean([M[("msp_min", e, "LOO-long")] for e in EVALS])),
        ("fixed Lehmer beta=1 (round2, rung-invariant)", np.mean([b1[e] for e in EVALS])),
        ("SAPLMA", np.mean([ood_mean(M, "SAPLMA", e) for e in EVALS])),
        ("attention pooler", np.mean([ood_mean(M, "armA(attention)", e) for e in EVALS])),
        ("wMSP-norm", np.mean([ood_mean(M, "wMSP-norm", e) for e in EVALS])),
        ("wMSP-shrink@2 (incumbent)", np.mean([ood_mean(M, "wMSP-shrink@2", e) for e in EVALS])),
    ]:
        lines.append(f"| {name} | {val:+.4f} |")
    for m in METHODS[:4]:
        lines.append(f"| {m} | {np.mean([ood_mean(g, m, e) for e in EVALS]):+.4f} |")

    # ---- predeclared axes: hard OOD, summarisation subgroup, per-rung table ----
    for rung in ("DiffTask-long", "1ds-Diff-long"):
        d_r = {e: g[(PRIMARY, e, rung)] - M[("msp_min", e, rung)] for e in EVALS}
        package(d_r, f"Hard OOD ({rung}): {PRIMARY} − msp_min", lines)
    summ = ["xsum", "cnn_dailymail", "samsum"]
    lines.append("\n### Summarisation subgroup (predeclared) — per-dataset OOD-mean deltas vs msp_min")
    for e in summ:
        lines.append(f"- {e}: {d_min[e]:+.4f}")

    lines.append("\n### Per-rung table — new methods (8-eval mean per rung)")
    lines.append("| method | " + " | ".join(RUNGS) + " |")
    lines.append("|---|" + "---|" * len(RUNGS))
    for m in METHODS:
        row = [float(np.mean([g[(m, e, r)] for e in EVALS])) for r in RUNGS]
        lines.append(f"| {m} | " + " | ".join(f"{v:+.4f}" for v in row) + " |")

    text = "\n".join(lines) + "\n"
    print(text)
    OUT.write_text(text)
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
