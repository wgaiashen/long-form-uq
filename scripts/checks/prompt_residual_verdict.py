#!/usr/bin/env python
"""PR1 -- the preregistered evidence package for prompt-residual probing.

Reads ONLY the committed per-eval CSVs written by scripts/checks/prompt_residual.py and emits the
exact package prereg/PR1_prompt_residual.md promised. No fitting, no compute, no choices.

    python scripts/checks/prompt_residual_verdict.py

⚠️ THE UNIT OF ANALYSIS IS THE DATASET, n = 8. Not "32 OOD cells", which is 8 values counted four
times and yields an interval roughly twice too narrow.

⚠️ THE PRIMARY COMPARISON IS R2 - R1 (residual vs generated-only), NOT R2 - R0. Comparing the
residual only against the anchor-inclusive mean would confound "subtracting h0" with "dropping h0",
which is the entire reason the R1 arm exists.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = Path("/rds/general/user/gs925/home/gs925-msc_project/msc-project-gs925/"
              "results/method_dev/prompt_residual")
OUT_MD = IN_DIR / "PROMPT_RESIDUAL_VERDICT.md"

EVALS = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa", "asqa", "factscore"]
OOD_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
HARD_RUNGS = ["DiffTask-long", "1ds-Diff-long"]
EASY_RUNGS = ["SameTask-long", "LOO-long"]
ALL_RUNGS = ["ID"] + OOD_RUNGS

R0, R1 = "resid_R0_anchormean", "resid_R1_genmean"
R2, R3 = "resid_R2_promptresid", "resid_R3_constanchor"
ALL_ARMS = [R0, R1, R2, R3]
BOOT_B, BOOT_SEED = 10000, 0

# The pre-registered gate. Fixed in prereg §4 before any R1/R2 PRR existed; not editable here.
GATE_MACRO, GATE_SIGNS = 0.010, 6


def load():
    """One PRR per (method, eval, rung), averaged over the seeds that actually ran."""
    files = sorted(IN_DIR.glob("prompt_residual_*__meta-llama_Meta-Llama-3.1-8B.csv"))
    files = [f for f in files if "SMOKE" not in f.name]
    if not files:
        raise SystemExit(f"no result CSVs in {IN_DIR}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if df.get("smoke", pd.Series([False])).any():
        raise SystemExit("a SMOKE row reached the verdict input -- refusing to score it")
    g = df.groupby(["method", "eval", "rung"])["prr"].agg(["mean", "std", "count"])
    return df, g


def cell(g, m, e, r):
    """PRR for one cell, or None if it was never measured. NEVER 0 -- a blank must not read as a
    measured-and-bad number."""
    try:
        return float(g.loc[(m, e, r), "mean"])
    except KeyError:
        return None


def scope_mean(g, m, e, rungs):
    vals = [cell(g, m, e, r) for r in rungs]
    return None if any(v is None for v in vals) else float(np.mean(vals))


def package(deltas, label, lines, gate=False):
    """The standing evidence package: per-dataset deltas, macro, signs, Wilcoxon, bootstrap, LODO."""
    have = [e for e in EVALS if deltas.get(e) is not None]
    if len(have) < 3:
        lines.append(f"\n### {label}\n\n⚠️ only {len(have)}/8 datasets measured -- not scored.")
        return None
    d = np.array([deltas[e] for e in have])
    _, p = wilcoxon(d, alternative="two-sided") if np.any(d != 0) else (None, 1.0)
    rng = np.random.RandomState(BOOT_SEED)
    boots = [np.mean(d[rng.randint(0, len(d), len(d))]) for _ in range(BOOT_B)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    lodo = {e: float(np.mean([deltas[k] for k in have if k != e])) for e in have}

    lines.append(f"\n### {label}")
    lines.append("")
    lines.append("| dataset | delta |")
    lines.append("|---|---|")
    for e in have:
        lines.append(f"| `{e}` | {deltas[e]:+.4f} |")
    if len(have) < 8:
        lines.append(f"\n⚠️ MISSING (not measured, not zero): {sorted(set(EVALS) - set(have))}")
    lines.append(f"\n- macro mean delta **{d.mean():+.4f}**, median {np.median(d):+.4f}, "
                 f"signs {(d > 0).sum()}/{len(d)}, exact two-sided Wilcoxon p = "
                 + (f"{p:.4f}" if p is not None else "n/a"))
    lines.append(f"- dataset-level bootstrap 95% CI for the macro mean: [{lo:+.4f}, {hi:+.4f}]")
    lines.append("- leave-one-dataset-out macro deltas: "
                 + ", ".join(f"−{e} {v:+.4f}" for e, v in lodo.items()))

    if gate:
        c1 = d.mean() > GATE_MACRO
        c2 = int((d > 0).sum()) >= GATE_SIGNS
        c3 = all(v > 0 for v in lodo.values())
        lines.append("")
        lines.append("**PRE-REGISTERED GATE (prereg §4, fixed before any result):**")
        lines.append("")
        lines.append("| condition | required | observed | |")
        lines.append("|---|---|---|---|")
        lines.append(f"| macro delta | > +{GATE_MACRO:.3f} | {d.mean():+.4f} | "
                     f"{'PASS' if c1 else 'FAIL'} |")
        lines.append(f"| positive datasets | ≥ {GATE_SIGNS}/8 | {(d > 0).sum()}/{len(d)} | "
                     f"{'PASS' if c2 else 'FAIL'} |")
        lines.append(f"| every LODO positive | all | "
                     f"{sum(v > 0 for v in lodo.values())}/{len(lodo)} | "
                     f"{'PASS' if c3 else 'FAIL'} |")
        lines.append("")
        lines.append(f"**VERDICT: {'PROMISING' if (c1 and c2 and c3) else 'NOT PROMISING'}** "
                     f"— a decision rule for spending remaining time, NOT a significance claim.")
    return d.mean()


def main():
    df, g = load()
    lines = ["# PR1 PROMPT-RESIDUAL PROBING — preregistered evidence package", "",
             "Population: `meta-llama/Meta-Llama-3.1-8B`, canonical ProbeDriftLong, carve `legacy`, "
             "layer 15, seeds 1/2/3. Prereg: `prereg/PR1_prompt_residual.md`, committed before any "
             "`R1`/`R2` PRR existed.", "",
             "| arm | representation |", "|---|---|",
             "| `resid_R0_anchormean` | `s.mean(0)` — anchor-inclusive mean (the existing control) |",
             "| `resid_R1_genmean` | `s[1:].mean(0)` — generated-only mean |",
             "| `resid_R2_promptresid` | `s[1:].mean(0) - s[0]` — prompt-residual (the method) |",
             "| `resid_R3_constanchor` | `s[1:].mean(0) - mean_train(h0)` — constant anchor "
             "(centring control, prereg §8) |"]

    # ---- coverage, stated BEFORE any number -----------------------------------------------------
    lines.append("\n## 0. Coverage")
    lines.append("")
    missing = []
    for e in EVALS:
        for r in ALL_RUNGS:
            for m in ALL_ARMS:
                if cell(g, m, e, r) is None:
                    missing.append(f"{m}/{e}/{r}")
    n_want = len(EVALS) * len(ALL_RUNGS) * len(ALL_ARMS)
    lines.append(f"- {n_want - len(missing)}/{n_want} (eval × rung × arm) cells present.")
    if missing:
        lines.append(f"- ⚠️ **MISSING (not measured, never zero):** {len(missing)} — "
                     + ", ".join(f"`{x}`" for x in missing[:12])
                     + (" …" if len(missing) > 12 else ""))
    else:
        lines.append("- Complete grid, no gaps.")
    seeds = sorted(df["seed"].unique().tolist())
    lines.append(f"- seeds present: {seeds}; source files: {len(sorted(IN_DIR.glob('prompt_residual_*.csv')))}")

    # ---- the reproduction gate ------------------------------------------------------------------
    lines.append("\n## 1. Reproduction gate (blocking)")
    lines.append("")
    # Read the gate quantities the driver persisted per row, so this is the evidence the run
    # actually recorded -- not a re-derivation from the rounded cell means.
    vg = df["gate_vec_gap"].astype(float)
    pg = df["gate_prr_gap"].astype(float)
    lines.append(f"`R0` is the canonical mean-pool control computed through the pseudo-sequence "
                 f"path, so the two must agree. Over {len(df['method'].eq(R0).index)} scored rows:")
    lines.append("")
    lines.append("| quantity | max observed | gate | |")
    lines.append("|---|---|---|---|")
    lines.append(f"| per-example uncertainty gap | **{vg.max():.3e}** | abort above 1e-5 | "
                 f"{'PASS' if vg.max() <= 1e-5 else 'FAIL'} |")
    lines.append(f"| PRR gap (rank-discretised) | {pg.max():.3e} | abort above 1e-3 | "
                 f"{'PASS' if pg.max() <= 1e-3 else 'FAIL'} |")
    lines.append("")
    lines.append("⚠️ The gate is on the **per-example uncertainties**, not PRR. PRR is a rank "
                 "statistic and therefore discontinuous in the scores: a measured `1e-7` "
                 "perturbation moves it by up to `1.57e-04`, and a single adjacent-pair swap by "
                 "`4.58e-06`. Gating a rank metric at `1e-6` tests 'no tie flipped', not 'same "
                 "computation'. Both quantities are reported so the substitution can be judged "
                 "rather than taken on trust — see prereg §9.")

    # ---- primary --------------------------------------------------------------------------------
    lines.append("\n## 2. PRIMARY — `R2 − R1` over the 4 OOD rungs")
    lines.append("")
    lines.append("Residual vs generated-only: the comparison that isolates *subtracting* the anchor "
                 "from *dropping* it.")
    d_primary = {e: (None if scope_mean(g, R2, e, OOD_RUNGS) is None
                     else scope_mean(g, R2, e, OOD_RUNGS) - scope_mean(g, R1, e, OOD_RUNGS))
                 for e in EVALS}
    package(d_primary, "`R2 − R1`, mean over the 4 OOD rungs", lines, gate=True)

    # ---- THE MECHANISM SEPARATOR ----------------------------------------------------------------
    lines.append("\n## 2b. MECHANISM — `R2 − R3` (per-example anchor vs constant anchor)")
    lines.append("")
    lines.append("`R3` subtracts the **training-set mean** `h0`, so it buys the *centring* that `R2` "
                 "also gets, without any task-relativity. This is the comparison that separates the "
                 "claimed mechanism from an optimisation artifact (prereg §8):")
    lines.append("")
    lines.append("- `R2 ≈ R3` → the gain is **centring/conditioning**, not prompt-relativity. "
                 "The mechanism claim fails even if the primary gate passes.")
    lines.append("- `R2 ≫ R3` → the **per-example** anchor carries real signal.")
    d_mech = {e: (None if scope_mean(g, R2, e, OOD_RUNGS) is None or scope_mean(g, R3, e, OOD_RUNGS) is None
                  else scope_mean(g, R2, e, OOD_RUNGS) - scope_mean(g, R3, e, OOD_RUNGS))
              for e in EVALS}
    package(d_mech, "`R2 − R3`, mean over the 4 OOD rungs", lines)

    # ---- decompositions -------------------------------------------------------------------------
    lines.append("\n## 3. Decomposition — where any movement comes from")
    for lab, (a, b) in {"`R2 − R0` (residual vs anchor-inclusive)": (R2, R0),
                        "`R1 − R0` (dropping the anchor alone)": (R1, R0),
                        "`R3 − R1` (centring alone, no task-relativity)": (R3, R1)}.items():
        dd = {e: (None if scope_mean(g, a, e, OOD_RUNGS) is None
                  else scope_mean(g, a, e, OOD_RUNGS) - scope_mean(g, b, e, OOD_RUNGS))
              for e in EVALS}
        package(dd, lab, lines)

    # ---- per-rung, and the predeclared directional axis ------------------------------------------
    lines.append("\n## 4. Per-rung breakdown, and the prereg §5 directional axis")
    lines.append("")
    lines.append("| rung | mean `R2 − R1` | mean `R2 − R3` | mean `R2 − R0` | mean `R1 − R0` | datasets |")
    lines.append("|---|---|---|---|---|---|")
    per_rung = {}
    for r in ALL_RUNGS:
        row = []
        for a, b in ((R2, R1), (R2, R3), (R2, R0), (R1, R0)):
            vals = [cell(g, a, e, r) - cell(g, b, e, r) for e in EVALS
                    if cell(g, a, e, r) is not None and cell(g, b, e, r) is not None]
            row.append(float(np.mean(vals)) if vals else None)
        n = sum(1 for e in EVALS if cell(g, R2, e, r) is not None)
        per_rung[r] = row[0]
        lines.append(f"| `{r}` | " + " | ".join(f"{x:+.4f}" if x is not None else "—" for x in row)
                     + f" | {n}/8 |")

    hard = [per_rung[r] for r in HARD_RUNGS if per_rung.get(r) is not None]
    easy = [per_rung[r] for r in EASY_RUNGS if per_rung.get(r) is not None]
    idv = per_rung.get("ID")
    lines.append("")
    if hard and easy and idv is not None:
        h, ez = float(np.mean(hard)), float(np.mean(easy))
        lines.append(f"Predicted ordering (prereg §5, stated before results): "
                     f"`hard OOD > easy OOD > ID`, with ID permitted to be negative.")
        lines.append("")
        lines.append(f"- hard OOD (DiffTask, 1ds-Diff): **{h:+.4f}**")
        lines.append(f"- easy OOD (SameTask, LOO): **{ez:+.4f}**")
        lines.append(f"- ID: **{idv:+.4f}**")
        ok = h > ez > idv
        lines.append("")
        lines.append(f"**Directional axis: {'SUPPORTED' if ok else 'NOT SUPPORTED'}.**")
        if not ok and h > idv:
            lines.append("(hard OOD does exceed ID, but the full ordering does not hold — "
                         "report as partial.)")
        lines.append("")
        lines.append("⚠️ prereg §5 also fixed the converse in advance: if `R2` improves ID but not "
                     "the hard OOD rungs, the mechanism claim is **refuted** even under a positive "
                     "macro mean.")

    # ---- external bar ---------------------------------------------------------------------------
    lines.append("\n## 5. Against the external bar")
    lines.append("")
    lines.append("| dataset | R0 OOD | R1 OOD | R2 OOD | R3 OOD | msp_min | perplexity |")
    lines.append("|---|---|---|---|---|---|---|")
    for e in EVALS:
        vals = [scope_mean(g, m, e, OOD_RUNGS) for m in ALL_ARMS]
        fl = [cell(g, f, e, "LOO-long") for f in ("floor_min", "floor_ppl")]
        lines.append(f"| `{e}` | " + " | ".join(f"{v:+.4f}" if v is not None else "—"
                                                for v in vals + fl) + " |")
    lines.append("")
    lines.append("⚠️ The floors are training-free, so they are rung-invariant by construction and "
                 "are entered once per dataset (from `LOO-long`), not once per rung. `SAPLMA` is "
                 "the standing OOD bar and is read from the canonical master, not refitted here.")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
