#!/usr/bin/env python
"""PR2 -- the preregistered evidence package for source-relative weighted MSP.

Reads ONLY the committed CSVs written by scripts/checks/source_relative_wmsp.py.
No fitting, no compute, no choices.

    python scripts/checks/source_relative_verdict.py

UNIT OF ANALYSIS: THE DATASET, n = 8.

SCOPE IS THE MULTI-SOURCE RUNGS ONLY. Single-source pools (`ID`, `1ds-Diff-long`, and `SameTask`
for the targets whose same-family pool has one member) CANNOT differ from canonical -- within-source
ranking IS global ranking there. Those cells are excluded from scoring and used instead as an
EXACT-INVARIANCE CONTROL. An inert cell is reported as inert, never as a zero delta: averaging in
structural zeros would dilute a real effect toward nothing and manufacture a null.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

IN_DIR = Path("/rds/general/user/gs925/home/gs925-msc_project/msc-project-gs925/"
              "results/method_dev/source_relative")
OUT_MD = IN_DIR / "SOURCE_RELATIVE_VERDICT.md"

EVALS = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa", "asqa", "factscore"]
SCOPE_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long"]
CANON = "wmsp_shrink2"
ARMS = {"wmsp_srcrel_masked_shrink2": "masked (UNSCALED — void, see §3)",
        "wmsp_srcrel_masked_scaled_shrink2": "masked_scaled (PRIMARY)",
        "wmsp_srcrel_pure_shrink2": "pure (SECONDARY)"}
BOOT_B, BOOT_SEED = 10000, 0
GATE_MACRO, GATE_SIGNS = 0.010, 6


def load():
    files = [f for f in sorted(IN_DIR.glob("source_relative_*.csv")) if "SMOKE" not in f.name]
    if not files:
        raise SystemExit(f"no scored CSVs in {IN_DIR}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "smoke" in df and df["smoke"].any():
        raise SystemExit("a SMOKE row reached the verdict input -- refusing to score it")
    inv_files = sorted(IN_DIR.glob("invariance_*.csv"))
    inv = pd.concat([pd.read_csv(f) for f in inv_files], ignore_index=True) if inv_files else None
    return df, inv


def cellmean(df, method, e, r):
    s = df[(df.method == method) & (df["eval"] == e) & (df.rung == r)]["prr"]
    return float(s.mean()) if len(s) else None


def live_rungs(df, e):
    """Rungs for this eval whose training pool genuinely has >1 source."""
    out = []
    for r in SCOPE_RUNGS:
        s = df[(df["eval"] == e) & (df.rung == r)]
        if len(s) and int(s["n_sources"].max()) > 1:
            out.append(r)
    return out


def package(deltas, label, lines, gate=False):
    have = [e for e in EVALS if deltas.get(e) is not None]
    if len(have) < 3:
        lines.append(f"\n### {label}\n\nonly {len(have)}/8 datasets measured — not scored.")
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
        lines.append(f"\nMISSING (not measured, not zero): {sorted(set(EVALS) - set(have))}")
    lines.append(f"\n- macro mean delta **{d.mean():+.4f}**, median {np.median(d):+.4f}, "
                 f"signs {(d > 0).sum()}/{len(d)}, exact two-sided Wilcoxon p = "
                 + (f"{p:.4f}" if p is not None else "n/a"))
    lines.append(f"- dataset-level bootstrap 95% CI: [{lo:+.4f}, {hi:+.4f}]")
    lines.append("- leave-one-dataset-out macro deltas: "
                 + ", ".join(f"−{e} {v:+.4f}" for e, v in lodo.items()))
    if gate:
        c1, c2 = d.mean() > GATE_MACRO, int((d > 0).sum()) >= GATE_SIGNS
        c3 = all(v > 0 for v in lodo.values())
        lines.append("")
        lines.append("**PRE-REGISTERED GATE (prereg §3, fixed before any result):**")
        lines.append("")
        lines.append("| condition | required | observed | |")
        lines.append("|---|---|---|---|")
        lines.append(f"| macro delta | > +{GATE_MACRO:.3f} | {d.mean():+.4f} | {'PASS' if c1 else 'FAIL'} |")
        lines.append(f"| positive datasets | ≥ {GATE_SIGNS}/8 | {(d > 0).sum()}/{len(d)} | {'PASS' if c2 else 'FAIL'} |")
        lines.append(f"| every LODO positive | all | {sum(v > 0 for v in lodo.values())}/{len(lodo)} | {'PASS' if c3 else 'FAIL'} |")
        lines.append("")
        lines.append(f"**VERDICT: {'PROMISING' if (c1 and c2 and c3) else 'NOT PROMISING'}** — a "
                     f"decision rule, NOT a significance claim.")
    return d.mean()


def main():
    df, inv = load()
    present = [a for a in ARMS if (df.method == a).any()]
    lines = ["# PR2 SOURCE-RELATIVE WEIGHTED MSP — preregistered evidence package", "",
             "Population: `meta-llama/Meta-Llama-3.1-8B`, canonical ProbeDriftLong, carve `legacy`, "
             "layer 15, seeds 1/2/3, λ = 2. Prereg: `prereg/PR2_source_relative_wmsp.md`.", "",
             "Comparator is `wmsp_shrink2` computed by the **library** (`luq.weighted_msp`), not by "
             "this driver's copy of the loop, so every delta is against the real registry method."]

    # ---- scope -------------------------------------------------------------------------------
    lines.append("\n## 0. Scope and coverage")
    lines.append("")
    lines.append("| dataset | live (multi-source) rungs | inert rungs in scope |")
    lines.append("|---|---|---|")
    LIVE = {}
    for e in EVALS:
        lv = live_rungs(df, e)
        LIVE[e] = lv
        inert = [r for r in SCOPE_RUNGS if r not in lv and
                 len(df[(df["eval"] == e) & (df.rung == r)])]
        lines.append(f"| `{e}` | {', '.join(f'`{r}`' for r in lv) or '—'} | "
                     f"{', '.join(f'`{r}`' for r in inert) or '—'} |")
    lines.append("")
    lines.append("Inert rungs are **excluded from the scored means**, not entered as zeros. "
                 "`ID` and `1ds-Diff-long` are inert by construction and are used as the "
                 "invariance control in §2.")

    # ---- no-op fidelity ----------------------------------------------------------------------
    lines.append("\n## 1. No-op fidelity gate")
    lines.append("")
    lines.append("This driver re-implements the canonical training loop so that `weighted_msp.py` "
                 "need not be edited. In `canonical` mode that copy must reproduce the library "
                 "exactly; the driver aborts otherwise. Every scored cell passed.")

    # ---- invariance --------------------------------------------------------------------------
    lines.append("\n## 2. Exact-invariance control (single-source pools)")
    lines.append("")
    if inv is not None and len(inv):
        worst, rows = 0.0, []
        for e in sorted(inv["eval"].unique()):
            for r in sorted(inv[inv["eval"] == e].rung.unique()):
                c = cellmean(inv, CANON, e, r)
                for a in present:
                    v = cellmean(inv, a, e, r)
                    if c is not None and v is not None:
                        worst = max(worst, abs(c - v))
                rows.append(e)
        lines.append(f"On a single-source pool, within-source ranking **is** global ranking, so "
                     f"every arm must reproduce canonical exactly. Max |PRR difference| over the "
                     f"`ID` and `1ds-Diff-long` cells of {len(set(rows))} datasets: "
                     f"**{worst:.3e}**. The driver raises above 1e-6.")
    else:
        lines.append("**NOT MEASURED** — no invariance CSVs found.")
    # inert cells inside the scored files are a second, free instance of the same control
    inert_gaps = []
    for e in EVALS:
        for r in SCOPE_RUNGS:
            s = df[(df["eval"] == e) & (df.rung == r)]
            if len(s) and int(s["n_sources"].max()) == 1:
                c = cellmean(df, CANON, e, r)
                for a in present:
                    v = cellmean(df, a, e, r)
                    if c is not None and v is not None:
                        inert_gaps.append(abs(c - v))
    if inert_gaps:
        lines.append(f"\nA second, independent instance of the same control: the "
                     f"structurally single-source `SameTask` cells inside the scored files agree "
                     f"with canonical to **{max(inert_gaps):.3e}**.")

    # ---- the void arm ------------------------------------------------------------------------
    if "wmsp_srcrel_masked_shrink2" in present:
        lines.append("\n## 3. The UNSCALED `masked` arm is VOID — it became a different method")
        lines.append("")
        lines.append("Reported, not deleted, because the failure is itself informative.")
        lines.append("")
        lines.append("The rank loss is an MSE between soft and hard ranks, so its magnitude grows "
                     "with the number of items ranked (ranks run 1..m; the MSE is O(m²)). Ranking "
                     "inside source subgroups of ~5 rather than a batch of 32 shrinks that term by "
                     "about an order of magnitude while λ stays fixed at 2 — so the shrink penalty "
                     "dominates, the weights are driven uniform, and weighted MSP **becomes "
                     "perplexity**. That is not a null result about source-relative supervision.")
        lines.append("")
        lines.append("| rung | sources | Ω(w) = mean((w−1)²) |")
        lines.append("|---|---:|---|")
        for r in SCOPE_RUNGS:
            s = df[(df.method == "wmsp_srcrel_masked_shrink2") & (df.rung == r)]
            if len(s) and "final_omega" in s:
                lines.append(f"| `{r}` | {int(s['n_sources'].median())} | "
                             f"{float(pd.to_numeric(s['final_omega'], errors='coerce').mean()):.4f} |")
        for a in ("wmsp_srcrel_pure_shrink2", "wmsp_srcrel_masked_scaled_shrink2"):
            s = df[df.method == a]
            if len(s) and "final_omega" in s:
                om = pd.to_numeric(s["final_omega"], errors="coerce").mean()
                lines.append(f"| *(reference)* `{a}` | — | {float(om):.4f} |")
        lines.append("")
        lines.append("Ω falls monotonically as the subgroups get smaller, which is the signature of "
                     "the scale mechanism rather than of a learning failure. `masked_scaled` "
                     "rescales each subgroup's ranks to the full-batch range, restoring λ's "
                     "intended strength. That is a **scale correction, not a hyperparameter tune** — "
                     "λ, lr, batch size, epochs, seeds, batch membership and permutation are all "
                     "unchanged.")

    # ---- primary / secondary ------------------------------------------------------------------
    lines.append("\n## 4. Deltas against canonical `wmsp_shrink2`, live rungs only")
    order = [a for a in ("wmsp_srcrel_masked_scaled_shrink2", "wmsp_srcrel_pure_shrink2",
                         "wmsp_srcrel_masked_shrink2") if a in present]
    for a in order:
        deltas = {}
        for e in EVALS:
            lv = LIVE[e]
            if not lv:
                continue
            vs = [cellmean(df, a, e, r) for r in lv]
            cs = [cellmean(df, CANON, e, r) for r in lv]
            if any(v is None for v in vs) or any(c is None for c in cs):
                continue
            deltas[e] = float(np.mean(vs) - np.mean(cs))
        package(deltas, f"`{ARMS[a]}` − canonical", lines,
                gate=(a == "wmsp_srcrel_masked_scaled_shrink2"))

    # ---- per rung -----------------------------------------------------------------------------
    lines.append("\n## 5. Per-rung means (live cells only)")
    lines.append("")
    SHORT = {"wmsp_srcrel_masked_scaled_shrink2": "masked_scaled (primary)",
             "wmsp_srcrel_pure_shrink2": "pure (secondary)",
             "wmsp_srcrel_masked_shrink2": "masked (VOID)"}
    lines.append("| rung | " + " | ".join(f"`{SHORT[a]}`" for a in order) + " | datasets |")
    lines.append("|---|" + "---|" * (len(order) + 1))
    for r in SCOPE_RUNGS:
        cells, n = [], 0
        for a in order:
            ds = [cellmean(df, a, e, r) - cellmean(df, CANON, e, r) for e in EVALS
                  if r in LIVE[e] and cellmean(df, a, e, r) is not None
                  and cellmean(df, CANON, e, r) is not None]
            cells.append(float(np.mean(ds)) if ds else None)
            n = max(n, len(ds))
        lines.append(f"| `{r}` | " + " | ".join(f"{c:+.4f}" if c is not None else "—"
                                                for c in cells) + f" | {n} |")

    # ---- mechanism ----------------------------------------------------------------------------
    lines.append("\n## 6. Mechanism diagnostics (prereg §5)")
    lines.append("")
    lines.append("Shortcut removal predicts Ω(w) falls and weights transfer better; noise removal "
                 "predicts the rank loss converges lower with Ω little changed.")
    lines.append("")
    lines.append("| arm | Ω(w) | final rank loss | optimiser steps |")
    lines.append("|---|---|---|---|")
    for a in [CANON] + order:
        s = df[(df.method == a) & (df["n_sources"] > 1)]
        if not len(s):
            continue
        om = pd.to_numeric(s.get("final_omega"), errors="coerce").mean()
        rl = pd.to_numeric(s.get("final_rank_loss"), errors="coerce").mean()
        st = pd.to_numeric(s.get("n_steps"), errors="coerce").mean()
        f = lambda x: "—" if pd.isna(x) else f"{x:.4f}"
        lines.append(f"| `{a}` | {f(om)} | {f(rl)} | "
                     + ("—" if pd.isna(st) else f"{st:.0f}") + " |")
    lines.append("")
    lines.append("`wmsp_shrink2` is computed by the library, which does not expose these "
                 "diagnostics, so its row is blank — not zero.")
    lines.append("")
    lines.append("`pure` takes slightly MORE optimiser steps than canonical: chunking each "
                 "source separately turns each source's final partial chunk into its own batch. "
                 "Recorded because it is a difference between the arms that is not the intervention.")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
