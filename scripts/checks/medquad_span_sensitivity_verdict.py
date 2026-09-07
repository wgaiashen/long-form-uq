#!/usr/bin/env python
"""med_quad clean-span SENSITIVITY verdict — assemble, control, compare, decide (2026-08-11).

Inputs: the 8 shadow per-eval CSVs (results/analysis/pdl_mqclean_<eval>__<slug>.csv, produced by
pbs/pdl_medquad_clean.pbs under LUQ_REGIME=med_quad=med_quad_clean) and the canonical
pdl_master__meta-llama_Meta-Llama-3.1-8B.csv (READ ONLY — never modified).

WHAT IT DOES, in order:
 1. Assembles the shadow grid → results/analysis/llama_medquad_clean_span_sensitivity.csv
    (one row per cell × method, shadow PRR next to canonical PRR and the delta).
 2. THE REPRODUCTION CONTROL: the cells whose train AND eval avoid med_quad (derived from
    cells_long, not hand-listed) must reproduce the canonical master within 4-dp rounding
    (|Δ| ≤ 2e-4). Any violation is listed and the verdict is BLOCKED — a shadow whose untouched
    cells move cannot attribute the moved cells to the span fix.
 3. Before/after on the report-level quantities: med_quad's own rows; macro OOD means per
    method; the per-dataset floor winners; wMSP-shrink@2 − SAPLMA on the two hardest rungs.
 4. Writes a SHADOW master in the canonical schema
    (results/analysis/pdl_master_SHADOW_mqclean.csv) and prints the command to re-run THE
    committed R-claim scorer on it (`replication_claims.py --population llama --csv <shadow>`)
    — the scorer stays the single test definition; nothing is re-derived here.
 5. Prints the decision-rule scaffold (conclusions stand → freeze + limitation paragraph;
    a headline flips → stop for a decision on canonical promotion).

    python scripts/checks/medquad_span_sensitivity_verdict.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_drift_long import cells_long  # noqa: E402  (via the installed library)

SLUG = "meta-llama_Meta-Llama-3.1-8B"
EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
OOD_RUNGS = RUNGS[1:]
ANALYSIS = ROOT / "results" / "analysis"
MASTER = ROOT / "results" / f"pdl_master__{SLUG}.csv"
CTRL_TOL = 2e-4                            # 4-dp rounding on both sides

# driver arm name -> canonical master display name (only methods present in the master compare)
NAME = {"floor_min": "msp_min", "floor_ppl": "perplexity", "floor_sum": "msp_sum",
        "saplma": "SAPLMA", "uniform": "armB(mean-pool)", "attention": "armA(attention)",
        "wmsp_norm": "wMSP-norm", "wmsp_shrink2": "wMSP-shrink@2", "wmsp_shrink10": "wMSP-shrink@10"}


def affected_cells():
    out = set()
    for rung, X, spec in cells_long(set(EVALS), EVALS):
        if rung not in RUNGS:
            continue                        # Long->Short sits outside the 40-cell master grid
        if X == "med_quad" or any(d == "med_quad" for d, _cap in spec):
            out.add((X, rung))
    return out


def load_shadow():
    g = {}
    missing = []
    for ev in EVALS:
        p = ANALYSIS / f"pdl_mqclean_{ev}__{SLUG}.csv"
        if not p.exists():
            missing.append(p.name)
            continue
        for r in csv.DictReader(open(p)):
            if r["method"].startswith("VERDICT:") or r["method"] == "fair_floor":
                continue
            try:
                g[(r["eval"], r["rung"], r["method"])] = float(r["prr_mean"])
            except (TypeError, ValueError):
                pass
    if missing:
        raise SystemExit(f"shadow grid incomplete — missing {missing}; refusing a partial verdict")
    return g


def load_master():
    g = {}
    for r in csv.DictReader(open(MASTER)):
        if r.get("seed_regime") == "3seed":
            g[(r["eval"], r["rung"], r["method"])] = float(r["prr"])
    return g


def ood_mean(g, disp, ev):
    return float(np.mean([g[(ev, rg, disp)] for rg in OOD_RUNGS]))


def main():
    shadow, canon = load_shadow(), load_master()
    touched = affected_cells()
    print(f"affected cells (derived from cells_long): {len(touched)}  "
          f"untouched: {40 - len(touched)}")

    # ---------------- 1. assemble the sensitivity CSV ----------------
    out_csv = ANALYSIS / "llama_medquad_clean_span_sensitivity.csv"
    rows = []
    for (ev, rg, arm), v in sorted(shadow.items()):
        disp = NAME.get(arm)
        cv = canon.get((ev, rg, disp)) if disp else None
        rows.append({"eval": ev, "rung": rg, "method_arm": arm, "method_display": disp or "",
                     "prr_shadow": round(v, 4),
                     "prr_canonical": "" if cv is None else round(cv, 4),
                     "delta": "" if cv is None else round(v - cv, 4),
                     "cell_touches_med_quad": int((ev, rg) in touched)})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")

    # ---------------- 2. reproduction control ----------------
    bad, n_ok = [], 0
    for (ev, rg, arm), v in shadow.items():
        disp = NAME.get(arm)
        if disp is None or (ev, rg) in touched:
            continue
        cv = canon.get((ev, rg, disp))
        if cv is None:
            continue
        if abs(v - cv) > CTRL_TOL:
            bad.append((ev, rg, disp, v, cv, v - cv))
        else:
            n_ok += 1
    print(f"\nREPRODUCTION CONTROL (untouched cells, |Δ| ≤ {CTRL_TOL}): {n_ok} comparisons OK, "
          f"{len(bad)} violations")
    if bad:
        for b in sorted(bad, key=lambda x: -abs(x[5]))[:20]:
            print(f"  VIOLATION {b[0]}/{b[1]}/{b[2]}: shadow {b[3]:+.4f} vs canon {b[4]:+.4f} "
                  f"(Δ {b[5]:+.4f})")
        print("CONTROL FAILED — the shadow moved cells the span fix cannot touch. The "
              "sensitivity is NOT interpretable until this is explained. STOP.")
        sys.exit(1)
    print("control PASS — every moved cell below is attributable to the med_quad span fix.")

    # ---------------- 3. report-level before/after ----------------
    print("\nMED_QUAD'S OWN ROWS (shadow vs canonical):")
    print(f"{'rung':16s}" + "".join(f"{NAME[a][:12]:>14s}" for a in NAME))
    for rg in RUNGS:
        line = f"{rg:16s}"
        for arm, disp in NAME.items():
            v, cv = shadow.get(("med_quad", rg, arm)), canon.get(("med_quad", rg, disp))
            line += (f"{v:+.3f}/{cv:+.3f}" if v is not None and cv is not None else "   —   ").rjust(14)
        print(line)

    print("\nMACRO OOD MEANS (8-eval mean of per-eval OOD means) — shadow vs canonical:")
    print(f"{'method':18s}{'shadow':>10s}{'canonical':>11s}{'delta':>9s}")
    for arm, disp in NAME.items():
        try:
            sv = float(np.mean([np.mean([shadow[(ev, rg, arm)] for rg in OOD_RUNGS])
                                for ev in EVALS]))
            cv = float(np.mean([ood_mean(canon, disp, ev) for ev in EVALS]))
        except KeyError:
            continue
        print(f"{disp:18s}{sv:>+10.4f}{cv:>+11.4f}{sv - cv:>+9.4f}")

    print("\nFLOOR WINNER PER DATASET (OOD; floors are rung-invariant):")
    for ev in EVALS:
        fl_s = {d: shadow[(ev, "LOO-long", a)] for a, d in NAME.items() if a.startswith("floor")}
        fl_c = {d: canon[(ev, "LOO-long", d)] for d in ("msp_min", "perplexity", "msp_sum")}
        ws, wc = max(fl_s, key=fl_s.get), max(fl_c, key=fl_c.get)
        flag = "  <-- WINNER CHANGED" if ws != wc else ""
        print(f"  {ev:15s} shadow {ws:11s} canonical {wc:11s}{flag}")

    print("\nwMSP-shrink@2 − SAPLMA on the two hardest rungs (8-eval mean):")
    for rg in ("DiffTask-long", "1ds-Diff-long"):
        sv = float(np.mean([shadow[(ev, rg, "wmsp_shrink2")] - shadow[(ev, rg, "saplma")]
                            for ev in EVALS]))
        cv = float(np.mean([canon[(ev, rg, "wMSP-shrink@2")] - canon[(ev, rg, "SAPLMA")]
                            for ev in EVALS]))
        print(f"  {rg:15s} shadow {sv:+.4f}   canonical {cv:+.4f}   delta {sv - cv:+.4f}")

    # ---------------- 4. shadow master for THE scorer ----------------
    shadow_master = ANALYSIS / "pdl_master_SHADOW_mqclean.csv"
    with open(shadow_master, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rung", "eval", "method", "prr", "n_seeds", "seed_regime", "source"])
        for r in csv.DictReader(open(MASTER)):
            if r.get("seed_regime") != "3seed":
                continue
            arm = {v: k for k, v in NAME.items()}.get(r["method"])
            key = (r["eval"], r["rung"], arm) if arm else None
            prr = shadow.get(key, float(r["prr"])) if key else float(r["prr"])
            w.writerow([r["rung"], r["eval"], r["method"], f"{prr:.4f}", r["n_seeds"],
                        "3seed", "SHADOW_mqclean" if key and key in shadow else r["source"]])
    print(f"\nwrote {shadow_master} — re-run THE scorer on it (SENSITIVITY label, never a gate):")
    print(f"  python scripts/checks/replication_claims.py --population llama --csv {shadow_master}")

    print("\nDECISION RULE: if the macro rankings, floor ordering, and the "
          "hardest-rung story stand → freeze the master, write the limitation paragraph. If a "
          "headline flips → STOP and refer canonical promotion for a decision.")


if __name__ == "__main__":
    main()
