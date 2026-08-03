#!/usr/bin/env python
"""STEP 4 -- assemble the CANONICAL ProbeDriftLong master table from every widened-`cells_long` source CSV.

Joins on (rung, eval, canonical_method). Handles the heterogeneous schemas (base/arms/mh/ensemble = long-format
`prr_mean`, 3-seed; 3A = `prr`, seed-1; router = WIDE, seed-1) by melting each to a common long form, mapping raw
method names to canonical rows, and tagging seed_regime + source + population. Duplicated cells (e.g. `floor_min`
appears in the base AND ensemble CSVs; `attention`==`armA` in base AND fixed_prior) are DEDUPED by source
priority, and the overlap is a CROSS-CHECK: if two 3-seed sources disagree by >0.02 on a shared cell it is flagged
LOUD (a population/seed mismatch), never silently averaged.

Outputs:
  results/pdl_master__<SLUG>.csv         long-format ground truth (rung,eval,method,prr,n_seeds,seed_regime,source)
  results/pdl_master__<SLUG>.md          rendered per-rung tables + the cross-dataset aggregate
Coverage is reported up front; genuinely-absent cells stay BLANK (never zero-filled). Off-population rows
(soft-Orgad restricted, the broad-LOO router) are NOT joined here -- they live in their own captioned tables.
"""
import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
EPHEM = Path("/rds/general/ephemeral/user/gs925/ephemeral/luq_overnight_results")
SLUG = "meta-llama_Meta-Llama-3.1-8B"

LONG_EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
OOD_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]

# canonical row order (free floors first, then supervised)
ORDER = ["msp_sum", "perplexity", "msp_min",
         "wMSP-norm", "wMSP-shrink@2", "wMSP-shrink@10",
         # The supervised baselines. ⚠️ A method absent from ORDER is not rendered even when ALIAS knows
         # it, so BOTH lists have to carry a new method -- adding it to only one is a silent half-fix.
         "SAPLMA", "linear probe", "P(True)", "Lookback Lens",
         "armB(mean-pool)", "armA(attention)",
         "armC:content-mass", "armC:NLL", "armD:content-mass", "armD:NLL",
         "multi-head(MH)", "multi-head-ablation(ABL)",
         "MultiMax", "Max-of-Rolling-Means(w10)",
         "score-std:V2(entropy-solve)", "score-std:LayerNorm-ctrl",
         "ens{MSP,SAPLMA}", "ens{wMSP,SAPLMA}", "ens{wMSP,MSP}",
         "ens-z{MSP,SAPLMA}", "ens-z{wMSP,SAPLMA}", "ens-z{wMSP,MSP}",
         "length-router", "length-blend"]
FLOORS = {"msp_sum", "perplexity", "msp_min"}

# raw method value -> canonical (None = intentionally dropped)
ALIAS = {
    "floor_sum": "msp_sum", "floor_ppl": "perplexity", "floor_min": "msp_min", "fair_floor": None,
    "wmsp_norm": "wMSP-norm", "wmsp_shrink2": "wMSP-shrink@2", "wmsp_shrink10": "wMSP-shrink@10",
    "wmsp_blondel": None, "wmsp_seg_flat": None, "wmsp_seg_softmax": None,
    "wmsp_shrink2_blondel": None, "wmsp_shrink10_blondel": None, "wmsp": "wMSP-norm",
    # ⭐ THE SUPERVISED BASELINES. Added 2026-08-03: probedriftlong.py has supported `--baselines
    # ptrue,lookback` for weeks (BASE_FEATS at probedriftlong.py:84) and writes the BARE KEY as the
    # method string — but ALIAS did not carry them, and `ALIAS.get(m, "__skip__")` SILENTLY DROPS an
    # unknown method. So the report's central claim ("our method beats existing probes") had no existing
    # probes in its table, and nothing anywhere said so. `linear` was missing for the same reason.
    "ptrue": "P(True)", "ptrue_accurate": "P(True)",
    "lookback": "Lookback Lens",
    "linear": "linear probe",
    "saplma": "SAPLMA", "uniform": "armB(mean-pool)", "armB": "armB(mean-pool)",
    "attention": "armA(attention)", "armA": "armA(attention)",
    "armC_content_mass": "armC:content-mass", "armC_nll": "armC:NLL",
    "armD_content_mass": "armD:content-mass", "armD_nll": "armD:NLL",
    "mh": "multi-head(MH)", "ablation": "multi-head-ablation(ABL)",
    "multimax": "MultiMax", "rolling_w10": "Max-of-Rolling-Means(w10)",
    "zstd_V2": "score-std:V2(entropy-solve)", "layernorm_ctrl": "score-std:LayerNorm-ctrl",
    "armA_s1": None, "rolling_wT": None, "zstd_V1": None,   # 3A internal reference / gate artifacts
    "rankavg_floor_min+saplma": "ens{MSP,SAPLMA}", "rankavg_wmsp+saplma": "ens{wMSP,SAPLMA}",
    "rankavg_wmsp+floor_min": "ens{wMSP,MSP}",
    # The z-average ensemble flavour. Found 2026-08-03 by the new loud-drop report: these were COMPUTED
    # in the same runs as the rank-average ones and silently discarded at assembly, so the ensemble
    # comparison only ever showed one of the two combination rules. Rendered distinctly (z vs rank) --
    # collapsing them onto one label would make two different methods look like one.
    "zavg_floor_min+saplma": "ens-z{MSP,SAPLMA}", "zavg_wmsp+saplma": "ens-z{wMSP,SAPLMA}",
    "zavg_wmsp+floor_min": "ens-z{wMSP,MSP}",
}

# (glob, priority, seed_regime). Lower priority number wins on a shared cell. Base is authoritative for
# floors/poolers/wMSP/SAPLMA; fixed_prior/mh add arm rows; ensemble adds ensemble rows; 3A/router add seed-1 rows.
SOURCES = [
    ("probedriftlong_*_widened_wmsp__" + SLUG + ".csv", 0, "3seed", "prr_mean"),
    ("fixed_prior_ladder__" + SLUG + ".csv", 1, "3seed", "prr_mean"),
    ("fixed_prior_ladder_factscore__" + SLUG + ".csv", 1, "3seed", "prr_mean"),
    ("fixed_prior_fill_*__" + SLUG + ".csv", 1, "3seed", "prr_mean"),          # all-5-rungs fills (this phase)
    ("multihead_ladder__" + SLUG + "_joined.csv", 1, "3seed", "prr_mean"),
    ("multihead_fill_*__" + SLUG + ".csv", 1, "3seed", "prr_mean"),            # all-5-rungs MH fills (this phase)
    ("ensemble_wmsp_saplma_full__" + SLUG + ".csv", 2, "3seed", "prr_mean"),   # 3E: all 8 evals + {wMSP,MSP}
    ("ensemble_wmsp_saplma__" + SLUG + ".csv", 2, "3seed", "prr_mean"),        # 6-eval predecessor (overlaps agree)
    ("aggregation_variants_3A__" + SLUG + ".csv", 3, "seed1", "prr"),
]
ROUTER_GLOB = "router_pdl__" + SLUG + ".csv"


def find(glob_pat):
    # UNION both dirs (was: first-dir-wins, which shadowed the 4-core widened+wMSP CSVs living in EPHEM behind
    # the 4 DoC base CSVs in RESULTS -> wMSP-shrink was 4/8). Dedupe by basename; RESULTS wins a name conflict.
    seen = {}
    for d in (RESULTS, EPHEM):
        for p in sorted(glob.glob(str(d / glob_pat))):
            b = Path(p).name
            if b not in seen:
                seen[b] = p
    return list(seen.values())


# Methods deliberately not rendered (internal references / gate artifacts), vs methods we have simply
# never heard of. The two must not be treated the same: the first is a decision, the second is a bug.
DROPPED_UNKNOWN = {}


def read_long(path, valcol):
    with open(path) as fh:
        for row in csv.DictReader(fh):
            m = (row.get("method") or "").strip()
            if not m or m.startswith("VERDICT:") or m == "method":
                continue
            if m not in ALIAS:
                # ⚠️ LOUD, not silent. A computed method that vanishes at assembly is invisible: the CSV
                # says it ran, the table says nothing, and nobody can tell the difference between "not
                # measured" and "measured then dropped". This is how ptrue/lookback/linear went missing.
                DROPPED_UNKNOWN.setdefault(m, set()).add(Path(path).name)
                continue
            canon = ALIAS[m]
            if canon is None:            # explicitly suppressed (see the ALIAS None entries)
                continue
            ev = (row.get("eval") or "").strip(); rg = (row.get("rung") or "").strip()
            v = row.get(valcol, "")
            if ev not in LONG_EVALS or rg not in RUNGS or v in ("", None):
                continue
            ns = row.get("n_seeds", "")
            yield rg, ev, canon, float(v), ns


def read_router(path):
    with open(path) as fh:
        for row in csv.DictReader(fh):
            ev = row["eval"].strip(); rg = row["rung"].strip()
            if ev not in LONG_EVALS or rg not in RUNGS:
                continue
            for col, canon in [("router2_floor_pool", "length-router"), ("blend", "length-blend")]:
                if row.get(col) not in ("", None):
                    yield rg, ev, canon, float(row[col]), "seed1(router)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / f"pdl_master__{SLUG}"))
    args = ap.parse_args()

    # NB: SLUG contains dots (Llama-3.1-8B) -> never use Path.with_suffix (it would eat ".1-8B"); append strings.
    csv_path = Path(str(args.out) + ".csv"); md_path = Path(str(args.out) + ".md")
    grid = {}          # (rung,eval,method) -> (prr, priority, seed_regime, source, n_seeds)
    conflicts = []
    for glob_pat, prio, seed, valcol in SOURCES:
        for path in find(glob_pat):
            src = Path(path).name
            for rg, ev, canon, v, ns in read_long(path, valcol):
                key = (rg, ev, canon)
                if key in grid:
                    pv, pp, ps, psrc, pns = grid[key]
                    if pp == prio and ps == seed and abs(pv - v) > 0.02:
                        conflicts.append(f"{rg}/{ev}/{canon}: {psrc} {pv:+.3f} vs {src} {v:+.3f} (Δ{v-pv:+.3f})")
                    if prio < pp:            # higher-priority (lower number) source wins
                        grid[key] = (v, prio, seed, src, ns)
                else:
                    grid[key] = (v, prio, seed, src, ns)
    for path in find(ROUTER_GLOB):
        for rg, ev, canon, v, seed in read_router(path):
            grid[(rg, ev, canon)] = (v, 3, "seed1", Path(path).name, "1")

    # ---- master long CSV ----
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rung", "eval", "method", "prr", "n_seeds", "seed_regime", "source"])
        for (rg, ev, m), (v, _p, seed, src, ns) in sorted(grid.items()):
            w.writerow([rg, ev, m, f"{v:.4f}", ns, seed, src])

    # ---- coverage report ----
    present = {(rg, ev, m) for (rg, ev, m) in grid}
    lines = []
    lines.append(f"# ProbeDriftLong MASTER table  (population: widened cells_long; SLUG={SLUG})")
    total = len(ORDER) * len(LONG_EVALS) * len(RUNGS)
    have = sum(1 for m in ORDER for ev in LONG_EVALS for rg in RUNGS if (rg, ev, m) in present)
    lines.append(f"\nCoverage: {have}/{total} target cells present "
                 f"({len(ORDER)} methods x {len(LONG_EVALS)} evals x {len(RUNGS)} rungs).")
    miss_ev = [e for e in LONG_EVALS if not any((rg, e, "msp_min") in present for rg in RUNGS)]
    if miss_ev:
        lines.append(f"⚠️ evals with NO base cells yet (DoC JOB 1 pending): {', '.join(miss_ev)}")
    if DROPPED_UNKNOWN:
        # Surfaced in the RENDERED table, not just on stdout: a run that computed a method the assembler
        # does not know about has silently lost it, and the table must say so where it will be read.
        lines.append("\n⚠️ **METHODS COMPUTED BUT NOT IN ALIAS — DROPPED FROM THIS TABLE.** They ran and "
                     "are in the source CSVs; add them to ALIAS to render them:")
        lines += [f"   `{m}`  (in {', '.join(sorted(srcs))})" for m, srcs in sorted(DROPPED_UNKNOWN.items())]
    if conflicts:
        lines.append("\n⚠️ CROSS-CHECK CONFLICTS (same-priority sources disagree >0.02 on a shared cell):")
        lines += [f"   {c}" for c in conflicts]
    else:
        lines.append("\nCross-check: no same-priority source disagreed >0.02 on any shared cell (armA/floor overlaps agree).")

    def fmt(v):
        return f"{v:+.3f}" if v is not None else "  ·  "

    # ---- per-rung tables (bold best per eval column) ----
    for rg in RUNGS:
        lines.append(f"\n## rung = {rg}   (population: widened cells_long)")
        header = "| method | " + " | ".join(LONG_EVALS) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(LONG_EVALS) + 1))
        # best per column -- over 3-SEED rows ONLY (bolding a seed-1 post-hoc read against 3-seed rows would be
        # a cross-seed-regime comparison; the seed-1 aggregation rows are supplementary, marked with a dagger).
        best = {}
        for ev in LONG_EVALS:
            vals = [(m, grid[(rg, ev, m)][0]) for m in ORDER
                    if (rg, ev, m) in grid and grid[(rg, ev, m)][2] == "3seed"]
            if vals:
                best[ev] = max(vals, key=lambda t: t[1])[0]
        prev_floor = True
        for m in ORDER:
            if prev_floor and m not in FLOORS:
                lines.append("| *— supervised —* |" + " |" * len(LONG_EVALS))
                prev_floor = False
            cells = []
            for ev in LONG_EVALS:
                if (rg, ev, m) in grid:
                    v, _p, seed, _s, _n = grid[(rg, ev, m)]
                    s = fmt(v)
                    if best.get(ev) == m:
                        s = f"**{s}**"
                    if seed.startswith("seed1"):
                        s += "†"
                    cells.append(s)
                else:
                    cells.append(" · ")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("\n† = seed-1 post-hoc read (3A/router); all other rows are 3-seed. Blank = not measured.")

    # ---- cross-dataset aggregate (THE OBJECTIVE): OOD rungs, on a COMMON cell set to avoid cross-population means ----
    def cell_val(rg, ev, m):
        return grid[(rg, ev, m)][0] if (rg, ev, m) in grid else None
    # COMMON = OOD cells where the three ANCHORS (msp_min, SAPLMA, armA) are ALL present -- so every 3-seed method's
    # mean is over the SAME cells (a fair ranking). Methods missing some COMMON cells are reported with coverage.
    anchors = ["msp_min", "SAPLMA", "armA(attention)"]
    COMMON = [(ev, rg) for ev in LONG_EVALS for rg in OOD_RUNGS
              if all((rg, ev, a) in grid for a in anchors)]
    rng = np.random.RandomState(0)
    lines.append("\n## CROSS-DATASET AGGREGATE over the long evals (OOD rungs) — the project objective")
    lines.append(f"Computed on a COMMON cell set of **{len(COMMON)}** OOD cells where msp_min+SAPLMA+armA are all "
                 "present, so the means ARE comparable (no cross-population). Methods covering <all COMMON cells "
                 "show coverage; win-counts are paired per cell. Caption: widened cells_long, OOD rungs, 3-seed.")
    if any(not any((rg, e, "msp_min") in present for rg in RUNGS) for e in LONG_EVALS):
        lines.append("⚠️ PRELIMINARY — the grid is incomplete (DoC JOB 1 base fills pending); COMMON will grow when they land.")
    lines.append("\n| method (3-seed) | cover | mean PRR (COMMON) | CI (dataset boot) | > msp_min | > SAPLMA |")
    lines.append("|---|---|---|---|---|---|")

    def agg_row(m, cellset):
        cov = [(ev, rg) for ev, rg in cellset if (rg, ev, m) in grid]
        if not cov:
            return None
        vals = np.array([cell_val(rg, ev, m) for ev, rg in cov])
        wmin = sum(1 for ev, rg in cov if cell_val(rg, ev, m) > (cell_val(rg, ev, "msp_min") or -9))
        wsap = sum(1 for ev, rg in cov if cell_val(rg, ev, "SAPLMA") is not None
                   and cell_val(rg, ev, m) > cell_val(rg, ev, "SAPLMA"))
        nsap = sum(1 for ev, rg in cov if cell_val(rg, ev, "SAPLMA") is not None)
        evs = sorted(set(ev for ev, _ in cov))
        by_ev = {e: [cell_val(rg, ev, m) for ev, rg in cov if ev == e] for e in evs}
        boot = [np.mean([np.mean(by_ev[e]) for e in rng.choice(evs, len(evs), replace=True)]) for _ in range(2000)]
        return (len(cov), vals.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), wmin, len(cov), wsap, nsap)

    seed1_methods = {m for (_r, _e, m), (_v, _p, s, _s, _n) in grid.items() if s.startswith("seed1")}
    core = [(m, agg_row(m, COMMON)) for m in ORDER if m not in seed1_methods]
    core = [(m, r) for m, r in core if r is not None]
    core.sort(key=lambda t: -t[1][1])
    for m, (nc, mean, lo, hi, wmin, nmin, wsap, nsap) in core:
        cov = f"{nc}/{len(COMMON)}" + ("" if nc == len(COMMON) else " ⚠")
        lines.append(f"| {m} | {cov} | {mean:+.3f} | [{lo:+.3f},{hi:+.3f}] | {wmin}/{nmin} | {wsap}/{nsap} |")

    # seed-1 supplementary methods: their OWN cell set, explicitly NOT comparable to the block above
    lines.append("\n**Seed-1 supplementary methods** (post-hoc reads / router; a DIFFERENT, smaller cell set and "
                 "seed regime — do NOT rank against the 3-seed block above):")
    lines.append("\n| method (seed-1) | n cells | mean PRR (own cells) | > msp_min | > SAPLMA |")
    lines.append("|---|---|---|---|---|")
    for m in ORDER:
        if m not in seed1_methods:
            continue
        own = [(ev, rg) for ev in LONG_EVALS for rg in OOD_RUNGS if (rg, ev, m) in grid]
        if not own:
            continue
        vals = np.array([cell_val(rg, ev, m) for ev, rg in own])
        wmin = sum(1 for ev, rg in own if cell_val(rg, ev, "msp_min") is not None
                   and cell_val(rg, ev, m) > cell_val(rg, ev, "msp_min"))
        nmin = sum(1 for ev, rg in own if cell_val(rg, ev, "msp_min") is not None)
        wsap = sum(1 for ev, rg in own if cell_val(rg, ev, "SAPLMA") is not None
                   and cell_val(rg, ev, m) > cell_val(rg, ev, "SAPLMA"))
        nsap = sum(1 for ev, rg in own if cell_val(rg, ev, "SAPLMA") is not None)
        lines.append(f"| {m} | {len(own)} | {vals.mean():+.3f} | {wmin}/{nmin} | {wsap}/{nsap} |")

    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {csv_path} and {md_path}")
    if conflicts:
        print(f"\n⚠️ {len(conflicts)} cross-check conflicts (see above) — resolve before trusting the table.")


if __name__ == "__main__":
    main()
