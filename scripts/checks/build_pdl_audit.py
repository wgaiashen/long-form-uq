#!/usr/bin/env python
"""STEP 2 -- ProbeDriftLong gap-audit matrix (READ-ONLY, no compute, no GPU).

For every (canonical method, long eval, cells_long rung) it records which result CSVs on disk provide that
cell and on WHICH population, then tags the cell:
    OK    present on the canonical WIDENED cells_long population
    OTHER present only on a DIFFERENT population (named: narrow / restricted / broad-LOO)
    MISS  absent everywhere
Emits results/pdl_audit_matrix__<SLUG>.csv + a rendered markdown block to stdout. This is the FIRST
deliverable; report it before running any fill job.

Population is inferred from the FILE FAMILY (+ env_hash where it disambiguates), because no CSV carries a
dedicated pool column, verified against the files:
  - base `probedriftlong_<eval>_widened[_wmsp]__` .......... WIDENED cells_long
  - `fixed_prior_ladder[_factscore]__`, `multihead_ladder__…_joined`, `ensemble_*__` (all import
    probedriftlong as pdl and use the post-Task-A widened `sampled_train_idx`) .... WIDENED cells_long
  - base `probedriftlong_<eval>__` (pre-widened per-dataset files) ................ NARROW
  - `fixed_prior_orgad_restricted__` (env 550aa0ee, 2-way 900+900 pools) ......... RESTRICTED
  - `cache/router_ood/*.npz` (broad leave-D-out over the 9-set universe, 1 rung) .. BROAD-LOO
Only files whose basename matches an explicit allow-regex are read, so wMSP-variant stems
(blondel/seg/universal) and label_* sidecars cannot leak base rows into the audit.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
EPHEM = Path(os.environ.get("EPHEMERAL", str(Path.home() / "ephemeral")) + "/luq_overnight_results")
ROUTER_OOD = ROOT / "cache" / "router_ood"
SLUG = "meta-llama_Meta-Llama-3.1-8B"

# ---- target grid ------------------------------------------------------------------------------------
LONG_EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]

# canonical method row order (target). NEW methods (this task) will show MISS everywhere -- that is the point.
TARGET_METHODS = [
    "msp_sum", "perplexity", "msp_min",                       # free floors
    "wMSP-norm", "wMSP-shrink@2", "wMSP-shrink@10",           # model-side learned
    "SAPLMA", "armB(mean-pool)", "armA(attention)",           # probe-side base
    "armC:content-mass", "armC:NLL", "armC:soft-Orgad",
    "armD:content-mass", "armD:NLL", "armD:soft-Orgad",
    "multi-head(MH)", "multi-head-ablation(ABL)",
    "MultiMax", "Max-of-Rolling-Means(w10)",                  # NEW (3A)
    "score-std:V1(zscore+fixedT)", "score-std:V2(entropy-solve)", "score-std:LayerNorm-ctrl",  # NEW (3A)
    "length-router", "continuous-length-blend",              # NEW (3B)
    "ens{MSP,SAPLMA}", "ens{wMSP,SAPLMA}", "ens{wMSP,MSP}",  # last is NEW (3E)
]

# Methods seen in a source CSV that ALIAS does not know about. Collected rather than swallowed and
# printed at the end -- see the note at the read loop.
_UNKNOWN_METHODS = set()

# raw method-column value -> canonical row (None => intentionally ignored, e.g. fair_floor == floor_min)
ALIAS = {
    "floor_sum": "msp_sum", "floor_ppl": "perplexity", "floor_min": "msp_min", "fair_floor": None,
    "wmsp_norm": "wMSP-norm", "wmsp_shrink2": "wMSP-shrink@2", "wmsp_shrink10": "wMSP-shrink@10",
    "saplma": "SAPLMA", "uniform": "armB(mean-pool)", "armB": "armB(mean-pool)",
    "attention": "armA(attention)", "armA": "armA(attention)",
    "armC_content_mass": "armC:content-mass", "armC_nll": "armC:NLL", "armC_orgad": "armC:soft-Orgad",
    "armD_content_mass": "armD:content-mass", "armD_nll": "armD:NLL", "armD_orgad": "armD:soft-Orgad",
    "mh": "multi-head(MH)", "ablation": "multi-head-ablation(ABL)",
    "rankavg_floor_min+saplma": "ens{MSP,SAPLMA}", "rankavg_wmsp+saplma": "ens{wMSP,SAPLMA}",
}
RUNG_ALIAS = {"RestrictedOOD-covered": None}  # not one of the 5 canonical rungs -> restricted-only, off-grid

# ---- allow-list of files to read, each tagged with its population -----------------------------------
BASE = "|".join(re.escape(e) for e in LONG_EVALS)
ALLOW = [
    (re.compile(rf"^probedriftlong_({BASE})_widened(_wmsp)?__{re.escape(SLUG)}\.csv$"), "widened"),
    (re.compile(rf"^probedriftlong_({BASE})__{re.escape(SLUG)}\.csv$"), "narrow"),
    (re.compile(rf"^fixed_prior_ladder(_factscore)?__{re.escape(SLUG)}\.csv$"), "widened"),
    (re.compile(rf"^fixed_prior_orgad_restricted__{re.escape(SLUG)}\.csv$"), "restricted"),
    (re.compile(rf"^multihead_ladder__{re.escape(SLUG)}_joined\.csv$"), "widened"),
    (re.compile(rf"^ensemble_(wmsp_saplma|saplma_fast)__{re.escape(SLUG)}\.csv$"), "widened"),
]


def population_of(name):
    for rx, pop in ALLOW:
        if rx.match(name):
            return pop
    return None


def scan_dir(d, presence, sources):
    if not d.is_dir():
        return
    for f in sorted(d.iterdir()):
        pop = population_of(f.name)
        if pop is None:
            continue
        with open(f) as fh:
            for row in csv.DictReader(fh):
                m = (row.get("method") or "").strip()
                if m.startswith("VERDICT:") or m == "":
                    continue
                # Third instance of the same mechanism found on 2026-08-03 (after assemble_pdl_table
                # and assemble_xl_table): an unrecognised method is dropped with no message, so a
                # computed method silently vanishes and "not measured" becomes indistinguishable from
                # "measured then discarded". Reported at the end of this script rather than swallowed.
                if m not in ALIAS:
                    _UNKNOWN_METHODS.add(m)
                canon = ALIAS.get(m, "__skip__")
                if canon is None or canon == "__skip__":
                    continue
                ev = (row.get("eval") or "").strip()
                rg = (row.get("rung") or "").strip()
                rg = RUNG_ALIAS.get(rg, rg)
                if rg is None or ev not in LONG_EVALS or rg not in RUNGS:
                    continue
                presence[(canon, ev, rg)].add(pop)
                sources[(canon, ev, rg)].add(f"{f.name}")


def main():
    presence = defaultdict(set)   # (method,eval,rung) -> {populations}
    sources = defaultdict(set)
    scan_dir(RESULTS, presence, sources)
    scan_dir(EPHEM, presence, sources)

    # router: broad-LOO, one rung, no CSV -- record as a single off-grid presence per eval it covers
    router_evals = sorted(p.name.split("__")[-1].replace(".npz", "")
                          for p in ROUTER_OOD.glob(f"{SLUG}__*.npz")) if ROUTER_OOD.is_dir() else []

    def cell_status(pops):
        if "widened" in pops:
            return "OK", "widened"
        if not pops:
            return "MISS", ""
        # name the best available non-canonical population
        for p in ("narrow", "restricted", "broad-LOO"):
            if p in pops:
                return "OTHER", p
        return "OTHER", sorted(pops)[0]

    # ---- write the machine matrix CSV ----
    out_csv = RESULTS / f"pdl_audit_matrix__{SLUG}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "eval", "rung", "status", "population", "sources"])
        for meth in TARGET_METHODS:
            for ev in LONG_EVALS:
                for rg in RUNGS:
                    pops = presence.get((meth, ev, rg), set())
                    st, pop = cell_status(pops)
                    w.writerow([meth, ev, rg, st, pop,
                                ";".join(sorted(sources.get((meth, ev, rg), []))) or ""])

    # ---- render markdown to stdout ----
    SYM = {"OK": "OK ", "OTHER": "~", "MISS": "."}
    print(f"\n# ProbeDriftLong gap-audit matrix  (SLUG={SLUG})")
    print(f"# grid = {len(TARGET_METHODS)} methods x {len(LONG_EVALS)} evals x {len(RUNGS)} rungs "
          f"= {len(TARGET_METHODS)*len(LONG_EVALS)*len(RUNGS)} cells")
    print("# legend:  OK = canonical widened cells_long   ~<pop> = other population   . = missing\n")

    # per-method coverage summary
    print("## Per-method coverage (of the 40 long eval x rung target cells)")
    print(f"{'method':30s}{'OK%':>6s}{'~%':>6s}{'.%':>6s}   other-populations")
    total = len(LONG_EVALS) * len(RUNGS)
    for meth in TARGET_METHODS:
        ok = other = miss = 0
        otherpops = set()
        for ev in LONG_EVALS:
            for rg in RUNGS:
                st, pop = cell_status(presence.get((meth, ev, rg), set()))
                if st == "OK":
                    ok += 1
                elif st == "OTHER":
                    other += 1
                    otherpops.add(pop)
                else:
                    miss += 1
        print(f"{meth:30s}{100*ok//total:>5d}%{100*other//total:>5d}%{100*miss//total:>5d}%   "
              + (",".join(sorted(otherpops)) if otherpops else ""))

    # compact matrix: one block per rung
    for rg in RUNGS:
        print(f"\n## rung = {rg}")
        print(f"{'method':30s}" + "".join(f"{e[:9]:>11s}" for e in LONG_EVALS))
        for meth in TARGET_METHODS:
            cells = []
            for ev in LONG_EVALS:
                st, pop = cell_status(presence.get((meth, ev, rg), set()))
                cells.append(SYM[st] if st != "OTHER" else f"~{pop[:6]}")
            print(f"{meth:30s}" + "".join(f"{c:>11s}" for c in cells))

    print(f"\n## Router (separate population -- NOT on the 5 canonical rungs)")
    print(f"broad-LOO npz present for {len(router_evals)} evals: {', '.join(router_evals)}")
    print("  -> the router has NO CSV and sits on the broad leave-D-out pool; it must be re-run on")
    print("     cells_long (task 3B) before it can enter the table.")

    print(f"\nwrote {out_csv}")


def _report_unknown_methods():
    """Loud, at the end. Silence here is how ptrue/lookback/linear went missing from the master table."""
    if _UNKNOWN_METHODS:
        print(f"\n{len(_UNKNOWN_METHODS)} METHOD(S) COMPUTED BUT NOT IN ALIAS -- dropped from this "
              f"audit: {', '.join(sorted(_UNKNOWN_METHODS))}")


if __name__ == "__main__":
    main()
    _report_unknown_methods()
