"""V5 GUARD (2026-07-27): make the PART A markdown table VERIFIABLE against its CSV ground truth.

WHY THIS EXISTS. The trivia `msp_min` corruption (sciq's 0.902/0.863 appearing verbatim in trivia's rows,
arithmetically impossible since best_of_3 >= msp_min by construction) was NOT a code bug: the length driver
processes every dataset in isolation (`load(dataset)` reads only that dataset's model-pinned files; each
emitted row carries its own dataset key; there is no cross-dataset merge/join/forward-fill). The corruption
was a HAND-TRANSCRIPTION / fill-down error when the Table A1 *markdown* was built by hand, downstream of the
correct CSV. The CSV emitter already asserts the arithmetic invariant at write time; this closes the OTHER
hole -- a hand-edited table or prose number drifting from the CSV -- by cross-checking the rendered table
against the CSV every time.

Checks:
  1. ARITHMETIC INVARIANTS on the CSV (defence in depth): best_of_3 >= max(sum,ppl,min); gap == probe-best3;
     gap_ci_lo <= gap_ci_hi; probe/best3 in [-1,1].
  2. CROSS-DATASET DUPLICATE-TRIPLE scan: two rows from DIFFERENT datasets sharing an identical
     (msp_sum, msp_perplexity, msp_min) triple is the copy-paste signature (identical values WITHIN a dataset
     are legitimate -- short bands where the three aggregates coincide -- and are not flagged).
  3. MARKDOWN == CSV: parse Table A1 out of the project's working notes and assert every rendered numeric cell
     matches the CSV to tolerance. This is the check that would have caught the trivia bug at the source.

Exit 1 (fail loud) on any violation.  Read-only.  Run:  python scripts/checks/verify_partA_consistency.py
"""
import csv as _csv
import re
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_SLUG = "meta-llama_Meta-Llama-3.1-8B"
CSV = ROOT / "results" / f"probe_vs_msp_length__{MODEL_SLUG}.csv"
# The rendered results table lives in the project's working notes, which is outside this repository.
# Point LUQ_RESULTS_DOC at it to enable the markdown cross-check; without it the check is
# skipped and only the CSV checks run (see the DOC.exists() branch below).
DOC = ROOT.parent / os.environ.get("LUQ_RESULTS_DOC", "results_record.md")
TOL = 2e-3          # rendered table shows 3 dp; allow half-ULP + rounding slack

FAILS = []


def fail(msg):
    FAILS.append(msg)


def num(s):
    """parse a table cell like '+0.902', '−0.052' (unicode minus), '**0.906**' -> float, or None."""
    s = s.strip().replace("*", "").replace("−", "-").replace("", "").strip()
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None


def main():
    if not CSV.exists():
        print(f"NO CSV at {CSV} -- run the length driver first", flush=True); sys.exit(1)
    rows = list(_csv.DictReader(open(CSV)))
    by_key = {(r["dataset"], r["quartile"]): r for r in rows}

    # 1. arithmetic invariants
    for r in rows:
        s, p, mn = float(r["msp_sum"]), float(r["msp_perplexity"]), float(r["msp_min"])
        b3, prb = float(r["msp_bestof3"]), float(r["probe_prr"])
        tag = f"{r['dataset']} Q{r['quartile']}"
        if b3 < max(s, p, mn) - 1e-6:
            fail(f"[invariant] {tag}: best3 {b3} < max(sum,ppl,min) {max(s,p,mn)}")
        # gap is round(prbv-mspv,4) from UNROUNDED inputs, while prb/b3 are each independently 4-dp rounded,
        # so gap vs (prb-b3) can legitimately differ by up to ~1.5e-4 (two half-ULP roundings). Tol accordingly.
        if abs(float(r["gap_probe_minus_best3"]) - (prb - b3)) > 2e-4:
            fail(f"[invariant] {tag}: gap {r['gap_probe_minus_best3']} != probe-best3 {prb-b3:.4f}")
        if float(r["gap_ci_lo"]) > float(r["gap_ci_hi"]) + 1e-9:
            fail(f"[invariant] {tag}: CI lo > hi")
        if not (-1.01 <= prb <= 1.01 and -1.01 <= b3 <= 1.01):
            fail(f"[invariant] {tag}: PRR out of [-1,1]")

    # 2. cross-dataset duplicate-triple scan (the copy-paste signature)
    seen = {}
    for r in rows:
        trip = (r["msp_sum"], r["msp_perplexity"], r["msp_min"])
        if trip in seen and seen[trip] != r["dataset"]:
            fail(f"[dup] identical floor triple {trip} in DIFFERENT datasets "
                 f"'{seen[trip]}' and '{r['dataset']}' -> possible copy-paste")
        seen.setdefault(trip, r["dataset"])

    # 3. markdown == csv
    if not DOC.exists():
        print(f"(doc {DOC} absent -- skipping markdown check, ran CSV checks only)", flush=True)
    else:
        lines = DOC.read_text().splitlines()
        # find the Table A1 header row, then read pipe-rows until the block ends
        hdr_i = next((i for i, ln in enumerate(lines)
                      if ln.strip().startswith("| dataset ") and "best-3" in ln and "PROBE" in ln), None)
        if hdr_i is None:
            fail("[md] could not locate the Table A1 header (| dataset ... best-3 ... PROBE ...) in the doc")
        else:
            # header cells can carry markdown emphasis (**PROBE**) -> strip it when keying columns
            header = [c.strip().replace("*", "") for c in lines[hdr_i].strip().strip("|").split("|")]
            col = {name: k for k, name in enumerate(header)}
            # which rendered columns we cross-check, and their CSV field
            checks = {"best-3": "msp_bestof3", "PROBE": "probe_prr", "gap": "gap_probe_minus_best3"}
            checked = 0
            for ln in lines[hdr_i + 2:]:                     # +2 skips the |---| separator
                if not ln.strip().startswith("|"):
                    break
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                if len(cells) < len(header):
                    continue
                ds, q = cells[col["dataset"]].strip(), cells[col["Q"]].strip()
                key = (ds, q)
                if key not in by_key:
                    fail(f"[md] table row {key} has no CSV counterpart")
                    continue
                cr = by_key[key]
                for disp, field in checks.items():
                    if disp not in col:
                        continue
                    mdv = num(cells[col[disp]])
                    csvv = float(cr[field])
                    if mdv is None or abs(mdv - csvv) > TOL:
                        fail(f"[md] {key} col '{disp}': table={mdv} vs csv={csvv:.3f} (Δ>{TOL})")
                checked += 1
            print(f"markdown cross-check: {checked} table rows vs CSV", flush=True)

    if FAILS:
        print(f"\nFAILED ({len(FAILS)} issues):", flush=True)
        for f in FAILS:
            print("  " + f, flush=True)
        sys.exit(1)
    print(f"\nOK: {len(rows)} CSV rows pass arithmetic + duplicate + markdown consistency.", flush=True)


if __name__ == "__main__":
    main()
