#!/usr/bin/env python
"""Union the corrected-span master with the published-baseline rows into one table per model.

WHY A NEW FILE RATHER THAN AN EDIT
----------------------------------
The corrected-span masters are the frozen record of what the ladder produced, and the pre-registration
for the baseline work states that its output must not be able to overwrite them. So this reads both
inputs and writes a THIRD file, and it verifies afterwards that neither input changed.

The union is legitimate because both sides describe the same population: the same eight datasets in the
same cache namespaces, the same forty cells, the same three seeds, and the same evaluation rows -- the
baseline driver took its probe and probability vectors from the master's own per-example sidecars and
its recomputed sequence-probability score matched the master's column exactly on every cell and seed.

Every row carries `source`, so a reader can always tell which file a number came from, and `family`,
so the published baselines are never mistaken for our own methods.

    python scripts/checks/assemble_consolidated_master.py
"""
import csv as _csv
import re
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "meta-llama/Meta-Llama-3.1-8B": {
        "slug": "meta-llama_Meta-Llama-3.1-8B",
        "master": "results/cleanv2/pdl_cleanv2_master__meta-llama_Meta-Llama-3.1-8B.csv",
    },
    "google/gemma-2-9b": {
        "slug": "google_gemma-2-9b",
        "master": "results/cleanv2/wmodels_sens8_cleanv2_master__google_gemma-2-9b.csv",
    },
    "Qwen/Qwen2.5-14B": {
        "slug": "Qwen_Qwen2.5-14B",
        "master": "results/analysis/pdl_master_qwenclean__Qwen_Qwen2.5-14B.csv",
        # This population has no corrected-span per-example vectors, so the hybrid run REFITTED the
        # probe instead of reading the master's own. The refit is a reproduction, and on this
        # population reproductions of the trained components drift measurably (documented, and a
        # matched-device test ruled out hardware as the cause). The duplicate rows are dropped in
        # favour of the master either way; what changes is that the agreement check below becomes a
        # reproducibility REPORT rather than an identity assertion. It is not a widened tolerance:
        # the strict bar still applies wherever the probe was taken from the master rather than refit.
        "probe_refitted": True,
    },
}

# Where the published-baseline rows come from. The corrected file is preferred and the fallback is
# announced, because the two differ on the supervised distance methods: the original run used a
# positivity-constrained ridge the reference does not use, which clipped a single-feature coefficient
# to zero and produced constant scores on a quarter of the cell-seeds. Silently consolidating those
# rows would put a superseded number in the report-facing table.
HYBRID_PREFERENCE = ["pdl_hybrids_unconstrained__{slug}.csv", "pdl_hybrids__{slug}.csv"]

# FURTHER published-baseline files, merged on top of the preferred one in this order. They exist
# because the family was produced in stages, not because they disagree.
#
#   the relative-distance file adds the methods that need a background corpus, which was generated
#   after the first run. Without it the relative distance, its supervised aggregation and its hybrid
#   are absent from every consolidated table, which is how they came to be missing from the tables
#   built from this file alone.
#
#   the all-layer file adds the reproductions at the published layer set. Its method names carry
#   their own suffix, so they sit beside the middle-layer rows rather than replacing them, which is
#   what makes the two readable as a fidelity ladder.
#
# A CELL PRESENT IN MORE THAN ONE FILE IS NOT SILENTLY RESOLVED. The files overlap: the
# relative-distance run recomputed the methods the first run already had. Where the same method and
# cell appears twice the two values must agree, and the assembler stops if they do not, because a
# preference order applied to a genuine disagreement would hide it.
HYBRID_EXTRA = ["pdl_hybrids_rmd__{slug}.csv", "pdl_mdlayers_alllayer__{slug}.csv"]
OVERLAP_TOL = 1e-6

# Which family each method belongs to, so a published baseline is never read as one of ours.
PUBLISHED = {"msp", "hbo", "satmd_mid", "satrmd_mid", "msp_satmd_mid", "msp_satrmd_mid",
             "huq_satmd_mid", "huq_satrmd_mid", "md_mean_mid", "rmd_mean_mid",
             # the reproductions at the published layer set
             "satmd_alllayer", "satrmd_alllayer", "msp_satmd_alllayer", "msp_satrmd_alllayer",
             "huq_satmd_alllayer", "huq_satrmd_alllayer"}
BASELINE = {"floor_sum", "floor_ppl", "floor_min", "fair_floor", "saplma", "ptrue", "lookback"}

FIELDS = ["model", "eval", "rung", "train", "method", "family", "prr_mean", "prr_std",
          "n_seeds", "degenerate_seeds", "source", "method_cells", "complete_grid"]

# The grid every method is supposed to cover: eight evaluation datasets by five training conditions.
# A method present on fewer cells is not a smaller version of the same measurement, and a mean over
# whatever cells happen to exist is not comparable to a mean over all of them.
FULL_GRID_CELLS = 40

# Methods the baseline driver recomputes that the master already owns. They are NOT emitted twice:
# the master's value is authoritative and the recomputation is used as a cross-check on it. `msp` is
# the published sequence-probability score, which is the same formula as the master's `floor_sum` --
# emitting both under two names would make one quantity look like two independent baselines.
DUPLICATE_OF = {"saplma": "saplma", "msp": "floor_sum"}
AGREE_TOL = 5e-3        # seed-level noise on a refitted probe
# The master stores its prediction-rejection values to four decimals, so a comparison made at the CSV
# level cannot resolve better than half of that no matter how identical the underlying scores are. The
# EXACT identity of the probability score was established elsewhere, on the per-example vectors, where
# the driver measured max |d| = 0.000e+00 across every cell and seed. This bar is the storage
# resolution of the file being read, not a claim about the method.
STORAGE_TOL = 1e-4


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def family_of(method):
    if method.startswith("VERDICT:"):
        return "verdict"
    if method in PUBLISHED:
        return "published baseline"
    if method in BASELINE:
        return "reference baseline"
    return "this project"


def main():
    for model, cfg in MODELS.items():
        slug = cfg["slug"]
        master = ROOT / cfg["master"]
        hybrid = None
        for pat in HYBRID_PREFERENCE:
            cand = ROOT / "results" / "hybrids" / pat.format(slug=slug)
            if cand.exists():
                hybrid = cand
                break
        extra = [ROOT / "results" / "hybrids" / pat.format(slug=slug) for pat in HYBRID_EXTRA]
        extra = [e for e in extra if e.exists()]
        if not master.exists() or hybrid is None:
            print(f"{model}: missing input (master={master.exists()}, hybrid=None) -> SKIPPED")
            continue
        for e in extra:
            print(f"  {model}: also merging {e.name}")
        if hybrid.name.startswith("pdl_hybrids__"):
            print(f"  {model}: WARNING -- using the PRE-CORRECTION hybrid file. Its supervised "
                  f"distance rows were produced with a positivity-constrained ridge the reference "
                  f"does not use and are superseded.")
        else:
            print(f"  {model}: using the corrected hybrid rows ({hybrid.name})")
        before = {p: sha(p) for p in [master, hybrid] + extra}

        rows, seen, master_vals, agree = [], {}, {}, []
        overlap_ok, overlap_bad = 0, []
        sources = [("corrected-span master", master), ("published baselines", hybrid)]
        sources += [("published baselines", e) for e in extra]
        for src, path in sources:
            for r in _csv.DictReader(open(path)):
                m = r.get("method", "")
                if not m:
                    continue
                if src.startswith("corrected"):
                    master_vals[(r.get("eval"), r.get("rung"), m)] = r.get("prr_mean", "")
                elif m in DUPLICATE_OF:
                    # Cross-check against the master rather than emitting a second row for one quantity.
                    ref = master_vals.get((r.get("eval"), r.get("rung"), DUPLICATE_OF[m]))
                    try:
                        d = abs(float(r.get("prr_mean")) - float(ref))
                    except (TypeError, ValueError):
                        d = float("nan")
                    agree.append((m, r.get("eval"), r.get("rung"), d))
                    continue
                # The per-layer raw distances are a diagnostic of the layer sweep, not comparison
                # methods. Thirty-two of them per family would swamp the table they are read beside.
                if re.match(r"^r?md_mean_L\d+$", m):
                    continue
                key = (r.get("eval"), r.get("rung"), m)
                if key in seen:
                    # The published-baseline files overlap by construction, because the later runs
                    # recomputed the methods the earlier ones already had. Agreement is expected and
                    # is therefore checked rather than assumed; a real disagreement stops the job
                    # instead of being resolved by whichever file happened to be read first.
                    prev = seen[key]
                    a, b = prev.get("prr_mean", ""), r.get("prr_mean", "")
                    if (a == "") != (b == ""):
                        overlap_bad.append(f"{key}: blank in one file, {a or b} in the other")
                    elif a != "":
                        d = abs(float(a) - float(b))
                        if d > OVERLAP_TOL:
                            overlap_bad.append(f"{key}: {a} vs {b} (|d| = {d:.3e})")
                        else:
                            overlap_ok += 1
                    else:
                        overlap_ok += 1
                    continue
                seen[key] = r
                rows.append({
                    "model": model, "eval": r.get("eval", ""), "rung": r.get("rung", ""),
                    "train": r.get("train", ""), "method": m, "family": family_of(m),
                    "prr_mean": r.get("prr_mean", ""), "prr_std": r.get("prr_std", ""),
                    "n_seeds": r.get("n_seeds", ""),
                    "degenerate_seeds": r.get("degenerate_seeds", ""), "source": src})

        if overlap_bad:
            print(f"\n{model}: {len(overlap_bad)} cell(s) disagree between published-baseline files:")
            for line in overlap_bad[:10]:
                print(f"    {line}")
            sys.exit(f"FATAL {model}: refusing to emit a table where one cell has two values.")
        if overlap_ok:
            print(f"  {model}: {overlap_ok} cell(s) appear in more than one published-baseline file "
                  f"and agree to within {OVERLAP_TOL:.0e}")

        # COVERAGE TRAVELS WITH EVERY ROW. A method on a partial grid is the one thing a reader of
        # this file cannot see by looking at a row, and averaging six cells as though they were forty
        # is the mistake the file would otherwise invite. The count is written into each row and the
        # partial methods are named in the log.
        counts = {}
        for r in rows:
            counts[r["method"]] = counts.get(r["method"], 0) + 1
        for r in rows:
            n = counts[r["method"]]
            r["method_cells"] = n
            r["complete_grid"] = "yes" if n == FULL_GRID_CELLS else "NO"
        partial = sorted((m, n) for m, n in counts.items() if n != FULL_GRID_CELLS)
        if partial:
            print(f"  {model}: INCOMPLETE COVERAGE, these methods are NOT on the full "
                  f"{FULL_GRID_CELLS}-cell grid and must not enter an aggregate:")
            for m, n in partial:
                print(f"    {m:<24} {n} of {FULL_GRID_CELLS} cells")

        out = ROOT / "results" / "hybrids" / f"pdl_consolidated_master__{slug}.csv"
        with open(out, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

        after = {p: sha(p) for p in [master, hybrid] + extra}
        if before != after:
            sys.exit(f"FATAL: an input file changed while assembling {model}.")
        # Report the cross-check on the de-duplicated quantities, and fail if any disagrees.
        for name in sorted({a[0] for a in agree}):
            ds = [a[3] for a in agree if a[0] == name]
            worst = max(ds) if ds else float("nan")
            tol = STORAGE_TOL if name == "msp" else AGREE_TOL
            ok = worst <= tol if worst == worst else False
            # Where the probe was REFITTED rather than read from the master's own per-example vectors,
            # this comparison measures reproducibility, not identity, and cannot assert the latter.
            # The distinction is declared per population in MODELS, never inferred from the size of
            # the disagreement -- deciding after the fact which failures count would make the check
            # worthless. The duplicate row is dropped in favour of the master in both cases.
            advisory = cfg.get("probe_refitted", False) and name == "saplma"
            verdict = "ok" if ok else ("REPRODUCTION DRIFT" if advisory else "DISAGREES")
            print(f"  cross-check {name:8s} vs master {DUPLICATE_OF[name]:10s} on {len(ds):3d} cells: "
                  f"max |d| = {worst:.2e} (bar {tol:.0e})  {verdict}")
            if not ok and advisory:
                print(f"    ADVISORY, not fatal: this population has no corrected-span per-example "
                      f"vectors, so the hybrid run refitted the probe. The master's value is kept and "
                      f"the refit is discarded. The drift itself is a documented property of re-fits "
                      f"on this population, not evidence of a wrong population -- the deterministic "
                      f"probability score above still agrees to storage resolution, which it could "
                      f"not do if the rows differed.")
            elif not ok:
                sys.exit(f"FATAL {model}: recomputed {name} disagrees with the master by {worst:.3e} "
                         f"(tolerance {tol:.0e}). The two are not the same population.")
        meths = sorted({r["method"] for r in rows if r["family"] != "verdict"})
        cells = {(r["eval"], r["rung"]) for r in rows}
        print(f"{model}\n  wrote {out.relative_to(ROOT)}  ({len(rows)} rows, {len(meths)} methods, "
              f"{len(cells)} cells)")
        for fam in ("this project", "reference baseline", "published baseline"):
            n = sorted({r['method'] for r in rows if r['family'] == fam})
            print(f"    {fam:20s} {len(n):2d}: {', '.join(n)}")
        print(f"  inputs unchanged: master sha {before[master][:12]}, hybrids sha {before[hybrid][:12]}")


if __name__ == "__main__":
    main()
