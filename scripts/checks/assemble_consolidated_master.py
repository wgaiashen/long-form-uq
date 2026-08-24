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
}

# Which family each method belongs to, so a published baseline is never read as one of ours.
PUBLISHED = {"msp", "hbo", "satmd_mid", "satrmd_mid", "msp_satmd_mid", "msp_satrmd_mid",
             "huq_satmd_mid", "huq_satrmd_mid", "md_mean_mid", "rmd_mean_mid"}
BASELINE = {"floor_sum", "floor_ppl", "floor_min", "fair_floor", "saplma", "ptrue", "lookback"}

FIELDS = ["model", "eval", "rung", "train", "method", "family", "prr_mean", "prr_std",
          "n_seeds", "degenerate_seeds", "source"]

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
        hybrid = ROOT / "results" / "hybrids" / f"pdl_hybrids__{slug}.csv"
        if not master.exists() or not hybrid.exists():
            print(f"{model}: missing input ({master.exists()=}, {hybrid.exists()=}) -> SKIPPED")
            continue
        before = {p: sha(p) for p in (master, hybrid)}

        rows, seen, master_vals, agree = [], set(), {}, []
        for src, path in (("corrected-span master", master), ("published baselines", hybrid)):
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
                key = (r.get("eval"), r.get("rung"), m)
                if key in seen:
                    sys.exit(f"FATAL {model}: method {m} appears for {key[:2]} in both inputs. "
                             f"Refusing to emit a table where one cell has two values.")
                seen.add(key)
                rows.append({
                    "model": model, "eval": r.get("eval", ""), "rung": r.get("rung", ""),
                    "train": r.get("train", ""), "method": m, "family": family_of(m),
                    "prr_mean": r.get("prr_mean", ""), "prr_std": r.get("prr_std", ""),
                    "n_seeds": r.get("n_seeds", ""),
                    "degenerate_seeds": r.get("degenerate_seeds", ""), "source": src})

        out = ROOT / "results" / "hybrids" / f"pdl_consolidated_master__{slug}.csv"
        with open(out, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

        after = {p: sha(p) for p in (master, hybrid)}
        if before != after:
            sys.exit(f"FATAL: an input file changed while assembling {model}.")
        # Report the cross-check on the de-duplicated quantities, and fail if any disagrees.
        for name in sorted({a[0] for a in agree}):
            ds = [a[3] for a in agree if a[0] == name]
            worst = max(ds) if ds else float("nan")
            tol = STORAGE_TOL if name == "msp" else AGREE_TOL
            ok = worst <= tol if worst == worst else False
            print(f"  cross-check {name:8s} vs master {DUPLICATE_OF[name]:10s} on {len(ds):3d} cells: "
                  f"max |d| = {worst:.2e} (bar {tol:.0e})  {'ok' if ok else 'DISAGREES'}")
            if not ok:
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
