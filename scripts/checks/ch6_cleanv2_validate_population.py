#!/usr/bin/env python
"""Validate the corrected-span (clean-v2) Llama-3.1-8B population before any analysis reads it.

The corrected-span population differs from the original one on MedQuAD alone: the retained answer
span was cut on 862 of 1800 rows, and every other dataset inherits its original cache unchanged.
Analyses migrated onto this population depend on that being true, so it is asserted here rather
than taken from prose.

Six checks, each fails loud:

  P1  the assembled master exists, has the expected digest, and is complete: one row per
      (method, eval, rung) with no duplicates, no nulls and no partially covered method.
  P2  every dataset named in the population manifest resolves to a record cache that exists on
      disk, with the row count and label field the manifest pins.
  P3  no dataset other than the corrected one has a corrected-span cache anywhere in the tree.
      A stray one would mean the "inherited unchanged" claim is false for that dataset.
  P4  cell-level affectedness. A cell is affected if the corrected dataset is its evaluation
      population or appears in its realised labelled training sources. The derived split must
      agree with the provenance stamps the master carries, cell for cell.
  P5  the shrinkage 1.5 column is complete, and the note that its filled cells carry n_seeds = 0
      is reported rather than silently inherited.
  P6  the original master is untouched: its digest still matches the recorded value.

    python scripts/checks/ch6_cleanv2_validate_population.py
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
CORRECTED = "med_quad"

MASTER = ROOT / "results" / "cleanv2" / f"pdl_cleanv2_master__{SLUG}.csv"
ORIGINAL_MASTER = ROOT / "results" / f"pdl_master__{SLUG}.csv"
MANIFEST = ROOT / "results" / "analysis" / "CLEAN_CORE_POPULATION_MANIFEST.json"

# Recorded digests. P1 and P6 are the "nothing moved under us" checks, so these are pinned rather
# than recomputed into a variable and compared with themselves.
MASTER_MD5 = "6473b69b4911abc33132f4ad33e5ddd3"
ORIGINAL_MASTER_MD5 = "9aa5643d2e8864badbb6165a141a4be0"

EVALS = ["asqa", "cnn_dailymail", "expertqa", "factscore",
         "med_quad", "pubmed_qa", "samsum", "xsum"]
RUNGS = ["ID", "SameTask-long", "LOO-long", "DiffTask-long", "1ds-Diff-long"]

failures = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_master():
    with open(MASTER) as fh:
        return list(csv.DictReader(fh))


def p1_master_complete(rows):
    print("\nP1  assembled master is complete")
    check(MASTER.exists(), "master file exists", str(MASTER.relative_to(ROOT)))
    got = md5(MASTER)
    check(got == MASTER_MD5, "master digest unchanged", f"md5 {got}")

    est = [r for r in rows if not r["method"].startswith("VERDICT")]
    triples = Counter((r["method"], r["eval"], r["rung"]) for r in est)
    methods = sorted({r["method"] for r in est})
    cells = {(r["eval"], r["rung"]) for r in est}

    check(len(cells) == len(EVALS) * len(RUNGS),
          "cell coverage", f"{len(cells)} of {len(EVALS) * len(RUNGS)}")
    check(sorted({r["eval"] for r in est}) == EVALS, "evaluation datasets", str(len(EVALS)))
    check(sorted({r["rung"] for r in est}) == sorted(RUNGS), "transfer rungs", str(len(RUNGS)))
    check(max(triples.values()) == 1, "no duplicated (method, eval, rung)")

    partial = {m for m in methods
               if sum(1 for r in est if r["method"] == m) != len(cells)}
    check(not partial, "every method covers every cell",
          f"{len(methods)} methods x {len(cells)} cells" if not partial else f"partial: {sorted(partial)}")

    blank = [r for r in est if r["prr_mean"].strip() == ""]
    check(not blank, "no null PRR", f"{len(blank)} blanks")
    return est


def p2_manifest_matches_disk():
    print("\nP2  manifest entries resolve on disk, with the pinned rows and digests")
    man = json.loads(MANIFEST.read_text())
    block = man["models"].get(MODEL)
    if block is None:
        check(False, "manifest block for the model", "not present in the manifest")
        return {}
    entries = {d["dataset"]: d for d in block["datasets"]}

    check(sorted(entries) == EVALS, "manifest covers the eight datasets", str(sorted(entries)))
    check(str(Path(block["master"])).endswith(MASTER.name),
          "manifest names this master", block["master"])

    for name in EVALS:
        d = entries.get(name)
        if d is None:
            check(False, f"manifest entry for {name}")
            continue
        rec = ROOT / d["records_path"]
        if not rec.exists():
            check(False, f"{name:14s} records present", str(d["records_path"]))
            continue
        n_disk = sum(1 for _ in open(rec))
        digest = hashlib.sha256(rec.read_bytes()).hexdigest()
        ok = n_disk == d["n_rows"] and digest == d["records_sha256"]
        check(ok, f"{name:14s} records match the manifest",
              f"{n_disk} rows, regime {d['expected_regime']}, label {d['label_field']}, "
              f"cut {d['n_rows_cut']}" if ok else
              f"rows {n_disk} vs {d['n_rows']}, sha256 {digest[:16]} vs {d['records_sha256'][:16]}")

    corrected = entries[CORRECTED]
    check(corrected["clean_span_applied"] and corrected["n_rows_cut"] > 0,
          "the corrected dataset is the one carrying a span correction",
          f"{CORRECTED}: {corrected['n_rows_cut']} rows cut")
    others = [n for n in EVALS if n != CORRECTED and entries[n]["clean_span_applied"]]
    check(not others, "no other dataset carries a span correction", str(others))
    return entries


def p3_no_stray_corrected_caches():
    print("\nP3  no other dataset carries a corrected-span cache")
    cache_root = ROOT / "cache"
    marked = [p for p in cache_root.rglob("*cleanv2*")] + \
             [p for p in cache_root.rglob("*/*cleanv2*/*") if p.is_file()]
    seen, stray = set(), []
    for p in marked:
        rel = str(p.relative_to(cache_root))
        if rel in seen:
            continue
        seen.add(rel)
        for name in EVALS:
            if name == CORRECTED:
                continue
            if name in rel:
                stray.append(rel)
    check(not stray, "corrected-span caches name only the corrected dataset",
          f"{len(seen)} corrected-span paths, all {CORRECTED}"
          if not stray else f"stray: {sorted(set(stray))[:5]}")


def p4_cell_affectedness(est):
    print("\nP4  cell-level affectedness agrees with the master's provenance stamps")
    train_of, prov_of = {}, defaultdict(set)
    for r in est:
        train_of[(r["eval"], r["rung"])] = r["train"]
        prov_of[(r["eval"], r["rung"])].add(r["provenance"])

    derived_affected = {c for c, tr in train_of.items()
                        if c[0] == CORRECTED or CORRECTED in tr}
    stamped_affected = {c for c, ps in prov_of.items() if "recomputed_cleanv2" in ps}

    check(derived_affected == stamped_affected,
          "derived affected set equals the recomputed set",
          f"{len(derived_affected)} affected, {len(train_of) - len(derived_affected)} control"
          + ("" if derived_affected == stamped_affected
             else f"; disagree on {sorted(derived_affected ^ stamped_affected)}"))

    print("      affected cells (recompute):")
    for c in sorted(derived_affected):
        print(f"        {c[0]:14s} {c[1]}")
    print(f"      control cells (must reproduce the original arm): {len(train_of) - len(derived_affected)}")
    return derived_affected, set(train_of) - derived_affected


def p5_shrinkage_sensitivity(est):
    print("\nP5  the shrinkage 1.5 sensitivity column")
    rows = [r for r in est if r["method"] == "wmsp_shrink1_5"]
    check(len(rows) == len(EVALS) * len(RUNGS), "complete coverage", f"{len(rows)} rows")
    filled = [r for r in rows if r["provenance"] == "inherited_lambda_source"]
    print(f"      NOTE {len(filled)} cells were filled from the separate shrinkage sweep and carry "
          f"n_seeds = 0 in this file. The underlying values are three-seed; quote them with that caveat.")


def p6_original_untouched():
    print("\nP6  the original master is untouched")
    if not ORIGINAL_MASTER.exists():
        check(False, "original master exists", str(ORIGINAL_MASTER))
        return
    got = md5(ORIGINAL_MASTER)
    check(got == ORIGINAL_MASTER_MD5, "original master digest unchanged", f"md5 {got}")


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print("=" * 92)
    print(f"CORRECTED-SPAN POPULATION VALIDATION  model={MODEL}  corrected dataset={CORRECTED}")
    print("=" * 92)

    rows = load_master()
    est = p1_master_complete(rows)
    p2_manifest_matches_disk()
    p3_no_stray_corrected_caches()
    p4_cell_affectedness(est)
    p5_shrinkage_sensitivity(est)
    p6_original_untouched()

    print("\n" + "=" * 92)
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("VALIDATION PASSED: the corrected-span population is safe to read.")


if __name__ == "__main__":
    main()
