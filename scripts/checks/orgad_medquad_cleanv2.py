#!/usr/bin/env python
"""Re-extract the med_quad answer spans on the corrected-span population, without losing the old ones.

WHY THIS IS NEEDED
------------------
The answer-span comparator reads a cache of spans that a model extracted from each generation. The
med_quad cache was built in July, before the corrected answer span existed, so it describes the
uncorrected generation. med_quad is the only one of the eight datasets whose span changed, so putting
the July cache in a table with the other seven would mix one uncorrected dataset into an otherwise
corrected comparison.

WHY IT ONLY RE-EXTRACTS PART OF THE DATASET
-------------------------------------------
862 of 1800 generations changed. The other 938 are character-identical, so their existing spans remain
valid and re-extracting them would be paying twice for the same answer. The extraction script is
resumable and skips anything already present, so this seeds it with the 938 unchanged entries and lets
it extract only the 862 that moved.

WHY IT IS A WRAPPER RATHER THAN A FLAG
--------------------------------------
The extraction script builds its output filename from the model, dataset and variant only, with no
component naming the cache namespace, so pointing it at the corrected records would overwrite the
uncorrected cache in place. This preserves the original first, files the new output under a name that
says which population it describes, and restores the original afterwards, so both survive.

    python scripts/checks/orgad_medquad_cleanv2.py --dry-run
    python scripts/checks/orgad_medquad_cleanv2.py
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache  # noqa: E402

SLUG = "meta-llama_Meta-Llama-3.1-8B"
DATASET = "med_quad"
VARIANT = "broad"          # the long-form claim-span prompt, which is what med_quad uses
BASE = ROOT / "cache" / "orgad_llm" / f"{SLUG}__{DATASET}__ID__{VARIANT}"
# NOT BASE.with_suffix(".json"): the filename contains dots (the model version), and with_suffix
# replaces from the LAST dot, which silently mangles it to "meta-llama_Meta-Llama-3.json".
LIVE = Path(str(BASE) + ".json")
PARK = Path(str(BASE) + "__uncorrected_span.json")
OUT = Path(str(BASE) + "__cleanv2.json")

RAW_REC = ROOT / "cache" / "records" / f"{SLUG}__{DATASET}__ID.jsonl"
CLEAN_REC = ROOT / "cache" / "cleanv2" / "records" / f"{SLUG}__{DATASET}__ID.jsonl"


def load(p):
    return [json.loads(l) for l in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report the scope and cost, call nothing")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    if OUT.exists():
        sys.exit(f"{OUT.name} already exists -- nothing to do.")
    for p in (LIVE, RAW_REC, CLEAN_REC):
        if not p.exists():
            sys.exit(f"FATAL: expected {p} and it is missing. Refusing to guess.")

    raw, clean = load(RAW_REC), load(CLEAN_REC)
    if len(raw) != len(clean):
        sys.exit(f"FATAL: {len(raw)} uncorrected rows vs {len(clean)} corrected -- not row-aligned.")
    existing = json.loads(LIVE.read_text())

    changed, unchanged = [], {}
    for a, b in zip(raw, clean):
        k = f"{b['split']}:{b['idx']}"
        if a["gen_text"] != b["gen_text"]:
            changed.append(k)
        elif k in existing:
            unchanged[k] = existing[k]

    n_chars = sum(len(b["gen_text"]) for a, b in zip(raw, clean) if a["gen_text"] != b["gen_text"])
    print(f"existing cache entries      : {len(existing)}")
    print(f"generations that changed    : {len(changed)}  -> re-extracted")
    print(f"generations unchanged       : {len(unchanged)}  -> reused, not paid for again")
    print(f"characters to send          : {n_chars:,} (~{n_chars // 4:,} tokens of generation text)")
    missing = [k for k in changed if k not in existing]
    print(f"changed rows absent from the old cache: {len(missing)} (expected 0)")
    if args.dry_run:
        print("\ndry run: nothing was called and no file was touched.")
        return

    # Preserve the uncorrected cache under a name that states what it is, BEFORE anything can overwrite it.
    if not PARK.exists():
        shutil.copy2(LIVE, PARK)
        print(f"\nuncorrected cache preserved as {PARK.name}")
    else:
        print(f"\nuncorrected cache already preserved at {PARK.name}")

    # Seed the working cache with only the unchanged entries, so the resumable extractor does the rest.
    LIVE.write_text(json.dumps(unchanged))
    print(f"seeded the working cache with {len(unchanged)} reusable entries")

    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "01o_orgad_llm_extract.py"),
           "--dataset", DATASET, "--variant", VARIANT,
           "--prompt-regime", "cleanv2", "--workers", str(args.workers)]
    print(f"running: {' '.join(cmd[1:])}\n", flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT))

    try:
        produced = json.loads(LIVE.read_text())
    except Exception as e:
        produced = None
        print(f"!!! could not read the produced cache: {e}")

    ok = rc == 0 and produced is not None and len(produced) == len(clean)
    if ok:
        OUT.write_text(json.dumps(produced))
        print(f"\ncorrected-span spans written to {OUT.name} ({len(produced)} entries)")
    else:
        print(f"\n!!! extraction did not complete cleanly (rc={rc}, "
              f"entries={None if produced is None else len(produced)} of {len(clean)}). "
              f"Nothing was filed under the corrected name.")

    # Restore the uncorrected cache to its canonical filename either way, so a failure cannot leave the
    # tree missing an artifact that other code still resolves by that name.
    shutil.copy2(PARK, LIVE)
    print(f"uncorrected cache restored to {LIVE.name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
