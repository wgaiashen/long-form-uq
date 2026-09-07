#!/usr/bin/env python
"""Regenerate scripts/checks/CANONICAL_v1_MANIFEST.{md,sha256} after a deliberate change to the records.

WHY THIS IS NEEDED, AND WHY IT DEMANDS A REASON
------------------------------------------------
The manifest exists because "unchanged" once rested on an mtime, and mtime lies. It works: a
`sha256sum -c` failure means the file is not what was pre-registered.

But that makes any INTENDED change to the records a problem. Merging the unsupervised-P(True) sidecars,
for example, adds three fields to six of the ten files — entirely additive, judge labels untouched — and
afterwards six checksums fail. Anyone later running `sha256sum -c` sees six failures and cannot tell
a deliberate additive merge from actual corruption. Silently regenerating the hashes is worse: it erases
the very evidence the manifest was created to preserve.

So this refreshes the manifest ONLY with a recorded reason, and APPENDS a dated amendment carrying the
old -> new hash for every file that moved. The chain of custody stays readable: what changed, when, why,
and what it was before.

    python scripts/checks/refresh_canonical_manifest.py --dry-run
    python scripts/checks/refresh_canonical_manifest.py \
        --reason "merged the unsupervised-P(True) sidecars (additive: ptrue_unsup, _mass, _model)" \
        --backup BACKUP_records_pre_ptrueunsup_2026-08-04 --date 2026-08-04
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MD = ROOT / "scripts" / "checks" / "CANONICAL_v1_MANIFEST.md"
SHA = ROOT / "scripts" / "checks" / "CANONICAL_v1_MANIFEST.sha256"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", help="what changed and why (required unless --dry-run)")
    ap.add_argument("--date", default="", help="ISO date for the amendment heading")
    ap.add_argument("--backup", default="", help="path holding the pre-change copies, recorded in the note")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not args.reason:
        sys.exit("--reason is required. A manifest refreshed without a recorded reason destroys the "
                 "only evidence distinguishing a deliberate change from corruption.")

    old = {}
    for line in SHA.read_text().splitlines():
        if not line.strip():
            continue
        h, _, rel = line.partition("  ")
        old[rel.strip()] = h.strip()

    moved, rows = [], []
    for rel, oh in old.items():
        p = ROOT / rel
        if not p.exists():
            sys.exit(f"{rel}: listed in the manifest but ABSENT on disk. Refusing to refresh a manifest "
                     "over a missing file — that would silently drop it from the record.")
        nh, nb = sha256(p), p.stat().st_size
        rows.append((nb, nh, rel))
        if nh != oh:
            moved.append((rel, oh, nh))

    print(f"  {len(old)} file(s) in the manifest; {len(moved)} changed:")
    for rel, oh, nh in moved:
        print(f"    {rel}\n        {oh}\n     -> {nh}")
    if not moved:
        print("  nothing changed — manifest is already current, no refresh needed.")
        return
    if args.dry_run:
        print("\n  (dry run — nothing written)")
        return

    SHA.write_text("".join(f"{h}  {rel}\n" for _b, h, rel in rows))

    md = MD.read_text()
    md = re.sub(r"\| (\d+) \| `([0-9a-f]{64})` \| `([^`]+)` \|",
                lambda m: next((f"| {b} | `{h}` | `{rel}` |" for b, h, rel in rows if rel == m.group(3)),
                               m.group(0)), md)
    note = [f"\n---\n\n## AMENDMENT — {args.date or 'undated'}: hashes refreshed after a deliberate change\n",
            f"**Reason:** {args.reason}\n",
            "The table above now holds the CURRENT hashes. The files below changed; their previous hashes "
            "are recorded here so a `sha256sum -c` failure against an older copy can still be resolved.\n"]
    if args.backup:
        note.append(f"**Pre-change copies:** `{args.backup}` (verified against the OLD hashes below "
                    f"before the change was made).\n")
    note.append("\n| file | old sha256 | new sha256 |\n|---|---|---|\n")
    for rel, oh, nh in moved:
        note.append(f"| `{rel}` | `{oh}` | `{nh}` |\n")
    MD.write_text(md + "".join(note))
    print(f"\n  wrote {SHA.name} and appended an amendment to {MD.name}")


if __name__ == "__main__":
    main()
