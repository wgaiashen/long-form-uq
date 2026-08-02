"""Emit a machine-diffable inventory of every cache artifact on THIS machine.

WHY A SCRIPT RATHER THAN AN AD-HOC `find`. The point of the inventory is to diff DoC against RCS and
see what exists on only one of them. Two tables built by two different ad-hoc commands cannot be
diffed: the column order, the rounding, the path prefixes and the sort order all differ, and the diff
fills with noise that hides the handful of rows that matter. Both machines run THIS, and the outputs
are directly comparable.

TWO THINGS IT DOES THAT A PLAIN `find cache/` MISSES, both of which bit us on DoC:

  - SYMLINKS OUT OF THE TREE. cache/pertok/ on DoC contains only symlinks into /vol/bitbucket/gs925,
    a scratch volume outside the CephFS quota. `find cache -type f` returns NOTHING for them, so a
    naive inventory reports the pertok caches as absent when they are present, and would have caused
    a transfer that silently skipped ~12 GB. Symlinks are resolved and reported with their target.
  - CACHES WITH NO LINK AT ALL. 6.9 GB of expertqa pertok and 5.7 GB of staged features sit on that
    same volume with nothing pointing at them from cache/. Pass --extra to sweep those roots too.

SIZE AND HASH. Both, always. A size difference localises a truncated transfer immediately; a hash
difference alone does not distinguish truncation from corruption. --quick skips hashing for a fast
look, and marks the column "-" rather than leaving it blank, so a quick run can never be mistaken
for a verified one.

    python scripts/checks/cache_inventory.py --out inventory_doc.tsv \
        --extra /vol/bitbucket/gs925
    python scripts/checks/cache_inventory.py --out inventory_rcs.tsv        # on RCS
    # then, on either machine:  diff <(cut -f2- inventory_doc.tsv) <(cut -f2- inventory_rcs.tsv)

Read-only. Never writes into, moves or deletes a cache.
"""
import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Directories that are downloaded rather than produced: regenerable from the internet, enormous, and
# nothing we would ever transfer between clusters. Skipped by default so the table stays about OUR data.
SKIP_DIRS = {"hf_cache", ".git", "__pycache__"}

DATASET_RE = re.compile(r"__([a-z_0-9]+)__(ID|OOD_[A-Z_]+|pilot)")


def sha256_of(path: Path, quick: bool) -> str:
    if quick:
        return "-"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def classify(path: Path, base: Path):
    """(regime, kind, dataset) for one file, from its path and filename.

    A regime is the cache-namespace directory (cache/<regime>/...); files directly under cache/ are
    the default v1 namespace. `kind` is records/features/pertok/probes/viz/... i.e. the tier.
    """
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = Path(path.name)
    parts = rel.parts
    known = {"records", "features", "pertok", "probes", "viz", "scores", "meta", "sar",
             "judge_agreement"}
    if len(parts) >= 2 and parts[0] in known:
        regime, kind = "(default)", parts[0]
    elif len(parts) >= 2:
        regime, kind = parts[0], (parts[1] if len(parts) > 2 else "-")
    else:
        regime, kind = "(default)", "-"
    m = DATASET_RE.search(path.name)
    return regime, kind, (m.group(1) if m else "-")


def walk(base: Path, quick: bool, label: str, seen: dict, rows: list):
    """Add a row per file under `base`, resolving symlinks and recording their target.

    ⚠️ DEDUPLICATION IS LOAD-BEARING, not tidiness. On DoC the same bytes are reachable twice: once
    through cache/pertok/<name> (a symlink) and again through the --extra sweep of the volume the
    symlink points into. Counting both double-counts ~12 GB, inflates the total by a quarter, and
    puts two rows for one file into a table whose whole purpose is a row-by-row diff. So a file is
    recorded ONCE, keyed by its resolved real path, and any further way of reaching it is appended
    to that row's alias column instead of becoming a new row.
    """
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                resolved = p.resolve()
                if not resolved.exists():
                    rows.append([label, *classify(p, base), 0, "DANGLING", str(p), str(resolved)])
                    continue
                p_read = resolved
            else:
                p_read = p.resolve()
            real = str(p_read)
            if real in seen:
                # Same bytes, another access path. Note the alias on the existing row; do not re-add.
                # Sweeping the volume a symlink points into re-finds the file under its OWN real
                # path, which the row already names -- saying it twice adds nothing.
                row = rows[seen[real]]
                if str(p) != real:
                    row[7] = (row[7] + " | " if row[7] else "") + f"also at {p}"
                continue
            try:
                size = p_read.stat().st_size
                digest = sha256_of(p_read, quick)
            except OSError as e:
                rows.append([label, *classify(p, base), 0, f"ERR:{type(e).__name__}", str(p), ""])
                continue
            seen[real] = len(rows)
            alias = f"-> {real}" if str(p) != real else ""
            rows.append([label, *classify(p, base), size, digest, str(p), alias])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional roots to sweep, e.g. an off-quota scratch volume")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="skip hashing (hash column becomes '-')")
    args = ap.parse_args()

    rows, seen = [], {}
    base = Path(args.cache)
    if base.exists():
        walk(base, args.quick, "cache", seen, rows)
    else:
        print(f"note: {base} does not exist here", file=sys.stderr)
    for x in args.extra:
        p = Path(x)
        if p.exists():
            walk(p, args.quick, p.name, seen, rows)
        else:
            print(f"note: extra root {p} does not exist here", file=sys.stderr)

    # Sort by the identifying columns, NOT by path: the two machines have different absolute path
    # prefixes, so sorting by path would order the two tables differently and make the diff useless.
    rows.sort(key=lambda r: (r[1], r[2], r[3], Path(r[5]).name))

    header = "location\tregime\tkind\tdataset\tbytes\tsha256\tpath\tsymlink_target"
    lines = [header] + ["\t".join(str(c) for c in r) for r in rows]
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}: {len(rows)} files")
    else:
        print(text)

    by = {}
    for r in rows:
        k = (r[0], r[1], r[2])
        n, b = by.get(k, (0, 0))
        by[k] = (n + 1, b + (r[4] if isinstance(r[4], int) else 0))
    print("\nlocation / regime / kind                       files        size", file=sys.stderr)
    for k in sorted(by):
        n, b = by[k]
        print(f"  {k[0]:<10s} {k[1]:<22s} {k[2]:<12s} {n:>5d}  {b/1073741824:>8.2f} GB",
              file=sys.stderr)
    tot = sum(v[1] for v in by.values())
    print(f"  {'TOTAL':<46s} {len(rows):>5d}  {tot/1073741824:>8.2f} GB", file=sys.stderr)
    dang = [r for r in rows if r[5] == "DANGLING"]
    if dang:
        print(f"\n⚠️ {len(dang)} DANGLING symlink(s) -- the cache LOOKS present but will fail at load:",
              file=sys.stderr)
        for r in dang:
            print(f"    {r[6]} -> {r[7]}", file=sys.stderr)


if __name__ == "__main__":
    main()
