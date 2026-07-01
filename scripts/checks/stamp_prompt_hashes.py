"""One-off: stamp a prompt-content hash onto every existing record cache.

The cache key is (model, dataset, ood) with no content hash, so before two prompt regimes
go live we lock each existing (frozen) cache to its own prompts. 01_extract refuses to
extend a cache whose stored hash differs from the current prompts, so this stamp is what
makes a "forgot --prompt-regime" run into the frozen namespace fail loudly instead of
silently appending new-prompt records.

The hash is computed from the cached records themselves (their prompt + target fields, in
stored order), so it reflects exactly what is on disk and needs no ProbeDrift load. Run
once, before installing the updated ProbeDrift. Idempotent: skips a key that already has a
sidecar (use --force to overwrite).

    python scripts/checks/stamp_prompt_hashes.py
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache  # noqa: E402
from luq.config import CACHE_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(CACHE_DIR),
                    help="namespace to stamp (default = the frozen original cache)")
    ap.add_argument("--force", action="store_true", help="overwrite existing sidecars")
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)

    rec_dir = cache_dir / "records"
    for f in sorted(rec_dir.glob("*.jsonl")):
        key = f.stem  # filename without .jsonl == the run key
        existing = cache.load_prompt_hash(cache_dir, key)
        if existing is not None and not args.force:
            print(f"skip  {key}  (already {existing[:12]})")
            continue
        records = cache.load_records(cache_dir, key)
        digest = cache.prompt_hash([r["prompt"] for r in records],
                                   [r.get("target") for r in records])
        cache.save_prompt_hash(digest, cache_dir, key)
        print(f"stamp {key}  {digest[:12]}  ({len(records)} records)")


if __name__ == "__main__":
    main()
