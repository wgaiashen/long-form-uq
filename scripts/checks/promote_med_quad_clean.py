"""Promote med_quad's clean answer-span label to the canonical `correctness` the ladder reads.

Background: `relabel_med_quad_clean.py` judged the CLEAN answer span into `correctness_clean` because
med_quad's raw labels are noisy (base-Llama's trailing junk made the judge disagree with itself --
cheap-vs-GPT-5 spearman 0.26 raw -> 0.81 clean). This makes the clean label the one the SameTask rung
trains on.

Non-destructive: the pre-clean label is preserved as `correctness_raw` (+ `correctness_raw_model`), so
the promotion is fully reversible. After this, `load_per_token(..., "correctness")` returns the clean
label with no cache re-extraction (y is read live from the records), so the ladder just needs a re-run.

    python scripts/checks/promote_med_quad_clean.py            # promote
    python scripts/checks/promote_med_quad_clean.py --revert   # undo (restore raw)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
DATASET = "med_quad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true", help="restore the raw label from correctness_raw")
    args = ap.parse_args()
    cfg = Config(model_name=MODEL, dataset=DATASET, ood_setting="ID")
    key = cache.run_key(MODEL, DATASET, "ID")
    recs = cache.load_records(cfg.cache_dir, key)

    if args.revert:
        miss = sum(1 for r in recs if r.get("correctness_raw") is None)
        if miss:
            sys.exit(f"{miss}/{len(recs)} rows have no correctness_raw backup -- nothing to revert to")
        for r in recs:
            r["correctness"] = r["correctness_raw"]
            r["correctness_model"] = r.get("correctness_raw_model")
        cache.save_records(recs, cfg.cache_dir, key)
        print(f"reverted med_quad correctness -> raw ({len(recs)} rows)")
        return

    # guard: every row must have a numeric clean label before we promote
    miss = sum(1 for r in recs if not isinstance(r.get("correctness_clean"), (int, float)))
    if miss:
        sys.exit(f"{miss}/{len(recs)} rows lack correctness_clean -- run relabel_med_quad_clean.py first")
    already = all(r.get("correctness_raw") is not None for r in recs)
    if already:
        sys.exit("correctness_raw already exists -- med_quad looks already promoted; use --revert to undo")

    raw = np.array([r["correctness"] for r in recs], float)
    clean = np.array([r["correctness_clean"] for r in recs], float)
    for r in recs:
        r["correctness_raw"] = r["correctness"]                 # backup the pre-clean label
        r["correctness_raw_model"] = r.get("correctness_model")
        r["correctness"] = r["correctness_clean"]               # canonical <- clean
        r["correctness_model"] = "gpt-5-mini"                   # clean stamp (all rows are gpt-5-mini)
    cache.save_records(recs, cfg.cache_dir, key)
    print(f"promoted med_quad: canonical correctness <- correctness_clean ({len(recs)} rows)")
    print(f"  mean raw={raw.mean():.3f} -> clean(canonical)={clean.mean():.3f}  (raw kept as correctness_raw)")
    print("  the ladder reads y live from records -> re-run contribution_ladder to pick this up (no re-extract).")


if __name__ == "__main__":
    main()
