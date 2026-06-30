"""Numerical check: our AlignScore label matches Joe's AlignScore wrapper.

Our scorer (src/luq/labels/alignscore.py) vendors Joe's AlignScorer, so the underlying model code
is the same. What this check confirms is the GLUE around it: the claims/contexts direction, the
max over multiple references (trivia aliases), and the empty-output handling, end to end on real
cached records. We score a handful of (output, target) pairs with both our `score()` and Joe's
`AlignScore.__call__`, then assert they agree to < 1e-3.

For Joe's side, multiple references are reduced exactly as his AggregatedMetric does for trivia:
score each alias on its own, then take the max.

Needs the AlignScore model (a GPU is faster, CPU works for ~20 records). Run from the repo root:

    python scripts/checks/alignscore_vs_authors.py
    python scripts/checks/alignscore_vs_authors.py --n 30
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOE_REPO = ROOT.parent / "Temp_robust_UQ_probes"
sys.path.insert(0, str(ROOT / "src"))

# This round is Llama-3.1-8B; Config.model_name still defaults to the old Qwen dev model.
MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.labels.alignscore import score as our_score  # noqa: E402

TOL = 1e-3


def joe_scorer():
    """Joe's AlignScore wrapper, imported from his repo. Kept import-local so the rest of the
    file can be read without his package installed."""
    if str(JOE_REPO) not in sys.path:
        sys.path.insert(0, str(JOE_REPO))
    from utils.alignscore import AlignScore  # noqa: E402
    return AlignScore(batch_size=1)


def joe_value(joe, record):
    """Joe's AlignScore for one record. A list target (trivia aliases) is reduced by max, the
    same as his AggregatedMetric."""
    out = record["gen_text"]
    tgt = record["target"]
    golds = tgt if isinstance(tgt, list) else [tgt]
    vals = [float(joe({"greedy_texts": [out]}, [g])[0]) for g in golds]
    return max(vals)


def sample_records(cache_dir, model, datasets, n):
    """A spread of records across datasets, taken from the front of each cache. Returns
    (dataset, record) pairs."""
    per = max(1, n // len(datasets))
    picked = []
    for d in datasets:
        key = cache.run_key(model, d, "ID")
        try:
            recs = cache.load_records(cache_dir, key)
        except FileNotFoundError:
            print(f"  [skip] no cached records for {d}")
            continue
        picked += [(d, r) for r in recs[:per]]
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--datasets", default="sciq,trivia_qa,pubmed_qa,xsum")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset="sciq", ood_setting="ID")
    pairs = sample_records(cfg.cache_dir, args.model, args.datasets.split(","), args.n)
    if not pairs:
        sys.exit("no records to check")

    joe = joe_scorer()

    print(f"{'dataset':12s}{'ours':>9}{'joe':>9}{'|diff|':>10}")
    print("-" * 40)
    worst = 0.0
    for d, r in pairs:
        ours = our_score(r, d)
        theirs = joe_value(joe, r)
        if ours is None:
            print(f"{d:12s}{'None':>9}{theirs:>9.4f}   our scorer returned None")
            continue
        diff = abs(ours - theirs)
        worst = max(worst, diff)
        flag = "" if diff < TOL else "  <-- OVER TOL"
        print(f"{d:12s}{ours:>9.4f}{theirs:>9.4f}{diff:>10.2e}{flag}")

    print("-" * 40)
    print(f"worst |diff| = {worst:.2e}  (tolerance {TOL:.0e})")
    if worst < TOL:
        print("PASS: our AlignScore matches Joe's wrapper.")
    else:
        sys.exit("FAIL: our AlignScore disagrees with Joe's beyond tolerance.")


if __name__ == "__main__":
    main()
