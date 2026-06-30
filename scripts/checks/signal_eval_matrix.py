"""Training-signal x eval-metric matrix for the SAPLMA baseline (ID setting).

Reproduces Joe Stacey's "Hidden Failures" training-signal ablation: train the SAPLMA probe on
one label, score it against another. Each cell here is

    train the probe on  <train signal>  ->  evaluate PRR against  <eval metric>

with the train signal and the eval metric each being either the AlignScore label
(`correctness_alignscore`) or the LLM-as-a-judge label (`correctness`). This is exactly the
swap the project already supports: 03_probe picks the training target with --label-field, and
PRR is computed here against whichever eval field we ask for.

How it runs each cell:
  1. call the real scripts/03_probe.py to TRAIN the SAPLMA MLP and write its test scores
     (so the probe is the actual pipeline probe, not a re-implementation),
  2. load those scores and compute PRR with luq.results.prr against each eval field.

We compute PRR in-process rather than shelling out to 04_eval on purpose: 04_eval loads every
cached method's scores and aborts if any are stale (e.g. a pre-relabel ptrue score on pubmed),
which is unrelated to this check. results.prr is the same function 04_eval uses, so the number
is identical.

Defaults sweep layer in {15, 16} (Joe's middle is index 15; our old default was 16) and the
SAPLMA batch size in {1, 32} (Joe fits batch_size=1; our A&M default is 32), so one run shows
which config matches Joe. Narrow with the flags for a quicker pass.

    python scripts/checks/signal_eval_matrix.py
    python scripts/checks/signal_eval_matrix.py --datasets sciq,pubmed_qa --layers 15 --saplma-batches 1

CPU only, no API calls, no GPU: it reuses the cached features and the cached AlignScore/judge
labels. At the end it retrains the canonical config (judge-trained, layer 15, default batch) so
the on-disk SAPLMA scores are left in the keystone state.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache, results  # noqa: E402
from luq.config import Config  # noqa: E402

PROBE_SCRIPT = ROOT / "scripts" / "03_probe.py"

# This round is Llama-3.1-8B. Config.model_name still defaults to the old Qwen dev model, so set
# the model here rather than inheriting that default.
MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"

# Joe's ID targets, keyed by (train signal short, eval metric short) -> per-dataset PRR.
# "align" = correctness_alignscore, "judge" = correctness. None where Joe gives no number.
TARGETS = {
    ("align", "align"): {"sciq": 0.63, "trivia_qa": 0.74, "pubmed_qa": 0.39, "xsum": 0.32},
    ("align", "judge"): {"sciq": 0.69, "trivia_qa": 0.79, "pubmed_qa": 0.36, "xsum": 0.38},
    ("judge", "judge"): {"sciq": 0.91, "trivia_qa": 0.81, "pubmed_qa": 0.69, "xsum": None},
    ("judge", "align"): {"sciq": None, "trivia_qa": None, "pubmed_qa": None, "xsum": None},
}

# Map a record label field to the short name used in TARGETS and the printout.
SHORT = {"correctness_alignscore": "align", "correctness": "judge"}


def run_probe(model, dataset, layer, batch, train_field):
    """Train SAPLMA on `train_field` at `layer` (batch `batch` if not None). Return True on
    success. A failure (e.g. xsum has no judge label, so training on it raises) leaves the cell
    blank rather than killing the sweep."""
    cmd = [sys.executable, str(PROBE_SCRIPT),
           "--model", model, "--dataset", dataset, "--ood", "ID",
           "--method", "saplma", "--layer", str(layer), "--label-field", train_field]
    if batch is not None:
        cmd += ["--saplma-batch", str(batch)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        tail = (res.stderr or res.stdout).strip().splitlines()
        print(f"    [skip] 03_probe failed: {tail[-1] if tail else 'unknown error'}")
        return False
    return True


def prr_against(cache_dir, key, eval_field):
    """PRR of the just-trained SAPLMA scores against `eval_field`. Returns None if the test
    split has no numeric label for that field (e.g. xsum has no judge label yet)."""
    records = cache.load_records(cache_dir, key)
    test_pos = [i for i, r in enumerate(records) if r["split"] == "test"]
    if not all(isinstance(records[i].get(eval_field), (int, float)) for i in test_pos):
        return None
    unc = cache.load_scores(cache_dir, key, method="saplma")["unc"]
    y_test = [records[i][eval_field] for i in test_pos]
    return results.prr(y_test, unc)


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--datasets", default="sciq,trivia_qa,pubmed_qa,xsum")
    ap.add_argument("--layers", default="15,16")
    ap.add_argument("--saplma-batches", default="1,32",
                    help="comma list; '1' matches Joe, '32' is our A&M default")
    ap.add_argument("--train-signals", default="correctness_alignscore,correctness")
    ap.add_argument("--eval-fields", default="correctness_alignscore,correctness")
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    layers = [int(x) for x in args.layers.split(",")]
    batches = [int(x) for x in args.saplma_batches.split(",")]
    train_signals = args.train_signals.split(",")
    eval_fields = args.eval_fields.split(",")

    cfg = Config(model_name=args.model, dataset="sciq", ood_setting="ID")
    cache_dir = cfg.cache_dir

    rows = []  # (dataset, layer, batch, train_short, eval_short, prr, target)
    for dataset in datasets:
        key = cache.run_key(args.model, dataset, "ID")
        print(f"\n==== {dataset} ====")
        for layer in layers:
            for batch in batches:
                for train_field in train_signals:
                    tag = f"L{layer} batch{batch} train={SHORT.get(train_field, train_field)}"
                    print(f"  {tag}")
                    if not run_probe(args.model, dataset, layer, batch, train_field):
                        continue
                    for eval_field in eval_fields:
                        prr = prr_against(cache_dir, key, eval_field)
                        ts, es = SHORT.get(train_field), SHORT.get(eval_field)
                        target = TARGETS.get((ts, es), {}).get(dataset)
                        rows.append((dataset, layer, batch, ts, es, prr, target))
                        delta = "" if (prr is None or target is None) else f"  Δ {prr - target:+.3f}"
                        joe = "n/a" if target is None else f"{target:.2f}"
                        print(f"      eval={es:5s}  ours {fmt(prr)}   Joe {joe}{delta}")

    # Summary table, grouped by (train, eval) row of Joe's ablation.
    print("\n\n================ SUMMARY (PRR, ID) ================")
    header = f"{'train->eval':14s}{'dataset':12s}{'layer':>6}{'batch':>6}{'ours':>8}{'Joe':>7}{'Δ':>8}"
    print(header)
    print("-" * len(header))
    for ts, es in [("align", "align"), ("align", "judge"), ("judge", "judge"), ("judge", "align")]:
        for (d, layer, batch, rts, res_, prr, target) in rows:
            if (rts, res_) != (ts, es):
                continue
            delta = "" if (prr is None or target is None) else f"{prr - target:+.3f}"
            joe = "n/a" if target is None else f"{target:.2f}"
            print(f"{ts + '->' + es:14s}{d:12s}{layer:>6}{batch:>6}{fmt(prr):>8}{joe:>7}{delta:>8}")

    # Leave the cache in the keystone state: judge-trained, layer 15, default batch.
    print("\nrestoring canonical SAPLMA scores (judge-trained, layer 15, default batch)...")
    for dataset in datasets:
        run_probe(args.model, dataset, 15, None, "correctness")
    print("done.")


if __name__ == "__main__":
    main()
