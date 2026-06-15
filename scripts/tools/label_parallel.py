"""Parallel long-form judge labelling (speed tool). Same labels as 02_label, concurrent.

The GPT-5 judge is I/O-bound (each call waits seconds on the API), so issuing many at once
gives a near-linear speedup. Resumable (skips already-scored records) and checkpointed, exactly
like 02_label; the only difference is a thread pool. Records are judged independently, so the
labels are identical to the sequential run -- order does not affect any label.

    export OPENAI_API_KEY=...
    python scripts/label_parallel.py --dataset xsum --ood ID --workers 16
"""
import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache, data  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.labels import llm_judge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xsum")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if args.dataset in data.SHORT_FORM:
        sys.exit("label_parallel is for long-form (judge) datasets only")

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)
    todo = [r for r in records if r.get("correctness") is None]
    print(f"{len(records)} records, {len(todo)} to judge, {args.workers} workers", flush=True)

    lock = threading.Lock()
    progress = {"done": 0, "failed": 0}

    def work(r):
        # judge() retries internally; wrap so one record's failure never kills the pool
        # (a None score is recorded and picked up by a later resume).
        try:
            score = llm_judge.judge(r, cfg.dataset)
        except Exception:
            score = None
        with lock:
            r["correctness"] = score
            progress["done"] += 1
            if score is None:
                progress["failed"] += 1
            if progress["done"] % 50 == 0:
                cache.save_records(records, cfg.cache_dir, key)  # checkpoint
                print(f"judged {progress['done']}/{len(todo)} "
                      f"(failed {progress['failed']})", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    cache.save_records(records, cfg.cache_dir, key)  # final save
    scores = [r["correctness"] for r in records if r["correctness"] is not None]
    mean = sum(scores) / len(scores) if scores else float("nan")
    print(f"DONE. labelled {len(scores)}/{len(records)}, failed {progress['failed']}, "
          f"mean correctness {mean:.3f}", flush=True)


if __name__ == "__main__":
    main()
