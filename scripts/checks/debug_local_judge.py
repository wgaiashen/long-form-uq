"""Show a LOCAL judge's RAW reply (before parsing) on the worst disagreements, to tell
whether a low agreement is the judge genuinely scoring differently, or our prompt/parser
mishandling the model's text. Needs the GPU (loads the model) + a prior judge_agreement run.

    python scripts/checks/debug_local_judge.py --dataset pubmed_qa --judge local:Qwen/Qwen2.5-7B-Instruct --top 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.labels.llm_judge import build_prompt, parse_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pubmed_qa")
    ap.add_argument("--judge", required=True, help="the local:MODEL string from judge_agreement")
    ap.add_argument("--top", type=int, default=10, help="how many worst disagreements to show")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    args = ap.parse_args()

    backend, _, model_name = args.judge.partition(":")
    if backend != "local":
        sys.exit("this debugger is for local judges (it shows raw model text)")

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)

    tag = args.judge.replace("/", "_").replace(":", "_")
    npz_path = Path(cfg.cache_dir) / "judge_agreement" / f"{key}__{tag}.npz"
    if not npz_path.exists():
        sys.exit(f"no saved scores at {npz_path} — run judge_agreement.py first")
    saved = np.load(npz_path)
    idx, gpt5, cand = saved["idx"], saved["gpt5"], saved["candidate"]

    records = cache.load_records(cfg.cache_dir, key)
    labelled = [r for r in records if isinstance(r.get("correctness"), (int, float))]

    diff = np.abs(gpt5 - cand)
    diff[np.isnan(cand)] = -1.0
    order = np.argsort(-diff)[:args.top]

    # Load the model and re-run it on these records, capturing the RAW reply.
    from luq.labels.local_judge import LocalJudge
    lj = LocalJudge(model_name)

    print(f"=== raw replies on worst disagreements: {args.dataset}, {args.judge} ===")
    for rank, k in enumerate(order, 1):
        r = labelled[int(idx[k])]
        prompt = build_prompt(r, args.dataset)
        raw = lj._reply(prompt)            # the model's text BEFORE parsing
        reparsed = parse_score(raw)        # what parse_score makes of it
        print(f"\n[{rank}] GPT-5={gpt5[k]:.2f}  saved_cand={cand[k]:.2f}  reparsed={reparsed}")
        print(f"  RAW REPLY   : {raw!r}")
        print(f"  gold answer : {str(r['target'])[:160]}")
        print(f"  model answer: {r['gen_text'][:160]}")
        print(f"  prompt tail : ...{prompt[-120:]!r}")


if __name__ == "__main__":
    main()
