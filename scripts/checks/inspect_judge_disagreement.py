"""Eyeball the biggest GPT-5-vs-candidate disagreements from a judge_agreement run.

Read-only, no API calls: it reloads the scores judge_agreement.py saved (the .npz) and,
for the records where the two judges disagree most, prints what each judge actually saw
(the trimmed question/context, the gold answer, and the model answer it scored). Use it
to decide WHO is right on a disagreement — is GPT-5 correctly catching a wrong answer
(so the candidate failed), or is GPT-5 being harsh (which would change our premise)?

    python scripts/checks/inspect_judge_disagreement.py --dataset pubmed_qa --judge openai:gpt-5-mini
    python scripts/checks/inspect_judge_disagreement.py --dataset pubmed_qa --judge openai:gpt-5-mini --top 8
"""
import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.data import JUDGE_NAME_MAP  # noqa: E402
from luq.labels.llm_judge import _extract_question  # noqa: E402


def _wrap(text, width=100, limit=900):
    """Wrap text for the terminal and clip very long context so the screen stays readable."""
    text = (text or "").strip()
    if len(text) > limit:
        text = text[:limit] + " …[clipped]"
    return textwrap.fill(text, width=width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pubmed_qa")
    ap.add_argument("--judge", required=True,
                    help="the same backend:model string you passed to judge_agreement.py")
    ap.add_argument("--top", type=int, default=8, help="how many worst disagreements to show")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)

    # Reload the saved scores (idx are positions into the LABELLED record list, the same
    # filtered order judge_agreement.py used — rebuild it the identical way below).
    tag = args.judge.replace("/", "_").replace(":", "_")
    npz_path = Path(cfg.cache_dir) / "judge_agreement" / f"{key}__{tag}.npz"
    if not npz_path.exists():
        sys.exit(f"no saved scores at {npz_path} — run judge_agreement.py first")
    saved = np.load(npz_path)
    idx, gpt5, cand = saved["idx"], saved["gpt5"], saved["candidate"]

    records = cache.load_records(cfg.cache_dir, key)
    labelled = [r for r in records if isinstance(r.get("correctness"), (int, float))]
    judge_name = JUDGE_NAME_MAP[args.dataset]

    # Rank by absolute disagreement, ignoring records the candidate failed to score (NaN).
    diff = np.abs(gpt5 - cand)
    diff[np.isnan(cand)] = -1.0          # push failures to the bottom
    order = np.argsort(-diff)[:args.top]

    print(f"=== worst disagreements: {args.dataset} ({args.ood}), {args.judge} "
          f"(top {len(order)}) ===")
    for rank, k in enumerate(order, 1):
        r = labelled[int(idx[k])]
        question, _ = _extract_question(r["prompt"], judge_name)
        print(f"\n[{rank}]  |Δ|={diff[k]:.2f}   GPT-5={gpt5[k]:.2f}   candidate={cand[k]:.2f}")
        print("--- question / context (what the judge saw) ---")
        print(_wrap(question))
        print("--- gold answer ---")
        print(_wrap(str(r["target"])))
        print("--- model answer (the text being scored) ---")
        print(_wrap(r["gen_text"]))


if __name__ == "__main__":
    main()
