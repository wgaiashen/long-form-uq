"""Score a method: write the results CSV and report PRR.

    python scripts/04_eval.py --dataset sciq --ood ID

MSP comes free here too: it needs no probe, just the cached token logprobs, so it is
the natural unsupervised anchor to print alongside SAPLMA.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, data, msp, results  # noqa: E402
from luq.config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    # TODO:
    #   - Build MSP scores: [msp.msp_uncertainty(r["token_logprobs"]) for r in records].
    #   - Gather correctness from the labelled records (see scripts/02_label.py).
    #   - Build CSV rows (dataset, task=data.TASK_OF[ds], split, correctness, msp) and
    #     results.write_csv(...). Print results.prr(correctness, msp).
    #   - Do the same for the SAPLMA uncertainty produced by scripts/03_probe.py.
    raise NotImplementedError("assemble CSV + PRR — follow the TODO above")


if __name__ == "__main__":
    main()
