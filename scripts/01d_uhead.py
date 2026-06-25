"""GPU step: UHead baseline scores from the cached records (no probe training).

Replays the Tier-1 token IDs through the base model + the pretrained uhead head and
writes ONE uncertainty score per TEST example, exactly like 03_probe does for the
supervised probes, so 04_eval picks it up as another column. Run AFTER 01_extract;
labels (02_label) are NOT required for this step, only for the PRR in 04_eval.

    python scripts/01d_uhead.py --dataset sciq --ood ID --model google/gemma-2-9b-it

Needs attention weights, so it loads the model with the eager backend. The head is
Gemma-specific (llm-uncertainty-head/uhead_gemma-2-9b-it), so this only makes sense on
google/gemma-2-9b-it. Heavy: all-layer attentions in memory -> use a40/a100.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Make `src/` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, generate  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import uhead  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    # The head reads attention weights, so eager backend (SDPA returns none). Leave
    # dtype to auto so Gemma loads in bf16 (its required dtype); bf16 eager attention
    # is numerically safe here, unlike fp16 (which is why Lookback forced fp32).
    model, tok = generate.load_model(cfg.model_name, attn_implementation="eager")
    head = uhead.load_uhead(model)

    # Score the TEST split only, in test-record order: that is what 04_eval expects
    # from a supervised method (it asserts len == number of test records).
    test_positions = [i for i, r in enumerate(records) if r["split"] == "test"]
    unc = []
    for n, i in enumerate(test_positions):
        unc.append(uhead.uhead_instance_score(model, tok, head, records[i]))
        if n % 10 == 0:
            print(f"{n}/{len(test_positions)} done", flush=True)

    unc = np.asarray(unc, dtype=np.float32)
    # layer=-1 is a sentinel: uhead reads many layers through its own head, so there is
    # no single layer choice the way SAPLMA/P(True) have one.
    path = cache.save_scores(unc, cfg.cache_dir, key, method="uhead", layer=-1)
    print(f"saved {len(unc)} uhead scores -> {path}")


if __name__ == "__main__":
    main()
