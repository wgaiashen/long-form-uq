"""GPU step: Lookback Lens attention features from the cached records.

Like 01b, this replays the Tier-1 token IDs (no regeneration). It needs ATTENTION
weights, so it loads the model with the eager backend. Run AFTER 01_extract.

    python scripts/01c_lookback.py --dataset sciq --ood ID

Saves a Tier-2 feature file keyed method="lookback", shape (n, 1, n_layers*n_heads):
one combined attention-ratio vector per example (Lookback Lens uses all layers at
once), stored as a single "layer" so 03_probe.py --method lookback treats it like the
other methods (it will report "layer 0", which here just means the whole vector).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make `src/` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, generate  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import lookback  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag (must match the one used by 01_extract).")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    # Attention weights need the eager backend; SDPA (the default) returns none.
    # And eager attention in fp16 overflows to NaN in the softmax, so use fp32 here.
    model, tok = generate.load_model(
        cfg.model_name, attn_implementation="eager", dtype=torch.float32)

    # Process every record in order so the feature rows stay aligned with the records
    # (03_probe asserts equal length). A small dev run is controlled upstream by
    # 01_extract --limit, which makes this records file small.
    vectors = []
    for i, r in enumerate(records):
        vectors.append(lookback.lookback_vector(model, tok, r))
        if i % 10 == 0:
            print(f"{i}/{len(records)} done", flush=True)

    feats = np.stack(vectors)        # (n, n_layers*n_heads)
    feats = feats[:, None, :]        # (n, 1, n_layers*n_heads): a single "layer"
    path = cache.save_features(feats, cfg.cache_dir, key, method="lookback")
    print(f"saved features {feats.shape} -> {path}")


if __name__ == "__main__":
    main()
