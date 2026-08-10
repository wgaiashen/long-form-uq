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
    # ---- multi-GPU, needed only for a big model over a long source -----------------------------
    # The forward holds EVERY block's attention at once: n_layers * n_heads * seq^2 * 4 B in fp32.
    # On Llama-3.1-8B that is 32*32*2048^2*4 = 17.2 GB on top of 32 GB of weights, so 49 GB, and one
    # 80 GB card was always enough. Qwen2.5-14B is 48 layers x 40 heads and 59 GB in fp32, so the
    # same 2048-token sequence costs 48*40*2048^2*4 = 32.2 GB and the total is ~91 GB -- past a
    # single A100-80GB. Only xsum and cnn_dailymail reach that length (max sequence 5964 / 2537, so
    # both saturate max_seq=2048); the other six Qwen sets peak at 1365 tokens and still fit one
    # card. Sharding the WEIGHTS across two cards is the fix that changes no number: max_seq, the
    # dtype, and the ratio itself are all untouched, so the features are the same ones a single
    # 128 GB card would have produced. Defaults are the previous behaviour exactly.
    ap.add_argument("--device-map", default="cuda",
                    help="passed to from_pretrained. 'cuda' (default) = one GPU, unchanged. "
                         "'auto' shards across the visible GPUs -- use with --max-memory.")
    ap.add_argument("--max-memory", default="",
                    help="force a real split, e.g. '0=40GiB,1=40GiB'. accelerate fills GPU 0 "
                         "first, so --device-map auto ALONE can silently put everything on one "
                         "card; when this is set the split is asserted below rather than assumed.")
    args = ap.parse_args()

    max_memory = None
    if args.max_memory:
        max_memory = {}
        for item in args.max_memory.split(","):
            k, v = item.split("=")
            max_memory[int(k.strip())] = v.strip()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    # Attention weights need the eager backend; SDPA (the default) returns none.
    # And eager attention in fp16 overflows to NaN in the softmax, so use fp32 here.
    model, tok = generate.load_model(
        cfg.model_name, attn_implementation="eager", dtype=torch.float32,
        device_map=args.device_map, max_memory=max_memory)

    # Prove the shard actually happened. A silent single-card placement would OOM half way through
    # the dataset, after hours of work, rather than here -- and worse, if it happened to survive it
    # would look like a successful run of a configuration we never got.
    if args.device_map == "auto":
        placed = getattr(model, "hf_device_map", {})
        n_dev = len({v for v in placed.values() if isinstance(v, int)})
        print(f"device map: {n_dev} GPU(s) hold weights", flush=True)
        if max_memory is not None and n_dev < 2:
            sys.exit("--max-memory asked for a split but every layer landed on one device; "
                     "the memory ceiling was too high or only one GPU is visible.")

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
