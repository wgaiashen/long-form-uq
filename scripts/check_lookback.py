"""Scratch one-example check for the Lookback Lens feature (safe to delete).

Mirrors what 01c does: load the model eager + fp32, run lookback_vector on the first
cached sciq ID record, print the shape and value range. The ratio must be in [0, 1].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from luq import cache, generate
from luq.config import Config
from luq.features import lookback

cfg = Config(dataset="sciq", ood_setting="ID")
key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
recs = cache.load_records(cfg.cache_dir, key)
print(f"loaded {len(recs)} records")

model, tok = generate.load_model(cfg.model_name, attn_implementation="eager", dtype=torch.float32)
v = lookback.lookback_vector(model, tok, recs[0])
n_expected = model.config.num_hidden_layers * model.config.num_attention_heads
print(f"shape {v.shape} | min {v.min():.3f} max {v.max():.3f} mean {v.mean():.3f}")
print("OK" if v.shape == (n_expected,) and 0.0 <= v.min() and v.max() <= 1.0 + 1e-6
      else "UNEXPECTED")
