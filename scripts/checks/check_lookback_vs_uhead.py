"""Numerical cross-check: our Lookback feature vs uhead's FeatureExtractorLookbackLens.

Both run on the SAME attention tensors (one forward pass on one cached record), so any
difference is purely the formula, not the attention. We report:
  (1) uhead's reference per-token ratios vs an exact reimplementation of uhead's formula
      -> confirms we understand uhead's extractor (should be ~0).
  (2) uhead's per-token ratios vs OUR production formula, at the overlapping positions
      -> shows the effect of our documented choices (divisor; edge tokens).
  (3) the response-level vectors (uhead mean vs our lookback_vector) -> practical impact.

Safe to delete.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch

from luq import cache, generate
from luq.config import Config
from luq.features import lookback
from luh.feature_extractors.lookback_lens import FeatureExtractorLookbackLens

dataset = sys.argv[1] if len(sys.argv) > 1 else "sciq"
cfg = Config(dataset=dataset, ood_setting="ID")
key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
r = cache.load_records(cfg.cache_dir, key)[0]
ctx_len = len(r["prompt_token_ids"])
n_out = len(r["gen_token_ids"])
full_ids = r["prompt_token_ids"] + r["gen_token_ids"]
seq_len = len(full_ids)
print(f"ctx_len={ctx_len} n_out={n_out} seq_len={seq_len}")

model, tok = generate.load_model(cfg.model_name, attn_implementation="eager", dtype=torch.float32)
L = model.config.num_hidden_layers
H = model.config.num_attention_heads

ids = torch.tensor(full_ids)[None].to(model.device)
with torch.no_grad():
    out = model(ids, output_attentions=True)
A = torch.stack([a[0].detach().cpu() for a in out.attentions])  # (L, H, seq, seq)

# ---- (A) uhead's reference extractor (training mode = single forward) ----
ext = FeatureExtractorLookbackLens.__new__(FeatureExtractorLookbackLens)  # bypass gemma tokenizer
ext._n_layers, ext._n_heads, ext._input_size = L, H, L * H


class _O:  # minimal llm_outputs: has .attentions + .context_lengths, NO .sequences
    pass


lo = _O()
lo.attentions = tuple(a.detach().clone().cpu() for a in out.attentions)  # cloned (extractor mutates)
lo.context_lengths = [ctx_len]
li = {"attention_mask": torch.ones(1, seq_len, dtype=torch.long)}
uhead_pp = ext(li, lo)[0]  # (seq_len-1, L*H): per-position ratios, abs position = index
print("uhead per-position output:", tuple(uhead_pp.shape))


# ---- (B) exact reimplementation of uhead's formula (to prove we read it right) ----
def uhead_formula(p):  # absolute position p, real ratio per uhead (p > ctx_len)
    row = A[:, :, p, :]                       # (L,H,seq)
    ctx = row[:, :, :ctx_len].sum(-1)         # (L,H)
    new = row[:, :, ctx_len:].sum(-1)         # new_mask = positions >= ctx_len
    mean_ctx = ctx / ctx_len
    mean_new = new / (p - ctx_len)            # uhead divisor = seq_idx - ctx_bound
    return (mean_new / (mean_ctx + mean_new)).reshape(-1)


overlap = list(range(ctx_len + 1, seq_len - 1))  # uhead computes real ratios here AND drops last token
ours_formula = uhead_formula  # alias for clarity
mismatch_AB = max(float((uhead_pp[p] - uhead_formula(p)).abs().max()) for p in overlap)
print(f"(1) uhead extractor vs our reimpl of uhead's formula: max abs diff = {mismatch_AB:.2e}")

# ---- (C) our PRODUCTION formula, per-position ----
rows = A[:, :, ctx_len:, :]
ctx_sum = rows[:, :, :, :ctx_len].sum(-1)
new_sum = rows[:, :, :, ctx_len:].sum(-1)
n_new = torch.arange(1, n_out + 1, dtype=A.dtype).view(1, 1, -1)
mean_ctx = ctx_sum / max(ctx_len, 1)
mean_new = new_sum / n_new
ours_ratio = mean_new / (mean_ctx + mean_new + 1e-8)       # (L,H,n_out)
ours_pp = ours_ratio.permute(2, 0, 1).reshape(n_out, -1)  # (n_out, L*H), abs pos = ctx_len + i

diff_ours_uhead = max(
    float((ours_pp[p - ctx_len] - uhead_pp[p]).abs().max()) for p in overlap)
print(f"(2) our production formula vs uhead, overlapping positions: max abs diff = {diff_ours_uhead:.2e}")

# ---- (D) response-level vectors ----
our_vec = lookback.lookback_vector(model, tok, r)             # our shipped vector
uhead_vec = uhead_pp[ctx_len + 1: seq_len - 1].mean(0).numpy()  # mean over uhead's real-ratio positions
print(f"(3) response-level: max abs diff = {float(np.max(np.abs(our_vec - uhead_vec))):.3e}"
      f"  mean abs diff = {float(np.mean(np.abs(our_vec - uhead_vec))):.3e}")
print(f"    our_vec[:4] = {np.round(our_vec[:4], 4)}")
print(f"    uhead_vec[:4]= {np.round(uhead_vec[:4], 4)}")
# sanity: our lookback_vector == mean of our per-position ratios
print(f"    lookback_vector == mean(our_pp)? max diff = "
      f"{float(np.max(np.abs(our_vec - ours_pp.mean(0).numpy()))):.2e}")
