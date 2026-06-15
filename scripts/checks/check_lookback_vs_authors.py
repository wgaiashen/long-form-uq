"""Verify our Lookback feature against the AUTHORS' code (lookback-src/step01_extract_attns.py).

Mirrors their lines 253-255 on our single teacher-forced forward pass:
    attn_on_context = attn[..., -1, :ctx].mean(-1)
    attn_on_new     = attn[..., -1, ctx:].mean(-1)
    ratio           = attn_on_context / (attn_on_context + attn_on_new)   # context-first
The authors attribute each generated token's ratio to the row of the token that PREDICTS it
(the previous position), with a plain .mean over the new-key slice (self of that predicting
token included). We then mean-pool over the generated tokens (our response-level adaptation).

Reports our lookback_vector vs this authors-reference, plus two own-row candidates for
diagnosis, so the NUMBERS decide which formulation is faithful (no prose). Safe to delete.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch

from luq import cache, generate
from luq.config import Config
from luq.features import lookback

dataset = sys.argv[1] if len(sys.argv) > 1 else "sciq"
cfg = Config(dataset=dataset, ood_setting="ID")
key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
r = cache.load_records(cfg.cache_dir, key)[0]
ctx = len(r["prompt_token_ids"])
n_out = len(r["gen_token_ids"])
full = r["prompt_token_ids"] + r["gen_token_ids"]
seq = len(full)
print(f"{dataset}: ctx={ctx} n_out={n_out} seq={seq}")

model, tok = generate.load_model(cfg.model_name, attn_implementation="eager", dtype=torch.float32)
ids = torch.tensor(full)[None].to(model.device)
with torch.no_grad():
    out = model(ids, output_attentions=True)
A = torch.stack([a[0].cpu() for a in out.attentions])   # (L,H,seq,seq)


def ratio_at(q, new_hi):
    """Context-first ratio from attention row q; new slice = keys [ctx, new_hi)."""
    a = A[:, :, q, :]
    mc = a[:, :, :ctx].mean(-1)
    mn = a[:, :, ctx:new_hi].mean(-1)
    return (mc / (mc + mn)).reshape(-1)


# AUTHORS' reference: predicting row q=ctx+i-1, new keys [ctx, ctx+i) (their .mean, self incl).
ref = torch.stack([ratio_at(ctx + i - 1, ctx + i) for i in range(1, n_out)]).mean(0).numpy()
# own-row INCLUDING self (the original new-bucket): row ctx+i, new keys [ctx, ctx+i+1).
own_incl = torch.stack([ratio_at(ctx + i, ctx + i + 1) for i in range(0, n_out)]).mean(0).numpy()
# own-row EXCLUDING self (current production): row ctx+i, new keys [ctx, ctx+i).
own_excl = torch.stack([ratio_at(ctx + i, ctx + i) for i in range(1, n_out)]).mean(0).numpy()
our = lookback.lookback_vector(model, tok, r)


def diff(a, b):
    return float(np.max(np.abs(a - b))), float(np.mean(np.abs(a - b)))


print("our      vs authors-ref : max %.2e  mean %.2e" % diff(our, ref))
print("own_excl vs authors-ref : max %.2e  mean %.2e" % diff(own_excl, ref))
print("own_incl vs authors-ref : max %.2e  mean %.2e" % diff(own_incl, ref))
print("our      vs own_excl     : max %.2e  mean %.2e   (sanity: production == own_excl?)"
      % diff(our, own_excl))
print("our range [%.3f, %.3f]" % (our.min(), our.max()))
