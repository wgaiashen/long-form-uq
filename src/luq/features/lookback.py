"""Lookback Lens feature (Chuang et al.): the attention lookback ratio.

A faithfulness signal read from attention, not from hidden states. For each generated
token, and each (layer, head), split that token's attention into mass on the CONTEXT
(the prompt) versus the PRIOR generated tokens. The lookback ratio (Chuang et al.) is

    mean_ctx / (mean_ctx + mean_new)

where each side is a plain MEAN: context attention over the N context tokens, "new" attention
over the generated tokens the row attends to, the token's own self-attention included (matching
the authors' `.mean` over the [ctx:] attention slice). Intuition: a token that keeps "looking
back" at the context has a HIGH ratio (well grounded); one that mostly attends to its own output
has a LOW ratio (more prone to drift / hallucinate).

Two differences from SAPLMA:
  * It needs attention weights, so the model must be loaded with
    attn_implementation="eager" (the default SDPA backend returns none).
  * Lookback Lens uses ALL layers x heads together as one vector, so there is no
    single layer to choose. We return one combined (n_layers*n_heads,) vector; the
    01c script stores it as a 1-"layer" feature so 03_probe trains on the whole thing.

Matches the authors' released code (Chuang et al., lookback-src/step01_extract_attns.py):
context/(ctx+new), where "new" is a plain MEAN over the row's new-token attention slice (the
token's own self-attention included). Verified to ~1e-6 against that code in
scripts/checks/check_lookback_vs_authors.py. Note that some re-implementations orient the ratio the
other way (new/(ctx+new) = 1 - authors); we follow the authors. Response-level adaptation: we mean-pool the
per-token ratio over the generated tokens (the paper scores spans / a sliding window), dropping
the last token to align with the authors' predicting-token indexing.
"""
import numpy as np
import torch

from .. import generate


def lookback_vector(model, tok, record: dict, max_seq: int = 2048) -> np.ndarray:
    """Return (n_layers*n_heads,): the mean lookback ratio over the output tokens.

    Built from the cached token IDs: context = the prompt tokens, new = the generated
    tokens. One teacher-forced forward pass with attentions, then a vectorised reduction.

    max_seq bounds the sequence length. The attention pass materialises an
    (n_layers, n_heads, seq, seq) tensor, which is O(seq^2) and OOMs on very long sources
    (some XSum/CNN articles exceed several thousand tokens -> tens of GB). For sequences over
    max_seq we keep ALL generated tokens and the most recent (max_seq - n_out) context tokens,
    truncating the oldest context. The lookback formula is unchanged; only the source length
    is bounded, and only for the longest few % of examples.
    """
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_out = len(record["gen_token_ids"])               # number of generated tokens

    # We drop the last generated token from the pool (see below), so we need >= 2 tokens.
    if n_out < 2:
        return np.zeros(n_layers * n_heads, dtype=np.float32)

    prompt_ids = record["prompt_token_ids"]
    ctx_len = len(prompt_ids)
    if ctx_len + n_out > max_seq:                      # bound very long sources (see docstring)
        keep = max_seq - n_out
        prompt_ids = prompt_ids[-keep:]
        ctx_len = keep
    full_ids = prompt_ids + record["gen_token_ids"]

    # We need attentions only, not hidden states, so request no layers' states.
    _, attentions = generate.recompute_states(
        model, tok, full_ids, layers=[], want_attentions=True
    )
    # attentions: tuple over the n_layers blocks, each (n_heads, seq, seq), already
    # causal (a token attends only to itself and earlier positions).
    A = torch.stack(list(attentions))                  # (L, H, seq, seq)

    # Rows for the generated tokens (positions ctx_len .. seq-1).
    rows = A[:, :, ctx_len:, :]                         # (L, H, n_out, seq)
    ctx_sum = rows[:, :, :, :ctx_len].sum(-1)          # attention on the context (L,H,n_out)
    new_sum = rows[:, :, :, ctx_len:].sum(-1)          # attention on the new tokens (incl. self)

    # Per-token MEANS, matching the authors' code (step01_extract_attns.py: a plain .mean over
    # the context slice and over the new slice). The i-th generated token's row attends to the
    # ctx_len context tokens and to the new keys ctx_len..ctx_len+i (i+1 of them, itself
    # included); causal masking zeroes the future so summing the whole [ctx_len:] slice is right.
    n_new = torch.arange(1, n_out + 1, dtype=A.dtype).view(1, 1, -1)  # i+1 new keys (self incl.)
    mean_ctx = ctx_sum / max(ctx_len, 1)
    mean_new = new_sum / n_new

    # Lookback ratio (Chuang et al.): context / (context + new). HIGH = grounded in the source,
    # LOW = attending to its own output (the hallucination direction). mean_ctx > 0 always
    # (softmax), so the denominator is never zero.
    lr = mean_ctx / (mean_ctx + mean_new)              # (L, H, n_out) per-token ratio

    # Response-level mean-pool over the generated tokens (our adaptation). Drop the LAST token:
    # the authors attribute each row's ratio to the token it predicts, so the final generated
    # token never contributes a within-response ratio. With this, the per-token ratios reproduce
    # the authors' released code (verified ~1e-6 in scripts/checks/check_lookback_vs_authors.py).
    feat = lr[:, :, :-1].mean(-1).reshape(-1)          # (L*H,)
    return feat.numpy()
