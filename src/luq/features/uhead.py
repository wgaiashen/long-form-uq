"""UHead baseline: score a generation with a PRE-TRAINED uncertainty head.

Unlike SAPLMA / P(True) / Lookback, this method trains NOTHING of ours. uhead ships a
pretrained head (here `llm-uncertainty-head/uhead_gemma-2-9b-it`, a token-level "luh"
head) that reads the base model's own hidden states + attention + token probabilities
and emits a per-token hallucination logit. We turn that into ONE instance-level
uncertainty score, so uhead slots in next to the other methods as a prior-work baseline.

How it works (mirrors the authors' `CalculatorInferLuh.infer_cached`, verified against
the installed package):
  1. Feed [prompt tokens + our cached generation tokens] through the base model in ONE
     teacher-forced forward pass (no regeneration), asking for hidden states AND
     attentions. The head needs attentions, so the base model must use the eager
     attention backend.
  2. Call the head on that forward. It returns a per-position logit; the forward/
     "training" branch drops the last position, so position p scores the token at p+1.
  3. Slice to the generated span (the authors' bound is begin-1 : end-1, because of that
     one-position shift) and reduce the per-token logits to a single number with
     mean(sigmoid(.)) — the response-level "mean" reduction from the repo's
     LuhClaimEstimator. Higher = more uncertain.

Why mean(sigmoid): sigmoid turns each token logit into a hallucination probability, and
the mean is the natural whole-response aggregate (the same set/sequence -> one-score
question the project is about; here we take the simplest fixed aggregator as the
baseline). Other reductions (max / min) are available in the repo if we want a
most-uncertain-token signal later.
"""
import torch

from luh import AutoUncertaintyHead

# The pretrained head that matches our scaled base model. One-to-one: a head is locked
# to the base model it was trained on (its input dims come from that model's config).
GEMMA_UHEAD = "llm-uncertainty-head/uhead_gemma-2-9b-it"


def load_uhead(base_model, uhead_name: str = GEMMA_UHEAD):
    """Load the pretrained head and bind it to an already-loaded base model.

    The head's weights load in float32 by default, but our Gemma base runs in bfloat16
    (Gemma-2 must — see generate.load_model). A dtype mismatch would crash the matmuls,
    so we cast the head to the base model's dtype. The head is a small Transformer
    encoder plus a linear layer, so bf16 inference is fine.
    """
    uhead = AutoUncertaintyHead.from_pretrained(uhead_name, base_model=base_model)
    uhead = uhead.to(device=base_model.device, dtype=base_model.dtype)
    uhead.eval()
    return uhead


@torch.no_grad()
def uhead_instance_score(model, tok, uhead, record: dict) -> float:
    """One uncertainty score for one cached record. Higher = more uncertain."""
    prompt_ids = record["prompt_token_ids"]
    gen_ids = record["gen_token_ids"]
    if len(gen_ids) == 0:
        # No generated tokens to score (degenerate record): maximally uncertain.
        return 1.0

    # Combined sequence = prompt then generation. batch_size=1, so no padding and the
    # attention mask is all ones. context_lengths tells the head where the prompt ends.
    combined = prompt_ids + gen_ids
    ids = torch.tensor([combined], device=model.device)
    attn_mask = torch.ones_like(ids)
    ctx_len = torch.tensor([len(prompt_ids)], device=model.device)

    out = model(
        input_ids=ids,
        attention_mask=attn_mask,
        output_hidden_states=True,
        output_attentions=uhead.output_attentions,  # the head says if it needs them
    )
    # The head reads context_lengths off BOTH the inputs dict and the model output
    # (it zeroes the prompt positions for some features). Matches infer_cached.
    out["context_lengths"] = ctx_len
    llm_inputs = {"input_ids": ids, "attention_mask": attn_mask, "context_lengths": ctx_len}

    # (1, T-1, 1) raw per-token logits -> (T-1,). out has no `.sequences`, so the head
    # takes its forward/training branch (the one infer_cached uses for cached tokens).
    logits = uhead(llm_inputs, out).squeeze(-1)[0].float()

    # Map back to the generated tokens. The one-position shift means the gen span lives
    # at positions begin-1 .. end-2 (the authors' (begin-1, end-1) half-open slice).
    begin = len(prompt_ids)
    end = begin + len(gen_ids)
    per_token = logits[begin - 1:end - 1]

    # mean over sigmoid(per-token hallucination prob) = response-level uncertainty.
    return torch.sigmoid(per_token).mean().item()
