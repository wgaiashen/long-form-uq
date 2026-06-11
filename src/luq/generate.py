"""Stage 1: generation + hidden states. The only GPU-heavy step.

Public functions:
    load_model(name)                       -> (model, tokenizer)
    generate(model, tok, prompt, mnt)      -> (record, pooled_all_layers)
    recompute_states(model, tok, ids, ...) -> per-token states (and attentions)

`generate` runs ONE forward pass per example with output_hidden_states=True and
captures everything inline (a single pass, not a second forward).
`recompute_states` is the teacher-forced regeneration pass that rebuilds any
representation from cached token IDs; decomposition, Lookback Lens, and P(True)
build on it.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name: str):
    """Load a frozen causal LM in fp16 on the GPU. We never train the base model."""
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int):
    """Generate one response; return (record, pooled_all_layers).

    record: dict with prompt, gen_token_ids, gen_text, token_logprobs.
    pooled_all_layers: tensor (n_layers, hidden) = mean over the OUTPUT tokens, per layer.

    TODO, step by step:
      1. Tokenise: inputs = tok(prompt, return_tensors="pt").to(model.device).
         Save prompt_len = inputs.input_ids.shape[1] (where the response begins).
      2. out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                 do_sample=False, output_hidden_states=True, output_scores=True,
                 return_dict_in_generate=True)
      3. gen_ids = out.sequences[0, prompt_len:]; gen_text = tok.decode(gen_ids,
         skip_special_tokens=True).
      4. token_logprobs (for MSP): out.scores is a tuple of (1, vocab) logits, one per
         generated step. log_softmax each and gather the chosen token's value.
      5. hidden states: out.hidden_states is a tuple over steps. step 0 holds the prompt
         pass (all prompt tokens); steps 1.. each hold ONE new token. For every layer,
         collect the generated-token hidden states (steps 1..) and mean over them.
         Stack the per-layer means -> (n_layers, hidden). (uhead's basic_hidden_states.py
         shows the exact tuple shapes if you want a reference.)
      6. Return the record dict and pooled.cpu().
    """
    raise NotImplementedError("fill in generate() — follow the numbered TODO above")


@torch.no_grad()
def recompute_states(model, tok, token_ids, layers, want_attentions: bool = False):
    """Teacher-forced forward over a saved token sequence; return per-token states.

    Use this when you need a representation you did NOT cache: raw per-token states
    for decomposition, attentions for Lookback Lens, or the appended-question pass
    for P(True). Inputs are token IDs from a Tier-1 record, so there is no
    re-tokenisation drift. This is one forward pass, no generation.

    TODO:
      1. ids = torch.tensor(token_ids)[None].to(model.device)
      2. out = model(ids, output_hidden_states=True, output_attentions=want_attentions)
      3. Return [out.hidden_states[l] for l in layers] (and out.attentions if asked).
    """
    raise NotImplementedError("fill in recompute_states() — follow the TODO above")
