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

from . import answer_span
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name: str, attn_implementation: str | None = None,
               dtype: torch.dtype | None = None):
    """Load a frozen causal LM on the GPU. We never train the base model.

    attn_implementation: pass "eager" when you need attention weights
    (output_attentions=True). The default fast backend (SDPA) does not return them,
    so Lookback Lens must load with "eager"; SAPLMA and P(True) leave this as None.

    dtype: None = auto-select per model. Gemma-2 MUST run in bfloat16: it applies
    soft-capping to its logits and attention scores, and in fp16 those overflow to
    NaN. Every other model defaults to fp16 (fast, half the memory), which keeps our
    existing Qwen runs bit-identical. Pass an explicit dtype to override -- e.g. the
    Lookback feature forces fp32, because eager attention in fp16 overflows to NaN in
    the softmax. Hidden-state features (SAPLMA, P(True)) are fine at the default.

    Gemma-2 also defaults to the EAGER attention backend here. Gemma-2 soft-caps its
    attention logits, and only the eager path applies that cap -- SDPA (the usual HF
    default) silently skips it. Since we probe the hidden states, and those are
    computed FROM the attention, SDPA would give subtly wrong internal features. So
    for Gemma we force eager unless the caller asks for something specific.
    """
    is_gemma = "gemma" in name.lower()
    if dtype is None:
        dtype = torch.bfloat16 if is_gemma else torch.float16
    if attn_implementation is None and is_gemma:
        attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(name)
    kwargs = dict(dtype=dtype, device_map="cuda")
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int,
             truncate_at_newline: bool = False,
             repetition_penalty: float | None = None,
             no_repeat_ngram_size: int | None = None,
             truncate_answer_span: str | None = None):
    """Generate one response; return (record, pooled_all_layers).

    record: dict with prompt, prompt_token_ids, gen_token_ids, gen_text, token_logprobs.
    pooled_all_layers: tensor (n_layers, hidden) = mean over the last-prompt (pre-answer)
    position + the output tokens, per layer (Joe's SAPLMA masked-mean; see the pooling note).

    truncate_at_newline: for few-shot short-form QA the answer ends at the first
    newline; what follows is the model imitating the prompt format (inventing the
    next question), not part of the answer. Truncating HERE means the record,
    the logprobs, the pooled features, and the correctness label all describe the
    same text. Long-form datasets must keep newlines, so it is off by default.

    repetition_penalty / no_repeat_ngram_size: OPT-IN decoding controls, default OFF so
    every existing frozen run is byte-identical (None => the arg is not passed, so HF uses
    its no-op defaults 1.0 / 0). Needed only for open-ended prompts where the base model has
    no natural stop and loops to the token budget (ExpertQA: the Stage-2 scan found 73% capped,
    ~half repetition-degenerate). Greedy decoding is unchanged; these only forbid the loop.
    """
    # 1. Tokenise. prompt_len marks where the response begins: generate() returns
    #    prompt + response as one sequence, and we only ever cache the response part.
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    # 2. The single generate pass. Greedy decoding (do_sample=False) keeps runs
    #    reproducible; the output_* flags make generate hand back the logits and
    #    hidden states it computed anyway. The two repetition controls are only added
    #    when explicitly set, so the default call is unchanged for the frozen runs.
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_hidden_states=True,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tok.eos_token_id,
    )
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if no_repeat_ngram_size is not None:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    out = model.generate(**inputs, **gen_kwargs)

    # 3. Slice off the response. n_gen is often < max_new_tokens (generation stops
    #    early at an end-of-sequence token), so always measure the actual length.
    gen_ids = out.sequences[0, prompt_len:]
    if truncate_at_newline:
        for i, tid in enumerate(gen_ids):
            if "\n" in tok.decode([tid]):
                gen_ids = gen_ids[: max(i, 1)]  # keep at least one token
                break
    if truncate_answer_span:
        # Long-form sibling of truncate_at_newline. Base Llama is not instruction-tuned: under a
        # few-shot prompt it finishes the answer and then CONTINUES THE FORMAT, writing a fresh
        # "Question:/Answer:" pair and inventing both, often on an unrelated topic. On med_quad that
        # affects 47.8% of generations at a 128-token budget and 92.6% at 768, where ~66% of the
        # average generation is the invented part -- and the judge scores the whole saved output.
        #
        # ⚠️ THE POINT OF CUTTING HERE rather than at scoring time: gen_ids is truncated BEFORE the
        # logprobs (out.scores[:n_gen]) and BEFORE the hidden-state pooling below, so the record, the
        # MSP floors, the pooled features and the label all describe the SAME text. Cutting later
        # would leave features computed over text the label never saw.
        #
        # `answer_span` owns the per-dataset rules and was already written for this exact failure; it
        # returns a CHARACTER index, so map it back to a token boundary by binary search on the
        # decoded prefix (monotone in k, ~10 decodes rather than one per token).
        full = tok.decode(gen_ids, skip_special_tokens=True)
        _clean, cut_char, _reason = answer_span.answer_span(full, truncate_answer_span)
        if cut_char < len(full):
            lo, hi = 1, len(gen_ids)
            while lo < hi:
                mid = (lo + hi) // 2
                if len(tok.decode(gen_ids[:mid], skip_special_tokens=True)) >= cut_char:
                    hi = mid
                else:
                    lo = mid + 1
            gen_ids = gen_ids[: max(lo, 1)]         # keep at least one token, as above
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    n_gen = len(gen_ids)

    # 4. Per-token logprobs (MSP's raw material). out.scores has one (1, vocab)
    #    logits tensor per generated token, from BEFORE the token was picked;
    #    log_softmax turns them into log-probabilities and we keep the chosen
    #    token's. .float() because softmax in fp16 loses precision.
    #    Only the first n_gen entries: everything past the truncation point is
    #    not part of the answer.
    token_logprobs = []
    for step, logits in enumerate(out.scores[:n_gen]):
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        token_logprobs.append(logp[gen_ids[step]].item())

    # 5. Pool the hidden states over the output tokens, per layer.
    #    out.hidden_states is a tuple over generation steps; each step is a tuple
    #    over layers (index 0 = embedding layer); each layer tensor is
    #    (1, n_tokens_in_step, hidden). Step 0 covers the whole prompt; steps 1..
    #    each cover the ONE token fed back from the previous step. So the
    #    generated tokens' states live at steps 1.., last position. The final
    #    generated token is never fed back in, so it has no state here; a mean
    #    over n_gen - 1 of n_gen tokens is fine.
    n_layers = len(out.hidden_states[0])
    pooled = []
    for layer in range(n_layers):
        # Average the last-prompt (pre-answer) position PLUS the generated-token states,
        # matching Joe's SAPLMA masked-mean: his output_mask aligns so the averaged window
        # starts at the last prompt position (the state that PREDICTS the first answer
        # token). That pre-answer state encodes the whole question and DOMINATES for short
        # answers -- excluding it (our earlier bug) meant a 2-token answer like "friday"
        # ['fr','iday'] was pooled from just ['fr'], and a 1-token answer from the prompt's
        # ':' alone. The final generated token has no fed-back state in generate(), so it is
        # dropped (Joe drops it too). Verified against compiled_features.py + full_seq_head_saplma.py.
        vecs = [out.hidden_states[0][layer][0, -1, :]]                       # last prompt token (pre-answer)
        # range(1, n_gen+1) -- include the LAST kept answer token's state too. We generate the
        # full budget with NO stop criterion and truncate at EXTRACTION, so out.hidden_states
        # extends PAST n_gen; out.hidden_states[n_gen] (the last kept answer token, whose
        # fed-back state exists because the now-truncated continuation followed) is real. The
        # old range(1, n_gen) dropped it -- catastrophic for short answers (trivia mean 2.9
        # tokens; a 1-token answer pooled ZERO answer tokens). Joe includes all answer-token
        # states. min(...) guards the rare case where generation stopped exactly at n_gen (EOS).
        last = min(n_gen + 1, len(out.hidden_states))
        vecs += [out.hidden_states[s][layer][0, -1, :] for s in range(1, last)]  # answer tokens 0..G-1
        pooled.append(torch.stack(vecs).mean(dim=0))
    # Back to float32 (numpy/sklearn-friendly) and off the GPU.
    pooled = torch.stack(pooled).float().cpu()

    # 6. The Tier-1 record. Token IDs (not just decoded text) on purpose:
    #    re-tokenising text is not guaranteed to round-trip, and later stages
    #    (P(True), decomposition) replay these exact IDs through recompute_states.
    #    .tolist() because JSON cannot store tensors.
    record = {
        "prompt": prompt,
        "prompt_token_ids": inputs.input_ids[0].tolist(),
        "gen_token_ids": gen_ids.tolist(),
        "gen_text": gen_text,
        "token_logprobs": token_logprobs,
    }
    return record, pooled


@torch.no_grad()
def recompute_states(model, tok, token_ids, layers, want_attentions: bool = False):
    """Teacher-forced forward over a saved token sequence; return per-token states.

    Use this when you need a representation you did NOT cache: raw per-token states
    for decomposition, attentions for Lookback Lens, or the appended-question pass
    for P(True). Inputs are token IDs from a Tier-1 record, so there is no
    re-tokenisation drift. This is one forward pass, no generation.

    Returns hidden_states: list over `layers`, each (seq_len, hidden), float32 cpu.
    If want_attentions, also returns attentions: tuple over ALL transformer blocks,
    each (n_heads, seq_len, seq_len).

    Unlike the generation-time structure, a teacher-forced pass yields the state of
    EVERY position at once, including the last token, because here we feed the full
    sequence in as input rather than building it one token at a time.
    """
    # [None] adds the batch dimension: (seq_len,) -> (1, seq_len).
    ids = torch.tensor(token_ids)[None].to(model.device)
    out = model(ids, output_hidden_states=True, output_attentions=want_attentions)

    # out.hidden_states is a tuple over layers (0 = embedding layer), each
    # (1, seq_len, hidden). Drop the batch dim and move off the GPU.
    states = [out.hidden_states[l][0].float().cpu() for l in layers]
    if want_attentions:
        # out.attentions: one (1, n_heads, seq_len, seq_len) tensor per block.
        attentions = tuple(a[0].float().cpu() for a in out.attentions)
        return states, attentions
    return states
