"""P(True) feature (append-and-read): probe the verdict-position hidden state.

Append a yes/no verification question after the response, run one forward pass, and read the
hidden state at the final position -- the state about to emit the verdict. The probe reads that
hidden state instead of the stated word: a probe-level P(True).
"""
import numpy as np
import torch

from .. import generate

# The verification question (a recorded hyperparameter: fix it, record it). Ends on an "Answer:"
# cue so the verdict is the unambiguous next token, distinguishable from ProbeDrift's few-shot
# "Answer:" markers. "accurate" is task-agnostic: it reads as factually-true for QA and as
# faithful-to-source for summarisation, so one wording works across all tasks (the earlier "is
# the above answer true?" was QA-only). Validate the yes/no verdict mass per wording with
# scripts/checks/check_ptrue_verdict.py. The wording is a function arg (below) so candidate
# wordings can be swept without code edits.
PTRUE_SUFFIX = "\nIs the above response accurate? Answer (yes or no):"


def ptrue_vector(model, tok, record: dict, layers=None, suffix: str = PTRUE_SUFFIX) -> np.ndarray:
    """Return (n_layers, hidden): the hidden state at the appended verdict position.

    Replay the cached prompt + generated token IDs (no re-tokenisation drift) plus the `suffix`
    tokens (add_special_tokens=False, so no stray BOS mid-sequence), run one teacher-forced
    forward via recompute_states, and read the LAST position at every layer -- the state the
    model would use to emit its yes/no verdict. `suffix` defaults to PTRUE_SUFFIX; pass a
    different string to test an alternative wording.
    """
    if layers is None:
        # All layers (embedding + every block), matching SAPLMA's all-layer cache, so choosing
        # the best layer stays a free sweep at probe time.
        layers = range(model.config.num_hidden_layers + 1)

    suffix_ids = tok(suffix, add_special_tokens=False).input_ids
    full_ids = record["prompt_token_ids"] + record["gen_token_ids"] + suffix_ids
    states = generate.recompute_states(model, tok, full_ids, layers)
    return np.stack([s[-1].numpy() for s in states])  # (n_layers, hidden)


def yes_no_token_ids(tok):
    """Single-token ids for the "yes" and "no" surface forms, returned as (yes_ids, no_ids).

    A verdict at the appended position is one token, so we only keep surface forms that tokenise
    to a single id. Leading-space variants are included because after the "Answer:" cue the model
    usually emits the word with a leading space (one BPE token). Casing variants cover Yes/yes/YES.
    """
    def single_ids(forms):
        out = set()
        for s in forms:
            ids = tok(s, add_special_tokens=False).input_ids
            if len(ids) == 1:
                out.add(ids[0])
        return sorted(out)

    yes = single_ids([" yes", " Yes", " YES", "yes", "Yes", "YES"])
    no = single_ids([" no", " No", " NO", "no", "No", "NO"])
    return yes, no


def ptrue_unsup_confidence(model, tok, record, suffix=PTRUE_SUFFIX, yes_ids=None, no_ids=None):
    """Unsupervised P(True): the model's OWN P(yes) / (P(yes) + P(no)) at the verdict position.

    This reads the next-token distribution the model emits, not a trained probe. Append the
    verification question, run one forward pass, take the next-token logits at the last position,
    and renormalise over the yes and no token ids. Returns (confidence, p_yes, p_no), where
    confidence in [0, 1] is the model's stated probability that the response is accurate.
    p_yes + p_no is the share of next-token mass on a yes/no verdict (a sanity signal: if it is
    tiny, the model is not answering yes/no and the confidence is unreliable).
    """
    if yes_ids is None or no_ids is None:
        yes_ids, no_ids = yes_no_token_ids(tok)
    suffix_ids = tok(suffix, add_special_tokens=False).input_ids
    full_ids = record["prompt_token_ids"] + record["gen_token_ids"] + suffix_ids
    ids = torch.tensor(full_ids)[None].to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    p_yes = float(probs[yes_ids].sum())
    p_no = float(probs[no_ids].sum())
    total = p_yes + p_no
    conf = p_yes / total if total > 0 else float("nan")
    return conf, p_yes, p_no
