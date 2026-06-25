"""P(True) feature (append-and-read): probe the verdict-position hidden state.

Append a yes/no verification question after the response, run one forward pass, and read the
hidden state at the final position -- the state about to emit the verdict. The probe reads that
hidden state instead of the stated word: a probe-level P(True).
"""
import numpy as np

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
