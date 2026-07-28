"""S3 prior builders — per-example, label-free, EXAMPLE-LOCAL weight vectors for the AttnPool `prior=` channel.

Each builder returns a length-(G+1) vector aligned to the SAPLMA window `[last_prompt_token] + gen_tokens`
(row 0 = the last-prompt anchor, set to 0 so pooling is over the answer tokens, matching `answer_states`).
The frozen/annealed forward renormalises internally, so builders return non-negative RAW weights.

FAIL-LOUD (the recurring silent-default class): a builder that cannot produce a genuine per-token weight for a
row must NOT silently emit uniform. content-mass has a documented degenerate case (a row with no content token)
-> it falls back to uniform for THAT row and the caller is handed a COUNT so it can report the fallback rate
(a blank/`fell_back` reads as "not measured", never as "measured and uniform"). A missing *cache* (soft-Orgad)
must RAISE at the loader (reuse `weighted_msp_orgad_ladder.build_masks`, which does) -- content-mass/NLL read no
cache, so their only absence is the degenerate row, handled here.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq.weighted_msp import per_token_nll                                   # noqa: E402
from pool_peak_tokens import classify_id                                     # noqa: E402


def _window(record, state):
    G = len(record["gen_token_ids"])
    T = int(np.asarray(state).shape[0])
    if T != G + 1:                     # the G+1 alignment guard (never pad/trim silently)
        raise SystemExit(f"prior alignment: state has {T} rows, expected G+1={G + 1} for this record")
    return G, T


def build_content_mass(record, state, tok, special_ids):
    """Weight 1 on CONTENT tokens (alphabetic ≥3 chars), 0 elsewhere. D2 says content transfers OOD."""
    G, T = _window(record, state)
    gw = np.array([1.0 if classify_id(int(t), tok, special_ids) == "content" else 0.0
                   for t in record["gen_token_ids"]], dtype=np.float32)
    fell_back = False
    if gw.sum() <= 0:                  # no content token -> LOUD uniform fallback (counted by the caller)
        gw[:] = 1.0
        fell_back = True
    w = np.zeros(T, dtype=np.float32)
    w[1:] = gw
    return w, fell_back


def build_nll(record, state):
    """Weight = per-token NLL (surprisal); upweights the tokens the model was least sure of. Label-free."""
    G, T = _window(record, state)
    nll = per_token_nll(record)
    if len(nll) != G:
        raise SystemExit(f"NLL prior: {len(nll)} logprobs != G={G} gen tokens")
    gw = np.clip(np.asarray(nll, dtype=np.float32), 0.0, None)
    fell_back = False
    if (not np.isfinite(gw).all()) or gw.sum() <= 0:
        gw = np.ones(G, dtype=np.float32)
        fell_back = True
    w = np.zeros(T, dtype=np.float32)
    w[1:] = gw
    return w, fell_back


BUILDERS = {"content_mass": "content", "nll": "nll"}   # names -> kind (for the driver's --priors)


def build_prior(name, records_list, states_list, tok=None, special_ids=None):
    """Return (list of length-(G+1) priors aligned to states_list, n_fallback_rows)."""
    out, nfb = [], 0
    for r, s in zip(records_list, states_list):
        if name == "content_mass":
            w, fb = build_content_mass(r, s, tok, special_ids)
        elif name == "nll":
            w, fb = build_nll(r, s)
        else:
            raise SystemExit(f"unknown prior '{name}' (have {list(BUILDERS)})")
        out.append(w); nfb += int(fb)
    return out, nfb
