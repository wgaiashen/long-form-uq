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


def build_topk_surprisal(record, state, k, floor=0.0, shuffle_rng=None):
    """S4 — mass on the k MOST SURPRISING generated tokens, `floor` elsewhere. Label-free.

    Pre-registered in `prereg/topk_surprisal_prior.md`. This turns PART C's k-sweep taxonomy (pubmed
    k=1 concentrated, cnn k=all spread) from a description of the unsupervised floors into a prior that
    points the pooler at the tokens the taxonomy says carry the signal.

    It fixes BOTH defects of the plain `nll` prior at once while holding the information source fixed:
      * DENSITY  -- `nll` puts weight on every position; this puts it on k of them.
      * SCALE    -- `nll` uses raw surprisal magnitudes, which are not comparable across datasets (an OOD
                    scale mismatch). This uses only the RANK of the NLL, so it is scale-free.
    So `topk` vs `nll` is a clean two-factor test, which is why they must be run in the SAME job.

    `k`: an int (absolute count) or a float in (0,1) (fraction of G, resolved per example so a fraction
    means the same thing on a 32-token answer and a 219-token one). k >= G degenerates to uniform, which
    is control #1 in the prereg (arm C must then reproduce mean-pool).

    `floor`: weight on the non-selected tokens. 0.0 is a hard filter. With floor=0 arm D's additive
    `beta*log(prior)` clamps at log(1e-9), so ANY beta > 0 hard-masks the rest and beta stops
    interpolating -- hence the registered floor=0.05 arm, where beta keeps its meaning.

    `shuffle_rng`: the registered POSITIONAL-SHUFFLE control. Keeps sparsity and the floor identical and
    destroys only WHICH tokens are selected, so a win that survives it is not about surprisal at all.

    Returns (w, fell_back) with w length G+1 (row 0 = the last-prompt anchor, always 0).
    """
    G, T = _window(record, state)
    nll = per_token_nll(record)
    if len(nll) != G:
        raise SystemExit(f"topk prior: {len(nll)} logprobs != G={G} gen tokens")
    nll = np.asarray(nll, dtype=np.float32)

    if G == 0:                                    # genuinely nothing to select over -> LOUD fallback
        return np.zeros(T, dtype=np.float32), True
    if not np.isfinite(nll).all():
        return np.zeros(T, dtype=np.float32), True

    if isinstance(k, float) and 0.0 < k < 1.0:
        k_eff = max(1, int(round(k * G)))
    else:
        k_eff = int(k)
    k_eff = max(1, min(k_eff, G))

    gw = np.full(G, float(floor), dtype=np.float32)
    if k_eff >= G:
        gw[:] = 1.0                               # degenerate -> uniform (prereg control #1)
    else:
        sel = np.argpartition(-nll, k_eff - 1)[:k_eff]      # indices of the k largest NLL
        if shuffle_rng is not None:                          # positional-shuffle control
            sel = shuffle_rng.choice(G, size=k_eff, replace=False)
        gw[sel] = 1.0

    if gw.sum() <= 0:                             # only reachable with floor=0 and k_eff=0, guarded above
        return np.zeros(T, dtype=np.float32), True
    w = np.zeros(T, dtype=np.float32)
    w[1:] = gw
    return w, False


def _parse_topk(name):
    """`topk:<k>[:floor=<f>][:shuf]` -> (k, floor, shuffled). Raises on anything it cannot parse rather
    than guessing, so a typo in --priors cannot silently become a different method."""
    parts = name.split(":")
    if parts[0] != "topk" or len(parts) < 2:
        raise SystemExit(f"bad topk prior spec {name!r}; expected topk:<k>[:floor=<f>][:shuf]")
    raw = parts[1]
    k = float(raw) if ("." in raw) else int(raw)
    floor, shuf = 0.0, False
    for p in parts[2:]:
        if p.startswith("floor="):
            floor = float(p.split("=", 1)[1])
        elif p == "shuf":
            shuf = True
        else:
            raise SystemExit(f"unknown topk option {p!r} in {name!r}")
    if not 0.0 <= floor < 1.0:
        raise SystemExit(f"topk floor must be in [0,1); got {floor}")
    return k, floor, shuf


BUILDERS = {"content_mass": "content", "nll": "nll", "orgad": "orgad"}   # names -> kind (for --priors)
_SLUG = "meta-llama_Meta-Llama-3.1-8B"


class OrgadCoverageError(Exception):
    """A cell's sources are not all Orgad-covered -> the driver skips that cell LOUDLY (never silent-uniform)."""


def build_soft_orgad(records_list, states_list, datasets, tok, variant="broad", floor=0.0):
    """S3.6 — soft-Orgad prior (the lead): reuse the τ soft-tier `build_masks` (RAISES on a missing cache),
    NOT the hard build_answer_masks. EXAMPLE-LOCAL + per-dataset, so EVERY source dataset must be covered; if
    any isn't, raise OrgadCoverageError (the driver skips the cell). floor=0 -> pool over located tokens."""
    from weighted_msp_orgad_ladder import build_masks
    uncovered = sorted({d for d in set(datasets)
                        if not (ROOT / "cache" / "orgad_llm" / f"{_SLUG}__{d}__ID__broad.json").exists()})
    if uncovered:
        raise OrgadCoverageError(f"no broad Orgad cache for {uncovered}")
    masks = [None] * len(records_list); loc = [False] * len(records_list)
    by_ds = {}
    for i, d in enumerate(datasets):
        by_ds.setdefault(d, []).append(i)
    for d, idxs in by_ds.items():
        recs = [records_list[i] for i in idxs]
        for r in recs:
            r["_dataset"] = d
        m_d, loc_d = build_masks(tok, recs, variant=variant, floor=floor)
        for j, i in enumerate(idxs):
            masks[i] = m_d[j]; loc[i] = bool(loc_d[j])
    out, nfb = [], 0
    for r, s, m, lc in zip(records_list, states_list, masks, loc):
        G, T = _window(r, s)
        gw = np.clip(np.asarray(m, float), 0.0, None)
        if len(gw) != G:
            raise SystemExit(f"orgad mask len {len(gw)} != G={G}")
        fell_back = not lc
        if gw.sum() <= 0:
            gw = np.ones(G, dtype=np.float32); fell_back = True    # degenerate row -> LOUD uniform fallback
        w = np.zeros(T, dtype=np.float32); w[1:] = gw
        out.append(w); nfb += int(fell_back)
    return out, nfb


def build_prior(name, records_list, states_list, tok=None, special_ids=None, datasets=None):
    """Return (list of length-(G+1) priors aligned to states_list, n_fallback_rows). `datasets` (per-record
    dataset name) is required for the soft-Orgad prior."""
    if name == "orgad":
        if datasets is None:
            raise SystemExit("orgad prior needs per-record `datasets`")
        return build_soft_orgad(records_list, states_list, datasets, tok)
    topk_spec = _parse_topk(name) if name.startswith("topk") else None
    rng = np.random.RandomState(12345) if (topk_spec and topk_spec[2]) else None
    out, nfb = [], 0
    for r, s in zip(records_list, states_list):
        if name == "content_mass":
            w, fb = build_content_mass(r, s, tok, special_ids)
        elif name == "nll":
            w, fb = build_nll(r, s)
        elif topk_spec is not None:
            k, floor, _ = topk_spec
            w, fb = build_topk_surprisal(r, s, k, floor=floor, shuffle_rng=rng)
        else:
            raise SystemExit(f"unknown prior '{name}' (have {list(BUILDERS)} or topk:<k>)")
        out.append(w); nfb += int(fb)
    return out, nfb
