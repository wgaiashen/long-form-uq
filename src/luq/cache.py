"""Three-tier cache.

    Tier 1  canonical record  ->  cache/records/<key>.jsonl   (sacred; token IDs in it)
    Tier 2  pooled features   ->  cache/features/<key>__<method>.npz
    Tier 3  raw per-token states are NEVER written here; recompute on demand.

A "key" identifies a run: model + dataset + ood_setting. Tier 2 adds the method.
This module is plain glue and is written out in full so you have a working example
to read.
"""
import json
from pathlib import Path

import numpy as np


def _slug(s: str) -> str:
    """Make a string safe for a filename (model names contain '/')."""
    return s.replace("/", "_")


def run_key(model_name: str, dataset: str, ood_setting: str) -> str:
    return f"{_slug(model_name)}__{dataset}__{ood_setting}"


# ---- Tier 1: canonical records ------------------------------------------------

def save_records(records: list[dict], cache_dir: Path, key: str) -> Path:
    """Write Tier-1 records as JSONL (one example per line).

    Each record should hold at least:
        idx, split, prompt, gen_token_ids (list[int]), gen_text,
        target, token_logprobs (list[float])
    Token IDs are part of the record on purpose: decoded text alone loses the exact
    tokenisation that the alignment / decomposition step will need.
    """
    out = Path(cache_dir) / "records" / f"{key}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return out


def load_records(cache_dir: Path, key: str) -> list[dict]:
    path = Path(cache_dir) / "records" / f"{key}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---- Tier 2: pooled features (all layers) -------------------------------------

def save_features(feats: np.ndarray, cache_dir: Path, key: str, method: str) -> Path:
    """feats shape: (n_examples, n_layers, hidden). Keeping all layers makes the
    layer choice a free sweep at probe time."""
    out = Path(cache_dir) / "features" / f"{key}__{method}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, feats=feats)
    return out


def load_features(cache_dir: Path, key: str, method: str) -> np.ndarray:
    path = Path(cache_dir) / "features" / f"{key}__{method}.npz"
    return np.load(path)["feats"]


# ---- method scores: per-example uncertainty on the test split ------------------

def save_scores(unc: np.ndarray, cache_dir: Path, key: str, method: str, **extras) -> Path:
    """unc: (n_test,) uncertainty scores, in test-record order. `extras` stores
    small run facts next to the scores (e.g. layer=14) so a saved result is never
    ambiguous about how it was produced."""
    out = Path(cache_dir) / "scores" / f"{key}__{method}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, unc=unc, **{k: np.asarray(v) for k, v in extras.items()})
    return out


def load_scores(cache_dir: Path, key: str, method: str) -> dict:
    path = Path(cache_dir) / "scores" / f"{key}__{method}.npz"
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
