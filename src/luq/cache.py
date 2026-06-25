"""Three-tier cache.

    Tier 1  canonical record  ->  cache/records/<key>.jsonl   (sacred; token IDs in it)
    Tier 2  pooled features   ->  cache/features/<key>__<method>.npz
    Tier 3  raw per-token states are NEVER written here; recompute on demand.

A "key" identifies a run: model + dataset + ood_setting. Tier 2 adds the method.
This module is plain glue and is written out in full so you have a working example
to read.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np


def _slug(s: str) -> str:
    """Make a string safe for a filename (model names contain '/')."""
    return s.replace("/", "_")


def _atomic_replace(tmp: Path, out: Path) -> None:
    """Move a fully-written temp file onto the real path in one step.

    os.replace is atomic on the same filesystem, so a reader (or a job killed
    mid-write) ever sees either the old complete file or the new complete file,
    never a half-written one. We write to `<path>.tmp` first, then swap.
    """
    os.replace(tmp, out)


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
    tmp = out.parent / (out.name + ".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    _atomic_replace(tmp, out)
    return out


def records_path(cache_dir: Path, key: str) -> Path:
    """The on-disk path of the Tier-1 records file (used for the label-freshness check: a
    probe's scores are stamped with this file's mtime, so 04_eval can detect a relabel
    that happened after the probe was trained)."""
    return Path(cache_dir) / "records" / f"{key}.jsonl"


def load_records(cache_dir: Path, key: str) -> list[dict]:
    path = Path(cache_dir) / "records" / f"{key}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---- Tier 2: pooled features (all layers) -------------------------------------

def features_path(cache_dir: Path, key: str, method: str) -> Path:
    """The on-disk path of a Tier-2 feature file (shared by save/load and the
    provenance/freshness checks in 03_probe / 04_eval)."""
    return Path(cache_dir) / "features" / f"{key}__{method}.npz"


def save_features(feats: np.ndarray, cache_dir: Path, key: str, method: str) -> Path:
    """feats shape: (n_examples, n_layers, hidden). Keeping all layers makes the
    layer choice a free sweep at probe time."""
    out = Path(cache_dir) / "features" / f"{key}__{method}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp handle, not the real path: passing a file object also stops
    # numpy from "helpfully" appending a second .npz to the temp name.
    tmp = out.parent / (out.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, feats=feats)
    _atomic_replace(tmp, out)
    return out


def load_features(cache_dir: Path, key: str, method: str) -> np.ndarray:
    path = Path(cache_dir) / "features" / f"{key}__{method}.npz"
    return np.load(path)["feats"]


# ---- checkpoint/resume: Tier-1 records + Tier-2 SAPLMA features together --------
# Generation at 7-9B is slow, so an extract job can hit the Slurm time limit before
# it finishes. These two helpers let scripts/01_extract.py save its progress every
# few hundred examples and pick up where it left off on the next run, instead of
# throwing away hours of GPU work. Records and the pooled SAPLMA features are kept
# in lockstep (same order, same length) so they stay index-aligned by construction.

def save_checkpoint(records: list[dict], pooled_list: list[np.ndarray],
                    cache_dir: Path, key: str):
    """Persist the run so far. `pooled_list` is a list of per-example feature arrays,
    each (n_layers, hidden); stacked into the (n_examples, n_layers, hidden) Tier-2
    array. Both files are written atomically, so a kill mid-checkpoint leaves the
    PREVIOUS good checkpoint intact. Returns (records_path, features_path)."""
    feats = np.stack(pooled_list) if pooled_list else np.empty((0, 0, 0), dtype=np.float32)
    rpath = save_records(records, cache_dir, key)
    fpath = save_features(feats, cache_dir, key, method="saplma")
    return rpath, fpath


def load_checkpoint(cache_dir: Path, key: str):
    """Load a partial extract run if one exists, as (records, pooled_list) aligned by
    position. Returns ([], []) when nothing is cached yet. If the two files somehow
    disagree in length (e.g. a crash landed between the two atomic writes), both are
    trimmed to the shorter length so the caller resumes from a consistent point."""
    rpath = Path(cache_dir) / "records" / f"{key}.jsonl"
    fpath = Path(cache_dir) / "features" / f"{key}__saplma.npz"
    if not rpath.exists() or not fpath.exists():
        return [], []
    records = load_records(cache_dir, key)
    feats = load_features(cache_dir, key, method="saplma")
    n = min(len(records), len(feats))
    return records[:n], list(feats[:n])


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


# ---- trained probes: the fitted probe object (weights + scaler) -----------------
# Persisted so a probe trained on dataset A can be re-applied to dataset B's features
# without retraining — the OOD cross-task experiment (train A, score B). The probe
# objects (SoftProbe / MLPProbe) pickle without torch (plain numpy weights + scaler).
# Source of truth stays the cached features + deterministic reproduce.py; a stored probe
# is a convenience/OOD artifact, always regenerable, never the authority.

def save_probe(clf, cache_dir: Path, key: str, method: str, layer: int) -> Path:
    """Pickle the fitted probe object, keyed by run + method + layer."""
    out = Path(cache_dir) / "probes" / f"{key}__{method}__L{layer}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(clf, f)
    return out


def load_probe(cache_dir: Path, key: str, method: str, layer: int):
    """Load a probe saved by save_probe (e.g. to score another dataset's features OOD)."""
    path = Path(cache_dir) / "probes" / f"{key}__{method}__L{layer}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
