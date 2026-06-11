"""Run configuration: one place for settings every stage shares.

A dataclass gives you sensible defaults plus editor autocomplete, and any field can
be overridden from a script or the command line. Paths point at the gitignored
cache/ and results/ folders.
"""
from dataclasses import dataclass
from pathlib import Path

# .../msc-project-gs925  (this file is src/luq/config.py, so go up three parents)
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "cache"
RESULTS_DIR = REPO_ROOT / "results"


@dataclass
class Config:
    # --- what to run ---
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"  # small dev model for fast iteration
    dataset: str = "sciq"                            # a ProbeDrift key
    ood_setting: str = "ID"
    seed: int = 1

    # --- generation ---
    # The real budget is per-example from ProbeDrift (train_ds.max_new_tokens). This
    # is only a safety ceiling.
    max_new_tokens_cap: int = 128

    # --- features ---
    # Which hidden layers to pool and cache (Tier 2). None = all layers, so picking
    # the best layer later is a free sweep on cached data.
    layers: tuple | None = None

    # --- paths ---
    cache_dir: Path = CACHE_DIR
    results_dir: Path = RESULTS_DIR
