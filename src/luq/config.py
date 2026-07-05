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
    model_name: str = "meta-llama/Meta-Llama-3.1-8B"  # the one model (frozen, middle layer 15)
    dataset: str = "sciq"                            # a ProbeDrift key
    ood_setting: str = "ID"
    seed: int = 1

    # Prompt regime = a namespace tag for one set of ProbeDrift prompts. Empty = the
    # original ("frozen") cache, left exactly where it is. A non-empty tag (e.g. "pdnew"
    # for the updated ProbeDrift, whose prompts differ) routes ALL of this run's
    # cache/results into a SEPARATE physical directory, so two prompt regimes can never
    # read each other's records or features. See __post_init__ and labels/.. provenance.
    prompt_regime: str = ""

    # --- generation ---
    # We own the per-dataset generation budget now (the updated ProbeDrift no longer
    # ships max_new_tokens); see data.MAX_NEW_TOKENS. This is only a safety ceiling.
    max_new_tokens_cap: int = 128

    # --- features ---
    # Which hidden layers to pool and cache (Tier 2). None = all layers, so picking
    # the best layer later is a free sweep on cached data.
    layers: tuple | None = None

    # --- paths ---
    cache_dir: Path = CACHE_DIR
    results_dir: Path = RESULTS_DIR

    def __post_init__(self):
        # A non-empty prompt_regime puts this run in its own cache/results namespace,
        # but only when the paths are the defaults (an explicit cache_dir is respected).
        # This is what keeps the frozen old-prompt cache and the new-prompt cache from
        # silently cross-contaminating, without changing run_key or its many callers.
        if self.prompt_regime:
            if self.cache_dir == CACHE_DIR:
                self.cache_dir = CACHE_DIR / self.prompt_regime
            if self.results_dir == RESULTS_DIR:
                self.results_dir = RESULTS_DIR / self.prompt_regime
