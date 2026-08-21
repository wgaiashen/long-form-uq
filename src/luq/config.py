"""Run configuration: one place for settings every stage shares.

A dataclass gives you sensible defaults plus editor autocomplete, and any field can
be overridden from a script or the command line. Paths point at the gitignored
cache/ and results/ folders.
"""
import os
from dataclasses import dataclass
from pathlib import Path

# .../msc-project-gs925  (this file is src/luq/config.py, so go up three parents)
REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- where the (large, regenerable) cache lives -----------------------------------------------
# Default: `cache/` beside the code, which is right on RCS (~900GB home).
#
# `LUQ_CACHE_ROOT` overrides it, because on DoC the repo sits on a CephFS allocation with a HARD
# 50GB quota (~4GB free), while `/vol/bitbucket/gs925` is NFS with no quota and 7.5TB free. The Qwen
# caches are ~90GB (16GB pooled features + ~74GB of per-token states), so they cannot live beside the
# code there. Set it in `pbs/_env.sh` per cluster, not per script.
#
# A MIS-SET CACHE ROOT MUST NOT BE SILENT. Pointing at a path that does not exist would otherwise
# look like "no cache found", and every driver would cheerfully regenerate from scratch into a new
# location -- or worse, split one dataset's cache across two roots. So the override must name a
# directory that already exists, and we crash if it does not.
_cache_env = os.environ.get("LUQ_CACHE_ROOT", "").strip()
if _cache_env:
    CACHE_DIR = Path(_cache_env)
    if not CACHE_DIR.is_dir():
        raise SystemExit(
            f"LUQ_CACHE_ROOT={_cache_env!r} is not an existing directory.\n"
            "Refusing to run: a wrong cache root does not fail, it silently regenerates everything "
            "into the wrong place (or splits one dataset across two roots). Create the directory "
            "deliberately, or unset the variable to use the default cache/ beside the code.")
else:
    CACHE_DIR = REPO_ROOT / "cache"

# Results stay WITH THE CODE on every machine: they are small, they are the precious artifact, and
# they are what gets committed and compared. Only the big regenerable caches move.
#
# Since 2026-08-14 `<repo>/results` is a SYMLINK to `../results`, i.e. to a directory outside
# this repo, where results are version-controlled (they previously sat in no repository at all).
# Nothing here changes: the path still resolves, so this constant, the ~130 per-file
# `ROOT / "results"` idioms, the 148 PBS scripts and the 6 hardcoded absolute paths all keep working.
# That transparency is exactly why the symlink was chosen over moving the directory.
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
