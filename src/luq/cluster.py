"""Cluster-resolution layer (Python twin of ``pbs/_env.sh``).

One place that answers "which cluster am I on, and where do the caches live?", so no
new code has to hardcode a ``/vol/...`` path. The existing pipeline does not need this:
``config.py`` already derives ``REPO_ROOT`` from this file's location, so the data path
is portable by construction. What is *not* portable is the **environment** -- the HF
cache root and the home of the python env -- which the batch wrappers set. This module
centralises those values and exposes them to:

  * any Python that wants a cluster-aware path (e.g. a scratch dir for a check script);
  * shell, via the tiny CLI at the bottom: ``python -m luq.cluster hf_home``.

We support two clusters:

  * ``doc`` -- the Department of Computing GPU cluster (Slurm). Everything sits on the
    ``/vol/gpudata`` CephFS allocation.
  * ``rcs`` -- the central RCS HPC, CX3 Phase 2 (PBSPro). Home has a large (~930GB)
    allocation, so the repo + caches live in ``$HOME`` (just not ``$EPHEMERAL``, which is
    wiped after 30 days).

Resolution mirrors ``pbs/_env.sh`` exactly; keep the two in step. Every value can be
overridden by the matching environment variable (``LUQ_CLUSTER``, ``HF_HOME``, ...).
"""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path

# .../msc-project-gs925 (this file is src/luq/cluster.py -> up three parents). Matches
# config.REPO_ROOT; kept independent so this module has no import side effects.
REPO_ROOT = Path(__file__).resolve().parents[2]


def detect_cluster() -> str:
    """Return ``"doc"`` or ``"rcs"``.

    Honours ``LUQ_CLUSTER`` if set, else guesses from the hostname. Unknown hosts fall
    back to ``"doc"`` here (the inverse of ``pbs/_env.sh``, which defaults to ``rcs``):
    a ``.pbs`` script only ever runs on RCS, but this Python module is most often
    imported on the existing DoC cluster, so DoC is the safer default for code paths.
    """
    env = os.environ.get("LUQ_CLUSTER", "").strip().lower()
    if env in {"doc", "rcs"}:
        return env
    host = socket.getfqdn() or socket.gethostname()
    host = host.lower()
    if "gpucluster" in host or host.endswith(".doc.ic.ac.uk"):
        return "doc"
    if "cx3" in host or "hpc" in host or ".rcs." in host:
        return "rcs"
    return "doc"


@dataclass(frozen=True)
class ClusterSettings:
    """Resolved cluster-specific settings. Build with :func:`get_settings`."""

    name: str           # "doc" | "rcs"
    repo_root: Path     # the repo checkout
    cache_root: Path    # repo cache/ (Tier-1/2 features); regenerable per cluster
    hf_home: Path       # HuggingFace cache root (a large persistent volume)
    venv_activate: str | None   # path to a venv activate script, if used (rcs)
    cuda_setup: str | None      # a CUDA setup.sh to source, if any (doc historically)


def get_settings(cluster: str | None = None) -> ClusterSettings:
    """Resolve all cluster settings, letting env vars win over the defaults."""
    name = (cluster or detect_cluster())
    repo_root = Path(os.environ.get("LUQ_REPO", REPO_ROOT))
    cache_root = repo_root / "cache"

    if name == "doc":
        hf_home = Path(os.environ.get(
            "HF_HOME", "/vol/gpudata/gs925-msc_project/hf_cache"))
        # DoC historically did NOT source system CUDA (PyTorch bundles its own; sourcing
        # caused libcusparse conflicts), so this stays None unless explicitly set.
        cuda_setup = os.environ.get("LUQ_CUDA_SETUP") or None
        venv_activate = os.environ.get("LUQ_VENV")  # doc uses conda, so usually None
        venv_activate = f"{venv_activate}/bin/activate" if venv_activate else None
    else:
        # ---- rcs ----
        # Mirror pbs/_env.sh: RCS home has a large (~930GB) allocation, so cache + venv
        # live in $HOME. NOT $EPHEMERAL (wiped after 30 days).
        home = Path(os.environ.get("HOME", repo_root.parent))
        hf_home = Path(os.environ.get("HF_HOME", home / "hf_cache"))
        cuda_setup = os.environ.get("LUQ_CUDA_SETUP") or None  # RCS uses module load
        venv = os.environ.get("LUQ_VENV", str(home / "venv"))
        venv_activate = f"{venv}/bin/activate"

    return ClusterSettings(
        name=name,
        repo_root=repo_root,
        cache_root=cache_root,
        hf_home=hf_home,
        venv_activate=venv_activate,
        cuda_setup=cuda_setup,
    )


# Convenience accessors -------------------------------------------------------
def cache_root() -> Path:
    return get_settings().cache_root


def hf_home() -> Path:
    return get_settings().hf_home


def scratch_dir(name: str) -> Path:
    """A per-cluster scratch path under the cache (regenerable, gitignored).

    Handy for check scripts that currently hardcode a ``/vol/...`` scratch dir.
    """
    return get_settings().cache_root / "scratch" / name


_CLI_KEYS = {
    "cluster": lambda s: s.name,
    "repo_root": lambda s: str(s.repo_root),
    "cache_root": lambda s: str(s.cache_root),
    "hf_home": lambda s: str(s.hf_home),
    "venv_activate": lambda s: s.venv_activate or "",
    "cuda_setup": lambda s: s.cuda_setup or "",
}


def _main(argv: list[str]) -> int:
    """Tiny CLI so shell can read a resolved value: ``python -m luq.cluster hf_home``."""
    s = get_settings()
    if not argv:
        for k, fn in _CLI_KEYS.items():
            print(f"{k}={fn(s)}")
        return 0
    key = argv[0]
    if key not in _CLI_KEYS:
        import sys
        print(f"unknown key {key!r}; choose from {', '.join(_CLI_KEYS)}", file=sys.stderr)
        return 2
    print(_CLI_KEYS[key](s))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
