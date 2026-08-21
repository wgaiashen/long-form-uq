"""ONE provenance stamp, shared by every grid driver.

WHY THIS EXISTS (the 2026-08-05 XL contamination, and why it was invisible)
---------------------------------------------------------------------------
`probedriftlong.py` has stamped `git_sha` + `cluster` + `env_hash` on every row for weeks, and refuses to
run at all on a dirty tracked tree. That is why the Long grid could be AUDITED after the fact: four jobs
on four commits, and the stamps proved no numerically-relevant code differed between them.

The two XL drivers had NO provenance of any kind. `contribution_ladder` never stamped, and
`ood_onegrid`'s CSV_FIELDS did not even contain the column. The consequence was not theoretical: on
2026-08-05 three onegrid jobs (med_quad, asqa, samsum) started at 20:52 and the unlabelled-row filter was
not written until ~22:15. They loaded ExpertQA's 292 and factscore's 45 NaN labels as TRAINING TARGETS and
produced NaN for every method in every cell whose pool contained those two datasets. Nothing in the
artifacts said which code had produced them. It was caught only by noticing the NaN and hand-checking job
start times against `git log`, which is exactly the manual check that keeps failing.

So: one implementation, imported, and every driver stamps.

TWO MODES, and the choice is deliberate:
  * `strict=True`  — refuse to run on a dirty tracked tree. Correct default for a driver launched fresh.
  * `strict=False` — stamp `dirty=1` instead of refusing. For drivers with jobs already queued, where a
    late refusal would kill a job that has waited hours for a slot. The staleness stays DETECTABLE in the
    artifact, which is the property that was missing; it just does not block.

Untracked files are ignored in both modes: they do not change the committed code the SHA points at.
"""
import hashlib
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _git(*a):
    return subprocess.run(["git", *a], cwd=str(ROOT), capture_output=True, text=True).stdout.strip()


def provenance(strict=True):
    """{'git_sha', 'cluster', 'env_hash', 'dirty'} for stamping onto every result row."""
    sha = _git("rev-parse", "HEAD")
    if not sha:
        raise SystemExit("provenance: could not read git HEAD (not a repo?) -- refusing to run unversioned")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if dirty and strict:
        raise SystemExit("provenance: TRACKED working tree is DIRTY -- refusing to stamp a git_sha that "
                         f"does not reproduce. Commit or stash first, then resubmit.\n{dirty}")
    if dirty:
        print(f"provenance: tracked tree is DIRTY; stamping dirty=1 so this run is identifiable later.\n"
              f"{dirty}", flush=True)
    root = str(ROOT)
    cluster = "DoC" if root.startswith("/vol/gpudata") else ("RCS" if "/rds/" in root else socket.gethostname())
    try:
        from importlib.metadata import distributions
        pkgs = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata.get('Name'))
        env_hash = hashlib.sha256("\n".join(pkgs).encode()).hexdigest()[:12]
    except Exception as e:                              # never let env-hashing crash the run
        env_hash = f"unknown:{type(e).__name__}"
    return {"git_sha": sha, "cluster": cluster, "env_hash": env_hash, "dirty": 1 if dirty else 0}
