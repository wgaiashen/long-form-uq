#!/usr/bin/env python
"""Progress of the position-prior runs, and whether any of them has died.

The prior ladder is run one evaluation target per job, and a target whose cost does not fit one
walltime is split further by transfer setting. That means progress is spread over many files, and
the failure mode that matters is not a crash but a SILENT one: a job killed at its walltime leaves
the settings it finished, and that file is indistinguishable from one still being written.

So completeness alone is not enough. This cross-references what is on disk against what is still in
the scheduler:

  complete   all five transfer settings present
  live       incomplete, but a job for it is still queued or running
  ORPHANED   incomplete and NO job for it exists -- its job died, and it will never finish on its own

An orphaned target must be resubmitted. Reporting it as "in progress" would mean waiting forever.

    python scripts/checks/ch6_prior_ladder_status.py
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLUG = "meta-llama_Meta-Llama-3.1-8B"
A = ROOT / "results" / "analysis"
RUNGS = {"ID", "SameTask-long", "LOO-long", "DiffTask-long", "1ds-Diff-long"}
EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
ARMS = {"corrected": "ch6_cleanv2", "original": "ch6_origspan"}


def live_combinations(jobname="luq_ch6fit"):
    """(tag, target) still queued or running. Empty on a machine with no scheduler."""
    try:
        out = subprocess.run(["qstat", "-u", os.environ.get("USER", "")],
                             capture_output=True, text=True, timeout=120).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    live = set()
    for line in out.splitlines():
        if jobname not in line:
            continue
        job = line.split()[0]
        f = subprocess.run(["qstat", "-f", job], capture_output=True, text=True).stdout.replace("\n\t", "")
        ev, arm = re.search(r"LUQ_EVAL=(\w+)", f), re.search(r"LUQ_ARM=(\w+)", f)
        if ev and arm and arm.group(1) in ARMS:
            live.add((ARMS[arm.group(1)], ev.group(1)))
    return live


def settings_on_disk(tag, ev):
    parts = glob.glob(str(A / f"{tag}_fixed_prior_ladder_{ev}__{SLUG}.csv")) \
        + glob.glob(str(A / f"{tag}_fixed_prior_ladder_{ev}_*__{SLUG}.csv"))
    got = set()
    for p in parts:
        got |= {r["rung"] for r in csv.DictReader(open(p))}
    return got, len(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobname", default="luq_ch6fit")
    args = ap.parse_args()
    live = live_combinations(args.jobname)

    print("=" * 84)
    print("POSITION-PRIOR RUN STATUS")
    if live is None:
        print("no scheduler reachable; orphan detection is unavailable and is NOT reported as clean")
    print("=" * 84)

    done = orphaned = inflight = 0
    for arm, tag in ARMS.items():
        cells = []
        for ev in EVALS:
            got, nparts = settings_on_disk(tag, ev)
            if got == RUNGS:
                state, done = "*", done + 1
            elif live is None:
                state = "?"
            elif (tag, ev) in live:
                state, inflight = "", inflight + 1
            else:
                state, orphaned = " ORPHANED", orphaned + 1
            cells.append(f"{ev}:{len(got)}/5{state}")
        print(f"  {arm:10s} " + "  ".join(cells))

    total = len(ARMS) * len(EVALS)
    print(f"\n  {done}/{total} complete   {inflight} still live   {orphaned} orphaned")
    if orphaned:
        print("\n  An orphaned target's job died. It will never finish on its own and must be")
        print("  resubmitted, or excluded from the assembled grid. Do not wait on it.")
        raise SystemExit(1)
    if live is not None and done == total:
        print("\n  All targets complete. Assemble with ch6_assemble_prior_ladder.py.")


if __name__ == "__main__":
    main()
