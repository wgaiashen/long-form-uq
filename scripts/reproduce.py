"""One-command reproduction of the cached ID results tables (Phase A #4).

From the cached Tier-1 records + Tier-2 features + labels, this regenerates every
supervised score (03_probe) and PRR table (04_eval) — NO GPU. This is the step a marker
or supervisor runs to reproduce the worklog ID tables from the cache.

It does NOT re-run the heavy GPU extraction (01_extract / 01b_ptrue / 01c_lookback) or the
labelling (02_label); those produce the cache this reads. See README.md for the full
from-scratch flow.

    python scripts/reproduce.py                  # all ID datasets, all methods
    python scripts/reproduce.py --dataset xsum   # just one

Each method uses 03_probe's default middle layer (saplma/linear/ptrue -> middle, lookback
-> 0), which is exactly the layer the worklog reports, so the printed PRRs should match it.
saplma is the A&M 4-layer MLP; linear is the linear-probe baseline on the same hidden states.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = ["sciq", "pubmed_qa", "xsum"]
METHODS = ["saplma", "linear", "ptrue", "lookback"]


def run(rel_cmd):
    """Run one pipeline script with the current Python, echoing the command first."""
    print("›", "python", *rel_cmd, flush=True)
    subprocess.run([sys.executable, *rel_cmd], cwd=REPO, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, help="just one dataset (default: all)")
    args = ap.parse_args()
    datasets = [args.dataset] if args.dataset else DATASETS

    for ds in datasets:
        # Retrain each probe from cached features (deterministic, sub-second, no GPU),
        # then score + print the PRR table for the dataset.
        for m in METHODS:
            run(["scripts/03_probe.py", "--dataset", ds, "--method", m])
        run(["scripts/04_eval.py", "--dataset", ds, "--ood", "ID"])


if __name__ == "__main__":
    main()
