"""Which generation-time processor setting reproduces each dataset, as a runnable answer.

Reads the measurements written by scripts/checks/m11_entropy_regime.py and prints, for each dataset,
the command-line flags that reproduce its cached token log-probabilities. A dataset whose best
configuration is still outside the tolerance is reported as unresolved and produces no flags, so the
caller cannot accidentally rebuild it under a setting that does not match.

The point of reading the measurement rather than a written-down list is that the two cannot drift
apart: whatever setting is used to rebuild an entropy cache is by construction the one that was shown
to reproduce that dataset.

    python scripts/checks/m11_entropy_settings.py --results <csv>            # human readable
    python scripts/checks/m11_entropy_settings.py --results <csv> --shell    # one line per dataset
"""
import argparse
import csv as _csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The flag spelling for each processor name used by the sweep.
FLAGS = {
    "none": "",
    "rep1.2": "--repetition-penalty 1.2",
    "rep1.3": "--repetition-penalty 1.3",
    "ngram3": "--no-repeat-ngram-size 3",
    "rep1.2+ngram3": "--repetition-penalty 1.2 --no-repeat-ngram-size 3",
    "rep1.3+ngram3": "--repetition-penalty 1.3 --no-repeat-ngram-size 3",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--shell", action="store_true",
                    help="emit 'dataset<TAB>regime<TAB>flags' for the datasets that are resolved")
    args = ap.parse_args()

    path = ROOT / args.results
    if not path.exists():
        sys.exit(f"FATAL: no measurements at {path}. Run the sweep first.")

    by = defaultdict(list)
    for r in _csv.DictReader(open(path)):
        by[r["dataset"]].append(r)

    resolved, unresolved = [], []
    for ds, rs in sorted(by.items()):
        scored = [r for r in rs if r["worst_abs_diff"]]
        if not scored:
            unresolved.append((ds, "nothing ran", None))
            continue
        best = min(scored, key=lambda r: float(r["worst_abs_diff"]))
        d, tol = float(best["worst_abs_diff"]), float(best["tolerance"])
        if d <= tol:
            resolved.append((ds, best["regime"], best["processors"], d, best["forward"]))
        else:
            unresolved.append((ds, f"best {best['processors']} at {d:.3e} > {tol:.1e}", best))

    if args.shell:
        for ds, regime, proc, _d, _fw in resolved:
            print(f"{ds}\t{regime}\t{FLAGS.get(proc, '')}")
        return 0 if resolved else 1

    print(f"{'dataset':<15} {'forward':<12} {'processors':<16} {'worst |d|':>12}  flags")
    for ds, _regime, proc, d, fw in resolved:
        print(f"{ds:<15} {fw:<12} {proc:<16} {d:>12.3e}  {FLAGS.get(proc, '(none)')}")
    if unresolved:
        print(f"\nUNRESOLVED, and no entropy may be rebuilt for these:")
        for ds, why, _ in unresolved:
            print(f"  {ds:<15} {why}")
        print("\nThey keep their partial coverage under the registered rule. The alignment tolerance")
        print("is not relaxed to move them into this table.")
    else:
        print("\nEvery dataset resolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
