#!/usr/bin/env python
"""Does letting the sharpening strength depend on response length improve the aggregation.

The fixed sharpening rule concentrates weight on high-surprisal tokens by a constant amount. This
variant lets that amount scale with the response's own length: strength(length) = tau0 *
(length / reference) ** gamma, with the reference fixed at 100 tokens. A positive gamma makes longer
responses weighted more extremely, a negative one less, and **gamma = 0 removes the length dependence
entirely and recovers the fixed rule**. So the question has a clean answer shape: if the best grid
point sits at gamma = 0, length adds nothing.

The grid is frozen and reported whole: tau0 in {0.5, 1, 2, 4} by gamma in {-1, -0.5, 0, 0.5, 1}.

THE BEST GRID POINT HERE IS DESCRIPTIVE. It is chosen against the targets' own labels, so it is an
upper bound on what any selector could reach. That is precisely why a gamma of zero is informative:
even given the answers, length does not earn its place.

    python scripts/checks/ch6_length_conditioned_table.py --sweep <csv> --out <csv>
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SLUG = "meta-llama_Meta-Llama-3.1-8B"
A = ROOT / "results" / "analysis"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
PARAM = re.compile(r"^t(?P<tau>[0-9.]+)_g(?P<gamma>-?[0-9.]+)$")



def _rel(p):
    """Path for display. A path given on the command line need not sit under the repository."""
    p = Path(p)
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default=str(A / f"ch6_cleanv2_sharpening_family__{SLUG}__lengthtau.csv"))
    ap.add_argument("--out", default=str(A / f"ch6_cleanv2_length_conditioned_sharpening__{SLUG}.csv"))
    ap.add_argument("--population", default="meta-llama/Meta-Llama-3.1-8B, corrected span")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.sweep)) if r["family"] == "length_tau"]
    if not rows:
        raise SystemExit(f"{args.sweep} carries no length-conditioned rows")
    prr = {(r["dataset"], r["param"]): float(r["prr"]) for r in rows}

    grid = sorted({r["param"] for r in rows})
    parsed = {p: PARAM.match(p) for p in grid}
    bad = [p for p, m in parsed.items() if m is None]
    if bad:
        raise SystemExit(f"unparsable grid points {bad}; the table reports the registered grid only")
    # Map (tau, gamma) back to the EXACT parameter string the sweep wrote. Reformatting the numbers
    # would be a guess about the writer's float formatting, and a near-miss key would look like a
    # missing grid point rather than a lookup bug.
    key_of = {(float(m["tau"]), float(m["gamma"])): p for p, m in parsed.items()}
    taus = sorted({float(parsed[p]["tau"]) for p in grid})
    gammas = sorted({float(parsed[p]["gamma"]) for p in grid})
    if len(grid) != len(taus) * len(gammas):
        raise SystemExit(f"grid is not complete: {len(grid)} points for {len(taus)}x{len(gammas)}")

    print("=" * 96)
    print(f"LENGTH-CONDITIONED SHARPENING   population: {args.population}")
    print(f"strength = tau0 * (length / 100) ** gamma;  gamma = 0 recovers the fixed rule")
    print(f"grid: tau0 in {taus}, gamma in {gammas}")
    print("=" * 96)
    print(f"{'gamma':>8s}" + "".join(f"{('tau0=' + str(t)):>12s}" for t in taus))

    out, macro = [], {}
    for g in gammas:
        line = f"{g:>8.1f}"
        for t in taus:
            p = key_of[(t, g)]
            vals = [prr[(d, p)] for d in LONG if (d, p) in prr]
            if len(vals) != len(LONG):
                raise SystemExit(f"grid point {p} covers {len(vals)} of {len(LONG)} targets")
            m = float(np.mean(vals))
            macro[(t, g)] = m
            line += f"{m:>+12.4f}"
            out.append({"scope": "grid point", "tau0": t, "gamma": g,
                        "macro_prr": round(m, 4),
                        **{d: round(prr[(d, p)], 4) for d in LONG},
                        "population": args.population})
        print(line)

    best = max(macro, key=macro.get)
    fixed_best = max(((t, 0.0) for t in taus), key=macro.get)
    print(f"\nBest grid point: tau0 = {best[0]:g}, gamma = {best[1]:g}, macro {macro[best]:+.4f}")
    print(f"Best point with the length dependence removed (gamma = 0): tau0 = {fixed_best[0]:g}, "
          f"macro {macro[fixed_best]:+.4f}")
    verdict = ("the selected length coefficient is ZERO, so allowing the sharpening strength to "
               "depend on response length does not improve the aggregation"
               if best[1] == 0.0 else
               f"the selected length coefficient is {best[1]:g}, and it gains "
               f"{macro[best] - macro[fixed_best]:+.4f} over removing the length dependence")
    print(f"Verdict: {verdict}")
    out.append({"scope": "best grid point (descriptive)", "tau0": best[0], "gamma": best[1],
                "macro_prr": round(macro[best], 4),
                **{d: "" for d in LONG}, "population": args.population})
    out.append({"scope": "best with gamma = 0 (the fixed rule)", "tau0": fixed_best[0],
                "gamma": 0.0, "macro_prr": round(macro[fixed_best], 4),
                **{d: "" for d in LONG}, "population": args.population})

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "tau0", "gamma", "macro_prr"] + LONG
                                          + ["population"])
        w.writeheader(); w.writerows(out)
    print(f"wrote {_rel(Path(args.out))}")


if __name__ == "__main__":
    main()
