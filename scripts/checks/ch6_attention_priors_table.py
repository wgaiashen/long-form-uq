#!/usr/bin/env python
"""Does biasing the learned attention towards promising positions help.

The learned pooler decides where to read the hidden-state sequence. These variants give it a fixed
prior about where to look, applied to the attention logits before normalisation, in two forms: the
prior frozen in place, and the prior used only as an initialisation the query may move away from.

The comparison that matters is against the two controls the variants must beat to be worth having:

  learned query   the incumbent pooler with no prior at all. A prior earns its place only by beating
                  this. Beating uniform pooling is not enough, because the learned query already does.
  uniform         every position weighted equally. The reference floor, not the bar.

Reported per variant: matched-setting and mean shifted PRR, and the paired comparison against each
control with the dataset as the unit of analysis.

    python scripts/checks/ch6_attention_priors_table.py --ladder <csv> --out <csv>
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "checks"))
from report_pack_stats import EVALS, OOD_RUNGS, paired            # noqa: E402

SLUG = "meta-llama_Meta-Llama-3.1-8B"
CONTROLS = {"armA": "learned query, no prior", "armB": "uniform pooling"}
VARIANTS = {
    "armC_content_mass": "frozen prior, semantic relevance",
    "armC_nll": "frozen prior, probability derived",
    "armD_content_mass": "prior as initialisation, semantic relevance",
    "armD_nll": "prior as initialisation, probability derived",
}


def _rel(p):
    p = Path(p)
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--population", default="meta-llama/Meta-Llama-3.1-8B, corrected span")
    args = ap.parse_args()

    d = defaultdict(dict)
    for r in csv.DictReader(open(args.ladder)):
        if not r["method"].startswith("VERDICT"):
            d[r["method"]][(r["rung"], r["eval"])] = float(r["prr_mean"])

    missing = [m for m in list(CONTROLS) + list(VARIANTS) if m not in d]
    if missing:
        raise SystemExit(f"ladder is missing {missing}; refusing a partial comparison")
    for m in list(CONTROLS) + list(VARIANTS):
        gaps = [(rg, e) for rg in ["ID"] + OOD_RUNGS for e in EVALS if (rg, e) not in d[m]]
        if gaps:
            raise SystemExit(f"{m} is missing {len(gaps)} cells, first {gaps[:3]}; "
                             "refusing to average a partial grid")

    def ood(m):
        return np.array([np.mean([d[m][(rg, e)] for rg in OOD_RUNGS]) for e in EVALS])

    def matched(m):
        return np.array([d[m][("ID", e)] for e in EVALS])

    print("=" * 112)
    print(f"POSITION PRIORS ON THE HIDDEN-STATE BRANCH   population: {args.population}")
    print("A prior earns its place by beating the LEARNED QUERY, not by beating uniform pooling.")
    print("=" * 112)
    print(f"{'variant':46s}{'matched':>10s}{'shifted':>10s}"
          f"{'vs learned query':>30s}{'vs uniform':>30s}")

    rows = []
    for m, label in CONTROLS.items():
        rows.append({"variant": m, "description": label,
                     "matched_prr": round(float(matched(m).mean()), 4),
                     "mean_shifted_prr": round(float(ood(m).mean()), 4),
                     "vs_learned_query": "", "vs_uniform": "", "role": "control",
                     "population": args.population})
        print(f"{label:46s}{matched(m).mean():>+10.4f}{ood(m).mean():>+10.4f}"
              f"{'(control)':>30s}")

    for m, label in VARIANTS.items():
        cmp_txt = {}
        for base in ("armA", "armB"):
            r = paired(list(ood(m) - ood(base)))
            cmp_txt[base] = (f"{r['macro_mean']:+.4f} {r['n_positive']}/{r['n']} "
                             f"p={r['wilcoxon_p_exact']:.3f}")
        rows.append({"variant": m, "description": label,
                     "matched_prr": round(float(matched(m).mean()), 4),
                     "mean_shifted_prr": round(float(ood(m).mean()), 4),
                     "vs_learned_query": cmp_txt["armA"], "vs_uniform": cmp_txt["armB"],
                     "role": "variant", "population": args.population})
        print(f"{label:46s}{matched(m).mean():>+10.4f}{ood(m).mean():>+10.4f}"
              f"{cmp_txt['armA']:>30s}{cmp_txt['armB']:>30s}")

    beats = [r["variant"] for r in rows if r["role"] == "variant"
             and r["vs_learned_query"].startswith("+")]
    print(f"\nVariants beating the learned query on the mean shifted setting: "
          f"{beats if beats else 'none'}")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {_rel(args.out)}")


if __name__ == "__main__":
    main()
