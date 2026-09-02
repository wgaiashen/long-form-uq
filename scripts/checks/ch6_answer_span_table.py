#!/usr/bin/env python
"""Answer-span restricted aggregation against unrestricted aggregation, on the corrected span.

The like-for-like comparison already computes every quantity this needs, on the corrected-span
population, with the span located by a language model and never by the gold answer. This reshapes
those rows into the form the mechanism chapter reports and adds the two coverage quantities that
must be read alongside the result:

  located rate       the share of responses in which the extractor found a span
  fallback rate      1 - located rate. On those responses the restricted score falls back to all
                     tokens, so the restricted and unrestricted scores are equal there by
                     construction. A restricted score that looks close to the unrestricted one on a
                     dataset with a low located rate is partly an artifact of that fallback, which
                     is why the rate is carried in the same table rather than in a footnote.

    python scripts/checks/ch6_answer_span_table.py
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLUG = "meta-llama_Meta-Llama-3.1-8B"
SRC = ROOT / "results" / "hybrids" / f"likeforlike__{SLUG}.csv"
OUT = ROOT / "results" / "analysis" / f"ch6_cleanv2_answer_span__{SLUG}.csv"
POPULATION = "meta-llama/Meta-Llama-3.1-8B, corrected span (med_quad), eight long-form targets"

EVALS = ["asqa", "cnn_dailymail", "expertqa", "factscore",
         "med_quad", "pubmed_qa", "samsum", "xsum"]
COLUMNS = [
    ("answer-span sequence NLL", "answer-span sequence NLL"),
    ("answer-span mean NLL", "answer-span mean NLL"),
    ("sequence NLL (published MSP)", "full-response sequence NLL"),
    ("mean token NLL", "full-response mean NLL"),
]


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    src = list(csv.DictReader(open(SRC)))
    by = {(r["dataset"], r["method"]): r for r in src}

    missing = [(d, m) for d in EVALS for m, _ in COLUMNS if (d, m) not in by]
    if missing:
        raise SystemExit(f"like-for-like table is missing {missing}; refusing a partial table")

    print("=" * 104)
    print("ANSWER-SPAN RESTRICTED AGGREGATION")
    print(f"population: {POPULATION}")
    print("Unlocated responses fall back to all tokens, so read the located rate with the PRR.")
    print("=" * 104)
    header = f"{'dataset':14s}" + "".join(f"{lbl[:22]:>24s}" for _, lbl in COLUMNS) \
             + f"{'located':>10s}{'fallback':>10s}"
    print(header)

    rows, acc = [], {lbl: [] for _, lbl in COLUMNS}
    for d in EVALS:
        loc = by[(d, "answer-span mean NLL")]["located_rate"]
        loc = float(loc) if loc not in ("", None) else float("nan")
        row = {"dataset": d, "n_test": by[(d, COLUMNS[0][0])]["n_test"]}
        for m, lbl in COLUMNS:
            v = float(by[(d, m)]["prr"])
            row[lbl] = round(v, 4)
            acc[lbl].append(v)
        row["span located rate"] = round(loc, 4)
        row["fallback rate"] = round(1.0 - loc, 4)
        row["population"] = POPULATION
        rows.append(row)
        print(f"{d:14s}" + "".join(f"{row[lbl]:>+24.4f}" for _, lbl in COLUMNS)
              + f"{loc:>10.3f}{1.0 - loc:>10.3f}")

    macro = {"dataset": "MACRO (eight targets)", "n_test": "",
             "span located rate": round(sum(r["span located rate"] for r in rows) / len(rows), 4),
             "fallback rate": round(sum(r["fallback rate"] for r in rows) / len(rows), 4),
             "population": POPULATION}
    for _, lbl in COLUMNS:
        macro[lbl] = round(sum(acc[lbl]) / len(acc[lbl]), 4)
    rows.append(macro)
    print(f"{'MACRO':14s}" + "".join(f"{macro[lbl]:>+24.4f}" for _, lbl in COLUMNS)
          + f"{macro['span located rate']:>10.3f}{macro['fallback rate']:>10.3f}")

    delta = macro["answer-span mean NLL"] - macro["full-response mean NLL"]
    print(f"\nRestricting to the located span changes the macro by {delta:+.4f} under the mean "
          f"aggregation, on a located rate of {macro['span located rate']:.3f}.")

    fields = ["dataset", "n_test"] + [lbl for _, lbl in COLUMNS] \
             + ["span located rate", "fallback rate", "population"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
