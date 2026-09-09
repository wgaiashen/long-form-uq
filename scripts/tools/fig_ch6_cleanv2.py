#!/usr/bin/env python

# AI assistance: the plotting code in this file was drafted with Claude Code (Anthropic),
# then reviewed, corrected and tested by the author. The figure design, the quantities
# plotted and their interpretation are the author's own. See ACKNOWLEDGEMENTS.md.
"""Mechanism-chapter figures, rebuilt on the corrected-span population.

Every plotted value is read from a result artifact and re-emitted as the figure's own data file, so
each panel can be checked against a CSV rather than trusted. No number is typed into this script.

The figures keep the scientific purpose of the ones they replace; this is a population migration,
not a redesign.

  probability_aggregation   the three fixed aggregators by target. Purpose: show that the preferred
                            endpoint varies by target, so aggregation is not a neutral choice.
  lehmer_regimes            the mean-to-extreme curve per target. Purpose: show that the interior
                            optimum sits in different places, and that the endpoints are the two
                            existing baselines.
  response_length           median retained length against the extreme-minus-mean gap. Purpose: test
                            whether length selects the regime at the dataset level. It does not.
  probe_drift               PRR by transfer rung for the supervised estimators. Purpose: show how
                            much matched-setting performance survives as the source moves away.
  relevance_uniformity      how close the external relevance weights are to uniform. Purpose: show
                            that this comparator is not independent of mean token NLL here.

    python scripts/tools/fig_ch6_cleanv2.py
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SLUG = "meta-llama_Meta-Llama-3.1-8B"
A = ROOT / "results" / "analysis"
FIGDIR = A / "CH6_CLEANV2_FIGURES"
DATADIR = A / "CH6_CLEANV2_FIGURE_DATA"
POPULATION = "Llama-3.1-8B, corrected span, eight long-form targets"

EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
PRETTY = {"pubmed_qa": "PubMedQA", "med_quad": "MedQuAD", "asqa": "ASQA", "xsum": "XSum",
          "cnn_dailymail": "CNN/DailyMail", "samsum": "SAMSum", "expertqa": "ExpertQA",
          "factscore": "FActScore"}
RUNGS = ["ID", "SameTask-long", "LOO-long", "DiffTask-long", "1ds-Diff-long"]
RUNG_LABEL = {"ID": "matched", "SameTask-long": "same task", "LOO-long": "leave one out",
              "DiffTask-long": "different task", "1ds-Diff-long": "one different-task source"}


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def emit(name, fieldnames, rows):
    p = DATADIR / f"{name}.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"  data  {p.relative_to(ROOT)}")


def save(fig, name):
    p = FIGDIR / f"{name}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig   {p.relative_to(ROOT)}")


def fig_probability_aggregation():
    src = read(A / "ch6_cleanv2_probability_aggregates.csv")
    per = {r["dataset"]: r for r in src if r["dataset"] in EVALS}
    cols = ["Sequence NLL", "Mean token NLL", "Minimum token probability"]
    rows = [{"dataset": d, **{c: per[d][c] for c in cols}} for d in EVALS]
    emit("fig_probability_aggregation_by_target", ["dataset"] + cols, rows)

    x = np.arange(len(EVALS)); w = 0.27
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    for j, c in enumerate(cols):
        ax.bar(x + (j - 1) * w, [float(per[d][c]) for d in EVALS], w, label=c)
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[d] for d in EVALS], rotation=20, ha="right")
    ax.set_ylabel("PRR"); ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Fixed probability aggregation by target\n{POPULATION}", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_probability_aggregation_by_target")


def fig_lehmer():
    src = read(A / f"ch6_cleanv2_lehmer_sweep__{SLUG}.csv")
    per = {r["dataset"]: r for r in src if r["scope"] == "per target"}
    macro = next(r for r in src if r["scope"].startswith("MACRO"))
    betas = ["0.0", "0.5", "1.0", "2.0", "4.0", "8.0", "16.0"]
    xs = [float(b) for b in betas]

    rows = []
    for d in EVALS:
        for b in betas + ["inf"]:
            rows.append({"dataset": d, "beta": b, "prr": per[d][f"beta={b}"]})
    for b in betas + ["inf"]:
        rows.append({"dataset": "MACRO", "beta": b, "prr": macro[f"beta={b}"]})
    emit("fig_lehmer_regimes", ["dataset", "beta", "prr"], rows)

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    cmap = plt.get_cmap("tab10")
    for j, d in enumerate(EVALS):
        ys = [float(per[d][f"beta={b}"]) for b in betas]
        ax.plot(xs, ys, "-o", ms=3.5, lw=1.3, color=cmap(j % 10), label=PRETTY[d])
        best = per[d]["best_finite_beta_descriptive"]
        ax.plot([float(best)], [float(per[d]["best_finite_prr_descriptive"])],
                marker="*", ms=12, color=cmap(j % 10), mec="k", mew=0.4, zorder=5)
    ax.plot(xs, [float(macro[f"beta={b}"]) for b in betas], "-s", ms=4.5, lw=2.4,
            color="k", label="macro")
    ax.set_xscale("symlog", linthresh=0.5)
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_xlabel("Lehmer coefficient   (0 is mean token NLL, the limit is minimum token probability)")
    ax.set_ylabel("PRR")
    ax.set_title("The mean-to-extreme aggregation continuum\n"
                 "stars mark each target's best coefficient, which is descriptive only\n"
                 f"{POPULATION}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(alpha=0.25)
    save(fig, "fig_lehmer_regimes")


def fig_response_length():
    src = read(A / f"ch6_cleanv2_length_dataset__{SLUG}.csv")
    per = {r["dataset"]: r for r in src if r["dataset"] in EVALS}
    stat = next(r for r in src if r["dataset"].startswith("SPEARMAN median"))
    rho = stat["prr_minimum_token_probability"]; pval = stat["prr_mean_token_nll"]

    rows = [{"dataset": d,
             "median_retained_length": per[d]["median_retained_length"],
             "mean_retained_length": per[d]["mean_retained_length"],
             "min_minus_mean": per[d]["min_minus_mean"]} for d in EVALS]
    emit("fig_response_length", ["dataset", "median_retained_length", "mean_retained_length",
                                 "min_minus_mean"], rows)

    x = [float(per[d]["median_retained_length"]) for d in EVALS]
    y = [float(per[d]["min_minus_mean"]) for d in EVALS]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.scatter(x, y, s=64, color="#3b6ea5", zorder=3)
    for d, xi, yi in zip(EVALS, x, y):
        ax.annotate(PRETTY[d], (xi, yi), textcoords="offset points", xytext=(7, 4), fontsize=8.5)
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_xlabel("median retained response length (tokens)")
    ax.set_ylabel("PRR(minimum token probability) - PRR(mean token NLL)")
    ax.set_title("Response length does not select the preferred aggregation endpoint\n"
                 f"Spearman {rho}, {pval}, n = 8\n{POPULATION}", fontsize=10)
    ax.grid(alpha=0.25)
    save(fig, "fig_response_length")


def fig_probe_drift():
    src = read(A / "ch6_cleanv2_probe_drift.csv")
    fitted = [r for r in src if r["fitted_on_source_pool"] == "True"]
    free = [r for r in src if r["fitted_on_source_pool"] == "False"]

    rows = []
    for r in fitted + free:
        for rung in RUNGS:
            rows.append({"method": r["method"], "rung": rung, "prr": r[rung],
                         "fitted_on_source_pool": r["fitted_on_source_pool"]})
    emit("fig_probe_drift", ["method", "rung", "prr", "fitted_on_source_pool"], rows)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(RUNGS))
    cmap = plt.get_cmap("tab10")
    for j, r in enumerate(fitted):
        ax.plot(x, [float(r[k]) for k in RUNGS], "-o", ms=5, lw=1.7,
                color=cmap(j % 10), label=r["method"])
    for r in free:
        ax.axhline(float(r["ID"]), ls=":", lw=1.1, color="0.45")
        ax.annotate(r["method"], (len(RUNGS) - 1, float(r["ID"])), fontsize=7.5, color="0.35",
                    textcoords="offset points", xytext=(6, -3))
    ax.set_xticks(x); ax.set_xticklabels([RUNG_LABEL[k] for k in RUNGS], rotation=15, ha="right")
    ax.set_ylabel("mean PRR over the eight targets")
    ax.set_title("How much matched-setting performance survives as the labelled source moves away\n"
                 "dotted lines are the training-free scores, which do not depend on the source\n"
                 f"{POPULATION}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    save(fig, "fig_probe_drift")


def fig_relevance_uniformity():
    src = [r for r in read(A / f"ch6_cleanv2_tokensar_diagnostic__{SLUG}.csv")
           if r["relevance_cache"] != "MISSING"]
    order = [d for d in EVALS if d in {r["dataset"] for r in src}]
    per = {r["dataset"]: r for r in src}
    rows = [{"dataset": d,
             "spearman_vs_mean_token_nll": per[d]["spearman_vs_mean_token_nll"],
             "median_normalised_weight_entropy": per[d]["median_normalised_weight_entropy"],
             "min_normalised_weight_entropy": per[d]["min_normalised_weight_entropy"]}
            for d in order]
    emit("fig_relevance_uniformity", list(rows[0].keys()), rows)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    y = np.arange(len(order))
    a1.barh(y, [float(per[d]["spearman_vs_mean_token_nll"]) for d in order], color="#3b6ea5")
    a1.set_yticks(y); a1.set_yticklabels([PRETTY[d] for d in order], fontsize=8.5)
    a1.set_xlim(0.9, 1.0); a1.invert_yaxis()
    a1.set_xlabel("Spearman against mean token NLL")
    a1.set_title("the relevance-weighted score barely differs\nin rank from plain averaging",
                 fontsize=9.5)
    a1.grid(axis="x", alpha=0.25)

    a2.barh(y, [float(per[d]["median_normalised_weight_entropy"]) for d in order], color="#8a5a9e")
    a2.set_yticks(y); a2.set_yticklabels([]); a2.invert_yaxis()
    a2.set_xlim(0.9, 1.005)
    a2.axvline(1.0, color="0.35", lw=1.0, ls="--")
    a2.set_xlabel("median normalised weight entropy   (1.0 is exactly uniform)")
    a2.set_title("because its weights are close to uniform\nover a long generation", fontsize=9.5)
    a2.grid(axis="x", alpha=0.25)
    # tight_layout fights the saver's bbox_inches="tight", so header room is reserved explicitly:
    # the axes stop at 0.72 and the two-line figure title sits above them.
    fig.subplots_adjust(top=0.72, wspace=0.08)
    fig.suptitle("External relevance is not an independent comparator here\n" + POPULATION,
                 fontsize=10.5, y=0.98)
    save(fig, "fig_relevance_uniformity")


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    DATADIR.mkdir(parents=True, exist_ok=True)
    for fn in (fig_probability_aggregation, fig_lehmer, fig_response_length,
               fig_probe_drift, fig_relevance_uniformity):
        print(fn.__name__)
        fn()


if __name__ == "__main__":
    main()
