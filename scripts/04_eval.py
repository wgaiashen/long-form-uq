"""Score a method: write the results CSV and report PRR.

    python scripts/04_eval.py --dataset sciq --ood ID

MSP comes free here too: it needs no probe, just the cached token logprobs, so it is
the natural unsupervised anchor to print alongside SAPLMA.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, data, msp, results  # noqa: E402
from luq.config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--label-field", default="correctness",
                    help="which correctness field to PRR against (e.g. correctness_alignscore "
                         "for AlignScore-eval, or correctness for the judge). Lets us run the "
                         "train x eval matrix. The CSV always writes it into the 'correctness' column.")
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag (must match the one used by 01_extract).")
    args = ap.parse_args()

    # Which label does the `correctness` column actually hold? Stamp it into the CSV so the
    # judge-vs-AlignScore mix-up (a recurring gotcha) can never happen again by reading a file.
    # See results/README_LABELS.md.
    JUDGE_MODEL = {"sciq": "gpt-5", "trivia_qa": "gpt-5", "coqa": "gpt-5",
                   "pubmed_qa": "gpt-5-mini", "xsum": "gpt-5-mini", "cnn_dailymail": "gpt-5-mini",
                   "med_quad": "gpt-5-mini", "expertqa": "gpt-5-mini"}

    def label_model_of(label_field, dataset):
        if label_field == "correctness":
            return JUDGE_MODEL.get(dataset, "gpt-5-mini(judge)")
        if label_field == "correctness_alignscore":
            return "yzha/AlignScore-large"
        if label_field == "correctness_strmatch":
            return "string-match"
        return label_field

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)
    # Eval ground truth = the chosen label field. Fail loudly if a test record lacks it.
    lf = args.label_field
    bad = [i for i, r in enumerate(records)
           if r["split"] == "test" and not isinstance(r.get(lf), (int, float))]
    if bad:
        sys.exit(f"ERROR: {len(bad)} test records have no numeric '{lf}' label "
                 f"(e.g. AlignScore/judge not run for this field). Label it first.")

    # MSP needs no probe: it comes straight off the cached token logprobs, for
    # every record. Both aggregates, since they can differ a lot: "mean" is the
    # whole-sequence view, "min" is the single least-confident token. SAPLMA's
    # scores were produced by 03_probe.py for the TEST split only, in test-record
    # order.
    msp_mean = [msp.msp_uncertainty(r["token_logprobs"], "mean") for r in records]
    msp_min = [msp.msp_uncertainty(r["token_logprobs"], "min") for r in records]
    msp_sum = [msp.msp_uncertainty(r["token_logprobs"], "sum") for r in records]
    # Perplexity = length-normalised MSP (mean per-token negative log-likelihood). Free from
    # the same cached logprobs; this is lm-polygraph's `Perplexity` estimator and the
    # "Perplexity" baseline (see msp.py for the mean-NLL-vs-exp naming note).
    perplexity = [msp.msp_uncertainty(r["token_logprobs"], "perplexity") for r in records]

    # Unsupervised P(True): the model's own yes/no answer at the verdict position, written into
    # the records by 01g_ptrue_unsup (a GPU step). Like MSP it is one score per example with no
    # probe, so surface it here when it has been extracted, else skip it.
    have_ptrue_unsup = all(isinstance(r.get("ptrue_unsup"), (int, float)) for r in records)
    ptrue_unsup = [r["ptrue_unsup"] for r in records] if have_ptrue_unsup else None

    test_positions = [i for i, r in enumerate(records) if r["split"] == "test"]

    # Supervised methods (03_probe) write one uncertainty per TEST example, in
    # test-record order. Load whichever have been run and skip the rest, so this
    # eval works after SAPLMA alone or after both SAPLMA and P(True).
    # saplma = the A&M MLP; linear = the linear-probe baseline on the same hidden states.
    # Each is loaded only if 03_probe has produced it, so the table grows as methods run.
    sup = {}  # method -> {"unc": array, "layer": int, "at": {record_pos: unc}}
    for m in ["saplma", "linear", "ptrue", "ptrue_accurate", "lookback"]:
        try:
            s = cache.load_scores(cfg.cache_dir, key, method=m)
        except FileNotFoundError:
            continue
        unc, layer = s["unc"], int(s["layer"])
        assert len(unc) == len(test_positions), \
            f"{m} scores out of step — rerun 03 --method {m}"

        # Freshness/provenance guard: refuse to serve a score whose inputs changed since it was
        # written. 03_probe stamps (a) the feature file + mtime — catches re-extraction; and
        # (b) the records file mtime — catches a RELABEL (correctness changed but features didn't,
        # which the feature check alone would miss; e.g. sciq string-match -> judge).
        if "feat_method" in s and "feat_mtime" in s:
            fp = cache.features_path(cfg.cache_dir, key, str(s["feat_method"]))
            if not fp.exists():
                sys.exit(f"ERROR: {m} scores reference missing features {fp.name}; "
                         f"rerun: scripts/03_probe.py --method {m}")
            if abs(fp.stat().st_mtime - float(s["feat_mtime"])) > 1e-6:
                sys.exit(f"ERROR: STALE {m} scores — its features ({fp.name}) were regenerated "
                         f"after the scores were written. Rerun: scripts/03_probe.py --method {m}")
        else:
            print(f"WARNING: {m} scores have no feature provenance stamp (written before this guard "
                  f"existed); cannot verify freshness — rerun scripts/03_probe.py --method {m}.")

        if "rec_mtime" in s:
            rp = cache.records_path(cfg.cache_dir, key)
            if abs(rp.stat().st_mtime - float(s["rec_mtime"])) > 1e-6:
                sys.exit(f"ERROR: STALE {m} scores — the records/labels ({rp.name}) were rewritten "
                         f"(e.g. relabelled) after the probe was trained, so the probe is fit to the "
                         f"OLD label. Rerun: scripts/03_probe.py --method {m}")
        else:
            print(f"WARNING: {m} scores have no label-provenance stamp; a relabel since training "
                  f"would go undetected — rerun scripts/03_probe.py --method {m} to stamp it.")

        sup[m] = {"unc": unc, "layer": layer, "at": dict(zip(test_positions, unc))}

    # One CSV row per example; both methods as columns (SAPLMA blank on train
    # rows, since the probe never scores its own training data).
    rows = []
    for i, r in enumerate(records):
        row = {
            "dataset": cfg.dataset,
            "task": data.TASK_OF[cfg.dataset],
            "split": r["split"],
            "correctness": r[lf],
            "label_field": lf,
            "label_model": label_model_of(lf, cfg.dataset),
            "msp_mean": msp_mean[i],
            "msp_min": msp_min[i],
            "msp_sum": msp_sum[i],
            "perplexity": perplexity[i],
        }
        if ptrue_unsup is not None:
            row["ptrue_unsup"] = ptrue_unsup[i]
        # Each supervised method is blank on train rows (the probe never scores its
        # own training data).
        for m in sup:
            row[m] = float(sup[m]["at"][i]) if i in sup[m]["at"] else ""
        rows.append(row)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.results_dir / f"{key}.csv"
    unsup_cols = ["msp_mean", "msp_min", "msp_sum", "perplexity"]
    if ptrue_unsup is not None:
        unsup_cols.append("ptrue_unsup")
    results.write_csv(csv_path, rows, [*unsup_cols, *sup.keys()],
                      meta_cols=["label_field", "label_model"])

    # PRR is computed on the test split only: SAPLMA has no train scores, and
    # MSP must be compared on the identical examples to be a fair anchor.
    y_test = [records[i][lf] for i in test_positions]
    print(f"{cfg.dataset} {cfg.ood_setting} | test n={len(y_test)} "
          f"| mean correctness {sum(y_test) / len(y_test):.3f}")
    unsup = [("MSP mean ", msp_mean), ("MSP min  ", msp_min),
             ("MSP sum  ", msp_sum), ("Perplexity", perplexity)]
    if ptrue_unsup is not None:
        unsup.append(("P(True) uns", ptrue_unsup))
    for name, unc in unsup:
        unc_test = [unc[i] for i in test_positions]
        print(f"PRR  {name}        : {results.prr(y_test, unc_test):.3f}")
    for m in sup:
        print(f"PRR  {m:8s} (layer {sup[m]['layer']}): "
              f"{results.prr(y_test, sup[m]['unc']):.3f}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
