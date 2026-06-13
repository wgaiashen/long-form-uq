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
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    # MSP needs no probe: it comes straight off the cached token logprobs, for
    # every record. Both aggregates, since they can differ a lot: "mean" is the
    # whole-sequence view, "min" is the single least-confident token. SAPLMA's
    # scores were produced by 03_probe.py for the TEST split only, in test-record
    # order.
    msp_mean = [msp.msp_uncertainty(r["token_logprobs"], "mean") for r in records]
    msp_min = [msp.msp_uncertainty(r["token_logprobs"], "min") for r in records]
    msp_sum = [msp.msp_uncertainty(r["token_logprobs"], "sum") for r in records]

    test_positions = [i for i, r in enumerate(records) if r["split"] == "test"]

    # Supervised methods (03_probe) write one uncertainty per TEST example, in
    # test-record order. Load whichever have been run and skip the rest, so this
    # eval works after SAPLMA alone or after both SAPLMA and P(True).
    sup = {}  # method -> {"unc": array, "layer": int, "at": {record_pos: unc}}
    for m in ["saplma", "ptrue", "lookback"]:
        try:
            s = cache.load_scores(cfg.cache_dir, key, method=m)
        except FileNotFoundError:
            continue
        unc, layer = s["unc"], int(s["layer"])
        assert len(unc) == len(test_positions), \
            f"{m} scores out of step — rerun 03 --method {m}"
        sup[m] = {"unc": unc, "layer": layer, "at": dict(zip(test_positions, unc))}

    # One CSV row per example; both methods as columns (SAPLMA blank on train
    # rows, since the probe never scores its own training data).
    rows = []
    for i, r in enumerate(records):
        row = {
            "dataset": cfg.dataset,
            "task": data.TASK_OF[cfg.dataset],
            "split": r["split"],
            "correctness": r["correctness"],
            "msp_mean": msp_mean[i],
            "msp_min": msp_min[i],
            "msp_sum": msp_sum[i],
        }
        # Each supervised method is blank on train rows (the probe never scores its
        # own training data).
        for m in sup:
            row[m] = float(sup[m]["at"][i]) if i in sup[m]["at"] else ""
        rows.append(row)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.results_dir / f"{key}.csv"
    results.write_csv(csv_path, rows, ["msp_mean", "msp_min", "msp_sum", *sup.keys()])

    # PRR is computed on the test split only: SAPLMA has no train scores, and
    # MSP must be compared on the identical examples to be a fair anchor.
    y_test = [records[i]["correctness"] for i in test_positions]
    print(f"{cfg.dataset} {cfg.ood_setting} | test n={len(y_test)} "
          f"| mean correctness {sum(y_test) / len(y_test):.3f}")
    for name, unc in [("MSP mean", msp_mean), ("MSP min ", msp_min), ("MSP sum ", msp_sum)]:
        unc_test = [unc[i] for i in test_positions]
        print(f"PRR  {name}        : {results.prr(y_test, unc_test):.3f}")
    for m in sup:
        print(f"PRR  {m:8s} (layer {sup[m]['layer']}): "
              f"{results.prr(y_test, sup[m]['unc']):.3f}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
