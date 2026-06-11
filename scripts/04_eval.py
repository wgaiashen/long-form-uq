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
    sap = cache.load_scores(cfg.cache_dir, key, method="saplma")
    sap_unc, layer = sap["unc"], int(sap["layer"])

    test_positions = [i for i, r in enumerate(records) if r["split"] == "test"]
    assert len(sap_unc) == len(test_positions), "scores out of step — rerun 03"
    # Map each test record's position in `records` to its SAPLMA score.
    sap_at = dict(zip(test_positions, sap_unc))

    # One CSV row per example; both methods as columns (SAPLMA blank on train
    # rows, since the probe never scores its own training data).
    rows = []
    for i, r in enumerate(records):
        rows.append({
            "dataset": cfg.dataset,
            "task": data.TASK_OF[cfg.dataset],
            "split": r["split"],
            "correctness": r["correctness"],
            "msp_mean": msp_mean[i],
            "msp_min": msp_min[i],
            "saplma": float(sap_at[i]) if i in sap_at else "",
        })
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.results_dir / f"{key}.csv"
    results.write_csv(csv_path, rows, ["msp_mean", "msp_min", "saplma"])

    # PRR is computed on the test split only: SAPLMA has no train scores, and
    # MSP must be compared on the identical examples to be a fair anchor.
    y_test = [records[i]["correctness"] for i in test_positions]
    print(f"{cfg.dataset} {cfg.ood_setting} | test n={len(y_test)} "
          f"| mean correctness {sum(y_test) / len(y_test):.3f}")
    for name, unc in [("MSP mean", msp_mean), ("MSP min ", msp_min)]:
        unc_test = [unc[i] for i in test_positions]
        print(f"PRR  {name}        : {results.prr(y_test, unc_test):.3f}")
    print(f"PRR  SAPLMA (layer {layer}): {results.prr(y_test, sap_unc):.3f}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
