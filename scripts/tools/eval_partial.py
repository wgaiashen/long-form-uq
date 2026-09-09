# AI assistance: this utility was drafted with Claude Code (Anthropic), then reviewed,
# corrected and tested by the author. See ACKNOWLEDGEMENTS.md.
"""Preliminary eval on LABELLED records only (for a partially-judged dataset).

If the judge didn't finish (e.g. ran out of OpenAI quota), this trains the probes on the labelled
TRAIN rows and computes PRR on the labelled TEST rows, skipping unlabelled (None) records. Same
metric as 04_eval, just restricted to what's labelled -- a clearly preliminary number. Re-run the
judge to fill the gaps, then use the normal 04_eval for the full result.

    python scripts/tools/eval_partial.py --dataset cnn_dailymail --ood ID
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from luq import cache, msp, probe, results
from luq.config import Config
from luq.features import saplma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cnn_dailymail")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    recs = cache.load_records(cfg.cache_dir, key)
    has = lambda r: r.get("correctness") is not None  # noqa: E731

    tr = [i for i, r in enumerate(recs) if r["split"] == "train" and has(r)]
    te = [i for i, r in enumerate(recs) if r["split"] == "test" and has(r)]
    n_test_total = sum(r["split"] == "test" for r in recs)
    y_tr = np.array([recs[i]["correctness"] for i in tr])
    y_te = [recs[i]["correctness"] for i in te]
    print(f"{cfg.dataset} {cfg.ood_setting} PRELIMINARY (labelled-only) | train {len(tr)} | "
          f"test {len(te)}/{n_test_total} | test mean correctness {np.mean(y_te):.3f}")

    for agg in ["mean", "min", "sum"]:
        u = [msp.msp_uncertainty(recs[i]["token_logprobs"], agg) for i in te]
        print(f"PRR  MSP {agg:4s}        : {results.prr(y_te, u):.3f}")

    for m in ["saplma", "ptrue", "lookback"]:
        try:
            feats = cache.load_features(cfg.cache_dir, key, method=m)
        except FileNotFoundError:
            continue
        layer = feats.shape[1] // 2
        X = saplma.select_layer(feats, layer)
        clf = probe.train_probe(X[tr], y_tr)
        u = probe.uncertainty(clf, X[te])
        print(f"PRR  {m:8s} (layer {layer}): {results.prr(y_te, u):.3f}")


if __name__ == "__main__":
    main()
