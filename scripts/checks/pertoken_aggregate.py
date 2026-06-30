"""Per-token SAPLMA, then aggregate: does scoring each token and combining beat mean-pool?

The mean-pool SAPLMA baseline averages the answer-token hidden states into one vector and probes
that, so a single bad clause is diluted by the rest. This experiment asks the aggregation question
directly: train a SAPLMA probe on INDIVIDUAL token states (each token carries its response's
correctness label), score every answer token at test time, then combine the per-token P(correct)
into one response score with several aggregators:

    mean      the average (the per-token analogue of mean-pooling)
    min       the least-confident token (weakest-link)
    max       the most-confident token
    bottom3   the mean of the three least-confident tokens (a robust weakest-link)
    geomean   the geometric mean = product of per-token P(correct) under independence
              (the "soft conjunction" baseline: one unsupported token drags the whole score down)

Everything uses the same SAPLMA MLP, so the only thing that varies is the aggregation. We report
PRR against the judge label and against AlignScore, with the mean-pool baseline alongside as the
bar to beat (it should also match the cached layer-16 SAPLMA number, a sanity gate).

Runs on the per-token cache (cache/pertok, layer 16) for sciq/trivia/pubmed -- CPU only, no GPU.
xsum has no per-token cache yet, so it is skipped until that is extracted.

    python scripts/checks/pertoken_aggregate.py
    python scripts/checks/pertoken_aggregate.py --datasets pubmed_qa
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402

MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"
AGGREGATORS = ["mean", "min", "max", "bottom3", "geomean"]


def aggregate(p, how):
    """Combine a response's per-token P(correct) into one confidence in [0, 1]."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    if how == "mean":
        return p.mean()
    if how == "min":
        return p.min()
    if how == "max":
        return p.max()
    if how == "bottom3":
        return np.sort(p)[:3].mean()
    if how == "geomean":
        return float(np.exp(np.log(p).mean()))
    raise ValueError(how)


def load_per_token(model, dataset, layer):
    """Return per-example token states aligned to records: (states_list, split, labels dict)."""
    path = (ROOT / "cache" / "pertok"
            / f"{cache._slug(model)}__{dataset}__ID__L{layer}.npz")
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    layer = int(z["layer"])
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID")
    records = cache.load_records(cfg.cache_dir, cache.run_key(model, dataset, "ID"))
    st = z["states"]
    # Align POSITIONALLY, not by idx: the cache is written in record order (01h iterates records
    # in order), and the record "idx" field is NOT unique -- train and test share idx 0..N, so a
    # {idx: states} map silently drops half the rows. Guard that the lengths match.
    if len(st) != len(records):
        raise SystemExit(f"{dataset}: per-token cache has {len(st)} rows but {len(records)} records "
                         f"-- re-extract with scripts/01h_pertoken.py")
    states = [np.asarray(st[k], dtype=np.float32) for k in range(len(records))]
    split = np.array([r["split"] for r in records])
    labels = {k: [r.get(k) for r in records] for k in ("correctness", "correctness_alignscore")}
    return states, split, labels, layer


def prr_for(y_test, conf_test):
    """PRR from per-response confidence (higher = more correct) -> uncertainty = 1 - confidence."""
    unc = [1.0 - c for c in conf_test]
    return results.prr(y_test, unc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--datasets", default="sciq,trivia_qa,pubmed_qa")
    ap.add_argument("--layer", type=int, default=15,
                    help="per-token cache layer to use (must have been extracted by 01h_pertoken)")
    ap.add_argument("--eval-fields", default="correctness,correctness_alignscore")
    args = ap.parse_args()

    eval_fields = args.eval_fields.split(",")

    for dataset in args.datasets.split(","):
        loaded = load_per_token(args.model, dataset, args.layer)
        if loaded is None:
            print(f"\n==== {dataset}: no per-token cache, skipping ====")
            continue
        states, split, labels, layer = loaded
        tr = split == "train"
        te = split == "test"
        print(f"\n==== {dataset} (layer {layer}, n_train={tr.sum()}, n_test={te.sum()}) ====")

        # Mean-pool baseline: average each response's token states, then SAPLMA on that.
        Xmean = np.stack([s.mean(axis=0) for s in states])
        ytrain = np.array(labels["correctness"], dtype=float)
        base_clf = probe.train_probe_mlp(Xmean[tr], ytrain[tr])
        base_conf_test = base_clf.p_correct(Xmean[te])

        # Cache gate: the mean-pool of the per-token states must reproduce the cached SAPLMA
        # feature's PRR (it IS that mean). If it does not, the per-token cache is stale or was
        # stored at the wrong precision, so the aggregation numbers below cannot be trusted.
        cfg = Config(model_name=args.model, dataset=dataset, ood_setting="ID")
        feat = cache.load_features(cfg.cache_dir, cache.run_key(args.model, dataset, "ID"), "saplma")
        feat_prr = results.prr(ytrain[te],
                               probe.uncertainty(probe.train_probe_mlp(feat[tr, layer, :], ytrain[tr]),
                                                 feat[te, layer, :]))
        base_prr_judge = results.prr(ytrain[te], probe.uncertainty(base_clf, Xmean[te]))
        gate = "OK" if abs(base_prr_judge - feat_prr) < 0.03 else "MISMATCH -- per-token cache suspect"
        print(f"  [cache gate] mean-pool {base_prr_judge:.3f} vs cached SAPLMA L{layer} {feat_prr:.3f}  {gate}")

        # Per-token probe: every answer token of a response inherits the response's label.
        Xtok_tr = np.concatenate([states[i] for i in np.where(tr)[0]], axis=0)
        ytok_tr = np.concatenate(
            [np.full(len(states[i]), ytrain[i]) for i in np.where(tr)[0]])
        tok_clf = probe.train_probe_mlp(Xtok_tr, ytok_tr)
        # Per-token P(correct) for each test response.
        test_idx = np.where(te)[0]
        per_tok_conf = [tok_clf.p_correct(states[i]) for i in test_idx]

        # Report PRR for the baseline and each aggregator, against each eval label.
        for ef in eval_fields:
            yall = labels[ef]
            if not all(isinstance(yall[i], (int, float)) for i in test_idx):
                continue
            y_test = [yall[i] for i in test_idx]
            base_prr = prr_for(y_test, base_conf_test)
            print(f"  eval={ef}")
            print(f"    {'mean-pool (baseline)':24s} {base_prr:+.3f}")
            for how in AGGREGATORS:
                conf = [aggregate(p, how) for p in per_tok_conf]
                tag = f"per-token {how}"
                delta = prr_for(y_test, conf) - base_prr
                print(f"    {tag:24s} {prr_for(y_test, conf):+.3f}   (vs baseline {delta:+.3f})")


if __name__ == "__main__":
    main()
