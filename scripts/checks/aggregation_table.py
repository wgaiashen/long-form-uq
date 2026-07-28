"""The consolidated aggregation table: one command, one explicit label, uniform seeds, a paired
significance verdict, and length-normalised weight diagnostics.

This is the CONTROLLED comparison for the contribution: every row is the SAME probe machinery on the
SAME per-token L15 states, so the only thing that differs across the ladder is how per-token
information is combined:

    mean-pool + SAPLMA MLP     uniform weights, MLP head            (the strong standard baseline)
    last-token + MLP           only the final token
    per-sentence (mean)        pool each sentence, probe each, average per-sentence P(correct)  (#5)
    per-token (mean)           probe each token, average per-token P(correct)                   (#4)
    uniform (frozen-q head)    uniform weights, the SAME torch linear head as attention
                               (the controlled mean-pool -- isolates aggregation from the head)
    attention (softmax+T)      a learned query weights the tokens; T selected on a val split
    attention + position       + a learned positional bias
    attention + answer-only    pooling over answer tokens only

Everything reuses the existing functions (attn_pool.py, aggregators.py, pertoken_aggregate.py) so
there is one implementation of each aggregator, not a copy.

Three things make this airtight rather than just reproducible (see the plan / worklog):
  1. EXPLICIT, STAMPED LABEL. The bare `correctness` field differs by dataset (gpt-5 short-form,
     gpt-5-mini long-form). We read a named --label-field and print the per-dataset model, and fail
     loudly on a missing/NaN label. AlignScore is a secondary field, read by regime (alive
     short-form, degenerate long-form).
  2. UNIFORM SEED PROTOCOL. EVERY row is trained over the SAME --seeds set and reported mean+/-std,
     so the small long-form margins are read against one band.
  3. PAIRED SIGNIFICANCE VERDICT. The attention-minus-baseline difference is computed PER SEED
     (paired), and a paired t-test gives an explicit SIGNIFICANT / NOT-SIGNIFICANT call per dataset.
     We report whatever it says -- if long-form comes back null, that is the finding.

Verification is anchored to the cache, NOT to the old ad-hoc numbers: the mean-pool row must match
the cached SAPLMA L15 PRR (a HARD assert). If a margin differs from the old worklog table, that is
the consolidation working -- investigate, do not force a match.

    python scripts/checks/aggregation_table.py --label-fields correctness,correctness_alignscore \
        --datasets sciq,trivia_qa,pubmed_qa,xsum --seeds 1,2,3,4,5
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/vol/gpudata/gs925-msc_project/hf_cache")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from luq import cache, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402
# Reuse the existing aggregator implementations -- do NOT reimplement them here.
from attn_pool import (  # noqa: E402
    load_per_token, train_attn, attn_prr, select_temperature, pad_batch, pad_prior, _mask_answer_only)
from aggregators import window_token_ids, sentence_ids  # noqa: E402

MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"
GATE_TOL = 0.03
BOOT_B = 1000        # paired bootstrap resamples for the test-set CI
BOOT_SEED = 12345


def paired_bootstrap(yte, unc_a, unc_b, b=BOOT_B):
    """The HONEST significance source: a paired bootstrap over TEST EXAMPLES, i.e. against the real
    data variance, NOT seed-reinit noise. (A paired t-test over 3-5 re-inits has a near-zero
    denominator -- attention/uniform seed-std ~0.001 -- so it labels any tiny gap "significant";
    that measures almost nothing. The dominant real variance for a PRR margin on a fixed test set is
    WHICH test examples you drew, which this resamples.)

    unc_a, unc_b: per-example uncertainty vectors (higher = more uncertain) for method A (attention)
    and baseline B, both from the SAME averaged-over-seeds model. We resample the SAME test indices
    for both (paired), recompute the PRR margin, and report the observed margin, the 95% CI, and a
    two-sided bootstrap p. Significant iff the CI excludes 0. Degenerate resamples (no label spread)
    are skipped -- relevant only on the near-dead long-form AlignScore label."""
    yte = np.asarray(yte, dtype=float)
    ua, ub = np.asarray(unc_a, dtype=float), np.asarray(unc_b, dtype=float)
    n = len(yte)
    margin = results.prr(yte, ua) - results.prr(yte, ub)
    rng = np.random.RandomState(BOOT_SEED)
    deltas = []
    for _ in range(b):
        idx = rng.randint(0, n, n)
        yb = yte[idx]
        if np.ptp(yb) < 1e-9:                      # a degenerate resample has no oracle -> skip
            continue
        deltas.append(results.prr(yb, ua[idx]) - results.prr(yb, ub[idx]))
    deltas = np.array(deltas)
    if len(deltas) < 2:
        return margin, float("nan"), float("nan"), float("nan"), False
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = min(1.0, 2.0 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0))))
    return margin, float(lo), float(hi), float(p), bool(lo > 0 or hi < 0)


@torch.no_grad()
def attn_unc(model, states, te_idx, device, answer_only=False, bs=64, prior_list=None):
    """Per-example uncertainty vector (1 - sigmoid(logit)) for a trained AttnPool on te_idx -- the
    same computation as attn_pool.attn_prr but returning the predictions (needed for the bootstrap).
    `prior_list` (S3): per-example prior weight vectors aligned to `states`; passed to forward for the
    frozen/annealed-prior arms. None = the plain learned/frozen-query pooler (unchanged)."""
    model.eval()
    preds = np.zeros(len(te_idx))
    for b in range(0, len(te_idx), bs):
        idx = te_idx[b: b + bs]
        X, mask, pos = pad_batch([states[i] for i in idx], device)
        if answer_only:
            mask = _mask_answer_only(mask)
        prior_b = (pad_prior([prior_list[i] for i in idx], X.shape[1], device)
                   if prior_list is not None else None)
        logit, _ = model(X, mask, pos, prior=prior_b)
        p = torch.sigmoid(logit)
        if p.dim() == 2:                     # S6 multi-head: ensemble by mean-of-sigmoids
            p = p.mean(dim=1)
        preds[b: b + len(idx)] = p.cpu().numpy()
    return 1.0 - preds


# ---- per-response confidence for each aggregator (probe trained at a given seed) ----

def conf_meanpool(Xmean, tr, te, y, seed):
    clf = probe.train_probe_mlp(Xmean[tr], y[tr], seed=seed)
    return clf.p_correct(Xmean[te])


def conf_lasttoken(Xlast, tr, te, y, seed):
    clf = probe.train_probe_mlp(Xlast[tr], y[tr], seed=seed)
    return clf.p_correct(Xlast[te])


def conf_persentence(sent_vecs, tr, te, y, seed):
    """Pool each sentence, probe each sentence (each carries its response label), average the
    per-sentence P(correct). MEAN aggregator (principled; no selecting the aggregator on test)."""
    Xtr = np.concatenate([sent_vecs[i] for i in tr], axis=0)
    ytr = np.concatenate([np.full(len(sent_vecs[i]), y[i]) for i in tr])
    clf = probe.train_probe_mlp(Xtr, ytr, seed=seed)
    return np.array([float(np.clip(clf.p_correct(sent_vecs[i]), 1e-6, 1 - 1e-6).mean()) for i in te])


def conf_pertoken(states, tr, te, y, seed):
    """Probe each token (each carries its response label), average the per-token P(correct). MEAN."""
    Xtr = np.concatenate([states[i] for i in tr], axis=0)
    ytr = np.concatenate([np.full(len(states[i]), y[i]) for i in tr])
    clf = probe.train_probe_mlp(Xtr, ytr, seed=seed)
    return np.array([float(np.clip(clf.p_correct(states[i]), 1e-6, 1 - 1e-6).mean()) for i in te])


def prr_from_conf(yte, conf):
    return results.prr(yte, [1.0 - c for c in conf])


def diag_lengthnorm(states, y, tr_idx, records, device, model_name):
    """Length-normalised weight diagnostic, averaged over the test split (comparable across datasets
    of different length). Trains the plain softmax-attention (T selected on val), then per test
    example computes:
        entropy_ratio  = H(weights) / log(n_tokens)           (1 = uniform, ->0 = peaked)
        answer_ratio   = mass_on_answer_tokens / ((n-1)/n)    (1 = chance share, >1 = concentrated)
        peak_ratio     = max_weight * n_tokens                 (1 = uniform, >1 = a spike)
    Returns the test-mean of each."""
    best_T, _ = select_temperature(states, y, tr_idx, device, 1, False, False)
    model = train_attn(states, y, tr_idx, device, seed=1, temperature=best_T)
    model.eval()
    er, ar, pr = [], [], []
    with torch.no_grad():
        for k in range(len(states)):
            if records[k]["split"] != "test":
                continue
            X, mask, pos = pad_batch([states[k]], device)
            _, a = model(X, mask, pos)
            a = a[0, : states[k].shape[0]].cpu().numpy()
            n = len(a)
            if n < 2:
                continue
            ent = float(-(a * np.log(a + 1e-12)).sum())
            er.append(ent / math.log(n))
            ar.append(float(a[1:].sum()) / (max(n - 1, 1) / n))
            pr.append(float(a.max()) * n)
    return best_T, float(np.mean(er)), float(np.mean(ar)), float(np.mean(pr))


def build_arrays(states, records, tok):
    """Precompute the seed-independent per-example arrays once: mean vector, last-token vector, and
    the per-sentence vectors (with a window/piece alignment guard)."""
    Xmean = np.stack([s.mean(axis=0) for s in states])
    Xlast = np.stack([s[-1] for s in states])
    pieces = [tok.convert_ids_to_tokens(window_token_ids(r)) for r in records]
    for k in range(len(records)):
        if len(pieces[k]) != states[k].shape[0]:
            raise SystemExit(f"row {k}: {len(pieces[k])} pieces != {states[k].shape[0]} states "
                             f"-- window misaligned")
    sent_vecs = []
    for k in range(len(records)):
        sid = sentence_ids(pieces[k])
        sent_vecs.append(np.stack([states[k][sid == s].mean(axis=0) for s in range(sid.max() + 1)]))
    return Xmean, Xlast, sent_vecs


def run_dataset(model_name, dataset, layer, label_field, seeds, device, tok):
    """Return (stamp dict, rows list of (name, prr_array), verdicts dict, diag tuple) or None."""
    loaded = load_per_token(model_name, dataset, layer, label_field)
    if loaded is None:
        print(f"  {dataset}: no per-token cache, skip")
        return None
    states, split, y, layer, records = loaded
    tr_idx = [i for i in range(len(states)) if split[i] == "train"]
    te_idx = [i for i in range(len(states)) if split[i] == "test"]
    yte = [y[i] for i in te_idx]

    # Fail loudly on a missing/NaN label for THIS field (mirror 04_eval's guard) -- no silent skip.
    bad = [i for i in range(len(states)) if not np.isfinite(y[i])]
    if bad:
        print(f"  {dataset}: {len(bad)} rows have no numeric '{label_field}' -- skip this field")
        return None
    model_stamp = (records[0].get(f"{label_field}_model")
                   or records[0].get(f"{label_field}_judge_model")
                   or records[0].get("correctness_judge_model")
                   or records[0].get("correctness_model") or "unknown")

    Xmean, Xlast, sent_vecs = build_arrays(states, records, tok)

    # HARD cache-anchored gate: mean-pool (seed 1) must reproduce the cached SAPLMA L15 PRR. This
    # checks the per-token cache + label wiring against an INDEPENDENT anchor, not the old table.
    prr_mean_s1 = prr_from_conf(yte, conf_meanpool(Xmean, tr_idx, te_idx, y, 1))
    feat = cache.load_features(Config(model_name=model_name, dataset=dataset, ood_setting="ID").cache_dir,
                               cache.run_key(model_name, dataset, "ID"), "saplma")
    prr_feat = prr_from_conf(yte, probe.train_probe_mlp(feat[tr_idx, layer, :], y[tr_idx], seed=1)
                             .p_correct(feat[te_idx, layer, :]))
    assert abs(prr_mean_s1 - prr_feat) < GATE_TOL, (
        f"{dataset}/{label_field}: cache gate FAILED mean-pool {prr_mean_s1:.3f} vs cached SAPLMA "
        f"{prr_feat:.3f} -- per-token cache or label wiring is wrong, aborting")
    print(f"  [{dataset}/{label_field}] gate ok (mean-pool {prr_mean_s1:.3f}); selecting T ...",
          flush=True)

    # Attention temperature is selected ONCE per variant on a val split (seed 1), then trained at
    # that T for every seed (so the temperature choice never sees test and is shared across seeds).
    attn_T = {}
    for key, use_pos, ans in [("attention", False, False), ("attention+pos", True, False),
                              ("attention+answer", False, True)]:
        attn_T[key], _ = select_temperature(states, y, tr_idx, device, 1, use_pos, ans)
    print(f"  [{dataset}/{label_field}] T={attn_T}; training {len(seeds)} seeds ...", flush=True)

    # Every row over the SAME seeds. For the three verdict methods we ALSO keep the per-example
    # uncertainty vectors (averaged over seeds) so significance can come from a test-set bootstrap,
    # not the near-zero seed-reinit noise.
    rows = {name: [] for name in
            ["mean-pool+MLP", "last-token", "per-sentence(mean)", "per-token(mean)",
             "uniform(frozen-q)", "attention", "attention+pos", "attention+answer"]}
    pred_mean, pred_uni, pred_attn = [], [], []
    for sd in seeds:
        um = 1.0 - conf_meanpool(Xmean, tr_idx, te_idx, y, sd)      # mean-pool+MLP uncertainty
        rows["mean-pool+MLP"].append(results.prr(yte, um)); pred_mean.append(um)
        rows["last-token"].append(prr_from_conf(yte, conf_lasttoken(Xlast, tr_idx, te_idx, y, sd)))
        rows["per-sentence(mean)"].append(prr_from_conf(yte, conf_persentence(sent_vecs, tr_idx, te_idx, y, sd)))
        rows["per-token(mean)"].append(prr_from_conf(yte, conf_pertoken(states, tr_idx, te_idx, y, sd)))
        um_mod = train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True)
        uu = attn_unc(um_mod, states, te_idx, device)
        rows["uniform(frozen-q)"].append(results.prr(yte, uu)); pred_uni.append(uu)
        a_mod = train_attn(states, y, tr_idx, device, seed=sd, temperature=attn_T["attention"])
        au = attn_unc(a_mod, states, te_idx, device)
        rows["attention"].append(results.prr(yte, au)); pred_attn.append(au)
        rows["attention+pos"].append(
            attn_prr(train_attn(states, y, tr_idx, device, seed=sd, temperature=attn_T["attention+pos"],
                                use_position=True), states, y, te_idx, device))
        rows["attention+answer"].append(
            attn_prr(train_attn(states, y, tr_idx, device, seed=sd, temperature=attn_T["attention+answer"],
                                answer_only=True), states, y, te_idx, device, answer_only=True))
        print(f"    [{dataset}/{label_field}] seed {sd} done", flush=True)
    rows = {k: np.array(v) for k, v in rows.items()}

    # Significance = a PAIRED BOOTSTRAP over test examples (real data variance), on the seed-averaged
    # predictions. HEADLINE = attention vs the strong mean-pool+MLP baseline (what a reader compares
    # to, what we reproduced against Joe). SUPPORTING = attention vs uniform (same head), which
    # isolates the mechanistic "learned weighting helps holding the head fixed" claim only.
    avg_attn, avg_mean, avg_uni = (np.mean(p, axis=0) for p in (pred_attn, pred_mean, pred_uni))
    verdicts = {
        "attention_vs_meanMLP": paired_bootstrap(np.array(yte), avg_attn, avg_mean),  # HEADLINE
        "attention_vs_uniform": paired_bootstrap(np.array(yte), avg_attn, avg_uni),   # supporting
    }
    diag = diag_lengthnorm(states, y, tr_idx, records, device, model_name)
    stamp = {"dataset": dataset, "label_field": label_field, "label_model": model_stamp,
             "n_train": len(tr_idx), "n_test": len(te_idx), "attn_T": attn_T["attention"]}
    return stamp, rows, verdicts, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--datasets", default="sciq,trivia_qa,pubmed_qa,xsum")
    ap.add_argument("--label-fields", default="correctness,correctness_alignscore",
                    help="primary first (the judge 'correctness'); secondary read by regime (alignscore)")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--out", default=None, help="CSV path (default results/aggregation_table__<model>.csv)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"device {device} | seeds {seeds} | label-fields {args.label_fields}")

    out_rows = []  # long-format CSV rows
    for field in args.label_fields.split(","):
        for dataset in args.datasets.split(","):
            print(f"\n[start] {dataset} / {field} ...", flush=True)
            res = run_dataset(args.model, dataset, args.layer, field, seeds, device, tok)
            if res is None:
                continue
            stamp, rows, verdicts, diag = res
            bt, er, ar, pr = diag
            print(f"\n==== {dataset} | label={field} ({stamp['label_model']}) | "
                  f"train {stamp['n_train']} test {stamp['n_test']} ====")
            base = rows["mean-pool+MLP"].mean()
            for name, arr in rows.items():
                d = f"  (vs mean-pool {arr.mean() - base:+.3f})" if name != "mean-pool+MLP" else ""
                print(f"  {name:22s} {arr.mean():+.3f} +/- {arr.std(ddof=1) if len(seeds)>1 else 0:.3f}{d}")
                out_rows.append({"dataset": dataset, "label_field": field, "label_model": stamp["label_model"],
                                 "aggregator": name, "prr_mean": round(arr.mean(), 4),
                                 "prr_std": round(float(arr.std(ddof=1)) if len(seeds) > 1 else 0.0, 4),
                                 "n_seeds": len(seeds)})
            # Print HEADLINE (vs mean-pool+MLP) first, then the supporting mechanistic one.
            for vk in ("attention_vs_meanMLP", "attention_vs_uniform"):
                m, lo, hi, p, sig = verdicts[vk]
                tag = "HEADLINE" if vk == "attention_vs_meanMLP" else "support "
                verdict = "SIGNIFICANT" if sig else "NOT SIGNIFICANT"
                print(f"  [{tag}] {vk:22s} margin {m:+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]  "
                      f"p={p:.3f} (test bootstrap)  -> {verdict}")
                out_rows.append({"dataset": dataset, "label_field": field, "label_model": stamp["label_model"],
                                 "aggregator": f"VERDICT:{vk}", "prr_mean": round(m, 4),
                                 "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "boot_p": round(p, 4),
                                 "n_seeds": len(seeds), "significant": sig})
            print(f"  [diag L-norm] T*={bt}  entropy/logn {er:.3f}  answer/uniform {ar:.3f}  peak*n {pr:.3f}")
            out_rows.append({"dataset": dataset, "label_field": field, "label_model": stamp["label_model"],
                             "aggregator": "DIAG:lengthnorm", "attn_T": bt,
                             "entropy_ratio": round(er, 3), "answer_ratio": round(ar, 3),
                             "peak_ratio": round(pr, 3)})

    out = Path(args.out) if args.out else (ROOT / "results" /
          f"aggregation_table__{cache._slug(args.model)}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    cols = ["dataset", "label_field", "label_model", "aggregator", "prr_mean", "prr_std", "n_seeds",
            "ci_lo", "ci_hi", "boot_p", "significant", "attn_T", "entropy_ratio", "answer_ratio",
            "peak_ratio"]
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
