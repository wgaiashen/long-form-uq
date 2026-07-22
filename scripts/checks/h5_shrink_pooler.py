"""H5 (co-headline): shrink-the-pooler. Does adding a shrink-to-uniform penalty on the ATTENTION pooler's
weights (pulling it toward mean-pool, its unsupervised prior) make it OOD-robust — generalising the
moderation-toward-the-prior mechanism from weighted-MSP to the hidden-state pooler? Runs the STANDARD
ProbeDrift ID+OOD ladder (xl_rungs.cells / get_training_spec) so it is directly comparable to §B.

Methods per cell: fair-floor (max of msp_sum/perplexity/msp_min) · uniform (mean-pool) · attention (plain,
best-T) · attn_shrink@2 · attn_shrink@10 (same best-T + shrink_lambda) · SAPLMA (mean-pool+MLP). 3 seeds,
paired bootstrap: shrink-vs-plain (does shrink help?), shrink-vs-fair-floor, plain-vs-fair-floor.

    python scripts/checks/h5_shrink_pooler.py --evals sciq,pubmed_qa --seeds 1   # smoke
    python scripts/checks/h5_shrink_pooler.py --evals <one>                      # per-eval (parallel)
"""
import argparse, csv as _csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "checks"))
import torch  # noqa: E402
from luq import cache, msp, results  # noqa: E402
from aggregation_table import load_per_token, attn_unc, paired_bootstrap, conf_meanpool  # noqa: E402
from attn_pool import train_attn, select_temperature  # noqa: E402
from xl_rungs import cells, build_rows, eval_split, label_of  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
CANDIDATE_SOURCES = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum"]
SHRINK = [2.0, 10.0]                       # the shrink-lambda sweep for the pooler
METHODS = ["floor_sum", "floor_ppl", "floor_min", "fair_floor", "uniform", "saplma", "attention",
           "attn_shrink2", "attn_shrink10"]


def sampled_train_idx(split, seed, cap):
    tr = np.where(split == "train")[0]
    if cap is None or cap >= len(tr):
        return tr
    return tr[np.random.RandomState(seed).permutation(len(tr))[:cap]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--evals", default="sciq,trivia_qa,pubmed_qa,xsum,cnn_dailymail")
    ap.add_argument("--sources", default=",".join(CANDIDATE_SOURCES))
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    evals = args.evals.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | seeds {seeds} | evals {evals} | shrink {SHRINK}", flush=True)

    PT = {}
    for d in sorted(set(args.sources.split(",")) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip", flush=True); continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} rows", flush=True)
    sources = set(PT)

    out_rows = []
    for rung, X, spec in cells(sources, evals):
        if X not in PT:
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        per = {m: [] for m in METHODS}; unc_acc = {m: [] for m in METHODS}; yte_ref = None
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows); tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float); yte_ref = yte
            states = [PT[d][0][i] for d, i in allrows]; records = [PT[d][3][i] for d, i in allrows]
            v = {}
            v["floor_sum"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "sum") for i in te_idx])
            v["floor_ppl"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity") for i in te_idx])
            v["floor_min"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "min") for i in te_idx])
            Xmean = np.stack([s.mean(axis=0) for s in states])
            v["saplma"] = 1.0 - conf_meanpool(Xmean, tr_idx, te_idx, y, sd)
            best_T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
            v["uniform"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True),
                                               states, te_idx, device), float)
            v["attention"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T),
                                                 states, te_idx, device), float)
            v["attn_shrink2"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd,
                                           temperature=best_T, shrink_lambda=2.0), states, te_idx, device), float)
            v["attn_shrink10"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd,
                                            temperature=best_T, shrink_lambda=10.0), states, te_idx, device), float)
            for m in v:
                per[m].append(results.prr(yte, v[m])); unc_acc[m].append(v[m])
        if yte_ref is None:
            continue
        stats = {m: (float(np.mean(per[m])), float(np.std(per[m]))) for m in per if per[m]}
        fair_name = max(("floor_sum", "floor_ppl", "floor_min"), key=lambda f: stats[f][0])
        stats["fair_floor"] = stats[fair_name]
        avg = {m: np.mean(np.stack(unc_acc[m]), 0) for m in unc_acc if unc_acc[m]}
        avg["fair_floor"] = avg[fair_name]
        srcs = "+".join(f"{d}:{c}" if c else d for d, c in spec)
        print(f"\n[{rung:18s}] eval={X}  fair_floor={fair_name} {stats['fair_floor'][0]:+.3f}", flush=True)
        for m in METHODS:
            if m in stats:
                print(f"    {m:14s} {stats[m][0]:+.3f} +/- {stats[m][1]:.3f}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": m,
                                 "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                                 "n_seeds": len(per[m])})
        for vk, a, b in [("shrink10_vs_attention", "attn_shrink10", "attention"),
                         ("shrink2_vs_attention", "attn_shrink2", "attention"),
                         ("shrink10_vs_fairfloor", "attn_shrink10", "fair_floor"),
                         ("attention_vs_fairfloor", "attention", "fair_floor")]:
            if a in avg and b in avg:
                mg, lo, hi, p, sig = paired_bootstrap(yte_ref, avg[a], avg[b])
                print(f"    [verdict] {vk:24s} margin {mg:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} {'SIG' if sig else 'ns'}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"VERDICT:{vk}",
                                 "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                 "boot_p": round(p, 4), "significant": bool(sig), "n_seeds": len(seeds)})

    out = Path(args.out) if args.out else (ROOT / "results" / f"h5_shrink_pooler__{cache._slug(MODEL)}.csv")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "train", "method", "prr_mean", "prr_std",
                                           "n_seeds", "ci_lo", "ci_hi", "boot_p", "significant"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
