"""ProbeDriftLong (W3 / H4): train EXCLUSIVELY on long-form datasets and evaluate on long-form, plus a
long->short transfer cell. Joe's steer: the current ProbeDrift ID/LOO settings let probes lean on easy
short-form training data (long_form_loo §J showed SAPLMA's long-form OOD collapses to the floor once
short-form is removed). This driver removes that crutch by construction and asks: in the realistic
long-only setting, does weighted-MSP / shrink overtake the probes, measured against the HONEST fair floor
(max of msp_sum, perplexity, msp_min)?

LONG UNIVERSE (correctness-labelled + cached): pubmed_qa, xsum, cnn_dailymail, med_quad, samsum.
  ExpertQA is EVAL-ONLY (faithfulness label -> cross-label OOD, never a training source).
  FINE families: long_qa = {pubmed_qa, med_quad, expertqa};  summ = {xsum, cnn_dailymail, samsum}.

RUNGS (long eval X): ID | SameTask-long (other long sets in X's family) | LOO-long (all other long sets)
  | DiffTask-long (opposite long family) | 1ds-Diff-long (one opposite-family long set).
  For a SHORT eval (sciq/trivia): the single "Long->Short" cell (train on the full long pool).

METHODS (the KEEP set): fair floor (msp_sum/perplexity/msp_min -> max), uniform + attention poolers,
  mean-pool SAPLMA (MLP), weighted-MSP {normalised, shrink@2, shrink@10, Blondel}. Paired bootstrap:
  best-wMSP vs fair-floor, best-wMSP vs best-pooler, attention vs fair-floor.

    python scripts/checks/probedriftlong.py --seeds 1,2,3
    python scripts/checks/probedriftlong.py --evals pubmed_qa,xsum --seeds 1   # smoke
"""
import argparse, csv as _csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "checks"))
import torch  # noqa: E402
from luq import cache, msp, results, weighted_msp, probe  # noqa: E402
from luq.weighting import shrink_to_uniform  # noqa: E402
from aggregation_table import load_per_token, attn_unc, paired_bootstrap, conf_meanpool, prr_from_conf  # noqa: E402
from attn_pool import train_attn, select_temperature  # noqa: E402
from xl_rungs import build_rows, eval_split, label_of, cross_label  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa"]
LONG_SRC = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum"]   # expertqa eval-only
SHORT = ["sciq", "trivia_qa"]
FINE = {"pubmed_qa": "long_qa", "med_quad": "long_qa", "expertqa": "long_qa",
        "xsum": "summ", "cnn_dailymail": "summ", "samsum": "summ"}
XL_TOTAL = 1800
EVALS = LONG + SHORT
# wMSP KEEP variants: (col name, kwargs to weighted_msp_unc)  [all length_normalise=True]
WMSP = [("wmsp_norm", {"weight_mode": "normalised"}),
        ("wmsp_shrink2", {"weight_mode": "normalised", "reg": shrink_to_uniform, "reg_lambda": 2.0}),
        ("wmsp_shrink10", {"weight_mode": "normalised", "reg": shrink_to_uniform, "reg_lambda": 10.0}),
        ("wmsp_blondel", {"weight_mode": "normalised", "loss": "blondel"})]
POOLERS = ["uniform", "attention"]
FLOORS = ["floor_sum", "floor_ppl", "floor_min"]
METHODS = FLOORS + ["fair_floor", "saplma"] + POOLERS + [w[0] for w in WMSP]


def rung_sources_long(X):
    same = [d for d in LONG_SRC if d != X and FINE[d] == FINE[X]]
    diff = [d for d in LONG_SRC if d != X and FINE[d] != FINE[X]]
    loo = [d for d in LONG_SRC if d != X]
    return {"SameTask-long": same, "DiffTask-long": diff, "LOO-long": loo, "1ds-Diff-long": diff[:1]}


def cells_long(sources, evals):
    out = []
    for X in evals:
        if X not in sources:
            continue
        if X in LONG:
            out.append(("ID", X, [(X, None)]))
            rs = rung_sources_long(X)
            for tag in ("SameTask-long", "DiffTask-long", "LOO-long"):
                srcs = [d for d in rs[tag] if d in sources]
                if srcs:
                    cap = max(1, XL_TOTAL // len(srcs))
                    out.append((tag, X, [(d, cap) for d in srcs]))
            one = [d for d in rs["1ds-Diff-long"] if d in sources]
            if one:
                out.append(("1ds-Diff-long", X, [(one[0], XL_TOTAL)]))
        elif X in SHORT:                                   # long -> short transfer
            srcs = [d for d in LONG_SRC if d in sources]
            if srcs:
                cap = max(1, XL_TOTAL // len(srcs))
                out.append(("Long->Short", X, [(d, cap) for d in srcs]))
    return out


def sampled_train_idx(split, seed, cap):
    tr = np.where(split == "train")[0]
    if cap is None or cap >= len(tr):
        return tr
    return tr[np.random.RandomState(seed).permutation(len(tr))[:cap]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--evals", default=",".join(EVALS))
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--include-expertqa", action="store_true",
                    help="add ExpertQA (faithfulness label) as a TRAINING source -> the UNIVERSAL mixed-label "
                         "ProbeDriftLong (train pool spans QA + summarisation + factuality). Default off keeps the "
                         "label-homogeneous baseline. When on, ExpertQA is still excluded from its OWN eval's sources.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global LONG_SRC
    if args.include_expertqa and "expertqa" not in LONG_SRC:
        LONG_SRC = LONG_SRC + ["expertqa"]
        print("UNIVERSAL mode: ExpertQA (faithfulness) added as a training source -> mixed-label long pool", flush=True)
    evals = args.evals.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | seeds {seeds} | evals {evals}", flush=True)

    PT = {}
    for d in sorted(set(LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip", flush=True); continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: fully unlabelled ({label_of(d)}) -> skip", flush=True); continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} rows (label={label_of(d)})", flush=True)
    sources = set(PT)

    out_rows = []
    for rung, X, spec in cells_long(sources, evals):
        if X not in PT:
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        xlbl = cross_label(X)
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
            # unsupervised floors (per-example vectors)
            v["floor_sum"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "sum") for i in te_idx])
            v["floor_ppl"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity") for i in te_idx])
            v["floor_min"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "min") for i in te_idx])
            # SAPLMA mean-pool + MLP
            Xmean = np.stack([s.mean(axis=0) for s in states])
            v["saplma"] = 1.0 - conf_meanpool(Xmean, tr_idx, te_idx, y, sd)
            # poolers
            best_T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
            v["uniform"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True),
                                               states, te_idx, device), float)
            v["attention"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T),
                                                 states, te_idx, device), float)
            # wMSP KEEP variants
            for name, kw in WMSP:
                v[name] = np.asarray(weighted_msp.weighted_msp_unc(states, records, y, tr_idx, te_idx, device,
                                     length_normalise=True, seed=sd, **kw), float)
            for m in v:
                per[m].append(results.prr(yte, v[m])); unc_acc[m].append(v[m])
        if yte_ref is None:
            continue
        stats = {m: (float(np.mean(per[m])), float(np.std(per[m]))) for m in per if per[m]}
        # fair floor = the best of the three unsupervised floors (by mean PRR)
        fair_name = max(FLOORS, key=lambda f: stats[f][0]); stats["fair_floor"] = stats[fair_name]
        avg = {m: np.mean(np.stack(unc_acc[m]), 0) for m in unc_acc if unc_acc[m]}
        avg["fair_floor"] = avg[fair_name]
        best_w = max((w[0] for w in WMSP), key=lambda m: stats[m][0])
        best_p = max(POOLERS, key=lambda m: stats[m][0])
        srcs = "+".join(f"{d}:{c}" if c else d for d, c in spec)
        xf = "  [CROSS-LABEL]" if (xlbl and rung != "ID") else ""
        print(f"\n[{rung:14s}] eval={X} ({label_of(X)}) train={srcs}{xf}  fair_floor={fair_name} {stats['fair_floor'][0]:+.3f}", flush=True)
        for m in METHODS:
            if m in stats:
                print(f"    {m:14s} {stats[m][0]:+.3f} +/- {stats[m][1]:.3f}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": m,
                                 "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                                 "n_seeds": len(per[m]), "cross_label": bool(xlbl and rung != "ID")})
        for vk, a, b in [("bestw_vs_fairfloor", best_w, "fair_floor"),
                         ("bestw_vs_bestpooler", best_w, best_p),
                         ("attention_vs_fairfloor", "attention", "fair_floor")]:
            if a in avg and b in avg:
                mg, lo, hi, p, sig = paired_bootstrap(yte_ref, avg[a], avg[b])
                print(f"    [verdict] {vk:24s} ({a} vs {b}) margin {mg:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} {'SIG' if sig else 'ns'}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"VERDICT:{vk}",
                                 "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                 "boot_p": round(p, 4), "significant": bool(sig), "n_seeds": len(seeds)})

    out = Path(args.out) if args.out else (ROOT / "results" / f"probedriftlong__{cache._slug(MODEL)}.csv")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "train", "method", "prr_mean", "prr_std",
                                           "n_seeds", "cross_label", "ci_lo", "ci_hi", "boot_p", "significant"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
