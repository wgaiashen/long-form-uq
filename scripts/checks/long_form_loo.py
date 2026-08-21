"""P1.5 — Long-form-LOO: does SAPLMA's leave-one-out strength come from SHORT-FORM training data?

the hypothesis (email): SAPLMA does well at LOO because the LOO mixture contains short-form sets it can
learn easy heuristics from; strip the short-form data (keep the training SIZE the same) and its edge should
shrink, while the OOD-robust log-prob methods (MSP, weighted-MSP) hold. If so, our methods win on the
cleaner, harder setting.

For each eval we build TWO leave-one-out training pools, both capped to the SAME total size (matched, so
the only difference is composition):
  normal_LOO    = all available OTHER datasets (includes short-form sciq/trivia)
  longform_LOO  = only the LONG-FORM other datasets (short-form removed)
and compare each method's PRR between them. The headline is the SAPLMA drop; a paired test-set bootstrap
(same test rows) gives a CI on normal-vs-longform per method. The MSP floor is composition-invariant, so it
is the control that should NOT move.

Methods (lean, fast, CPU): saplma (mean-pool + SAPLMA MLP), uniform (frozen-q torch head = mean-pool
baseline), weighted_msp_norm (our contribution), msp_sum (floor). The learned attention pooler is omitted
here for speed (its temperature selection dominates runtime and it is not the headline) — add later with
--with-attention if wanted.

    python scripts/checks/long_form_loo.py --seeds 1,2,3
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402

from luq import cache, msp, probe, results, weighted_msp  # noqa: E402
from aggregation_table import load_per_token, paired_bootstrap, attn_unc  # noqa: E402
from attn_pool import train_attn  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAB = "correctness"
LONG_FORM = {"pubmed_qa", "xsum", "med_quad", "samsum"}
SHORT_FORM = {"sciq", "trivia_qa"}
EVALS = ["sciq", "trivia_qa", "pubmed_qa", "xsum"]
CANDIDATES = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "med_quad", "samsum"]
TOTAL = 1800   # matched training size across both pools (Hidden-Failures convention)


def sampled(split_arr, seed, cap):
    tr = np.where(split_arr == "train")[0]
    if cap >= len(tr):
        return tr
    return tr[np.random.RandomState(seed).permutation(len(tr))[:cap]]


def pool_rows(pool, X, seed, PT):
    """Even split of TOTAL across the sources in `pool` (excluding eval X)."""
    srcs = [s for s in pool if s != X and s in PT]
    if not srcs:
        return None, srcs
    cap = max(1, TOTAL // len(srcs))
    rows = [(d, i) for d in srcs for i in sampled(PT[d][1], seed, cap)]
    return rows, srcs


def saplma_unc(states, y, tr_idx, te_idx, seed):
    """Mean-pool the per-token window (incl the last-prompt anchor, as attn_pool's mean baseline does),
    train the SAPLMA MLP, return 1-P(correct) on test."""
    Xmean = np.stack([s.mean(axis=0) for s in states])
    clf = probe.train_probe_mlp(Xmean[tr_idx], y[tr_idx], seed=seed)
    return probe.uncertainty(clf, Xmean[te_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | seeds {seeds}", flush=True)

    PT = {}
    for d in CANDIDATES:
        loaded = load_per_token(MODEL, d, args.layer, LAB)
        if loaded is None:
            print(f"  {d}: no pertok -> skip"); continue
        states, split, y, _, records = loaded
        if np.isnan(y).any():
            print(f"  {d}: unlabelled -> skip"); continue
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} rows", flush=True)

    methods = ["saplma", "uniform", "weighted_msp_norm", "msp_sum", "fair_floor"]
    out_rows = []
    for X in EVALS:
        if X not in PT:
            continue
        te_idx_global = np.where(PT[X][1] == "test")[0]
        yte = np.array([PT[X][2][i] for i in te_idx_global], dtype=float)
        # matched_LOO (2026-07-12 confound-killer): SAME number of training sources as longform_LOO and
        # same total size, but with the short-form sources forced IN (replacing long-form ones). Comparing
        # longform_LOO vs matched_LOO isolates "short-form helps" from "fewer / less-diverse sources" (the
        # confound flagged in the overnight log: pubmed's longform pool had only 3 sources vs normal's 5).
        long_srcs = [s for s in LONG_FORM if s != X and s in PT]
        short_srcs = [s for s in SHORT_FORM if s != X and s in PT]
        pools = {"normal_LOO": CANDIDATES, "longform_LOO": list(LONG_FORM)}
        if short_srcs and len(long_srcs) >= 2:
            k_long = max(0, len(long_srcs) - len(short_srcs))          # keep count == longform count
            pools["matched_LOO"] = short_srcs + long_srcs[:k_long]     # short forced in, same #sources
        # per (pool, method): list of per-seed PRR + per-seed uncertainty vecs (for the bootstrap)
        res = {p: {m: {"prr": [], "unc": []} for m in methods} for p in pools}
        srcs_used = {}
        for pool_name, pool in pools.items():
            for sd in seeds:
                train_rows, srcs = pool_rows(pool, X, sd, PT)
                srcs_used[pool_name] = srcs
                if not train_rows:
                    continue
                test_rows = [(X, i) for i in te_idx_global]
                allrows = train_rows + test_rows
                n_tr = len(train_rows)
                tr_idx = list(range(n_tr))
                te_idx = list(range(n_tr, n_tr + len(test_rows)))
                y = np.array([PT[d][2][i] for d, i in allrows], dtype=float)
                states = [PT[d][0][i] for d, i in allrows]
                records = [PT[d][3][i] for d, i in allrows]

                vecs = {}
                vecs["saplma"] = np.asarray(saplma_unc(states, y, tr_idx, te_idx, sd), dtype=float)
                vecs["uniform"] = np.asarray(attn_unc(
                    train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True),
                    states, te_idx, device), dtype=float)
                vecs["weighted_msp_norm"] = np.asarray(weighted_msp.weighted_msp_unc(
                    states, records, y, tr_idx, te_idx, device, weight_mode="normalised",
                    length_normalise=True, seed=sd), dtype=float)
                vecs["msp_sum"] = np.asarray(
                    [msp.msp_uncertainty(records[i]["token_logprobs"], "sum") for i in te_idx], dtype=float)
                # ADD the fair floor beside the honestly-named msp_sum (see canonical_ladder note).
                vecs["fair_floor"], _fname = msp.primary_floor(
                    [records[i] for i in te_idx])  # PRE-REGISTERED msp_min bar (2026-07-24)
                for m, u in vecs.items():
                    res[pool_name][m]["prr"].append(results.prr(yte, u))
                    res[pool_name][m]["unc"].append(u)

        print(f"\n==== eval={X}  normal={srcs_used.get('normal_LOO')}  "
              f"longform={srcs_used.get('longform_LOO')} ====", flush=True)
        for m in methods:
            row = {"eval": X, "method": m}
            for p in pools:
                vals = res[p][m]["prr"]
                if vals:
                    mean, std = float(np.mean(vals)), float(np.std(vals))
                    row[f"{p}_prr"] = round(mean, 4)
                    row[f"{p}_std"] = round(std, 4)
            # paired bootstrap: normal vs longform on the same test rows (seed-averaged uncertainty)
            if res["normal_LOO"][m]["unc"] and res["longform_LOO"][m]["unc"]:
                an = np.mean(np.stack(res["normal_LOO"][m]["unc"]), axis=0)
                al = np.mean(np.stack(res["longform_LOO"][m]["unc"]), axis=0)
                mg, lo, hi, p_, sig = paired_bootstrap(yte, an, al)  # + => normal better than longform
                row["delta_normal_minus_longform"] = round(mg, 4)
                row["ci_lo"], row["ci_hi"], row["boot_p"], row["significant"] = (
                    round(lo, 4), round(hi, 4), round(p_, 4), sig)
                print(f"    {m:18s} normal {row.get('normal_LOO_prr')}  longform {row.get('longform_LOO_prr')}"
                      f"  Δ(norm-long) {mg:+.3f} [{lo:+.3f},{hi:+.3f}] {'SIG' if sig else 'ns'}", flush=True)
            # the confound-killer: matched (same #sources, short forced in) vs longform. + => short-form
            # helps even at fixed source COUNT -> the hypothesis holds cleanly, not just a diversity effect.
            if "matched_LOO" in res and res["matched_LOO"][m]["unc"] and res["longform_LOO"][m]["unc"]:
                am = np.mean(np.stack(res["matched_LOO"][m]["unc"]), axis=0)
                al2 = np.mean(np.stack(res["longform_LOO"][m]["unc"]), axis=0)
                mg2, lo2, hi2, p2, sig2 = paired_bootstrap(yte, am, al2)
                row["delta_matched_minus_longform"] = round(mg2, 4)
                row["m_ci_lo"], row["m_ci_hi"], row["m_boot_p"], row["m_significant"] = (
                    round(lo2, 4), round(hi2, 4), round(p2, 4), sig2)
                print(f"    {'':18s} matched {row.get('matched_LOO_prr')} (srcs={srcs_used.get('matched_LOO')})"
                      f"  Δ(match-long) {mg2:+.3f} [{lo2:+.3f},{hi2:+.3f}] {'SIG' if sig2 else 'ns'}", flush=True)
            out_rows.append(row)

    out = Path(args.out) if args.out else (ROOT / "results" / f"long_form_loo__{cache._slug(MODEL)}.csv")
    cols = ["eval", "method", "normal_LOO_prr", "normal_LOO_std", "longform_LOO_prr", "longform_LOO_std",
            "matched_LOO_prr", "matched_LOO_std",
            "delta_normal_minus_longform", "ci_lo", "ci_hi", "boot_p", "significant",
            "delta_matched_minus_longform", "m_ci_lo", "m_ci_hi", "m_boot_p", "m_significant"]
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
