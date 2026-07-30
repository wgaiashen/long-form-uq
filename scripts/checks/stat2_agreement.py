#!/usr/bin/env python
"""STAT 2 (§D.7) -- method-agreement matrix on the OOD cells + disagreement-vs-router-gain test.

Per dataset, pairwise Spearman between per-example scores of {msp_min, perplexity, wMSP-shrink@10,
attention pooler, SAPLMA} on the broad-LEAVE-D-OUT OOD population (the SAME population the router uses:
train the probes on the OTHER 8 datasets, apply to D's test set). All 5 vectors are computed together on
one population -- no cross-cache join. GATE: my floor_min + pooler PRR reproduce the cached router_ood npz.

WHY: §D.6 shows the best floor VARIANT differs by dataset -> a THREE-WAY router {msp_min, perplexity, probe}
may beat the binary one. Low agreement is where per-example selection can pay. We report corr(per-dataset mean
disagreement, router's per-dataset gain over always-pooler) -- the direct test of whether extending is worth it.
No GPU (CPU probe training over cached pertok). No sampling.
"""
import sys, csv as _csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "checks"))
import torch                                                                    # noqa: E402
from luq import msp, results, weighted_msp                                      # noqa: E402
from luq.weighting import shrink_to_uniform                                     # noqa: E402
from attn_pool import load_per_token, train_attn, select_temperature           # noqa: E402
from aggregation_table import attn_unc, conf_meanpool                          # noqa: E402
from xl_rungs import eval_split, label_of                                       # noqa: E402
from length_router import CACHE_OOD, MODEL_SLUG                                 # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15
DATASETS = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail",
            "med_quad", "samsum", "expertqa", "asqa"]
METHODS = ["msp_min", "perplexity", "wmsp_shrink10", "pooler", "saplma"]

# router (LENGTH gate) per-dataset gain over always-pooler -- from §D.3 (3494057 LODO OOD table).
ROUTER_GAIN_VS_POOL = {
    "sciq": 0.854 - 0.770, "trivia_qa": 0.746 - 0.616, "pubmed_qa": 0.343 - 0.313,
    "xsum": 0.230 - 0.230, "cnn_dailymail": 0.248 - 0.248, "med_quad": 0.322 - 0.322,
    "samsum": 0.343 - 0.343, "expertqa": 0.167 - 0.183, "asqa": 0.347 - 0.388,
}


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_ci(x, y, fn, n=2000):
    x = np.asarray(x, float); y = np.asarray(y, float); m = len(x)
    rng = np.random.RandomState(0); vals = []
    for _ in range(n):
        idx = rng.randint(0, m, m)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        vals.append(fn(x[idx], y[idx]))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    DATA = {}
    for d in DATASETS:
        loaded = load_per_token(MODEL, d, LAYER, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok -> ABORT (need all 9 for the broad pool)"); sys.exit(1)
        states, split, y, _lyr, records = loaded
        DATA[d] = (states, split, y, records)
        print(f"  loaded {d}: {len(states)} rows", flush=True)

    cap = 225
    mats = {}            # dataset -> 5x5 spearman matrix
    disagree = {}        # dataset -> mean pairwise (1 - spearman)
    gate_rows = []
    all_rows = []
    for D in DATASETS:
        s_D, sp_D, y_D, rec_D = DATA[D]
        _, te = eval_split(sp_D); te = [i for i in te if np.isfinite(y_D[i])]
        rng = np.random.RandomState(1)                                     # IDENTICAL to extract_ood
        pool_states, pool_records, pool_y = [], [], []
        for o in DATASETS:
            if o == D:
                continue
            s_o, sp_o, y_o, rec_o = DATA[o]
            tr_o, _ = eval_split(sp_o); tr_o = [i for i in tr_o if np.isfinite(y_o[i])]
            for i in rng.choice(tr_o, min(cap, len(tr_o)), replace=False):
                pool_states.append(s_o[i]); pool_records.append(rec_o[i]); pool_y.append(float(y_o[i]))
        allstates = pool_states + [s_D[i] for i in te]
        allrecords = pool_records + [rec_D[i] for i in te]
        ally = np.array(pool_y + [float(y_D[i]) for i in te], float)
        tr_idx = list(range(len(pool_states))); te_idx = list(range(len(pool_states), len(allstates)))
        yte = ally[te_idx]

        v = {}
        v["msp_min"] = np.array([msp.msp_uncertainty(rec_D[i]["token_logprobs"], "min") for i in te])
        v["perplexity"] = np.array([msp.msp_uncertainty(rec_D[i]["token_logprobs"], "perplexity") for i in te])
        best_T, _ = select_temperature(allstates, ally, tr_idx, device, 1, False, False)
        pooler = train_attn(allstates, ally, tr_idx, device, seed=1, temperature=best_T)
        v["pooler"] = np.asarray(attn_unc(pooler, allstates, te_idx, device), float)
        Xmean = np.stack([s.mean(axis=0) for s in allstates])
        v["saplma"] = 1.0 - conf_meanpool(Xmean, tr_idx, te_idx, ally, 1)
        v["wmsp_shrink10"] = np.asarray(weighted_msp.weighted_msp_unc(
            allstates, allrecords, ally, tr_idx, te_idx, device,
            weight_mode="normalised", reg=shrink_to_uniform, reg_lambda=10.0,
            length_normalise=True, seed=1), float)

        # GATE: reproduce the cached router_ood floor_min + pooler PRR on the same population
        cpath = CACHE_OOD / f"{MODEL_SLUG}__{D}.npz"
        gate = "no-cache"
        if cpath.exists():
            z = np.load(cpath)
            same_y = (len(z["y"]) == len(yte)) and np.allclose(z["y"], yte)
            d_floor = abs(results.prr(yte, v["msp_min"]) - results.prr(z["y"], z["floor_min"]))
            d_pool = abs(results.prr(yte, v["pooler"]) - results.prr(z["y"], z["pooler_unc"]))
            gate = f"y_match={same_y} ΔfloorPRR={d_floor:.4f} ΔpoolPRR={d_pool:.4f} " \
                   f"{'PASS' if (same_y and d_floor < 1e-6 and d_pool < 0.03) else 'CHECK'}"
        gate_rows.append((D, gate))

        # 5x5 Spearman matrix + per-example PRRs (context)
        M = np.ones((5, 5))
        for a in range(5):
            for b in range(a + 1, 5):
                r = spearman(v[METHODS[a]], v[METHODS[b]])
                M[a, b] = M[b, a] = r
        mats[D] = M
        off = [M[a, b] for a in range(5) for b in range(a + 1, 5)]
        disagree[D] = float(np.mean([1 - r for r in off if r == r]))
        prrs = {m: results.prr(yte, v[m]) for m in METHODS}
        all_rows.append((D, len(te), disagree[D], ROUTER_GAIN_VS_POOL[D], prrs, M))
        print(f"  {D}: n_te={len(te)} mean_disagree={disagree[D]:.3f} "
              f"router_gain={ROUTER_GAIN_VS_POOL[D]:+.3f} | GATE {gate}", flush=True)

    # ---- report ----
    print("\n" + "=" * 82); print("STAT 2 -- per-dataset mean pairwise disagreement (1 - Spearman) vs router gain"); print("=" * 82)
    print(f"{'dataset':14s}{'disagree':>10s}{'router_gain':>13s}   PRRs [msp_min, ppl, wmsp10, pooler, saplma]")
    for D, n, dis, rg, prrs, M in all_rows:
        print(f"{D:14s}{dis:>10.3f}{rg:>13.3f}   "
              + "[" + ", ".join(f"{prrs[m]:+.3f}" for m in METHODS) + "]")
    xs = [dis for _, _, dis, _, _, _ in all_rows]; ys = [rg for _, _, _, rg, _, _ in all_rows]
    pear = float(np.corrcoef(xs, ys)[0, 1]); spear = spearman(xs, ys)
    ci = boot_ci(xs, ys, lambda a, b: float(np.corrcoef(a, b)[0, 1]))
    print(f"\ncorr(mean disagreement, router gain over always-pooler): "
          f"Pearson {pear:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] | Spearman {spear:+.3f}  (n=9)")
    print("READ: positive -> the binary router already captures the disagreement value; "
          "high disagreement + ~0 router gain -> room for a THREE-WAY router.")

    print("\n" + "=" * 82); print("MEAN agreement matrix across datasets (Spearman)"); print("=" * 82)
    MM = np.mean([mats[D] for D in DATASETS], axis=0)
    print(f"{'':14s}" + "".join(f"{m[:9]:>11s}" for m in METHODS))
    for a, m in enumerate(METHODS):
        print(f"{m:14s}" + "".join(f"{MM[a, b]:>+11.2f}" for b in range(5)))

    outp = Path("/rds/general/ephemeral/user/gs925/ephemeral/luq_overnight_results/stat2_agreement__" + MODEL_SLUG + ".csv")
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["dataset", "n_te", "mean_disagree", "router_gain"] + [f"prr_{m}" for m in METHODS]
                   + [f"sp_{METHODS[a]}_{METHODS[b]}" for a in range(5) for b in range(a + 1, 5)])
        for D, n, dis, rg, prrs, M in all_rows:
            w.writerow([D, n, dis, rg] + [prrs[m] for m in METHODS]
                       + [M[a, b] for a in range(5) for b in range(a + 1, 5)])
    print(f"\nGATE summary:")
    for D, g in gate_rows:
        print(f"  {D:14s} {g}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
