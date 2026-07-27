"""STEP 7: is the samsum DiffTask gain DIVERSITY or just DATA VOLUME?

{samsum,xsum} has two datasets AND twice the data of {xsum}. Controlled grid at matched N, same eval
sets / seeds / bootstrap (all CPU, cached features):
    xsum-only N        samsum-only N        {xsum,samsum} N/2 each (=N)        {xsum,samsum} full (=2N)
If the mixed-at-equal-N pool beats BOTH singles at N  -> diversity helps.
If only the 2N pool wins                              -> it's just data volume, and the samsum
                                                          DiffTask "finding" is deflated.

Reported for each QA eval and each method (uniform / attention / weighted-MSP), with a paired bootstrap
of mixed-N vs each single-N. N = 600 (the DiffTask per-source cap).

    python scripts/checks/diversity_vs_size.py --n 600 --seeds 1,2,3
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

from luq import cache, msp, results, weighted_msp  # noqa: E402
from aggregation_table import load_per_token, attn_unc, paired_bootstrap  # noqa: E402
from attn_pool import train_attn, select_temperature  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAB, LAYER = "correctness", 15
EVALS = ["sciq", "trivia_qa", "pubmed_qa"]
SUMM = ["xsum", "samsum"]


def sample(split, seed, n):
    tr = np.where(split == "train")[0]
    return tr[np.random.RandomState(seed).permutation(len(tr))[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seeds", default="1,2,3")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    N = args.n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    PT = {}
    for d in SUMM + EVALS:
        loaded = load_per_token(MODEL, d, LAYER, LAB)
        if loaded is None or np.isnan(loaded[2]).any():
            print(f"  {d}: missing/unlabelled -> abort" if d in SUMM else f"  {d}: skip eval");
            if d in SUMM: sys.exit(1)
            continue
        PT[d] = loaded[0], loaded[1], loaded[2], loaded[4]

    # training pools: (name, [(source, n_per_source)])
    POOLS = [("xsum_N", [("xsum", N)]), ("samsum_N", [("samsum", N)]),
             ("mix_N", [("xsum", N // 2), ("samsum", N // 2)]),
             ("mix_2N", [("xsum", N), ("samsum", N)])]

    def method_unc(spec, X, method, sd):
        train_rows = [(d, i) for d, npd in spec for i in sample(PT[d][1], sd, npd)]
        test_rows = [(X, i) for i in np.where(PT[X][1] == "test")[0]]
        n_tr = len(train_rows)
        tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
        allrows = train_rows + test_rows
        y = np.array([PT[d][2][i] for d, i in allrows], dtype=float)
        states = [PT[d][0][i] for d, i in allrows]
        records = [PT[d][3][i] for d, i in allrows]
        if method == "uniform":
            return attn_unc(train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True), states, te_idx, device)
        if method == "attention":
            bt, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
            return attn_unc(train_attn(states, y, tr_idx, device, seed=sd, temperature=bt), states, te_idx, device)
        return weighted_msp.weighted_msp_unc(states, records, y, tr_idx, te_idx, device,
                                             weight_mode="normalised", length_normalise=True, seed=sd, loss="pairwise")

    rows = []
    for X in EVALS:
        te = np.where(PT[X][1] == "test")[0]
        yte = PT[X][2][te]
        # FAIR floor (2026-07-22): best of {msp_sum, perplexity, msp_min}; msp_sum is the weakest on all 9.
        _fv, _fname = msp.primary_floor([PT[X][3][i] for i in te])  # PRE-REGISTERED msp_min bar (2026-07-24)
        floor = results.prr(yte, _fv)
        print(f"\n=== eval={X}  (floor msp_sum={floor:+.3f}) ===", flush=True)
        for method in ["uniform", "attention", "weighted_msp"]:
            prr, unc = {}, {}
            for pname, spec in POOLS:
                vv, uu = [], []
                for sd in seeds:
                    u = np.asarray(method_unc(spec, X, method, sd), dtype=float)
                    vv.append(results.prr(yte, u)); uu.append(u)
                prr[pname] = float(np.mean(vv)); unc[pname] = np.mean(np.stack(uu), axis=0)
            # verdicts: mix_N vs each single at N
            mg_x, lo_x, hi_x, p_x, sx = paired_bootstrap(yte, unc["mix_N"], unc["xsum_N"])
            mg_s, lo_s, hi_s, p_s, ss = paired_bootstrap(yte, unc["mix_N"], unc["samsum_N"])
            print(f"  {method:12s} xsum_N {prr['xsum_N']:+.3f}  samsum_N {prr['samsum_N']:+.3f}  "
                  f"mix_N {prr['mix_N']:+.3f}  mix_2N {prr['mix_2N']:+.3f}  | mixN-vs-xsumN {mg_x:+.3f}"
                  f"{'*' if sx else ''}  mixN-vs-samsumN {mg_s:+.3f}{'*' if ss else ''}", flush=True)
            rows.append({"eval": X, "method": method, "floor": round(floor, 4),
                         "xsum_N": round(prr["xsum_N"], 4), "samsum_N": round(prr["samsum_N"], 4),
                         "mix_N": round(prr["mix_N"], 4), "mix_2N": round(prr["mix_2N"], 4),
                         "mixN_vs_xsumN": round(mg_x, 4), "sig_vs_xsum": sx,
                         "mixN_vs_samsumN": round(mg_s, 4), "sig_vs_samsum": ss})

    out = ROOT / "results" / f"diversity_vs_size__{cache._slug(MODEL)}.csv"
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}", flush=True)
    print("READ: mix_N beating BOTH singles at N = diversity; only mix_2N winning = data volume "
          "(deflates the samsum finding).", flush=True)


if __name__ == "__main__":
    main()
