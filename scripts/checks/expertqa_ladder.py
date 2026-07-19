"""ExpertQA as a FIRST-CLASS OOD ladder eval (ProbeDrift-XL factuality axis) — normal eval, not LOCO.

ExpertQA becomes an eval TARGET on the ladder like pubmed/xsum, scored on `faithfulness`:
  ID        train + test on ExpertQA (faithfulness)                       -> the clean same-label CEILING
  SameTask  train on the long-QA neighbours (pubmed_qa + med_quad, correctness), test ExpertQA
  LOO       train on ALL other cached datasets (correctness), test ExpertQA
  DiffTask  train on the summarisation family (xsum + samsum + cnn, correctness), test ExpertQA

CROSS-LABEL CAVEAT (state it in every table): the OOD rungs train a probe on the other datasets'
`correctness` and score it against ExpertQA's `faithfulness`, so the ID->OOD drop conflates a TASK shift
AND a LABEL shift. The ID rung (same-label faithfulness) is the reference that isolates it. The MSP floor
is label-agnostic and transfers cleanly.

Methods per rung: msp_floor, saplma (mean-pool+MLP), uniform (frozen-q pooler), attention (pooler,
temperature selected ONCE on ID), wMSP-norm (weighted-MSP with the EOS fix). 3 seeds, paired bootstrap
vs floor. CPU only. ExpertQA labels: faithfulness (1724 rows; 292 all-uncovered None dropped).

    python scripts/checks/expertqa_ladder.py --seeds 1,2,3
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
from aggregation_table import load_per_token, attn_unc, paired_bootstrap  # noqa: E402
from attn_pool import train_attn, select_temperature  # noqa: E402
from expertqa_loco_agg import load_pertok as load_eqa_pertok  # noqa: E402
from expertqa_loco import stratified_id_split, load_data as load_eqa_saplma  # noqa: E402
from expertqa_label_validation import load_labelled_with_group  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15
EQA_LABEL = "faithfulness"
# OOD training sources (all carry `correctness`), and the rung composition.
SOURCES = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "med_quad", "samsum", "cnn_dailymail"]
RUNGS = {"SameTask": ["pubmed_qa", "med_quad"],
         "LOO": SOURCES,
         "DiffTask": ["xsum", "samsum", "cnn_dailymail"]}
TOTAL = 1800   # matched training size per OOD rung (split across its sources)


def sampled(split_arr, seed, cap):
    tr = np.where(split_arr == "train")[0]
    if cap is None or cap >= len(tr):
        return tr
    return tr[np.random.RandomState(seed).permutation(len(tr))[:cap]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | seeds {seeds} | ExpertQA label={EQA_LABEL}", flush=True)

    # ---- ExpertQA: per-token states + cluster-joined records + faithfulness (drop the 292 None) ----
    eqa_states, eqa_idx = load_eqa_pertok()
    recs = load_labelled_with_group()
    assert len(eqa_states) == len(recs), f"{len(eqa_states)} states vs {len(recs)} recs"
    valid = [i for i in range(len(recs))
             if isinstance(recs[i].get(EQA_LABEL), (int, float)) and np.isfinite(recs[i].get(EQA_LABEL))]
    E_st = [eqa_states[i] for i in valid]
    E_rec = [recs[i] for i in valid]
    E_y = np.array([float(recs[i][EQA_LABEL]) for i in valid], float)
    E_cl = np.array([recs[i]["cluster"] for i in valid])
    print(f"  ExpertQA: {len(E_st)}/{len(recs)} kept ({EQA_LABEL}); clusters {dict(zip(*np.unique(E_cl, return_counts=True)))}", flush=True)

    # integrity gate: mean-pool-pertok PRR vs cached SAPLMA PRR on one stratified ID split
    Xsap, sap_recs = load_eqa_saplma()
    Xsap = Xsap[valid]
    tr0, te0 = stratified_id_split(E_cl, seeds[0])
    Xmean = np.stack([np.asarray(s).mean(axis=0) for s in E_st])
    prr_pt = results.prr(E_y[te0], probe.uncertainty(probe.train_probe_mlp(Xmean[tr0], E_y[tr0], seed=seeds[0]), Xmean[te0]))
    prr_sap = results.prr(E_y[te0], probe.uncertainty(probe.train_probe_mlp(Xsap[tr0], E_y[tr0], seed=seeds[0]), Xsap[te0]))
    print(f"  [integrity] mean-pool-pertok PRR {prr_pt:+.4f} vs cached-SAPLMA PRR {prr_sap:+.4f} (diff {abs(prr_pt-prr_sap):.4f})", flush=True)

    # ---- other datasets (correctness) as OOD training sources ----
    PT = {}
    for d in SOURCES:
        loaded = load_per_token(MODEL, d, LAYER, "correctness")
        if loaded is None:
            print(f"  {d}: no pertok -> skip"); continue
        st, split, y, _, rc = loaded
        if np.isnan(np.asarray(y, float)).any():
            print(f"  {d}: NaN correctness -> skip"); continue
        PT[d] = (st, split, y, rc)
    have = set(PT)
    print(f"  OOD sources cached: {sorted(have)}", flush=True)

    def get(d, i):
        """(state, label, record) for a row. ExpertQA rows index into the E_* arrays.
        PT[d] = (states, split, y, records) -> the LABEL is index [2] (y), NOT [1] (the split strings)."""
        if d == "expertqa":
            return E_st[i], E_y[i], E_rec[i]
        return PT[d][0][i], PT[d][2][i], PT[d][3][i]

    # temperature selected ONCE on the ExpertQA ID split (reused across rungs).
    best_T, _ = select_temperature(E_st, E_y, list(tr0), device, seeds[0], False, False)
    print(f"  [pooler] temperature selected once on ExpertQA ID: T={best_T}", flush=True)

    METHODS = ["msp_floor", "saplma", "uniform", "attention", "wMSP-norm"]
    out_rows = []
    for rung in ["ID"] + list(RUNGS):
        per = {m: [] for m in METHODS}
        for sd in seeds:
            # rung train rows + fixed ExpertQA test rows
            tr_e, te_e = stratified_id_split(E_cl, sd)
            test_rows = [("expertqa", i) for i in te_e]
            if rung == "ID":
                train_rows = [("expertqa", i) for i in tr_e]
            else:
                srcs = [d for d in RUNGS[rung] if d in have]
                cap = max(1, TOTAL // max(len(srcs), 1))
                train_rows = [(d, i) for d in srcs for i in sampled(PT[d][1], sd, cap)]
            if not train_rows:
                continue
            allrows = train_rows + test_rows
            n_tr = len(train_rows)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            st = [get(d, i)[0] for d, i in allrows]
            y = np.array([get(d, i)[1] for d, i in allrows], float)
            rc = [get(d, i)[2] for d, i in allrows]
            yte = E_y[te_e]                                     # always faithfulness

            # msp floor (label-agnostic)
            per["msp_floor"].append(results.prr(yte, np.array(
                [msp.msp_uncertainty(rc[i]["token_logprobs"], "sum") for i in te_idx])))
            # saplma (mean-pool + MLP)
            Xm = np.stack([np.asarray(s).mean(axis=0) for s in st])
            per["saplma"].append(results.prr(yte, probe.uncertainty(
                probe.train_probe_mlp(Xm[tr_idx], y[tr_idx], seed=sd), Xm[te_idx])))
            # uniform + attention poolers
            per["uniform"].append(results.prr(yte, attn_unc(
                train_attn(st, y, tr_idx, device, seed=sd, freeze_query=True), st, te_idx, device)))
            per["attention"].append(results.prr(yte, attn_unc(
                train_attn(st, y, tr_idx, device, seed=sd, temperature=best_T), st, te_idx, device)))
            # weighted-MSP (EOS fix on by default)
            per["wMSP-norm"].append(results.prr(yte, np.asarray(weighted_msp.weighted_msp_unc(
                st, rc, y, tr_idx, te_idx, device, weight_mode="normalised", length_normalise=True, seed=sd), float)))

        line = f"[{rung:9s}] n_test={len(te_e):4d}"
        for m in METHODS:
            if per[m]:
                mean, std = float(np.mean(per[m])), float(np.std(per[m]))
                out_rows.append({"rung": rung, "method": m, "prr_mean": round(mean, 4),
                                 "prr_std": round(std, 4), "n_test": int(len(te_e))})
                line += f"  {m} {mean:+.3f}"
        print(line, flush=True)

    out = Path(args.out) if args.out else (ROOT / "results" / "expertqa" / "expertqa_ladder_faithfulness.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "method", "prr_mean", "prr_std", "n_test"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\nwrote {out}\nNOTE: OOD rungs are CROSS-LABEL (train correctness, test faithfulness); ID is same-label.", flush=True)


if __name__ == "__main__":
    main()
