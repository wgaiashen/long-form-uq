"""Quantify what the med_quad RELABEL changed: train the SameTask-rung probes on med_quad with the
CLEAN answer-span label vs the RAW label, everything else held equal (same L15 states, same QA eval
sets, same seeds), and compare PRR.

This isolates the LABEL change only -- unlike diffing two ladder CSVs from different eras (which also
differ in med_quad-in-LOO, code, etc.). med_quad records carry both `correctness` (clean, promoted) and
`correctness_raw` (the pre-clean backup), so we just toggle the training label field.

For each QA eval (sciq, trivia_qa, pubmed_qa): train on med_quad (1800), test on the eval's test split.
Methods: SAPLMA mean-pool, attention pooler, weighted-MSP (the ladder's supervised methods). The MSP
floor is label-independent, so it is identical clean-vs-raw and printed once as a sanity anchor.

    python scripts/checks/med_quad_clean_vs_raw.py --seeds 1 2 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402

from luq import msp, probe, results, weighted_msp  # noqa: E402
from aggregation_table import load_per_token, attn_unc  # noqa: E402
from attn_pool import train_attn, select_temperature  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15
EVALS = ["sciq", "trivia_qa", "pubmed_qa"]


def saplma_mean_prr(states, y, tr, te, yte, seeds):
    Xtr = np.stack([states[i].mean(axis=0) for i in tr])
    Xte = np.stack([states[i].mean(axis=0) for i in te])
    return float(np.mean([results.prr(yte, probe.uncertainty(
        probe.train_probe_mlp(Xtr, y[tr], seed=sd), Xte)) for sd in seeds]))


def attention_prr(states, y, tr, te, yte, device, seeds):
    best_T, _ = select_temperature(states, y, tr, device, seeds[0], False, False)
    vals = []
    for sd in seeds:
        mdl = train_attn(states, y, tr, device, seed=sd, temperature=best_T)
        vals.append(results.prr(yte, np.asarray(attn_unc(mdl, states, te, device), dtype=float)))
    return float(np.mean(vals))


def wmsp_prr(states, records, y, tr, te, yte, device, seeds):
    vals = []
    for sd in seeds:
        u = weighted_msp.weighted_msp_unc(states, records, y, tr, te, device,
                                          weight_mode="normalised", length_normalise=True, seed=sd)
        vals.append(results.prr(yte, np.asarray(u, dtype=float)))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = args.seeds

    # med_quad states are label-independent; load once with each label field for the training targets.
    mq_clean = load_per_token(MODEL, "med_quad", LAYER, "correctness")       # promoted clean label
    mq_raw = load_per_token(MODEL, "med_quad", LAYER, "correctness_raw")     # pre-clean backup
    if mq_raw is None or np.isnan(mq_raw[2]).all():
        sys.exit("med_quad has no correctness_raw -> was it promoted? run promote_med_quad_clean.py first")
    mq_states, mq_split, y_clean, _, mq_records = mq_clean
    _, _, y_raw, _, _ = mq_raw
    mq_tr = np.where(mq_split == "train")[0]
    print(f"med_quad train rows: {len(mq_tr)} | mean label raw {np.nanmean(y_raw):.3f} -> clean "
          f"{np.nanmean(y_clean):.3f}", flush=True)

    print(f"\n{'eval':10s} {'method':14s} {'raw':>8s} {'clean':>8s} {'delta':>8s}   {'floor':>8s}", flush=True)
    for ev in EVALS:
        loaded = load_per_token(MODEL, ev, LAYER, "correctness")
        if loaded is None:
            print(f"{ev}: no pertok -> skip", flush=True); continue
        ev_states, ev_split, ev_y, _, ev_records = loaded
        ev_te = np.where(ev_split == "test")[0]

        # assemble the SameTask cell: med_quad train rows + eval test rows
        states = [mq_states[i] for i in mq_tr] + [ev_states[i] for i in ev_te]
        records = [mq_records[i] for i in mq_tr] + [ev_records[i] for i in ev_te]
        n_tr = len(mq_tr)
        tr = list(range(n_tr)); te = list(range(n_tr, n_tr + len(ev_te)))
        yte = ev_y[ev_te]                                   # eval's OWN label (unchanged); PRR target

        floor = results.prr(yte, np.asarray(
            [msp.msp_uncertainty(records[i]["token_logprobs"], "sum") for i in te], dtype=float))

        for name, fn in [("saplma_mean", "sap"), ("attention", "attn"), ("weighted_msp", "wmsp")]:
            row = {}
            for tag, ymq in [("raw", y_raw), ("clean", y_clean)]:
                y = np.concatenate([ymq[mq_tr], yte])       # train uses med_quad label; test uses eval label
                if fn == "sap":
                    row[tag] = saplma_mean_prr(states, y, tr, te, yte, seeds)
                elif fn == "attn":
                    row[tag] = attention_prr(states, y, tr, te, yte, device, seeds)
                else:
                    row[tag] = wmsp_prr(states, records, y, tr, te, yte, device, seeds)
            d = row["clean"] - row["raw"]
            flag = "  <--" if abs(d) >= 0.03 else ""
            print(f"{ev:10s} {name:14s} {row['raw']:+8.3f} {row['clean']:+8.3f} {d:+8.3f}   "
                  f"{floor:+8.3f}{flag}", flush=True)


if __name__ == "__main__":
    main()
