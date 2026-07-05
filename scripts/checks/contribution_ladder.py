"""The contribution OOD ladder: weighted-MSP vs the attention pooler vs the MSP floor, across an
INCREMENTAL shift ladder (ID -> SameTask -> LOO -> DiffTask), all judge-labelled.

WHY A SEPARATE DRIVER (not ood_onegrid)
---------------------------------------
ood_onegrid.py is the verified BASELINE ladder (SAPLMA/linear/ptrue/lookback + poolers) and we leave
it untouched. This driver adds the two things the contribution needs and isolates their risk:
  * the two new methods -- weighted-MSP (Track 2) and the plain-MSP floor -- so a bug here cannot break
    the baseline ladder;
  * the SameTask rung (train on a same-task NEIGHBOUR dataset, e.g. med_quad for the QA evals), the mild
    shift step that ood_onegrid skips.

Every method here reads only the per-token states + the record logprobs, so a training SOURCE needs
only its pertok cache + records -- NOT the extra baseline features (lookback/ptrue). That is exactly
what lets med_quad (for which we extracted only generation + SAPLMA) slot in as a training source.

METHODS (all on the same per-token L15 states / same records, so the axis is aggregation only):
  uniform            frozen-query attention == mean-pool, torch linear head (the controlled baseline)
  attention          learned-query softmax attention, T selected on a val split
  weighted_msp_norm  learned per-token weight on NLL, softmax-normalised   (Track 2, the contribution)
  weighted_msp_unc   learned per-token weight on NLL, unconstrained
  msp_sum            plain MSP (unsupervised floor; shift-invariant)
  perplexity         length-normalised MSP (unsupervised floor)

RUNGS (get_training_spec-faithful, restricted to the sources we have a pertok cache for):
  ID        train on eval's own train split
  SameTask  OOD_ONE_DATASET_SAME_TASK  (QA -> med_quad)         <- the new mild rung
  LOO       OOD_LEAVE_ONE_OUT
  DiffTask  OOD_DIFF_TASK

PAIRED seeds + ID-diagonal gate, same discipline as ood_onegrid.

    python scripts/checks/contribution_ladder.py --seeds 1,2,3
    python scripts/checks/contribution_ladder.py --sources sciq,trivia_qa,pubmed_qa --seeds 1  # smoke
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
from transformers import AutoTokenizer  # noqa: E402

from luq import cache, msp, results, weighted_msp  # noqa: E402
from probe_drift.ood_settings import get_training_spec  # noqa: E402
from aggregation_table import load_per_token, build_arrays, attn_unc  # noqa: E402
from attn_pool import train_attn, select_temperature  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAB = "correctness"
EVALS = ["sciq", "trivia_qa", "pubmed_qa"]
# Candidate training sources; each included only if its pertok cache actually loads.
CANDIDATE_SOURCES = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "med_quad"]
# ID anchors (judge, from the aggregation table) the ID cells must reproduce.
ID_ANCHOR = {"sciq": {"uniform": 0.913, "attention": 0.932},
             "trivia_qa": {"uniform": 0.815, "attention": 0.844},
             "pubmed_qa": {"uniform": 0.683, "attention": 0.735}}
GATE_TOL = 0.03
SETTINGS = [("SameTask", "OOD_ONE_DATASET_SAME_TASK"),
            ("LOO", "OOD_LEAVE_ONE_OUT"),
            ("DiffTask", "OOD_DIFF_TASK")]


def sampled_train_idx(split, seed, cap):
    tr = np.where(split == "train")[0]
    if cap is None or cap >= len(tr):
        return tr
    return tr[np.random.RandomState(seed).permutation(len(tr))[:cap]]


def cells(sources):
    """(rung, eval, [(src, cap)]) using only sources we have a pertok cache for."""
    out = [("ID", X, [(X, None)]) for X in EVALS]
    for X in EVALS:
        for tag, setting in SETTINGS:
            spec = [(s, n) for s, n in get_training_spec(X, setting) if s in sources and s != X]
            if spec:  # skip a rung whose sources we do not have (e.g. SameTask before med_quad lands)
                out.append((tag, X, spec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--sources", default=",".join(CANDIDATE_SOURCES),
                    help="training sources to try; each used only if its pertok cache loads")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--length-normalise", default="yes", choices=["yes", "no"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ln = args.length_normalise == "yes"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"device {device} | seeds {seeds} | length_normalise={ln}", flush=True)

    # Load per-token states + records for every source whose cache exists (tolerant).
    PT = {}
    for d in args.sources.split(","):
        loaded = load_per_token(MODEL, d, args.layer, LAB)
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip as a source", flush=True)
            continue
        states, split, y, _, records = loaded
        if np.isnan(y).any():
            print(f"  {d}: unlabelled (NaN correctness) -> skip as a source", flush=True)
            continue
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} rows loaded", flush=True)
    sources = set(PT)
    methods = ["uniform", "attention", "weighted_msp_norm", "weighted_msp_unc", "msp_sum", "perplexity"]

    out_rows = []
    for rung, X, spec in cells(sources):
        if X not in PT:
            continue
        per_method = {m: [] for m in methods}
        for sd in seeds:
            train_rows = [(d, i) for d, cap in spec for i in sampled_train_idx(PT[d][1], sd, cap)]
            test_rows = [(X, i) for i in np.where(PT[X][1] == "test")[0]]
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], dtype=float)
            yte = [y[i] for i in te_idx]
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]

            # poolers (uniform / attention) reuse the verified attn machinery
            best_T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
            res = {}
            res["uniform"] = results.prr(yte, attn_unc(
                train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True), states, te_idx, device))
            res["attention"] = results.prr(yte, attn_unc(
                train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T), states, te_idx, device))
            # weighted-MSP (contribution)
            res["weighted_msp_norm"] = results.prr(yte, weighted_msp.weighted_msp_unc(
                states, records, y, tr_idx, te_idx, device, weight_mode="normalised",
                length_normalise=ln, seed=sd))
            res["weighted_msp_unc"] = results.prr(yte, weighted_msp.weighted_msp_unc(
                states, records, y, tr_idx, te_idx, device, weight_mode="unconstrained",
                length_normalise=ln, seed=sd))
            # plain MSP floor (unsupervised -> identical across seeds/rungs, computed per cell for the table)
            res["msp_sum"] = results.prr(yte, [msp.msp_uncertainty(records[i]["token_logprobs"], "sum")
                                               for i in te_idx])
            res["perplexity"] = results.prr(yte, [msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity")
                                                  for i in te_idx])
            for m, v in res.items():
                per_method[m].append(v)

        stats = {m: (float(np.mean(v)), float(np.std(v))) for m, v in per_method.items() if v}
        srcs = "+".join(f"{d}:{c}" if c else d for d, c in spec)
        print(f"\n[{rung:9s}] eval={X}  train={srcs}", flush=True)
        for m in methods:
            if m in stats:
                print(f"    {m:18s} {stats[m][0]:+.3f} +/- {stats[m][1]:.3f}", flush=True)
        # ID-diagonal gate on the controlled poolers
        if rung == "ID" and X in ID_ANCHOR:
            for m in ("uniform", "attention"):
                d = abs(stats[m][0] - ID_ANCHOR[X][m])
                assert d < GATE_TOL, f"ID-GATE FAIL {X}/{m}: {stats[m][0]:.3f} vs {ID_ANCHOR[X][m]} (|d|={d:.3f})"
            print(f"    [ID-GATE OK] poolers reproduce anchors within {GATE_TOL}", flush=True)
        for m in methods:
            if m in stats:
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": m,
                                 "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                                 "n_seeds": len(per_method[m])})

    out = Path(args.out) if args.out else (ROOT / "results" / f"contribution_ladder__{cache._slug(MODEL)}.csv")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "train", "method", "prr_mean", "prr_std", "n_seeds"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
