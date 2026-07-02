"""ProbeDriftLight OOD for the aggregation family: does the learned attention pooler generalise
across tasks, or collapse toward mean-pool off-distribution?

Light vocabulary only: datasets {sciq, trivia_qa, pubmed_qa} x settings {ID, LOO, DiffTask}.
  ID(X)       train on X's train,                     test X's test   (== the ID aggregation table)
  LOO(X)      train on the pooled OTHER two datasets, test X's test
  DiffTask(X) train on the DIFFERENT-task dataset(s), test X's test
              (short QA = sciq, trivia_qa ; long QA = pubmed_qa)

Realised by pooling EXISTING ID per-token caches + EXISTING ID labels -- no generation, no
labelling, no spend (the free ProbeDriftLight route; the protocol-faithful 1800-example ProbeDrift
mixtures would need re-extraction and are deferred). Reuses every aggregator from
aggregation_table.py so the OOD table is apples-to-apples with the ID table.

TWO GUARDS (do not trust an off-diagonal cell without them):
  * PROMPT REGIME = OLD, matching the ID table (default namespace per-token caches). Never mix.
  * ID-DIAGONAL GATE: each ID cell must reproduce the ID aggregation table (mean-pool AND attention),
    a HARD assert -- if the cross-dataset assembly is wrong, ID won't reproduce and no LOO/DiffTask
    cell is trustworthy.

Labels are the existing per-dataset judge (gpt-5 sciq/trivia, gpt-5-mini pubmed), same as the ID
table; a POOLED train set therefore mixes the two judge models -- documented, and an all-mini
re-label is a morning option (mini only, never gpt-5). Significance for the attention-minus-mean-pool
margin is the paired TEST-SET BOOTSTRAP (CI excludes 0), never a near-zero-seed t-stat.

    python scripts/checks/aggregation_ood.py --seeds 1,2,3
"""
import argparse
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

from luq import results  # noqa: E402
# Reuse the ID driver's aggregators + bootstrap wholesale -- one implementation, apples-to-apples.
from aggregation_table import (  # noqa: E402
    load_per_token, build_arrays, conf_meanpool, conf_lasttoken, conf_persentence, conf_pertoken,
    attn_unc, paired_bootstrap)
from attn_pool import train_attn, select_temperature  # noqa: E402

MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"
LIGHT = ["sciq", "trivia_qa", "pubmed_qa"]
TASK = {"sciq": "shortQA", "trivia_qa": "shortQA", "pubmed_qa": "longQA"}
# The ID aggregation-table numbers (judge label) the ID cells MUST reproduce (gate).
ID_ANCHOR = {"sciq": {"mean": 0.907, "attn": 0.932},
             "trivia_qa": {"mean": 0.807, "attn": 0.844},
             "pubmed_qa": {"mean": 0.713, "attn": 0.735}}
GATE_TOL = 0.02


def light_cells():
    """(setting, eval_ds, train_dss) for the whole Light grid."""
    out = []
    for X in LIGHT:
        out.append(("ID", X, [X]))
        out.append(("LOO", X, [d for d in LIGHT if d != X]))
        out.append(("DiffTask", X, [d for d in LIGHT if TASK[d] != TASK[X]]))
    return out


def assemble(data, train_dss, eval_ds):
    """Combine the TRAIN split rows of train_dss with the TEST split rows of eval_ds into one
    (states, records, y, tr_idx, te_idx). For ID (train_dss == [eval_ds]) this is exactly that
    dataset's own train/test, so the ID cell reproduces the ID table."""
    st_all, rec_all, y_all = [], [], []
    for d in train_dss:
        states, split, y, records = data[d]
        for i in range(len(states)):
            if split[i] == "train":
                st_all.append(states[i]); rec_all.append(records[i]); y_all.append(y[i])
    n_train = len(st_all)
    states, split, y, records = data[eval_ds]
    for i in range(len(states)):
        if split[i] == "test":
            st_all.append(states[i]); rec_all.append(records[i]); y_all.append(y[i])
    tr_idx = list(range(n_train))
    te_idx = list(range(n_train, len(st_all)))
    return st_all, rec_all, np.array(y_all, dtype=float), tr_idx, te_idx


def run_cell(states, records, y, tr_idx, te_idx, seeds, device, tok):
    """PRR for every aggregator on one assembled cell, plus the seed-averaged attention/mean-pool
    uncertainty vectors (for the bootstrap). Temperature selected on a val split from THIS cell's
    train (never the test)."""
    Xmean, Xlast, sent_vecs = build_arrays(states, records, tok)
    yte = [y[i] for i in te_idx]
    best_T, _ = select_temperature(states, y, tr_idx, device, 1, False, False)

    rows = {m: [] for m in ["mean-pool+MLP", "last-token", "per-sentence", "per-token", "uniform",
                            "attention"]}
    pred_mean, pred_attn, pred_uni = [], [], []
    for sd in seeds:
        um = 1.0 - conf_meanpool(Xmean, tr_idx, te_idx, y, sd)
        rows["mean-pool+MLP"].append(results.prr(yte, um)); pred_mean.append(um)
        rows["last-token"].append(results.prr(yte, [1 - c for c in conf_lasttoken(Xlast, tr_idx, te_idx, y, sd)]))
        rows["per-sentence"].append(results.prr(yte, [1 - c for c in conf_persentence(sent_vecs, tr_idx, te_idx, y, sd)]))
        rows["per-token"].append(results.prr(yte, [1 - c for c in conf_pertoken(states, tr_idx, te_idx, y, sd)]))
        uu = attn_unc(train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True), states, te_idx, device)
        rows["uniform"].append(results.prr(yte, uu)); pred_uni.append(uu)
        au = attn_unc(train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T), states, te_idx, device)
        rows["attention"].append(results.prr(yte, au)); pred_attn.append(au)
    rows = {m: float(np.mean(v)) for m, v in rows.items()}
    avg_attn, avg_mean, avg_uni = (np.mean(p, axis=0) for p in (pred_attn, pred_mean, pred_uni))
    ya = np.array(yte)
    # CLEAN aggregation comparison (same head) = attention vs uniform: this is the honest OOD claim.
    # attention vs mean-pool+MLP is HEAD-CONFOUNDED OOD (the MLP head overfits off-distribution), so
    # it is reported only as a caveat, never as the aggregation verdict.
    v_uniform = paired_bootstrap(ya, avg_attn, avg_uni)      # HEADLINE: isolates aggregation
    v_meanmlp = paired_bootstrap(ya, avg_attn, avg_mean)     # confounded by head OOD
    return rows, best_T, v_uniform, v_meanmlp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"device {device} | seeds {seeds} | OLD prompt regime (default namespace) | Light OOD", flush=True)

    data = {}
    for d in LIGHT:
        loaded = load_per_token(args.model, d, args.layer, "correctness")
        if loaded is None:
            sys.exit(f"{d}: no per-token cache -- cannot run Light OOD")
        states, split, y, layer, records = loaded
        if not np.isfinite(y).all():
            sys.exit(f"{d}: non-finite 'correctness' labels")
        data[d] = (states, split, y, records)
        print(f"  loaded {d}: {len(states)} rows", flush=True)

    out_rows = []
    for setting, X, train_dss in light_cells():
        st, rec, y, tr, te = assemble(data, train_dss, X)
        print(f"\n[{setting:9s}] eval={X:10s} train={'+'.join(train_dss):22s} "
              f"(train {len(tr)}, test {len(te)}) ...", flush=True)
        rows, best_T, v_uni, v_mean = run_cell(st, rec, y, tr, te, seeds, device, tok)
        for meth, prr in rows.items():
            print(f"    {meth:16s} {prr:+.3f}", flush=True)
        (mu, lou, hiu, pu, sigu) = v_uni
        (m, lo, hi, p, sig) = v_mean
        print(f"    HEADLINE attn-UNIFORM (clean, same head) {mu:+.4f}  95%CI [{lou:+.4f},{hiu:+.4f}]  "
              f"-> {'SIGNIFICANT' if sigu else 'not sig'}  (T*={best_T})", flush=True)
        print(f"    (caveat) attn-meanMLP (HEAD-confounded OOD) {m:+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]  "
              f"-> {'SIGNIFICANT' if sig else 'not sig'}", flush=True)
        # HARD ID-diagonal gate: an ID cell must reproduce the ID aggregation table.
        if setting == "ID":
            dm = abs(rows["mean-pool+MLP"] - ID_ANCHOR[X]["mean"])
            da = abs(rows["attention"] - ID_ANCHOR[X]["attn"])
            assert dm < GATE_TOL and da < GATE_TOL, (
                f"ID-GATE FAIL {X}: mean {rows['mean-pool+MLP']:.3f} vs anchor "
                f"{ID_ANCHOR[X]['mean']} (|d|={dm:.3f}); attn {rows['attention']:.3f} vs "
                f"{ID_ANCHOR[X]['attn']} (|d|={da:.3f}) -- cross-dataset wiring is wrong, aborting")
            print(f"    [ID-GATE OK] reproduces the ID table within {GATE_TOL}", flush=True)
        for meth, prr in rows.items():
            out_rows.append({"setting": setting, "eval": X, "train": "+".join(train_dss),
                             "method": meth, "prr": round(prr, 4), "attn_T": best_T})
        out_rows.append({"setting": setting, "eval": X, "train": "+".join(train_dss),
                         "method": "VERDICT:attn_vs_uniform_CLEAN", "prr": round(mu, 4),
                         "ci_lo": round(lou, 4), "ci_hi": round(hiu, 4), "significant": sigu})
        out_rows.append({"setting": setting, "eval": X, "train": "+".join(train_dss),
                         "method": "VERDICT:attn_vs_meanMLP_headconfound", "prr": round(m, 4),
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "significant": sig})

    out = Path(args.out) if args.out else (ROOT / "results" /
          f"aggregation_ood_light__{args.model.replace('/', '_')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    cols = ["setting", "eval", "train", "method", "prr", "attn_T", "ci_lo", "ci_hi", "significant"]
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
