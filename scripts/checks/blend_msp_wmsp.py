#!/usr/bin/env python
"""W7b -- per-instance length blend of msp_min <-> wMSP-shrink@1.5 (the length-blend registration; expectation NULL).

u_i = w_i*z(msp_min)_i + (1-w_i)*z(wMSP@1.5)_i, w_i = exp(-len_i/L), ONE LODO-selected L.
Controls: shuffled-length (does length carry anything?) and constant w=0.5 (is it just
decorrelation?). Join gate: the wMSP@1.5 endpoint must reproduce §15.2b to seed noise.
"""
import argparse, csv as _csv, os, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "checks"))
from luq import msp, results, weighted_msp
from luq.weighting import shrink_to_uniform
from aggregation_table import load_per_token
from xl_rungs import eval_split, label_of, build_rows
import probedriftlong as pdl

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LGRID = [8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]


def z(v):
    s = np.std(v)
    return (v - np.mean(v)) / s if s > 0 else v * 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="pubmed_qa")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    evals = args.evals.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    carve = os.environ.get("LUQ_CARVE", "legacy")
    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        got = load_per_token(MODEL, d, args.layer, label_of(d))
        if got is None:
            print(f"  {d}: no pertok -> SKIPPED LOUDLY")
            continue
        st, sp, y, _, rc = got
        f = np.isfinite(y)
        if not f.all():
            k = np.where(f)[0]
            st = [st[i] for i in k]; rc = [rc[i] for i in k]; sp = sp[k]; y = y[k]
        PT[d] = (st, sp, y, rc)
    rows = []
    rng0 = np.random.RandomState(7)
    for rung, X, spec in pdl.cells_long(set(PT), evals):
        if rung == "ID" or X not in PT:
            continue
        per = {("blend", L): [] for L in LGRID}
        per.update({("shufL", L): [] for L in LGRID})
        per[("const", 0.5)] = []
        ends = {"msp_min": [], "wmsp15": []}
        for sd in seeds:
            tr, te = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not tr or not te:
                continue
            n_tr = len(tr)
            tri = list(range(n_tr)); tei = list(range(n_tr, n_tr + len(te)))
            allr = tr + te
            y = np.array([PT[d][2][i] for d, i in allr], float)
            yte = np.array([y[i] for i in tei])
            states = [PT[d][0][i] for d, i in allr]
            records = [PT[d][3][i] for d, i in allr]
            m = weighted_msp.train_weighted_msp(states, records, y, tri, device,
                                                weight_mode="normalised", length_normalise=True,
                                                seed=sd, reg=shrink_to_uniform, reg_lambda=1.5)
            uw = np.asarray(weighted_msp.predict_weighted_msp(m, states, records, tei, device,
                            weight_mode="normalised", length_normalise=True), float)
            um = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "min") for i in tei])
            ln = np.array([len(records[i]["token_logprobs"]) for i in tei], float)
            ends["msp_min"].append(results.prr(yte, um))
            ends["wmsp15"].append(results.prr(yte, uw))
            zm, zw = z(um), z(uw)
            lsh = rng0.permutation(ln)
            for L in LGRID:
                w = np.exp(-ln / L)
                per[("blend", L)].append(results.prr(yte, w * zm + (1 - w) * zw))
                ws = np.exp(-lsh / L)
                per[("shufL", L)].append(results.prr(yte, ws * zm + (1 - ws) * zw))
            per[("const", 0.5)].append(results.prr(yte, 0.5 * zm + 0.5 * zw))
        if not ends["msp_min"]:
            continue
        print(f"[{rung:16s} {X:14s}] msp_min={np.mean(ends['msp_min']):+.4f} "
              f"wMSP@1.5={np.mean(ends['wmsp15']):+.4f} const0.5={np.mean(per[('const', 0.5)]):+.4f}")
        print("   L:     " + "  ".join(f"{int(L)}:{np.mean(per[('blend', L)]):+.3f}" for L in LGRID))
        print("   shufL: " + "  ".join(f"{int(L)}:{np.mean(per[('shufL', L)]):+.3f}" for L in LGRID),
              flush=True)
        for (k, Lv), vals in per.items():
            rows.append((rung, X, k, Lv, f"{np.mean(vals):.4f}", len(vals), carve))
        for k, vals in ends.items():
            rows.append((rung, X, "endpoint", k, f"{np.mean(vals):.4f}", len(vals), carve))
    ev = evals[0] if len(evals) == 1 else "multi"
    outp = Path(args.out) if args.out else ROOT / "results" / f"blend_msp_wmsp_{ev}__meta-llama_Meta-Llama-3.1-8B.csv"
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["rung", "eval", "kind", "param", "prr", "n_seeds", "carve"])
        for r in rows:
            w.writerow(r)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
