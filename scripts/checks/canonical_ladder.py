"""The CANONICAL 5-rung ladder for the slide deck: ALL nine methods on the SAME samsum-updated pool.

WHY THIS EXISTS
---------------
Two verified drivers already cover pieces of the ladder, but on DIFFERENT pools:
  * contribution_ladder.py  -> uniform, attention, weighted-MSP(norm/unc), MSP floor, across the full
                               samsum-updated ladder (ID/SameTask/LOO/OneDatasetDiffTask/DiffTask).
  * aggregation_table.py    -> the SAPLMA aggregators (mean-pool+MLP, last-token, per-sentence,
                               per-token) but ID ONLY.
  * ood_onegrid.py          -> those SAPLMA aggregators at OOD, but on the OLD xsum-only pool
                               (pre-samsum) and only ID/LOO/DiffTask (no SameTask/OneDatasetDiffTask).
So the SAPLMA-aggregator OOD cells on the CURRENT (samsum) pool do not exist anywhere. This driver
computes every method on one pool so slides 8/9 read from a single, consistent table.

It REUSES the verified building blocks -- it does not reimplement any estimator:
  * cell/spec/seed-sampling logic          <- contribution_ladder (cells, sampled_train_idx)
  * SAPLMA aggregators + build_arrays       <- aggregation_table (conf_*, build_arrays, attn_unc, prr_from_conf)
  * poolers                                 <- attn_pool (train_attn, select_temperature)
  * weighted-MSP                            <- weighted_msp.weighted_msp_unc

Rows: msp_sum (floor), mean-pool+MLP, last-token, per-sentence(mean), per-token(mean),
      uniform(frozen-q), attention, weighted_msp_norm (pairwise), weighted_msp_blondel.
Cols (rungs): ID, SameTask, LOO, OneDatasetDiffTask, DiffTask.  Evals: sciq, trivia_qa, pubmed_qa.
Each cell = 3-seed mean +/- std, and beats_floor = (mean PRR > that cell's msp_sum PRR).

ID-diagonal gates (same discipline as the parents): mean-pool ID must reproduce the cached SAPLMA
L15 PRR, and the poolers must reproduce the aggregation-table ID anchors.

    python scripts/checks/canonical_ladder.py --seeds 1,2,3
    python scripts/checks/canonical_ladder.py --sources sciq,trivia_qa,pubmed_qa,med_quad --seeds 1  # smoke
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

from luq import cache, msp, probe, results, weighted_msp  # noqa: E402
from luq.config import Config  # noqa: E402
from attn_pool import load_per_token, train_attn, select_temperature  # noqa: E402
from aggregation_table import (  # noqa: E402
    conf_meanpool, conf_lasttoken, conf_persentence, conf_pertoken,
    build_arrays, attn_unc, prr_from_conf)
import contribution_ladder as cl  # noqa: E402
from xl_rungs import cells as xl_cells, build_rows as xl_build_rows, KEYSTONES  # noqa: E402
# (rung builder + XL-aware train/test row assembly; contribution_ladder no longer re-exports cells)

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAB = "correctness"
GATE_TOL = 0.03
# The SAPLMA aggregators, keyed by the --aggregators short name -> display label. mean-pool and
# last-token are cheap (one vector per example); per-sentence and per-token train a probe over EVERY
# sentence/token of the pooled train set, which is very slow on long-form -- hence opt-in.
SAPLMA_AGG = {"meanpool": "mean-pool+MLP", "lasttoken": "last-token",
              "persentence": "per-sentence(mean)", "pertoken": "per-token(mean)"}
CONTRIB = ["uniform(frozen-q)", "attention", "weighted_msp_norm", "weighted_msp_blondel"]


def id_gate_meanpool(model_name, dataset, layer, y, tr_idx, te_idx, yte, prr_meanpool):
    """mean-pool (seed 1) must reproduce the cached SAPLMA L15 PRR -- the same cache-anchored gate
    aggregation_table uses. Only meaningful on the ID rung (single-dataset train)."""
    feat = cache.load_features(Config(model_name=model_name, dataset=dataset, ood_setting="ID").cache_dir,
                               cache.run_key(model_name, dataset, "ID"), "saplma")
    prr_feat = prr_from_conf(yte, probe.train_probe_mlp(feat[tr_idx, layer, :], y[tr_idx], seed=1)
                             .p_correct(feat[te_idx, layer, :]))
    ok = abs(prr_meanpool - prr_feat) < GATE_TOL
    return ok, prr_feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--sources", default=",".join(cl.CANDIDATE_SOURCES))
    ap.add_argument("--evals", default="",
                    help="override the eval datasets (comma list). Default = the 3 QA sets from "
                         "contribution_ladder; pass e.g. 'xsum' to add summarisation as an OOD eval.")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--length-normalise", default="yes", choices=["yes", "no"])
    ap.add_argument("--aggregators", default="meanpool,lasttoken",
                    help="which SAPLMA aggregators to compute (comma list of "
                         "meanpool,lasttoken,persentence,pertoken). per-token/per-sentence are SLOW.")
    ap.add_argument("--skip-contrib", action="store_true",
                    help="skip the poolers + weighted-MSP (uniform/attention/wMSP) and compute ONLY the "
                         "SAPLMA aggregators + MSP floor. Makes the run CPU-only (no GPU needed) -- use it "
                         "to fill just the SAPLMA-aggregator OOD cells, since the contribution rows already "
                         "exist in contribution_ladder__*.csv / weighted_msp_blondel__*.csv.")
    ap.add_argument("--out", default=str(ROOT / "results" / f"canonical_ladder__{cache._slug(MODEL)}.csv"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ln = args.length_normalise == "yes"
    agg = [a for a in args.aggregators.split(",") if a in SAPLMA_AGG]
    if args.evals:                       # override contribution_ladder's default EVALS (adds xsum, etc.)
        cl.EVALS = [e for e in args.evals.split(",") if e]
        print(f"eval override -> {cl.EVALS}", flush=True)
    # Table row order: floor, the selected SAPLMA aggregators (canonical order), then the contribution.
    methods_order = (["msp_sum"] + [SAPLMA_AGG[k] for k in ("meanpool", "lasttoken", "persentence", "pertoken")
                                    if k in agg] + ([] if args.skip_contrib else CONTRIB))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    have_bl = weighted_msp._HAVE_TORCHSORT
    print(f"device {device} | seeds {seeds} | length_normalise={ln} | torchsort={have_bl}", flush=True)

    # Per-token states + records for every source whose cache exists and is labelled (tolerant).
    PT = {}
    for d in args.sources.split(","):
        loaded = load_per_token(MODEL, d, args.layer, LAB)
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip", flush=True)
            continue
        states, split, y, _, records = loaded
        if np.isnan(y).any():
            print(f"  {d}: NaN labels -> skip", flush=True)
            continue
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} rows", flush=True)
    sources = set(PT)

    out_rows = []
    for rung, X, spec in xl_cells(sources, cl.EVALS):
        if X not in PT:
            continue
        per_method = {m: [] for m in methods_order}
        for sd in seeds:
            # XL-aware row assembly (same as contribution_ladder): eval_split gives the eval target's
            # FIXED test set -- baked split for keystones, deterministic seed=0 carve for the split-less XL
            # sets (med_quad/samsum all-train, ExpertQA all-test). The old np.where(split=="test") was empty
            # for med_quad/samsum, silently skipping them; build_rows fixes that (keystone numbers unchanged).
            train_rows, test_rows = xl_build_rows(X, spec, PT, sd, cl.sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], dtype=float)
            yte = np.array([y[i] for i in te_idx], dtype=float)
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]

            # SAPLMA aggregators (reuse aggregation_table's exact implementations). mean-pool and
            # last-token are cheap per-example vectors; per-sentence/per-token are opt-in (slow).
            if "meanpool" in agg:
                Xmean = np.stack([s.mean(axis=0) for s in states])
                per_method["mean-pool+MLP"].append(prr_from_conf(yte, conf_meanpool(Xmean, tr_idx, te_idx, y, sd)))
            if "lasttoken" in agg:
                Xlast = np.stack([s[-1] for s in states])
                per_method["last-token"].append(prr_from_conf(yte, conf_lasttoken(Xlast, tr_idx, te_idx, y, sd)))
            if "persentence" in agg:
                _, _, sent_vecs = build_arrays(states, records, tok)   # only build sentence vecs when needed
                per_method["per-sentence(mean)"].append(prr_from_conf(yte, conf_persentence(sent_vecs, tr_idx, te_idx, y, sd)))
            if "pertoken" in agg:
                per_method["per-token(mean)"].append(prr_from_conf(yte, conf_pertoken(states, tr_idx, te_idx, y, sd)))

            # poolers + weighted-MSP (the contribution rows) -- skipped with --skip-contrib because they
            # already live in contribution_ladder__*.csv / weighted_msp_blondel__*.csv; skipping makes the
            # run CPU-only (only the SAPLMA aggregators + floor remain, and those use a CPU MLP).
            if not args.skip_contrib:
                best_T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
                per_method["uniform(frozen-q)"].append(results.prr(yte, attn_unc(
                    train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True), states, te_idx, device)))
                per_method["attention"].append(results.prr(yte, attn_unc(
                    train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T), states, te_idx, device)))
                per_method["weighted_msp_norm"].append(results.prr(yte, np.asarray(weighted_msp.weighted_msp_unc(
                    states, records, y, tr_idx, te_idx, device, weight_mode="normalised",
                    length_normalise=ln, seed=sd, loss="pairwise"), float)))
                if have_bl:
                    per_method["weighted_msp_blondel"].append(results.prr(yte, np.asarray(weighted_msp.weighted_msp_unc(
                        states, records, y, tr_idx, te_idx, device, weight_mode="normalised",
                        length_normalise=ln, seed=sd, loss="blondel"), float)))

            # plain-MSP floor (unsupervised; identical across seeds, computed per cell for the table).
            per_method["msp_sum"].append(results.prr(yte, np.array(
                [msp.msp_uncertainty(records[i]["token_logprobs"], "sum") for i in te_idx])))
            print(f"    [{rung}/{X}] seed {sd} done", flush=True)

        stats = {m: (float(np.mean(v)), float(np.std(v))) for m, v in per_method.items() if v}
        if not stats:
            continue
        floor = stats.get("msp_sum", (float("nan"), 0.0))[0]
        srcs = "+".join(f"{d}:{c}" if c else d for d, c in spec)
        print(f"\n[{rung:18s}] eval={X}  train={srcs}", flush=True)
        for m in methods_order:
            if m in stats:
                beats = stats[m][0] > floor
                print(f"    {m:20s} {stats[m][0]:+.4f} +/- {stats[m][1]:.4f}  "
                      f"{'>' if beats else '<='}floor", flush=True)
                out_rows.append({"eval": X, "rung": rung, "train": srcs, "method": m,
                                 "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                                 "beats_floor": bool(stats[m][0] > floor), "n_seeds": len(per_method[m])})

        # ID-diagonal gates on the controlled rows.
        if rung == "ID":
            # mean-pool ID must reproduce the cached SAPLMA L15 PRR (only if mean-pool was computed).
            # Keystones only: the "saplma" feature cache the gate reloads exists for the core datasets,
            # not the XL sets (whose pooled features live in a different cache path) -- skip the gate there.
            if "mean-pool+MLP" in stats and X in KEYSTONES:
                te_rows = np.where(PT[X][1] == "test")[0]
                yX = PT[X][2]
                ok, prr_feat = id_gate_meanpool(
                    MODEL, X, args.layer, yX,
                    list(np.where(PT[X][1] == "train")[0]), list(te_rows),
                    np.array([yX[i] for i in te_rows], float),
                    stats["mean-pool+MLP"][0])
                if not ok:
                    print(f"    [ID-GATE WARN] {X} mean-pool {stats['mean-pool+MLP'][0]:.3f} vs cached "
                          f"SAPLMA {prr_feat:.3f} (|d|>={GATE_TOL})", flush=True)
                else:
                    print(f"    [ID-GATE OK] mean-pool reproduces cached SAPLMA ({prr_feat:.3f})", flush=True)
            if X in cl.ID_ANCHOR and not args.skip_contrib:
                for m, key in (("uniform(frozen-q)", "uniform"), ("attention", "attention")):
                    d = abs(stats[m][0] - cl.ID_ANCHOR[X][key])
                    flag = "OK" if d < GATE_TOL else "WARN"
                    print(f"    [ID-ANCHOR {flag}] {X}/{m} {stats[m][0]:.3f} vs {cl.ID_ANCHOR[X][key]} "
                          f"(|d|={d:.3f})", flush=True)

    out = Path(args.out)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["eval", "rung", "train", "method", "prr_mean", "prr_std",
                                           "beats_floor", "n_seeds"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
