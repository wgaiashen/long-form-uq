"""S2 (the idea 4) — ENSEMBLE a model-side score (MSP / wMSP) with a probe (attention pooler / SAPLMA).

Design note: "combine it with saplma / attention probes, as they would probably ensemble really well (they both learn
in different ways)." NO NEW TRAINING beyond the components' own: the ensemble is a LABEL-FREE combination at
scoring time (rank-average or z-average) of two per-example uncertainty vectors on the SAME test rows.

Motivation after the P0 correction: on long-form OOD the attention pooler BEATS mean-pool but does NOT reliably
beat the free MSP FLOOR. If the probe and a floor-like score fail on DIFFERENT examples, combining them is the
direct route to beating the floor. We therefore also report the per-cell rank-correlation of the two component
uncertainties -- the ensemble should help most where they are LEAST correlated (checked, not assumed).

Reuses the §C.3 harness EXACTLY (probedriftlong build_rows / cells_long / per-example recipes / paired_bootstrap),
so the component numbers match the ladder. Long-form OOD rungs first.

    python scripts/checks/ensemble_ladder.py --evals pubmed_qa,xsum,cnn_dailymail,med_quad,samsum,expertqa
    python scripts/checks/ensemble_ladder.py --skip-wmsp        # MSP-only ensembles (fast first read)

CPU job -> qsub, NOT the login node (it trains the attention pooler + wMSP per cell).
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, msp, results                                          # noqa: E402
from luq import weighted_msp                                                 # noqa: E402
from luq.features import sar                                                 # noqa: E402
from aggregation_table import attn_unc, conf_meanpool, paired_bootstrap, load_per_token  # noqa: E402
from attn_pool import train_attn, select_temperature                        # noqa: E402
from xl_rungs import build_rows, eval_split, label_of                        # noqa: E402
import probedriftlong as pdl                                                 # noqa: E402
from transformers import AutoTokenizer                                       # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
# long-form evals (the goal); the OOD-long rungs are where the floor is weak and the ensemble matters
DEFAULT_EVALS = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa"]


def rankavg(*us):
    """rank-average of uncertainty vectors (higher = more uncertain). PRR is rank-based, so this is well-defined."""
    from scipy.stats import rankdata
    return np.mean([rankdata(u) for u in us], axis=0)


def zavg(*us):
    def z(u):
        s = np.std(u)
        return (u - np.mean(u)) / s if s > 0 else u * 0.0
    return np.mean([z(u) for u in us], axis=0)


def spearman(a, b):
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--evals", default=",".join(DEFAULT_EVALS))
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--rungs", default="", help="comma rung filter (base names, e.g. DiffTask,LOO); '' = all OOD-long + ID")
    ap.add_argument("--skip-wmsp", action="store_true", help="MSP-only ensembles (skip the wMSP training bottleneck)")
    ap.add_argument("--probes", default="attention,saplma",
                    help="probe-side components. 'saplma' alone = fast RCS-CPU first pass (no pooler temperature "
                         "sweep); add 'attention' for the pooler ensemble (slow on CPU -> DoC GPU).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    evals = args.evals.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    want_rungs = set(r for r in args.rungs.split(",") if r)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | seeds {seeds} | evals {evals} | wmsp={'off' if args.skip_wmsp else 'on'}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok -> skip", flush=True); continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: unlabelled -> skip", flush=True); continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]
        PT[d] = (states, split, y, records)   # (wMSP uses plain wmsp_norm now — no segment ids needed)
        print(f"  {d}: {len(states)} rows (label={label_of(d)})", flush=True)
    sources = set(PT)

    # model-side vs probe-side components; ensembles = one of each, combined label-free
    MODEL_SIDE = ["floor_min"] + ([] if args.skip_wmsp else ["wmsp"])
    PROBE_SIDE = [p for p in args.probes.split(",") if p]
    use_attn = "attention" in PROBE_SIDE
    ENS = []                          # (name, comp_a, comp_b, mode)
    for ms in MODEL_SIDE:
        for ps in PROBE_SIDE:
            ENS.append((f"rankavg_{ms}+{ps}", ms, ps, "rank"))
            ENS.append((f"zavg_{ms}+{ps}", ms, ps, "z"))
    # {wMSP, MSP} -- model-side x model-side (the actual suggestion; the MODEL_SIDE x PROBE_SIDE loop above
    # never pairs two model-side scores, so this ensemble had never been formed). Both components are computed
    # below (v["wmsp"], v["floor_min"]).
    if not args.skip_wmsp:
        ENS.append(("rankavg_wmsp+floor_min", "wmsp", "floor_min", "rank"))
        ENS.append(("zavg_wmsp+floor_min", "wmsp", "floor_min", "z"))
    base_methods = ["floor_sum", "floor_ppl", "floor_min"] + PROBE_SIDE + ([] if args.skip_wmsp else ["wmsp"])
    all_methods = base_methods + [e[0] for e in ENS]

    out_rows = []
    for rung, X, spec in pdl.cells_long(sources, evals):
        base_rung = rung.replace("-long", "")
        if X not in PT or (want_rungs and base_rung not in want_rungs and rung != "ID"):
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        per = {m: [] for m in all_methods}; acc = {m: [] for m in all_methods}
        corr = {}                    # component rank-correlations (seed-averaged inputs)
        yte_ref = None
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows); tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float); yte_ref = yte
            states = [PT[d][0][i] for d, i in allrows]; records = [PT[d][3][i] for d, i in allrows]
            v = {}
            v["floor_sum"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "sum") for i in te_idx])
            v["floor_ppl"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity") for i in te_idx])
            v["floor_min"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "min") for i in te_idx])
            if "saplma" in PROBE_SIDE:
                Xmean = np.stack([s.mean(axis=0) for s in states])
                v["saplma"] = 1.0 - conf_meanpool(Xmean, tr_idx, te_idx, y, sd)
            if use_attn:
                best_T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
                v["attention"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T),
                                                     states, te_idx, device), float)
            if not args.skip_wmsp:
                # BUGFIX (2026-07-28): plain wmsp_norm (the "wMSP"), NOT the segmented variant. The earlier
                # `segment_ids=seg_cell` made this wmsp_seg_flat, which was broken (pubmed ID 0.023 vs
                # wmsp_norm's 0.537) and produced a spurious −0.42 ensemble. weight_mode default is normalised.
                v["wmsp"] = np.asarray(weighted_msp.weighted_msp_unc(states, records, y, tr_idx, te_idx, device,
                                       weight_mode="normalised", length_normalise=True, seed=sd), float)
            # ensembles (label-free combination of the two component vectors)
            for name, a, b, mode in ENS:
                v[name] = rankavg(v[a], v[b]) if mode == "rank" else zavg(v[a], v[b])
            for m in all_methods:
                per[m].append(results.prr(yte, v[m])); acc[m].append(v[m])
        if yte_ref is None:
            continue
        stats = {m: (float(np.mean(per[m])), float(np.std(per[m]))) for m in per if per[m]}
        avg = {m: np.mean(np.stack(acc[m]), 0) for m in acc if acc[m]}
        # pre-registered bar = msp_min; dual-report the strongest free floor
        FLOORS = ["floor_sum", "floor_ppl", "floor_min"]
        fair_name = "floor_min" if "floor_min" in stats else max(FLOORS, key=lambda f: stats[f][0])
        strongest = max(FLOORS, key=lambda f: stats[f][0])
        # component correlations (the "helps where uncorrelated" check)
        for ms in MODEL_SIDE:
            for ps in PROBE_SIDE:
                corr[f"{ms}~{ps}"] = spearman(avg[ms], avg[ps])
        if not args.skip_wmsp and "wmsp" in avg and "floor_min" in avg:
            corr["wmsp~floor_min"] = spearman(avg["wmsp"], avg["floor_min"])
        _real = {}
        for _d, _i in train_rows:
            _real[_d] = _real.get(_d, 0) + 1
        srcs = "+".join(f"{d}:{_real.get(d, 0)}" for d in dict.fromkeys(d for d, _c in spec))
        dual = "" if strongest == fair_name else f"  (strongest free={strongest} {stats[strongest][0]:+.3f})"
        print(f"\n[{rung:14s}] eval={X} ({label_of(X)}) train={srcs}  bar={fair_name} {stats[fair_name][0]:+.3f}{dual}", flush=True)
        print("    corr(model~probe): " + "  ".join(f"{k} {corr[k]:+.2f}" for k in corr), flush=True)
        for m in all_methods:
            if m in stats:
                tag = "  <== ENSEMBLE" if m in [e[0] for e in ENS] else ""
                print(f"    {m:22s} {stats[m][0]:+.3f} +/- {stats[m][1]:.3f}{tag}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": m,
                                 "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                                 "n_seeds": len(per[m]), "bar": fair_name, "bar_prr": round(stats[fair_name][0], 4)})
        # verdicts: each ensemble vs the floor bar, and vs its best single component
        for name, a, b, _mode in ENS:
            best_comp = a if stats[a][0] >= stats[b][0] else b
            for vk, other in [(f"{name}_vs_floor", fair_name), (f"{name}_vs_bestcomp", best_comp)]:
                mg, lo, hi, p, sig = paired_bootstrap(yte_ref, avg[name], avg[other])
                print(f"    [verdict] {vk:34s} margin {mg:+.3f} CI[{lo:+.3f},{hi:+.3f}] {'SIG' if sig else 'ns'}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"VERDICT:{vk}",
                                 "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                 "boot_p": round(p, 4), "significant": bool(sig), "n_seeds": len(seeds)})

    out = Path(args.out) if args.out else (ROOT / "results" / f"ensemble_ladder__{cache._slug(MODEL)}.csv")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "train", "method", "prr_mean", "prr_std", "n_seeds",
                                           "bar", "bar_prr", "ci_lo", "ci_hi", "boot_p", "significant"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
