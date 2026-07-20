"""Track A: does SAR relevance (unsupervised 'which tokens matter') help weighted-MSP, ID and OOD?

Reads the cached SAR per-token relevance (scripts/01s_sar_relevance.py) and uses it as a SOFT [0,1]
importance mask into weighted-MSP (the same `masks=` path Orgad's hard mask uses; SAR is the soft,
long-form-capable version). Per cell we report:
  * weighted-MSP (no mask)            -- the learned-weight baseline
  * weighted-MSP x SAR soft mask      -- learned weight modulated by the unsupervised SAR importance
  * TokenSAR (unsupervised scalar)    -- pure SAR, no learned weight (the faithful SAR baseline)
  * MSP floor
with a paired bootstrap for (+SAR vs no-mask). A cell runs only if every source in it has a SAR cache.

The soft mask is R~ / max(R~) (so the most-relevant token = 1, filler down-weighted) -- a soft analogue
of Orgad's 0/1 answer-span mask.

    python scripts/checks/weighted_msp_sar.py --granularity token --seeds 1,2,3
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
from weighted_msp_blondel import sampled_train_idx, LAB, LAYER  # noqa: E402
# use the 5-eval ladder (incl xsum + cnn_dailymail) so SAR is tested on the SUMMARISATION sets, where
# relevance-weighting should matter most -- the light 3-eval set skipped exactly those. All 7 candidate
# sources now have SAR caches, and this script filters cells to SAR-available sources, so this is safe.
from weighted_msp_all_variants import cells, CANDIDATES as CANDIDATE_SOURCES  # noqa: E402  (XL-aware cells)
import xl_rungs  # noqa: E402
from aggregation_table import load_per_token, paired_bootstrap  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SAR_DIR = ROOT / "cache" / "sar"


def load_sar(dataset, granularity=None):
    """Return (relevance_list, tokensar_array, granularity) for a dataset, or None if not cached.
    Auto-detects granularity when not given: short-form uses `token`, long-form uses `sentence`, so we
    try token then sentence and take whichever exists."""
    grans = [granularity] if granularity else ["token", "sentence"]
    for g in grans:
        # prefer the sharper answer-only (__noprepend) caches -- the prepend dilutes long-form relevance;
        # the DoC token-level no-prepend run cut entropy from ~0.99 to ~0.89-0.93 on the long-form sets.
        for suf in ("__noprepend", ""):
            p = SAR_DIR / f"{cache._slug(MODEL)}__{dataset}__ID__{g}{suf}.npz"
            if p.exists():
                z = np.load(p, allow_pickle=True)
                return list(z["relevance"]), np.asarray(z["tokensar"], dtype=float), f"{g}{suf}"
    return None


def soft_mask(Rn):
    """R~ -> soft [0,1] importance mask (max token = 1). Uniform R~ -> all-ones (no-op)."""
    Rn = np.asarray(Rn, dtype=np.float32)
    m = Rn.max()
    return (Rn / m).astype(np.float32) if m > 0 else np.ones(len(Rn), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--sources", default=",".join(CANDIDATE_SOURCES))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | seeds {seeds} | SAR granularity auto (token short / sentence long)", flush=True)

    PT, MASKS, TSAR = {}, {}, {}
    for d in args.sources.split(","):
        loaded = load_per_token(MODEL, d, LAYER, LAB)
        if loaded is None:
            continue
        states, split, y, _, records = loaded
        if np.isnan(y).any():
            continue
        s = load_sar(d)
        if s is None:
            print(f"  {d}: no SAR cache -> not a SAR source", flush=True)
            continue
        rel, tsar, gran = s
        if len(rel) != len(records):
            print(f"  {d}: SAR cache len {len(rel)} != records {len(records)} -> skip", flush=True)
            continue
        PT[d] = (states, split, y, records)
        MASKS[d] = [soft_mask(r) for r in rel]
        TSAR[d] = tsar
        avg_keep = float(np.mean([soft_mask(r).mean() for r in rel]))
        print(f"  {d}: {len(states)} rows | SAR loaded (granularity={gran}, avg mask keeps "
              f"{100*avg_keep:.0f}% -> {'selective' if avg_keep < 0.85 else 'near-uniform'})", flush=True)
    sources = set(PT)

    def wmsp(spec, X, use_mask):
        vals, uacc = [], []
        for sd in seeds:
            train_rows, test_rows = xl_rungs.build_rows(X, spec, PT, sd, sampled_train_idx)
            if not train_rows or not test_rows:
                return None, None
            n_tr = len(train_rows)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], dtype=float)
            yte = np.array([y[i] for i in te_idx], dtype=float)
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]
            masks = [MASKS[d][i] for d, i in allrows] if use_mask else None
            u = np.asarray(weighted_msp.weighted_msp_unc(
                states, records, y, tr_idx, te_idx, device, weight_mode="normalised",
                length_normalise=True, seed=sd, loss="pairwise", masks=masks), dtype=float)
            vals.append(results.prr(yte, u)); uacc.append(u)
        return float(np.mean(vals)), np.mean(np.stack(uacc), axis=0)

    print(f"\n{'rung':9s} {'eval':10s} {'no-mask':>9s} {'+SAR':>8s} {'delta':>8s} {'tokensar':>9s} "
          f"{'floor':>7s}  verdict", flush=True)
    rows = []
    for rung, X, spec in cells(sources):
        if X not in PT:
            continue
        base, ub = wmsp(spec, X, False)
        sar_prr, us = wmsp(spec, X, True)
        if base is None or sar_prr is None:
            continue
        _, te = xl_rungs.eval_split(PT[X][1])          # XL-aware test indices (baked core / carved XL)
        yte = PT[X][2][te]
        floor = results.prr(yte, np.asarray(
            [msp.msp_uncertainty(PT[X][3][i]["token_logprobs"], "sum") for i in te], dtype=float))
        tsar_prr = results.prr(yte, TSAR[X][te])            # pure TokenSAR scalar (unsupervised)
        mg, lo, hi, p, sig = paired_bootstrap(yte, us, ub)  # +SAR vs no-mask
        d = sar_prr - base
        print(f"{rung:9s} {X:10s} {base:+9.3f} {sar_prr:+8.3f} {d:+8.3f} {tsar_prr:+9.3f} {floor:+7.3f}  "
              f"CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} {'SIG' if sig else 'ns'}", flush=True)
        rows.append({"rung": rung, "eval": X, "no_mask": round(base, 4), "sar": round(sar_prr, 4),
                     "delta": round(d, 4), "tokensar": round(tsar_prr, 4), "floor": round(floor, 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "boot_p": round(p, 4), "significant": sig})

    out = ROOT / "results" / f"weighted_msp_sar__{cache._slug(MODEL)}.csv"
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "no_mask", "sar", "delta", "tokensar",
                                           "floor", "ci_lo", "ci_hi", "boot_p", "significant"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
