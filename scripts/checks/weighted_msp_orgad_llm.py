"""Orgad the RIGHT way: leak re-check + the (now valid) +/-mask weighted-MSP comparison.

Uses the LLM-extracted MODEL-OWN answer span (cache/orgad_llm, from 01o_orgad_llm_extract.py) instead of
the gold-string-match span. First we RE-CHECK the leak: located rate by correct vs incorrect, and judge
correctness located vs unlocated. If 'located' no longer tracks correctness (unlike the gold version:
located|correct 99% vs |incorrect 0.2%), the +/-mask comparison is finally interpretable. Then we run it:
weighted-MSP masked vs unmasked, ID sciq/trivia, on ALL rows and on the LOCATED subset, paired bootstrap.

    python scripts/checks/weighted_msp_orgad_llm.py --seeds 1,2,3
"""
import argparse
import csv as _csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from luq import cache, results, weighted_msp  # noqa: E402
from luq.features import orgad_llm  # noqa: E402
from aggregation_table import load_per_token, paired_bootstrap  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
DATASETS = ["sciq", "trivia_qa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)

    PT, MASKS, LOC = {}, {}, {}
    print(f"{'dataset':10s} {'n':>5s} {'loc%':>6s} {'loc|corr%':>9s} {'loc|incorr%':>11s} "
          f"{'judge|loc':>9s} {'judge|unloc':>11s}", flush=True)
    for d in DATASETS:
        states, split, y, _, records = load_per_token(MODEL, d, 15, "correctness")
        ex_path = ROOT / "cache" / "orgad_llm" / f"{cache._slug(MODEL)}__{d}__ID.json"
        if not ex_path.exists():
            print(f"  {d}: no LLM extraction cache -> run 01o_orgad_llm_extract.py first", flush=True)
            continue
        extracted = json.loads(ex_path.read_text())
        masks, loc = [], np.zeros(len(records), bool)
        for i, r in enumerate(records):
            rows, found = orgad_llm.locate_extracted_rows(tok, r["gen_token_ids"], extracted.get(str(r["idx"]), ""))
            m = np.zeros(len(r["gen_token_ids"]), np.float32)
            if found:
                for row in rows:
                    if 0 <= row - 1 < len(m):
                        m[row - 1] = 1.0
            loc[i] = found and m.sum() > 0
            if m.sum() == 0:
                m[:] = 1.0
            masks.append(m)
        PT[d], MASKS[d], LOC[d] = (states, split, y, records), masks, loc
        # leak re-check (uses correctness_strmatch if present else judge>=0.5)
        strm = [r.get("correctness_strmatch") for r in records]
        corr = np.array([1 if (isinstance(x, (int, float)) and x >= 0.5) else 0 for x in strm]) \
            if any(isinstance(x, (int, float)) for x in strm) else (y >= 0.5).astype(int)
        print(f"{d:10s} {len(records):5d} {100*loc.mean():6.1f} "
              f"{100*loc[corr==1].mean():9.1f} {100*loc[corr==0].mean():11.1f} "
              f"{np.nanmean(y[loc]):9.3f} {np.nanmean(y[~loc]):11.3f}", flush=True)

    print("\nREAD (leak fix): loc|corr should now be CLOSE to loc|incorr (the gold version was 99 vs 0). "
          "If so, the comparison below is interpretable.\n", flush=True)

    def prr_arms(d, restrict_located):
        states, split, y, records = PT[d]
        tr = np.where(split == "train")[0]
        te = np.where(split == "test")[0]
        if restrict_located:
            tr = tr[LOC[d][tr]]; te = te[LOC[d][te]]
        yte = y[te]
        out = {}
        for tag, use in [("unmasked", False), ("masked", True)]:
            allrows = list(tr) + list(te)
            n_tr = len(tr)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(te)))
            st = [states[i] for i in allrows]; rc = [records[i] for i in allrows]
            yy = np.array([y[i] for i in allrows], float)
            mk = [MASKS[d][i] for i in allrows] if use else None
            uacc = []
            for sd in seeds:
                u = np.asarray(weighted_msp.weighted_msp_unc(st, rc, yy, tr_idx, te_idx, device,
                              weight_mode="normalised", length_normalise=True, seed=sd, loss="pairwise",
                              masks=mk), float)
                uacc.append(u)
            out[tag] = np.mean(np.stack(uacc), axis=0)
        return yte, out

    print(f"{'dataset':10s} {'subset':10s} {'n_te':>5s} {'corr_std':>8s} {'unmasked':>9s} {'+mask':>8s} "
          f"{'delta':>8s}  verdict", flush=True)
    rows = []
    for d in [x for x in DATASETS if x in PT]:
        for label, restr in [("all", False), ("located", True)]:
            yte, arms = prr_arms(d, restr)
            base = results.prr(yte, arms["unmasked"]); msk = results.prr(yte, arms["masked"])
            cs = float(np.std(yte))
            mg, lo, hi, p, sig = paired_bootstrap(yte, arms["masked"], arms["unmasked"])
            note = "corr_std<0.15 low-power" if cs < 0.15 else (f"CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} {'SIG' if sig else 'ns'}")
            print(f"{d:10s} {label:10s} {len(yte):5d} {cs:8.3f} {base:+9.3f} {msk:+8.3f} {msk-base:+8.3f}  {note}",
                  flush=True)
            rows.append({"dataset": d, "subset": label, "n_test": len(yte), "corr_std": round(cs, 4),
                         "unmasked": round(base, 4), "masked": round(msk, 4), "delta": round(msk - base, 4),
                         "boot_p": round(p, 4), "significant": sig})
    out = ROOT / "results" / f"weighted_msp_orgad_llm__{cache._slug(MODEL)}.csv"
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
