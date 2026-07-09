"""STEP 3: the ONLY valid Orgad comparison -- masked vs unmasked weighted-MSP on the SAME located subset.

Because 'located' tracks the correctness label (STEP 2: located|correct ~99%, located|incorrect ~0%),
comparing the ±mask arms over ALL rows scores correct vs incorrect rows with different estimators. The
only clean test restricts BOTH arms to the identical LOCATED rows. It is low-powered and the subset
skews heavily correct (we report the label variance to show it).

Also a MASK ALIGNMENT UNIT TEST: for a few located records, decode the masked (mask==1) tokens and check
they contain the gold answer -- validating the row->index -1 shift lines the mask up with answer_states.

    python scripts/checks/orgad_located_subset.py --seeds 1,2,3
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

from luq import cache, results, weighted_msp  # noqa: E402
from luq.features import orgad  # noqa: E402
from weighted_msp_blondel import cells, CANDIDATE_SOURCES, LAB, LAYER  # noqa: E402
from aggregation_table import load_per_token, paired_bootstrap  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)

    PT, MASKS, LOC = {}, {}, {}
    for d in CANDIDATE_SOURCES:
        loaded = load_per_token(MODEL, d, LAYER, LAB)
        if loaded is None or np.isnan(loaded[2]).any():
            continue
        states, split, y, _, records = loaded
        masks, loc = [], np.zeros(len(records), bool)
        for i, r in enumerate(records):
            rows, found = orgad.locate_answer_rows(tok, r["prompt_token_ids"], r["gen_token_ids"], r["target"])
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
        print(f"  {d}: {len(states)} rows, located {int(loc.sum())} ({100*loc.mean():.0f}%)", flush=True)

    # ---- MASK ALIGNMENT UNIT TEST ----
    print("\n=== mask alignment unit test (masked tokens should decode to the gold answer) ===", flush=True)
    n_pass = n_tot = 0
    for d in ["sciq", "trivia_qa"]:
        if d not in PT:
            continue
        _, _, _, records = PT[d]
        checked = 0
        for i in np.where(LOC[d])[0]:
            r = records[i]
            masked_ids = [t for t, mm in zip(r["gen_token_ids"], MASKS[d][i]) if mm > 0]
            dec = tok.decode(masked_ids, skip_special_tokens=True).lower()
            gold = r["target"] if isinstance(r["target"], (list, tuple)) else [r["target"]]
            ok = any(str(g).lower().strip() in dec or dec.strip() in str(g).lower() for g in gold if str(g).strip())
            n_pass += ok; n_tot += 1; checked += 1
            if checked >= 30:
                break
    print(f"  masked-tokens-decode-to-gold: {n_pass}/{n_tot} pass "
          f"({'ALIGNMENT OK' if n_tot and n_pass/n_tot > 0.8 else 'ALIGNMENT SUSPECT'})", flush=True)

    # ---- located-subset PRR (masked vs unmasked), ID + LOO ----
    def unc(spec, X, use_mask, restrict_located):
        vals, uacc, yte_ref = [], [], None
        for sd in seeds:
            tr_src = [(d, i) for d, cap in spec
                      for i in np.where((PT[d][1] == "train") & (LOC[d] if restrict_located else True))[0][:cap]]
            te = np.where((PT[X][1] == "test") & (LOC[X] if restrict_located else np.ones(len(LOC[X]), bool)))[0]
            test_rows = [(X, i) for i in te]
            if not tr_src or not test_rows:
                return None, None, None
            n_tr = len(tr_src)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            allrows = tr_src + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float); yte_ref = yte
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]
            masks = [MASKS[d][i] for d, i in allrows] if use_mask else None
            u = np.asarray(weighted_msp.weighted_msp_unc(states, records, y, tr_idx, te_idx, device,
                           weight_mode="normalised", length_normalise=True, seed=sd, loss="pairwise",
                           masks=masks), float)
            vals.append(results.prr(yte, u)); uacc.append(u)
        return float(np.mean(vals)), np.mean(np.stack(uacc), axis=0), yte_ref

    print("\n=== located-subset PRR (masked vs unmasked; same located rows both arms) ===", flush=True)
    print(f"{'rung':6s} {'eval':10s} {'n_te':>5s} {'corr_std':>8s} {'unmasked':>9s} {'+mask':>8s} "
          f"{'delta':>8s}  verdict", flush=True)
    rows = []
    for rung, X, spec in cells(set(PT)):
        if rung not in ("ID", "LOO") or X not in ("sciq", "trivia_qa"):
            continue
        base, ub, yref = unc(spec, X, False, True)
        msk, um, _ = unc(spec, X, True, True)
        if base is None or msk is None:
            continue
        corr_std = float(np.std(yref))
        mg, lo, hi, p, sig = paired_bootstrap(yref, um, ub)
        note = "DEGENERATE (corr_std<0.15; PRR unstable)" if corr_std < 0.15 else \
               (f"CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} {'SIG' if sig else 'ns'}")
        print(f"{rung:6s} {X:10s} {len(yref):5d} {corr_std:8.3f} {base:+9.3f} {msk:+8.3f} {msk-base:+8.3f}  {note}",
              flush=True)
        rows.append({"rung": rung, "eval": X, "n_test": len(yref), "corr_std": round(corr_std, 4),
                     "unmasked": round(base, 4), "masked": round(msk, 4), "delta": round(msk - base, 4),
                     "boot_p": round(p, 4), "significant": sig})
    out = ROOT / "results" / f"orgad_located_subset__{cache._slug(MODEL)}.csv"
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "n_test", "corr_std", "unmasked", "masked",
                                           "delta", "boot_p", "significant"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}\nREAD: if corr_std is tiny, the located subset is near-all-correct -> PRR has "
          "almost nothing to rank -> the ±mask comparison is UNINTERPRETABLE even within the subset.", flush=True)


if __name__ == "__main__":
    main()
