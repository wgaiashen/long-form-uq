#!/usr/bin/env python
"""Clean-vs-flagged SENSITIVITY on the two hardest OOD rungs — a diagnostic, not a result.

THE QUESTION. On Qwen, ExpertQA's entire length-label relationship turns out to be produced by
degenerate/bleed generations: on clean rows rho(len,label) falls from -0.577 to -0.040 and a
length-only rule falls from PRR +0.666 to -0.060. The same collapse happens on Llama. So the
obvious follow-up is whether the SUPERVISED comparison is also partly degeneracy detection: does
wMSP@2-vs-SAPLMA on the two hardest OOD rungs survive when the flagged rows are removed from the
TEST set?

WHAT IT DOES. For each (rung, eval) in {DiffTask-long, 1ds-Diff-long} x 8 evals, 3 seeds:
train SAPLMA and wMSP-shrink@2 EXACTLY as `probedriftlong.py:493-495,533-534` does — same pools,
same draws, same seeds, same kwargs — then score each one TWICE:
    ALL    every test row (this must reproduce the master, and is checked against it)
    CLEAN  test rows that are neither severe (luq.degeneracy) nor bleed (the fixed diagnostic)
The model is trained ONCE per cell and scored on both row sets, so the two columns differ only in
which rows are scored — never in what was learned. That isolates "how much of this PRR is the
method detecting degeneracy?" from "how much is it detecting uncertainty?".

⚠️ SCOPE, STATED PLAINLY. Flagged rows are removed from the TEST set only, not the training pool.
Removing them from training too is a different and larger counterfactual (it changes what the probe
learns and the pool size); it is NOT done here and no claim is made about it.

⚠️ THIS IS A DIAGNOSTIC SENSITIVITY. It writes its own CSV, it does not touch `pdl_master`, and no
example is dropped from the canonical population. A clean-subset PRR is not a corrected result —
the clean subset is a different, easier population (mean label 0.896 vs 0.652 on expertqa), so the
two columns are not directly comparable as method scores. The informative quantity is the COLLAPSE.

    python scripts/checks/qwen_clean_sensitivity.py
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache, degeneracy, msp, results, weighted_msp   # noqa: E402
from luq.config import Config                                   # noqa: E402
from luq.weighting import shrink_to_uniform                     # noqa: E402
from transformers import AutoTokenizer                          # noqa: E402
from aggregation_table import load_per_token, conf_meanpool     # noqa: E402
from attn_pool import PROMPT_REGIME                             # noqa: E402
from xl_rungs import eval_split, label_of, build_rows           # noqa: E402
import probedriftlong as pdl                                    # noqa: E402

QWEN = "Qwen/Qwen2.5-14B"
LLAMA = "meta-llama/Meta-Llama-3.1-8B"
HARDEST = ["DiffTask-long", "1ds-Diff-long"]        # the two hardest rungs, as named on Llama
BLEED = re.compile(
    r"^\s*(?:Question:|Answer:|Available choices:|\(\d+\)\.|Summary:|Article:|Story:|Abstract:"
    r"|Text:|Document:|Dialogue:|Conversation:|A single-select problem|Is the question answered)",
    re.MULTILINE)

FIELDS = ["model", "rung", "eval", "method", "row_set", "prr_mean", "prr_std", "n_seeds",
          "n_rows", "frac_flagged", "carve", "git_sha", "cluster", "env_hash", "provenance"]


def flagged_of(record):
    t = record.get("gen_text", "") or ""
    return degeneracy.is_severe(t) or bool(BLEED.search(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN)
    ap.add_argument("--evals", default=",".join(pdl.LONG))
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=23)
    args = ap.parse_args()

    slug = cache._slug(args.model)
    out = ROOT / "results" / "analysis" / f"clean_sensitivity__{slug}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    evals = [e for e in args.evals.split(",") if e]
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prov = pdl._provenance()
    carve = os.environ.get("LUQ_CARVE", "legacy")

    print("=" * 110)
    print(f"CLEAN-vs-FLAGGED SENSITIVITY   model={args.model} layer={args.layer} seeds={seeds}")
    print(f"rungs: {HARDEST}   device={device}   LUQ_CARVE={carve}")
    print("DIAGNOSTIC ONLY. Trained once per cell, scored on ALL rows and on CLEAN rows. Flagged =")
    print("  luq.degeneracy severe OR the fixed bleed pattern. Removed from TEST only, not training.")
    print("=" * 110, flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.model != LLAMA:
        from luq import token_subsets
        weighted_msp.set_special_ids(tok.all_special_ids)
        token_subsets.set_special_ids(tok.all_special_ids)
        print(f"special-token ids registered ({len(tok.all_special_ids)} ids)", flush=True)

    PT, FLAG = {}, {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        loaded = load_per_token(args.model, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> cells needing it stay ABSENT, not 0")
            continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]
        PT[d] = (states, split, y, records)
        FLAG[d] = np.array([flagged_of(r) for r in records])
        print(f"  {d}: {len(states)} rows, {100*FLAG[d].mean():.1f}% flagged", flush=True)
    sources = set(PT)

    rows = []
    for rung, X, spec in pdl.cells_long(sources, evals):
        if rung not in HARDEST or X not in PT:
            continue
        if len(eval_split(PT[X][1])[1]) == 0:
            continue
        acc = {(m, rs): [] for m in ("saplma", "wmsp_shrink2") for rs in ("ALL", "CLEAN")}
        n_all = n_clean = 0
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows)
            tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float)
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]
            clean = ~np.array([FLAG[d][i] for d, i in test_rows])

            v = {}
            Xmean = np.stack([s.mean(axis=0) for s in states])
            v["saplma"] = 1.0 - conf_meanpool(Xmean, tr_idx, te_idx, y, sd)
            v["wmsp_shrink2"] = np.asarray(weighted_msp.weighted_msp_unc(
                states, records, y, tr_idx, te_idx, device, weight_mode="normalised",
                reg=shrink_to_uniform, reg_lambda=2.0, length_normalise=True, seed=sd), float)
            n_all, n_clean = len(yte), int(clean.sum())
            for m, vec in v.items():
                acc[(m, "ALL")].append(results.prr(yte, vec))
                # A clean subset with no label spread has no oracle -> PRR is undefined, and a
                # number there would be invented. Skip loudly rather than emit one.
                if clean.sum() >= 20 and np.ptp(yte[clean]) > 1e-9:
                    acc[(m, "CLEAN")].append(results.prr(yte[clean], vec[clean]))
        if not acc[("saplma", "ALL")]:
            continue
        frac = 1 - n_clean / max(n_all, 1)
        for (m, rs), vals in acc.items():
            rows.append({"model": args.model, "rung": rung, "eval": X, "method": m, "row_set": rs,
                         "prr_mean": f"{np.mean(vals):.6f}" if vals else "",
                         "prr_std": f"{np.std(vals):.6f}" if vals else "",
                         "n_seeds": len(vals),
                         "n_rows": n_all if rs == "ALL" else n_clean,
                         "frac_flagged": f"{frac:.4f}", "carve": carve,
                         "provenance": "diagnostic-sensitivity", **prov})
        g = lambda m, rs: (np.mean(acc[(m, rs)]) if acc[(m, rs)] else float("nan"))
        print(f"  [{rung:15s} {X:14s}] flagged {100*frac:4.1f}%  "
              f"saplma {g('saplma','ALL'):+.4f}->{g('saplma','CLEAN'):+.4f}   "
              f"wmsp2 {g('wmsp_shrink2','ALL'):+.4f}->{g('wmsp_shrink2','CLEAN'):+.4f}   "
              f"margin {g('wmsp_shrink2','ALL')-g('saplma','ALL'):+.4f}->"
              f"{g('wmsp_shrink2','CLEAN')-g('saplma','CLEAN'):+.4f}", flush=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")
    print("⚠️ The CLEAN column is a DIFFERENT, EASIER population, not a corrected score. Read the")
    print("   collapse (or its absence), not the level.")


if __name__ == "__main__":
    main()
