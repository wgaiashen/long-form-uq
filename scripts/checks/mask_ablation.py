#!/usr/bin/env python
"""MASK ABLATION -- does excluding end-of-text tokens from the weight logits actually HELP wMSP?


THE QUESTION, AND WHY IT IS OPEN
--------------------------------
`weighted_msp._weights_from_raw` masks special tokens to -inf before the softmax. The stated reason
(weighted_msp.py:216-219) is that ~70% of xsum/cnn generations end in EOS and the learned weighter
otherwise concentrates its softmax mass on that content-free "I'm done" token.

BUT THAT IS AN OBSERVATION ABOUT WHERE THE ATTENTION GOES, NOT A MEASUREMENT THAT GOING THERE
HURTS. The mask was introduced because concentration on EOS looked wrong, never because the
unmasked variant was run and scored worse. That is the same inference pattern that turned out to be
wrong about punctuation (prereg/R0_punctuation_ablation.md). And §13 has since shown the completion
indicator carries real signal -- +0.268 on expertqa -- so "the EOS token is content-free" is not
obviously true either.

It also matters because the mask makes weighted MSP the ONLY method on the ladder scored on a
different token set from the floor it is compared against (§12).

ALL THREE HEADLINE VARIANTS INHERIT IT. `train_weighted_msp` / `predict_weighted_msp` default
`exclude_special=True`, and `probedriftlong.py`'s WMSP list never overrides it, so wmsp_norm,
wmsp_shrink2 AND wmsp_shrink10 are all masked. All three are ablated here.

THE DESIGN
----------
Six arms per (cell, seed): {norm, shrink@2, shrink@10} x {masked, unmasked}. Identical data, identical
seeds, identical everything else -- the ONLY thing that varies is `exclude_special`. So a difference
is the mask and nothing else.

NO-OP CONTROL: the MASKED arms must reproduce `pdl_master`'s wMSP-norm / shrink@2 / shrink@10 to
4 dp. That is the same outside check Track 2 passed on 25/25 cells, and without it a difference
between arms could be this driver rather than the mask.

    qsub -v LUQ_EVAL=pubmed_qa pbs/mask_ablation.pbs
"""
import argparse
import csv as _csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import msp, results, weighted_msp                           # noqa: E402
from luq.weighting import shrink_to_uniform                          # noqa: E402
from aggregation_table import load_per_token                         # noqa: E402
from xl_rungs import eval_split, label_of, build_rows                # noqa: E402
import probedriftlong as pdl                                         # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15
# (name, reg_lambda) -- the three variants that appear in the master table
VARIANTS = [("norm", 0.0), ("shrink2", 2.0), ("shrink10", 10.0)]
OUT = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="pubmed_qa")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--smoke", action="store_true", help="one cell, one seed; NEVER enters a table")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    evals = [e for e in args.evals.split(",") if e]
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.smoke:
        seeds = seeds[:1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    carve = os.environ.get("LUQ_CARVE", "legacy")
    print("=" * 100)
    print(f"MASK ABLATION -- does excluding EOS from the weight logits help?  "
          f"[{'SMOKE, NOT A RESULT' if args.smoke else 'full grid'}]")
    print(f"device={device} seeds={seeds} evals={evals} LUQ_CARVE={carve}")
    print("Six arms per cell: {norm, shrink@2, shrink@10} x {masked, unmasked}. Only the mask varies.")
    print("=" * 100, flush=True)

    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> SKIPPED LOUDLY (absent, never zero)")
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
        print(f"  {d}: {len(states)} rows (label={label_of(d)})", flush=True)
    sources = set(PT)

    rows = []
    for rung, X, spec in pdl.cells_long(sources, evals):
        if X not in PT:
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        acc = {(v, m): [] for v, _ in VARIANTS for m in ("masked", "unmasked")}
        floors = {"msp_min": [], "perplexity": []}
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
            floors["msp_min"].append(results.prr(yte, np.array(
                [msp.msp_uncertainty(records[i]["token_logprobs"], "min") for i in te_idx])))
            floors["perplexity"].append(results.prr(yte, np.array(
                [msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity") for i in te_idx])))
            for vname, lam in VARIANTS:
                for mode, excl in (("masked", True), ("unmasked", False)):
                    v = weighted_msp.weighted_msp_unc(
                        states, records, y, tr_idx, te_idx, device,
                        weight_mode="normalised", length_normalise=True, seed=sd,
                        reg=(shrink_to_uniform if lam > 0 else None), reg_lambda=lam,
                        exclude_special=excl)
                    acc[(vname, mode)].append(results.prr(yte, np.asarray(v, float)))
        if not acc[("norm", "masked")]:
            continue
        fl = {k: float(np.mean(v)) for k, v in floors.items()}
        print(f"\n[{rung:16s} {X:14s}]  msp_min {fl['msp_min']:+.4f}  ppl {fl['perplexity']:+.4f}")
        print(f"   {'variant':12s}{'masked':>10s}{'unmasked':>10s}{'unmask-mask':>13s}")
        for vname, _ in VARIANTS:
            a = float(np.mean(acc[(vname, "masked")]))
            b = float(np.mean(acc[(vname, "unmasked")]))
            print(f"   {vname:12s}{a:>+10.4f}{b:>+10.4f}{b - a:>+13.4f}")
            for mode, val in (("masked", a), ("unmasked", b)):
                rows.append((rung, X, vname, mode, f"{val:.4f}",
                             f"{np.std(acc[(vname, mode)]):.4f}", len(acc[(vname, mode)]), carve))
        print("   The MASKED column must match pdl_master's wMSP-norm/@2/@10 to 4 dp "
              "(outside check).", flush=True)
        for k, v in fl.items():
            rows.append((rung, X, "floor", k, f"{v:.4f}", "0.0000", len(seeds), carve))
        if args.smoke:
            print("\nSMOKE -- one cell, one seed. NOT A RESULT.")
            break

    ev = evals[0] if len(evals) == 1 else "multi"
    outp = Path(args.out) if args.out else OUT / f"mask_ablation_{ev}__meta-llama_Meta-Llama-3.1-8B.csv"
    if args.smoke:
        outp = outp.with_name(outp.stem + "__SMOKE" + outp.suffix)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["rung", "eval", "variant", "mode", "prr", "prr_std", "n_seeds", "carve"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {outp}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
