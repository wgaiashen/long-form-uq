#!/usr/bin/env python
"""W2 -- A LENGTH-CONDITIONED TEMPERATURE ON WEIGHTED MSP  (the 7 August ask, §2b + §6).


WHAT WAS ASKED FOR (the supervision meeting notes)
------------------------------------------------------------------------------------------------
    "Is there some axis where at one extreme it would be MSP-Min and at the other it would be MSP
     probes?" ... "Can we within weighted-MSP kind of go towards MSP-Min where the length is short?
     Some kind of temperature control on which tokens you include."
plus §6 (:144-166): length should enter PER TEST INSTANCE, not per dataset.

`_weights_from_raw` is `softmax(raw) * n`. This divides `raw` by a temperature that is a function of
the example's OWN generation length:

    raw  ->  raw / T(len)          T(len) = T0 * (len / len_ref) ** gamma

    gamma = 0, T0 = 1   the EXACT current weighted MSP (the required no-op control)
    T -> inf            weights flatten to uniform -> plain length-normalised MSP = perplexity
    T -> 0              all mass on ONE token: the one the LEARNED weighter ranks highest

THE T -> 0 LIMIT IS **NOT** msp_min, AND IS NEVER REPORTED AS msp_min.
`_seq_q` sharpens the LEARNED logits, so `T -> 0` concentrates on `argmax(raw)`, which coincides with
`argmax(nll)` only if the weighter happened to learn `raw ~ nll`. It was never asked to. So this arm
is a LEARNED WEAKEST-LINK, which is its own object. What W1 and W2 share is the FORM
`q = sum_t w_t * nll_t` with one sharpening parameter, NOT a single continuous path that literally
passes through msp_min. The control for this is computed below: the per-example agreement rate
between argmax(raw) and argmax(nll). See prereg discussion in the project plan

TWO VARIANTS, AND WHY THE CHEAP ONE COMES FIRST
-----------------------------------------------
  T2a (this file, default) POST-HOC. Train the weighter ONCE per (cell, seed), then re-score the whole
      (T0, gamma) grid off that one model. No retraining. There is precedent for exactly this shape of
      knob: `predict_weighted_msp` documents `smooth_n` as something that "can smooth the weights at
      scoring time even for a model trained without it" (weighted_msp.py:421-422). This isolates the
      SHARPENING from the TRAINING, which is the cleaner experiment anyway.
  T2b (not implemented) temperature inside the training loop. Retrains per grid point, so it costs the
      grid size times as much. Only worth building if T2a shows something.

ZERO SHARED-FILE EDITS. This driver imports from `luq.weighted_msp` and runs its own 15-line
scoring loop with the temperature in it. It does NOT modify `weighted_msp.py` or `probedriftlong.py`,
both of which are on the Qwen port's edit list. Keep it that way.

SELECTION HONESTY. `T0` and `gamma` are chosen by LEAVE-ONE-DATASET-OUT over the 8 evals, never on
the cell being reported. `len_ref` is the median generation length of that cell's TRAINING pool --
label-free, train-only, fixed in advance, never touched by the test rows.

Population: the COMPLETE ProbeDriftLong grid, 8 long evals x 5 rungs, 3 seeds. Reuses probedriftlong's
own cells_long / build_rows / eval_split, so the no-op arm is comparable with the master ladder cell
for cell.

    python scripts/checks/sharpening_wmsp.py --evals pubmed_qa --seeds 1 --smoke   # SMOKE TEST
    python scripts/checks/sharpening_wmsp.py --evals pubmed_qa                     # one eval, full
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

from luq import msp, results, weighted_msp                       # noqa: E402
from luq.weighting import reduces_to_uniform                     # noqa: E402
from aggregation_table import load_per_token                     # noqa: E402
from xl_rungs import eval_split, label_of, build_rows            # noqa: E402
import probedriftlong as pdl                                     # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15

# The registered grid. gamma = 0 with T0 = 1 is the no-op control and MUST be present.
T0S = [0.25, 0.5, 1.0, 2.0, 4.0]
GAMMAS = [-1.0, -0.5, 0.0, 0.5, 1.0]
NOOP = (1.0, 0.0)

OUT = ROOT / "results" / f"sharpening_wmsp__{'meta-llama_Meta-Llama-3.1-8B'}.csv"


def score_at_T(model, states, records, te_idx, device, T0, gamma, len_ref):
    """The whole method, in one loop. `q = sum_t softmax(raw/T)_t * n * nll_t / n_kept`, with T set by
    THIS example's own generated length. Everything except the `raw / T` is `_seq_q` unchanged, so the
    (T0=1, gamma=0) row is byte-identical to `predict_weighted_msp`.
    """
    model.eval()
    out = np.zeros(len(te_idx))
    argmax_agree = []                     # the control: does the learned weakest-link match the free one?
    with torch.no_grad():
        for k, i in enumerate(te_idx):
            nll_np = weighted_msp.per_token_nll(records[i])
            nll = torch.from_numpy(nll_np).to(device)
            raw = model(torch.from_numpy(weighted_msp.answer_states(states[i])).to(device))
            T = float(T0) * (max(len(nll_np), 1) / len_ref) ** float(gamma)
            kmask = torch.from_numpy(weighted_msp.content_keep(records[i])).to(device)
            out[k] = float(weighted_msp._seq_q(raw / T, nll, "normalised", True, keep=kmask).item())
            argmax_agree.append(int(int(torch.argmax(raw).item()) == int(np.argmax(nll_np))))
    return out, float(np.mean(argmax_agree)) if argmax_agree else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default=",".join(
        ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]))
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--smoke", action="store_true",
                    help="SMOKE TEST: one cell, one seed. Output is LABELLED as such and must NEVER "
                         "enter a results table.")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    evals = [e for e in args.evals.split(",") if e]
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.smoke:
        seeds = seeds[:1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    carve = os.environ.get("LUQ_CARVE", "legacy")
    tag = "SMOKE TEST -- NOT A RESULT" if args.smoke else "full grid"
    print("=" * 100)
    print(f"W2 -- length-conditioned temperature on weighted MSP   [{tag}]")
    print(f"device={device}  layer={args.layer}  seeds={seeds}  evals={evals}  LUQ_CARVE={carve}")
    print("T->0 is a LEARNED weakest-link, NOT msp_min. See the module docstring.")
    print("=" * 100, flush=True)

    # ---- load per-token states for every eval AND every training source the rungs can draw on ----
    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> SKIPPED LOUDLY (cells needing it will be absent, not zero)")
            continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: no finite labels -> skipped")
            continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]
            records = [records[k] for k in keep]
            split = split[keep]
            y = y[keep]
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
        per = {}                      # (T0, gamma) -> [prr per seed]
        floors = {"msp_min": [], "perplexity": []}
        agree = []
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows)
            tr_idx = list(range(n_tr))
            te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float)
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]

            # len_ref: LABEL-FREE, TRAIN-ONLY, fixed in advance. Never the test rows.
            len_ref = float(np.median([len(records[i]["token_logprobs"]) for i in tr_idx])) or 1.0

            floors["msp_min"].append(results.prr(yte, np.array(
                [msp.msp_uncertainty(records[i]["token_logprobs"], "min") for i in te_idx])))
            floors["perplexity"].append(results.prr(yte, np.array(
                [msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity") for i in te_idx])))

            # ONE training pass per (cell, seed). The whole grid is then a re-score off this model.
            model = weighted_msp.train_weighted_msp(states, records, y, tr_idx, device,
                                                    weight_mode="normalised", length_normalise=True,
                                                    seed=sd)

            # THE EXACT NO-OP IDENTITY. `score_at_T(T0=1, gamma=0)` divides `raw` by exactly 1.0, so
            # it MUST equal the library's own `predict_weighted_msp` to floating-point noise, on the
            # SAME trained model. This is the real control, and it is decisive in a way that comparing
            # a single seed against a 3-seed mean is NOT: seed noise on this cell is +/-0.076, wide
            # enough to hide a small wiring bug. Same model, same rows, no seeds involved -> any
            # difference is the temperature plumbing and nothing else. Checked EVERY cell and seed.
            v_noop, _ = score_at_T(model, states, records, te_idx, device, 1.0, 0.0, len_ref)
            v_lib = weighted_msp.predict_weighted_msp(model, states, records, te_idx, device,
                                                      weight_mode="normalised", length_normalise=True)
            dmax = float(np.max(np.abs(v_noop - v_lib)))
            if dmax > 1e-6:
                raise SystemExit(
                    f"NO-OP IDENTITY FAILED [{rung}/{X}/seed{sd}]: max|score_at_T(1,0) - "
                    f"predict_weighted_msp| = {dmax:.3e} > 1e-6. The temperature plumbing has changed "
                    f"the base method. STOPPING -- every W2 number would be meaningless.")
            print(f"    [{rung}/{X}/seed{sd}] no-op identity vs predict_weighted_msp: "
                  f"max|diff| = {dmax:.2e}  PASS", flush=True)
            for T0 in T0S:
                for gm in GAMMAS:
                    v, ag = score_at_T(model, states, records, te_idx, device, T0, gm, len_ref)
                    per.setdefault((T0, gm), []).append(results.prr(yte, v))
                    if (T0, gm) == NOOP:
                        agree.append(ag)
        if not per:
            continue

        # ---- the no-op control, printed BEFORE the grid is readable ----
        noop = float(np.mean(per[NOOP]))
        print(f"\n[{rung:16s} {X:14s}] no-op (T0=1, gamma=0) = {noop:+.4f}   "
              f"msp_min {np.mean(floors['msp_min']):+.4f}  ppl {np.mean(floors['perplexity']):+.4f}  "
              f"| argmax(raw)==argmax(nll) on {np.mean(agree):.1%} of examples", flush=True)
        print("   The no-op MUST match this cell's wMSP-norm entry in the master ladder to within")
        print("      seed noise. If it does not, that is a BUG and the grid below is not readable.")
        best = max(per, key=lambda k: float(np.mean(per[k])))
        print(f"   grid best (ORACLE, not a result): T0={best[0]} gamma={best[1]} "
              f"-> {float(np.mean(per[best])):+.4f}   delta vs no-op {float(np.mean(per[best]))-noop:+.4f}")
        for T0 in T0S:
            print(f"     T0={T0:<5}" + "".join(
                f"  g={gm:+.1f}:{float(np.mean(per[(T0, gm)])):+.4f}" for gm in GAMMAS))

        for (T0, gm), vals in per.items():
            rows.append((rung, X, T0, gm, float(np.mean(vals)), float(np.std(vals)),
                         len(vals), float(np.mean(agree)), carve,
                         "smoke" if args.smoke else "full"))
        for f, vals in floors.items():
            rows.append((rung, X, "", "", float(np.mean(vals)), float(np.std(vals)),
                         len(vals), "", carve, f"floor:{f}"))
        if args.smoke:
            print("\nSMOKE TEST -- one cell, one seed. NOT A RESULT. Does not enter any table.")
            break

    outp = Path(args.out)
    if args.smoke:
        outp = outp.with_name(outp.stem + "__SMOKE" + outp.suffix)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["rung", "eval", "T0", "gamma", "prr", "prr_std", "n_seeds",
                    "argmax_agree", "carve", "kind"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {outp}  ({len(rows)} rows)")
    print("\nNOTHING above is a selected result. T0/gamma must be chosen by LEAVE-ONE-DATASET-OUT "
          "over the 8 evals\n   before any number here is quoted as the method's score. The per-cell "
          "grid best is an ORACLE.")


if __name__ == "__main__":
    main()
