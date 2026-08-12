#!/usr/bin/env python
"""PR1 -- PROMPT-RESIDUAL HIDDEN-STATE PROBING.

Pre-registration: ../prereg/PR1_prompt_residual.md (committed BEFORE any R1/R2 PRR existed).
Output:           results/method_dev/prompt_residual/

THE QUESTION
------------
Response-level hidden-state probes pool ABSOLUTE states, which carry prompt/domain/task identity --
exactly the component that should not transfer when the eval target changes task family. Does
representing the response as its CHANGE FROM THE PROMPT ANCHOR improve cross-task ranking?

The per-token cache window is [last_prompt_token] + gen_tokens (scripts/01h_pertoken.py:71,
`lo, hi = P - 1, P + G`). So for a cached array `s` of shape (G+1, d):

    s[0]           = h0, the hidden state at the FINAL PROMPT position (the "anchor")
    s[1:]          = h_1 .. h_T, the generated-token states

Three representations, one classifier:

    R0  s.mean(0)                    anchor-inclusive mean  = the EXISTING mean-pool control
    R1  s[1:].mean(0)                generated-only mean    = the matched partner
    R2  s[1:].mean(0) - s[0]         prompt-residual mean   = THE METHOD UNDER TEST
    R3  s[1:].mean(0) - mean_train(h0)   CONSTANT anchor    = the centring control (see build_reps)

⚠️ R1 IS NOT OPTIONAL. Without it, a win for R2 over R0 could simply be "dropping h0" rather than
"subtracting h0". The pre-registered primary comparison is therefore R2 - R1, never R2 - R0.

⚠️ R3 IS NOT OPTIONAL EITHER. Subtracting a per-example anchor also CENTRES the input, which by
itself can help an unstandardised linear head. R3 buys the centring without the task-relativity, so
R2 - R3 is what separates the claimed mechanism from an optimisation artifact.

WHY R2 IS A RESTRICTION, NOT EXTRA CAPACITY
-------------------------------------------
A linear probe on R2 computes w·(mean_gen) - w·h0. That is the TIED-WEIGHT (+w, -w) special case of
a probe on the concatenation [mean_gen, h0]. So R2 constrains the representation rather than
enlarging it -- the same shape as the one intervention that has worked on this benchmark (shrinking
learned token weights toward uniform buys OOD robustness at an ID cost).

ZERO SHARED-FILE EDITS
----------------------
This driver imports from `attn_pool` / `aggregation_table` / `xl_rungs` / `probedriftlong` and runs
its own scoring loop. It does NOT modify attn_pool.py, probedriftlong.py, weighted_msp.py or
xl_rungs.py -- attn_pool.py in particular is on the Qwen port's no-edit list. Same discipline as
scripts/checks/sharpening_wmsp.py.

HOW THE CANONICAL RECIPE IS REUSED ON A PRE-POOLED VECTOR
---------------------------------------------------------
The canonical mean-pool control pools INSIDE train_attn (probedriftlong.py:579,
`train_attn(states, ..., freeze_query=True)`), so it cannot be handed a pre-pooled vector directly
without editing attn_pool.py. Instead each vector `z` is passed as a LENGTH-1 PSEUDO-SEQUENCE of
shape (1, d). With freeze_query=True the query stays at zeros, so every score is 0 and the softmax
over a single real position is exactly 1.0 -- giving pooled == z into the identical Linear(d,1)
head, optimiser, schedule, weight decay and RNG draw order.

⚠️ THAT EQUIVALENCE IS NOT ASSUMED. Every cell re-derives the canonical mean-pool control the
ORIGINAL way (full states through train_attn) and asserts it matches the pseudo-sequence R0 on the
PER-EXAMPLE UNCERTAINTIES (see the GATE_VEC_TOL note below for why that, and not PRR, is the right
instrument). If it disagrees the run ABORTS: R0 is not a new measurement, so a drift there means the
harness is wrong and no arm is reportable. (Verified offline first at 1.19e-07 on synthetic data,
then at exactly 0.00e+00 on real cached states.)

    python scripts/checks/prompt_residual.py --evals pubmed_qa --seeds 1 --smoke   # SMOKE TEST
    python scripts/checks/prompt_residual.py --evals xsum                          # one eval, full
"""
import argparse
import csv as _csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch                                                          # noqa: E402
from luq import msp, results                                          # noqa: E402
from aggregation_table import load_per_token, attn_unc                # noqa: E402
from attn_pool import train_attn                                      # noqa: E402
from xl_rungs import build_rows, eval_split, label_of, different_label_projection  # noqa: E402
import probedriftlong as pdl                                          # noqa: E402

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B"

# The three representations under test, plus the canonical control they are gated against.
# `meanpool_canonical` is computed the ORIGINAL way (full states into train_attn); `resid_R0_anchormean`
# is the same quantity through the pseudo-sequence path. They must agree -- see GATE below.
ARMS = ["meanpool_canonical", "resid_R0_anchormean", "resid_R1_genmean", "resid_R2_promptresid",
        "resid_R3_constanchor"]

# ---- THE REPRODUCTION GATE, AND WHY IT MEASURES VECTORS RATHER THAN PRR ------------------------
# Respecified 2026-08-12 after the first grid attempt (prereg §10). The gate's CLAIM is "the
# pseudo-sequence path computes the same thing as the canonical path". The first version tested
# that claim on PRR -- which was the wrong instrument, because PRR is a RANK statistic and therefore
# a DISCONTINUOUS function of the scores: two examples whose uncertainties differ by fp32 round-off
# can swap order, and the metric jumps.
#
# Measured, not assumed (200 trials at n=1140, the cnn_dailymail held-out size):
#   * a PURE 1e-7 perturbation of the scores moves PRR by up to 1.57e-04
#   * swapping ONE adjacent pair moves PRR by 4.58e-06
# The observed failure was 9.66e-06, i.e. squarely inside what tie-flipping alone produces, and the
# ID cells of the same run agreed at EXACTLY 0.00e+00 -- same computation, no tie happened to flip.
#
# So the gate now runs on the PER-EXAMPLE UNCERTAINTIES, which is the quantity the identity is
# actually about, at a tolerance appropriate to fp32 accumulation over d=4096. The PRR gap is still
# computed, logged and stored per cell -- it is informative, not ignored -- and still aborts if it
# grows large enough to threaten a conclusion (effects of interest here are ~0.010, so 1e-3 is two
# orders below anything that could matter).
GATE_VEC_TOL = 1e-5      # PRIMARY: max |uncertainty_canonical - uncertainty_R0| per example
GATE_PRR_INFO = 1e-3     # secondary backstop on the PRR gap; far above pure rank discreteness


def build_reps(states, tr_idx):
    """Turn each cached (G+1, d) per-token array into the four pooled representations.

    Returns a dict of {arm_name: list of (1, d) pseudo-sequences}, ready for train_attn.

    ⚠️ WHY R3 EXISTS (added 2026-08-12 after the smoke, see prereg §9 amendment).
    R2 subtracts the example's OWN anchor, which does two things at once: it makes the
    representation task-relative (the claim), AND it removes a large offset that hidden states
    share, which on its own could just be better CONDITIONING for an unstandardised linear head
    trained for 60 epochs. Those are different mechanisms with the same sign, so a win for R2 alone
    cannot distinguish them. R3 subtracts a CONSTANT anchor -- the mean h0 over the TRAINING rows --
    which delivers the conditioning benefit and nothing else:

        R2 ≈ R3   ->  the gain is centring/conditioning, NOT prompt-relativity
        R2 >> R3  ->  the per-example anchor carries real signal

    ⚠️ The constant is computed from TRAINING ROWS ONLY. Using all rows would leak the test set
    into the representation.

    ⚠️ EMPTY-GENERATION GUARD. If a record has G == 0 the window is a single row (just the anchor),
    so `s[1:]` is an EMPTY slice and numpy returns NaN with only a RuntimeWarning -- a plausible
    number in place of an absence, which would then be ranked arbitrarily by PRR. We abort instead.
    Falling back to h0 would silently turn R1/R2 into a different method.
    """
    bad = [i for i, s in enumerate(states) if np.asarray(s).shape[0] < 2]
    if bad:
        raise SystemExit(
            f"FATAL: {len(bad)} record(s) have a per-token window of <2 rows (G == 0), so the "
            f"generated-only mean is undefined. First offenders: {bad[:5]}. Refusing to run: an "
            f"empty-slice mean is NaN, and NaN in a PRR ranking returns a plausible-looking number "
            f"for output nobody measured.")
    # train-only constant anchor for R3
    const_anchor = np.mean([np.asarray(states[i], dtype=np.float32)[0] for i in tr_idx], axis=0)
    R0, R1, R2, R3 = [], [], [], []
    for s in states:
        a = np.asarray(s, dtype=np.float32)
        anchor = a[0]                       # h0: hidden state at the final PROMPT position
        gen_mean = a[1:].mean(axis=0)       # mean over the generated-token states only
        R0.append(a.mean(axis=0)[None, :])              # anchor-inclusive mean (existing control)
        R1.append(gen_mean[None, :])                    # generated-only mean
        R2.append((gen_mean - anchor)[None, :])         # prompt-residual (per-example anchor)
        R3.append((gen_mean - const_anchor)[None, :])   # constant anchor = centring control
    return {"resid_R0_anchormean": R0, "resid_R1_genmean": R1,
            "resid_R2_promptresid": R2, "resid_R3_constanchor": R3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=15,
                    help="probed layer. 15 = Llama-3.1-8B canonical middle; Qwen2.5-14B needs 23.")
    ap.add_argument("--evals", default="", help="comma-separated eval targets (default: all cached)")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--rungs", default="", help="comma-separated BASE rung names, e.g. ID,DiffTask")
    ap.add_argument("--out", default="", help="output CSV path")
    ap.add_argument("--smoke", action="store_true",
                    help="SMOKE TEST: subsample the pools hard. Output is labelled SMOKE and must "
                         "never enter a results table.")
    args = ap.parse_args()

    MODEL = args.model
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    want_rungs = set(s.strip() for s in args.rungs.split(",") if s.strip()) or None
    if want_rungs:
        valid = {"ID", "SameTask", "DiffTask", "LOO", "1ds-Diff", "Long->Short"}
        bad = want_rungs - valid
        if bad:
            raise SystemExit(f"--rungs: unknown {sorted(bad)}; valid are {sorted(valid)}")

    # Provenance FIRST: this aborts on a dirty tracked tree, before the (slow) cache load. A git_sha
    # stamped from a modified tree asserts a reproducibility that does not hold.
    prov = pdl._provenance()
    print(f"PROVENANCE: git_sha={prov['git_sha'][:12]} cluster={prov['cluster']} "
          f"env_hash={prov['env_hash']} carve={prov['carve']}", flush=True)
    if args.smoke:
        print("⚠️  SMOKE TEST -- results are NOT reportable", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={MODEL} layer={args.layer} seeds={seeds}", flush=True)

    evals = [e.strip() for e in args.evals.split(",") if e.strip()] or list(pdl.LONG_SRC)

    # ---- load the per-token caches -------------------------------------------------------------
    # Only the layer under test is loaded. Absence is LOUD and leaves the cell blank -- never zero.
    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: NO pertok cache -> cells touching {d} will be ABSENT", flush=True)
            continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: fully unlabelled ({label_of(d)}) -> skip", flush=True)
            continue
        if not finite.all():
            # Drop unlabelled rows so every downstream index sits on ONE basis (the filtered one).
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]
            records = [records[k] for k in keep]
            split = split[keep]
            y = y[keep]
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} labelled rows", flush=True)

    cells = pdl.cells_long(set(PT), evals)
    out_rows = []
    n_gate = 0

    for rung, X, spec in cells:
        if want_rungs is not None and rung.replace("-long", "") not in want_rungs:
            continue
        if X not in PT:
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        xlbl = different_label_projection(X)

        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            if args.smoke:                      # hard subsample; labelled SMOKE, never reportable
                train_rows, test_rows = train_rows[:120], test_rows[:60]
            n_tr = len(train_rows)
            tr_idx = list(range(n_tr))
            te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows

            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float)
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]

            v = {}
            # Unsupervised floors, free from the cached logprobs. Carried so every table can show
            # the method beside the bar it has to clear, rather than alone.
            v["floor_min"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "min")
                                       for i in te_idx])
            v["floor_ppl"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity")
                                       for i in te_idx])

            # THE CANONICAL CONTROL, computed exactly as probedriftlong.py:579 does it: full
            # per-token states into train_attn with a frozen (zero) query, which makes the attention
            # uniform over real tokens = a plain mean pool.
            m_can = train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True)
            v["meanpool_canonical"] = np.asarray(attn_unc(m_can, states, te_idx, device), float)

            # The pre-pooled representations through the IDENTICAL recipe.
            reps = build_reps(states, tr_idx)
            for arm, seq in reps.items():
                m = train_attn(seq, y, tr_idx, device, seed=sd, freeze_query=True)
                v[arm] = np.asarray(attn_unc(m, seq, te_idx, device), float)

            # ---- GATE (blocking) -------------------------------------------------------------
            # R0 and the canonical control are THE SAME QUANTITY by two code paths. The identity is
            # tested on the per-example uncertainties; the PRR gap is recorded alongside because it
            # is discretised by rank ties (see the GATE_VEC_TOL note at the top of this file).
            vec_gap = float(np.abs(v["meanpool_canonical"] - v["resid_R0_anchormean"]).max())
            prr_can = results.prr(yte, v["meanpool_canonical"])
            prr_r0 = results.prr(yte, v["resid_R0_anchormean"])
            prr_gap = abs(prr_can - prr_r0)
            if vec_gap > GATE_VEC_TOL:
                raise SystemExit(
                    f"FATAL GATE FAILURE at [{rung}/{X}/seed{sd}]: the canonical mean-pool control "
                    f"and the pseudo-sequence R0 differ by {vec_gap:.3e} > {GATE_VEC_TOL:g} on the "
                    f"PER-EXAMPLE uncertainties. These are the same quantity computed two ways, so "
                    f"a disagreement of that size means the length-1 pooling identity does NOT hold "
                    f"here. Refusing to report ANY arm from a harness that cannot reproduce its own "
                    f"control.")
            if prr_gap > GATE_PRR_INFO:
                raise SystemExit(
                    f"FATAL GATE FAILURE at [{rung}/{X}/seed{sd}]: per-example vectors agree "
                    f"({vec_gap:.3e}) but PRR differs by {prr_gap:.3e} > {GATE_PRR_INFO:g}. That is "
                    f"far more than rank-tie discreteness can explain at this scale, so something "
                    f"other than fp32 round-off is moving the ranking.")
            n_gate += 1

            for m_name, vec in v.items():
                out_rows.append({
                    "model": MODEL, "layer": args.layer, "eval": X, "rung": rung, "seed": sd,
                    "method": m_name, "prr": results.prr(yte, vec),
                    "n_train": n_tr, "n_test": len(te_idx),
                    "gate_vec_gap": vec_gap, "gate_prr_gap": prr_gap,
                    "different_label_projection": xlbl,
                    "smoke": bool(args.smoke),
                    "git_sha": prov["git_sha"], "cluster": prov["cluster"],
                    "env_hash": prov["env_hash"], "carve": prov["carve"],
                })
            print(f"  [{rung}/{X}/s{sd}] gate ok (vec {vec_gap:.1e}, prr {prr_gap:.1e})  "
                  + "  ".join(f"{a}={results.prr(yte, v[a]):+.4f}" for a in ARMS), flush=True)

    if not out_rows:
        raise SystemExit("no cells produced -- refusing to write an empty CSV")

    slug = MODEL.replace("/", "_")
    default = ROOT / "results" / "method_dev" / "prompt_residual" / (
        f"prompt_residual__{slug}" + ("__SMOKE" if args.smoke else "") + ".csv")
    out = Path(args.out) if args.out else default
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"FATAL: {out} exists, refusing to clobber")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n[prompt-residual] {len(out_rows)} rows, {n_gate} cells passed the reproduction gate "
          f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
