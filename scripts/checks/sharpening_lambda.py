#!/usr/bin/env python
"""W5 -- (B) lambda = 1 and 1.5 from a July prediction, and (C) NLL inside the weight logits.

Pre-registration: prereg/W5_lambda_and_nll_prior.md (committed BEFORE this was run).
Plan: ../PLAN_sharpening_axis.md.   Results: ../STOCKTAKE_sharpening_axis.md §8.

TWO QUESTIONS, ONE JOB (they share the training pass)
-----------------------------------------------------
B. THE SHRINKAGE lambda. wMSP@2 is the best method at 3 of the 4 OOD rungs, including both hardest.
   lambda in {2, 10} was never chosen on the long grid at all -- it came from a coarse July grid on
   three short/QA sets. A finer sweep on 2026-07-12 concluded the sweet spot was 1-2 and named the
   SAME two rungs (DiffTask, 1ds-Diff) where wMSP@2 now leads, and that refinement was never
   propagated. lambda = 1 and 1.5 have never been run on the long grid.
   ⭐ lambda = 1.5 is PRE-COMMITTED in the prereg, justified ONLY by that dated prediction.

C. NLL IN THE WEIGHT LOGITS. Track 2 sharpened the LEARNED logits, and its T -> 0 limit lands on
   argmax(raw), which agrees with argmax(nll) only 6-13% of the time -- so it never spanned
   wMSP <-> msp_min. Putting the NLL inside the softmax fixes that:

       exponential :  w = softmax( raw + tau * z(nll) )
       power       :  w = softmax( raw + beta * log(clip(nll, 1e-9)) )

   Both reach msp_min as their parameter -> inf REGARDLESS of what the query learned.
   ⚠️ armD (attn_pool.py:463) already adds beta*log(prior) -- but to the ATTENTION POOLER, which
   pools hidden states, so its limit is a probe on the worst token's state and NOT msp_min. Only a
   tilt inside weighted MSP, which sums NLLs, reaches msp_min. This is not a re-run of S8.
   ⚠️ Both tilts are run because with raw = 0 they ARE round 1's softmax-tau and Lehmer families,
   and round 1 found Lehmer the stronger. Running only one would rest the arm on the weaker form.

THREE CONTROLS, PRINTED BEFORE ANY CURVE IS READABLE (prereg §3.3)
------------------------------------------------------------------
  1. tau = beta = 0 reproduces plain wMSP to < 1e-6 on the SAME trained model.
  2. tau, beta -> inf reproduces msp_min's PRR exactly.  ⚠️ THIS IS THE CLAIM. If it fails the arm
     is dead on arrival -- it is the check that would have caught the Track 2 error.
  3. argmax(w) vs argmax(nll) agreement -> 100% in the limit, against Track 2's 6-13%.
  Plus the OUTSIDE check: the lambda=norm rows must reproduce pdl_master to 4 dp (Track 2 passed
  this on 25/25 cells and it is the strongest control in the workstream).

⚠️ ZERO SHARED-FILE EDITS. Imports from luq.weighted_msp and runs its own scoring loop.
`probedriftlong.py` and `weighted_msp.py` are both on the Qwen port's list and stay untouched.

    python scripts/checks/sharpening_lambda.py --evals pubmed_qa --seeds 1 --smoke
    python scripts/checks/sharpening_lambda.py --evals pubmed_qa
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
from luq.weighting import shrink_to_uniform                      # noqa: E402
from aggregation_table import load_per_token                     # noqa: E402
from xl_rungs import eval_split, label_of, build_rows            # noqa: E402
import probedriftlong as pdl                                     # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15

# (B) the lambda arms. norm (=0) is trained as the VERIFICATION ANCHOR against pdl_master and as the
# base model for (C). 2 and 10 already exist on the master grid and are not recomputed.
LAMBDAS = [("norm", 0.0), ("lam1", 1.0), ("lam1.5", 1.5)]
LAMBDA_PRIMARY = 1.5                    # PRE-COMMITTED in prereg/W5 §2.1, before the run

# (C) the tilt grids. T = 1 and gamma = 0 throughout -- a DIFFERENT axis from Track 2's temperature.
TAUS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf]
BETAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, np.inf]
CLIP = 1e-9                             # reuses attn_pool.py:463's convention, not a new one

OUT = ROOT / "results"


def tilt_logits(raw, nll_np, kind, param, device, keep_np=None):
    """raw + the NLL tilt. Returns the tilted logits, or a one-hot marker for the infinite limit.

    ⚠️ BUGFIX 2026-08-09, caught by CONTROL 2 aborting the first smoke run. The infinite limit used
    `argmax(nll)` over ALL tokens. But `_weights_from_raw` masks SPECIAL tokens to -inf before the
    softmax (weighted_msp.py:206-221, because ~70% of xsum/cnn generations end in EOS and the learned
    weighter otherwise dumps its mass there). So when the largest NLL fell on a special token, EVERY
    logit was -1e30-or-masked, the softmax came out nearly UNIFORM over the kept tokens, and the score
    collapsed to roughly `perplexity` instead of the max -- catastrophic on pubmed_qa, where
    perplexity is -0.174. That happens on 10.9% of pubmed_qa examples and it is most of why the
    control failed by 0.078.

    The anchor must therefore be `argmax(nll)` among the KEPT tokens, which is what
    prereg/F5 §2 registered. The consequence, stated plainly: the limit is
    **`msp_min` restricted to content tokens**, not `msp_min` over all tokens. Measured difference:
    pubmed_qa 0.3478 vs 0.3710. Both are reported.
    """
    if not np.isfinite(param):
        z = torch.full_like(raw, -1e30)
        if keep_np is not None and keep_np.any():
            idx = int(np.flatnonzero(keep_np)[np.argmax(nll_np[keep_np.astype(bool)])])
        else:
            idx = int(np.argmax(nll_np))
        z[idx] = 0.0
        return z
    if param == 0.0:
        return raw
    if kind == "exp":
        sd = float(nll_np.std())
        if sd < 1e-12:                                  # every token equally surprising -> no tilt
            return raw
        zz = (nll_np - nll_np.mean()) / sd
        return raw + float(param) * torch.from_numpy(zz.astype(np.float32)).to(device)
    lg = np.log(np.clip(nll_np, CLIP, None)).astype(np.float32)
    return raw + float(param) * torch.from_numpy(lg).to(device)


def score_tilted(model, states, records, te_idx, device, kind, param):
    """Score the test rows with the NLL-tilted weights. Also returns the argmax-agreement rate."""
    model.eval()
    out = np.zeros(len(te_idx))
    agree = []
    with torch.no_grad():
        for k, i in enumerate(te_idx):
            nll_np = weighted_msp.per_token_nll(records[i])
            nll = torch.from_numpy(nll_np).to(device)
            raw = model(torch.from_numpy(weighted_msp.answer_states(states[i])).to(device))
            keep_np = weighted_msp.content_keep(records[i])
            kmask = torch.from_numpy(keep_np).to(device)
            lg = tilt_logits(raw, nll_np, kind, param, device, keep_np=keep_np)
            out[k] = float(weighted_msp._seq_q(lg, nll, "normalised", True, keep=kmask).item())
            # ⚠️ Compare against the argmax over KEPT tokens. The weighter cannot place mass on a
            # special token, so scoring it against the all-token argmax caps this diagnostic at
            # ~89% by construction (10.9% of pubmed_qa examples have their largest NLL on a special)
            # and would look like a failure when nothing is wrong. Same token set, per the project conventions.
            kb = keep_np.astype(bool)
            a_ref = int(np.flatnonzero(kb)[np.argmax(nll_np[kb])]) if kb.any() else int(np.argmax(nll_np))
            agree.append(int(int(torch.argmax(lg).item()) == a_ref))
    return out, (float(np.mean(agree)) if agree else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="pubmed_qa")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--smoke", action="store_true",
                    help="SMOKE TEST: one cell, one seed. Labelled as such; never enters a table.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    evals = [e for e in args.evals.split(",") if e]
    seeds = [int(s) for s in args.seeds.split(",")][:1] if args.smoke else \
        [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    carve = os.environ.get("LUQ_CARVE", "legacy")
    tag = "SMOKE TEST -- NOT A RESULT" if args.smoke else "full grid"
    print("=" * 100)
    print(f"W5 -- (B) lambda sweep + (C) NLL in the weight logits   [{tag}]")
    print(f"device={device} layer={args.layer} seeds={seeds} evals={evals} LUQ_CARVE={carve}")
    print(f"⭐ PRE-COMMITTED PRIMARY: lambda = {LAMBDA_PRIMARY} (prereg/W5 §2.1, from the 2026-07-12 "
          f"prediction)")
    print("=" * 100, flush=True)

    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> SKIPPED LOUDLY (cells needing it will be ABSENT, not 0)")
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
        lam_prr = {n: [] for n, _ in LAMBDAS}
        exp_prr = {t: [] for t in TAUS}
        pow_prr = {b: [] for b in BETAS}
        floors = {"msp_min": [], "msp_min_kept": [], "perplexity": []}
        agree_inf, agree_zero = [], []
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

            v_min = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "min") for i in te_idx])
            floors["msp_min"].append(results.prr(yte, v_min))
            # ⚠️ SAME TOKEN SET. The learned weighter is forbidden special tokens, so the endpoint
            # this family can reach is msp_min restricted to CONTENT tokens. Checking against the
            # all-token msp_min is a TOKEN-SET CONFOUND (the standing project rule), not a failure of
            # the method. Measured difference on pubmed_qa: 0.3478 (kept) vs 0.3710 (all), because
            # 10.9% of its examples have their largest NLL on a special token. Both are reported.
            v_mk = []
            for i in te_idx:
                nl = weighted_msp.per_token_nll(records[i])
                kk = weighted_msp.content_keep(records[i]).astype(bool)
                v_mk.append(float(nl[kk].max()) if kk.any() else float(nl.max()))
            floors["msp_min_kept"].append(results.prr(yte, np.array(v_mk)))
            floors["perplexity"].append(results.prr(yte, np.array(
                [msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity") for i in te_idx])))

            base_model = None
            for name, lam in LAMBDAS:
                reg = shrink_to_uniform if lam > 0 else None
                model = weighted_msp.train_weighted_msp(
                    states, records, y, tr_idx, device, weight_mode="normalised",
                    length_normalise=True, seed=sd, reg=reg, reg_lambda=lam)
                v = weighted_msp.predict_weighted_msp(model, states, records, te_idx, device,
                                                     weight_mode="normalised", length_normalise=True)
                lam_prr[name].append(results.prr(yte, v))
                if lam == 0.0:
                    base_model = model
                    # CONTROL 1: tau = 0 must reproduce plain wMSP EXACTLY on this same model.
                    v0, a0 = score_tilted(base_model, states, records, te_idx, device, "exp", 0.0)
                    dmax = float(np.max(np.abs(v0 - v)))
                    if dmax > 1e-6:
                        raise SystemExit(
                            f"CONTROL 1 FAILED [{rung}/{X}/seed{sd}]: tau=0 vs predict_weighted_msp "
                            f"max|diff|={dmax:.3e} > 1e-6. The tilt plumbing changed the base method.")
                    agree_zero.append(a0)

            # (C) the two tilts, re-scored off the SAME base model -> free
            for t in TAUS:
                v, a = score_tilted(base_model, states, records, te_idx, device, "exp", t)
                exp_prr[t].append(results.prr(yte, v))
                if not np.isfinite(t):
                    agree_inf.append(a)
                    # CONTROL 2: the infinite limit MUST equal msp_min's PRR. This is the claim.
                    d2 = abs(results.prr(yte, v) - floors["msp_min_kept"][-1])
                    if d2 > 1e-9:
                        raise SystemExit(
                            f"CONTROL 2 FAILED [{rung}/{X}/seed{sd}]: tau->inf PRR {results.prr(yte, v):+.6f} "
                            f"!= msp_min-on-CONTENT-tokens {floors['msp_min_kept'][-1]:+.6f} "
                            f"(d={d2:.2e}). The family does NOT span wMSP <-> the content-token floor.")
            for b in BETAS:
                v, _ = score_tilted(base_model, states, records, te_idx, device, "pow", b)
                pow_prr[b].append(results.prr(yte, v))

        if not lam_prr["norm"]:
            continue
        m = {k: float(np.mean(v)) for k, v in lam_prr.items() if v}
        fl = {k: float(np.mean(v)) for k, v in floors.items()}
        print(f"\n[{rung:16s} {X:14s}]  msp_min {fl['msp_min']:+.4f}  ppl {fl['perplexity']:+.4f}")
        print(f"   (B) lambda: " + "  ".join(f"{n}={m[n]:+.4f}" for n, _ in LAMBDAS))
        print(f"   ⚠️ `norm` MUST match this cell's wMSP-norm in pdl_master to 4 dp (outside check).")
        print(f"   (C) exp tilt:  " + "  ".join(
            f"{('inf' if not np.isfinite(t) else f'{t:g}')}={float(np.mean(exp_prr[t])):+.4f}" for t in TAUS))
        print(f"   (C) pow tilt:  " + "  ".join(
            f"{('inf' if not np.isfinite(b) else f'{b:g}')}={float(np.mean(pow_prr[b])):+.4f}" for b in BETAS))
        print(f"   CONTROL 3 argmax(w)==argmax(nll):  at tilt=0 {np.mean(agree_zero):.1%}  "
              f"at tilt=inf {np.mean(agree_inf):.1%} (must be 100%)", flush=True)

        for n, lam in LAMBDAS:
            rows.append((rung, X, "lambda", n, float(np.mean(lam_prr[n])), float(np.std(lam_prr[n])),
                         len(lam_prr[n]), carve, "smoke" if args.smoke else "full"))
        for t in TAUS:
            rows.append((rung, X, "tilt_exp", "inf" if not np.isfinite(t) else f"{t:g}",
                         float(np.mean(exp_prr[t])), float(np.std(exp_prr[t])), len(exp_prr[t]),
                         carve, "smoke" if args.smoke else "full"))
        for b in BETAS:
            rows.append((rung, X, "tilt_pow", "inf" if not np.isfinite(b) else f"{b:g}",
                         float(np.mean(pow_prr[b])), float(np.std(pow_prr[b])), len(pow_prr[b]),
                         carve, "smoke" if args.smoke else "full"))
        for k, v in fl.items():
            rows.append((rung, X, "floor", k, v, 0.0, len(seeds), carve, "floor"))
        if args.smoke:
            print("\n⚠️ SMOKE TEST -- one cell, one seed. NOT A RESULT.")
            break

    ev = evals[0] if len(evals) == 1 else "multi"
    outp = Path(args.out) if args.out else OUT / f"sharpening_lambda_{ev}__meta-llama_Meta-Llama-3.1-8B.csv"
    if args.smoke:
        outp = outp.with_name(outp.stem + "__SMOKE" + outp.suffix)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["rung", "eval", "kind", "param", "prr", "prr_std", "n_seeds", "carve", "tag"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {outp}  ({len(rows)} rows)")
    print("\n⚠️ lambda = 1.5 is the PRE-COMMITTED primary. If it misses the bar that is a FAILURE OF "
          "THE\n   PRE-COMMITTED VALUE, even if another lambda clears it -- promoting a different one "
          "afterwards\n   is banned by the pre-registration. The tilt arms have NO pre-committed "
          "parameter and are\n   exploratory; their per-cell grid best is an ORACLE, never a result.")


if __name__ == "__main__":
    main()
