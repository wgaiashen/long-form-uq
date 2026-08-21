#!/usr/bin/env python
"""Diagnose the asqa wMSP outlier: all 8 normalised variants return EXACTLY -0.0439, sd 0.0.

WHY THIS IS A SUSPECTED BUG, NOT A RESULT. On pubmed the same 8 variants span +0.64 to -0.14. Eight
different weighting schemes (plain softmax, two segment modes, two shrink strengths, Blondel loss, and
two shrink+Blondel combinations) cannot agree to four decimal places unless the weighting is not being
applied. sd=0.0 across 3 seeds says no randomness reaches the score either. The unconstrained variant
DOES differ (+0.4985), so the collapse is specific to weight_mode="normalised".

Reports the score distribution, which distinguishes the two candidate explanations:
  * near-CONSTANT score  -> PRR is meaningless and the cell must be BLANK, not -0.0439
  * varied score         -> the ranking really is anti-correlated and it is a finding about asqa
"""
import sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "checks"))
from attn_pool import load_per_token          # noqa: E402
from xl_rungs import eval_split               # noqa: E402
from luq import weighted_msp, results         # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"

for ds in ("asqa", "med_quad"):               # med_quad = the control (its variants DO differ)
    st, split, y, layer, recs = load_per_token(MODEL, ds, 15)
    tr, te = eval_split(split)
    print(f"\n===== {ds}: {len(st)} rows, train {len(tr)}, test {len(te)} =====", flush=True)
    for mode in ("normalised", "unconstrained", "constant"):
        for sd in (1, 2):
            u = np.asarray(weighted_msp.weighted_msp_unc(st, recs, y, tr, te, "cpu",
                                                         length_normalise=True, seed=sd,
                                                         weight_mode=mode), float)
            uniq = len(np.unique(np.round(u, 10)))
            print(f"  {mode:14s} seed{sd}  PRR={results.prr(y[te], u):+.4f}  "
                  f"unique={uniq}/{len(u)}  std={u.std():.6g}  range=[{u.min():.4g},{u.max():.4g}]",
                  flush=True)
            if uniq <= 3:
                print(f"      SCORE IS (NEAR-)CONSTANT -> the PRR is an artifact; the cell should be "
                      f"BLANK (not measured), never a number.", flush=True)
