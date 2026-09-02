#!/usr/bin/env python
"""wMSP shrinkage mechanism -- decomposition verification + the pre-registered diagnostic.

Pre-registration: prereg/shrinkage_mechanism.md (written before any arm comparison was
inspected). Report: results/analysis/WMSP_SHRINKAGE_MECHANISM.md.

MECHANISM-ONLY. Every quantity here comes from ONE seed at the ID rung, read from the persisted
production weights. It is NOT a performance measurement and must never be quoted as one; all PRR
values in the report remain the existing 3-seed ladder numbers.

The persisted dumps predate the 2026-08-03 all-excluded-softmax NaN fix, and asqa's vectors are
100% NaN as a result. asqa is reported UNAVAILABLE rather than skipped silently.

Read-only. Uses the PERSISTED production weights in cache/viz/*__wmsp_weights.npz (written by
visualise_token_weights.py --dump-weights through the production `_weights_from_raw`), so no model
is retrained here.

Verifies, per response, over the content-token set C used by wMSP (n = |C|):
    (i)   w_t == 0 exactly off C, and sum_{t in C} w_t == n
    (ii)  sum_{t in C} delta_t == 0            (delta_t = w_t - 1)
    (iii) U == mu_C + C_learned                (to float tolerance)
    (iv)  |C_learned| <= sqrt(Omega_C) * sigma_nll     (Cauchy-Schwarz)
and quantifies mu_C (content-token mean NLL) against the canonical all-token mean NLL.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, msp                                   # noqa: E402
from luq.config import Config                                # noqa: E402
from luq.weighted_msp import content_keep, per_token_nll     # noqa: E402
from attn_pool import PROMPT_REGIME                          # noqa: E402


def load_full_records(dataset):
    """The SAME unfiltered, positionally-indexed record list the weight dump used (load_per_token
    builds it exactly this way). `record_pos` in the npz indexes into THIS list, not a test-only
    subset -- indexing the filtered list silently pairs weights with the wrong responses."""
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    return cache.load_records(cfg.cache_dir, cache.run_key(MODEL, dataset, "ID"))

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = cache._slug(MODEL)
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
VIZ = ROOT / "cache" / "viz"


# WHERE THE PERSISTED PRODUCTION WEIGHTS ARE READ FROM, and why it is overridable.
#
# A run on a corrected population must not read the original population's weights for the dataset
# that changed, and must not overwrite them either: the two are the paired arms of one comparison,
# and losing either destroys the control. LUQ_VIZ_DIR names a directory holding a freshly dumped
# file for the corrected dataset alongside links to the unchanged ones, whose matched-setting cells
# train only on themselves and are therefore controls.
#
# The override must name a directory that ALREADY EXISTS. A mistyped path would otherwise report
# every dataset as missing weights, which reads as "not measured" rather than as a wrong path.
_viz_env = os.environ.get("LUQ_VIZ_DIR", "").strip()
if _viz_env:
    VIZ_DIR = Path(_viz_env)
    if not VIZ_DIR.is_dir():
        raise SystemExit(f"LUQ_VIZ_DIR={_viz_env!r} is not an existing directory. Create it "
                         "deliberately, or unset the variable to use the default weight directory.")
    print(f"[LUQ_VIZ_DIR] reading persisted weights from {VIZ_DIR}", flush=True)
else:
    VIZ_DIR = VIZ

# Output tag, so a second population's report cannot overwrite the first's.
OUT_TAG = os.environ.get("LUQ_OUT_TAG", "").strip()
ARMS = {"wMSP_pairwise": "lambda=0", "wMSP_shrink10": "lambda=10"}

rows, worst = [], {}
print(f"{'dataset':14s}{'arm':11s}{'n_ex':>6s}{'max|sumw-n|':>13s}{'max|sum d|':>12s}"
      f"{'max|U-(mu+C)|':>15s}{'CS viol':>9s}")
for d in LONG:
    p = VIZ_DIR / f"{SLUG}__{d}__ID__wmsp_weights.npz"
    if not p.exists():
        print(f"{d:14s} MISSING {p.name}")
        continue
    z = np.load(p, allow_pickle=True)
    pos = z["record_pos"].tolist()
    recs = load_full_records(d)

    for key, arm in ARMS.items():
        if key not in z.files:
            print(f"{d:14s}{arm:11s} arm absent from dump")
            continue
        W = z[key]
        e_sumw = e_sumd = e_id = 0.0
        n_cs = 0
        per = []
        for k, i in enumerate(pos):
            r = recs[i]
            w = np.asarray(W[k], dtype=np.float64)
            nll = per_token_nll(r).astype(np.float64)
            keep = content_keep(r).astype(bool)
            if len(w) != len(nll):                       # length guard, fail loud
                raise SystemExit(f"FATAL {d}/{i}: |w|={len(w)} vs |nll|={len(nll)}")
            n = int(keep.sum())
            if n < 2:
                continue                                  # sigma undefined; skip, counted below
            # (i) off-C weights must be exactly zero, on-C weights must sum to n
            e_sumw = max(e_sumw, abs(w[~keep]).max() if (~keep).any() else 0.0,
                         abs(w[keep].sum() - n))
            wc, lc = w[keep], nll[keep]
            delta = wc - 1.0
            e_sumd = max(e_sumd, abs(delta.sum()))
            mu_C = lc.mean()
            U = float((w * nll).sum() / n)                # production score: sum over ALL, /n_kept
            C_learned = float((delta * (lc - mu_C)).sum() / n)
            e_id = max(e_id, abs(U - (mu_C + C_learned)))
            Omega_C = float((delta ** 2).mean())          # the SPEC's Omega, over C
            Omega_impl = float(((w - 1.0) ** 2).mean())   # what shrink_to_uniform actually computes
            sigma = float(np.sqrt(((lc - mu_C) ** 2).mean()))
            bound = np.sqrt(Omega_C) * sigma
            if abs(C_learned) > bound + 1e-9:
                n_cs += 1
            per.append(dict(dataset=d, arm=arm, pos=int(i), n_content=n, n_tokens=len(nll),
                            mu_C=mu_C, U=U, C_learned=C_learned, absC=abs(C_learned),
                            Omega_C=Omega_C, Omega_impl=Omega_impl, sigma_nll=sigma, bound=float(bound),
                            alignment=(C_learned / bound) if bound > 0 else None,
                            ppl_all=float(msp.msp_uncertainty(r["token_logprobs"], "perplexity"))))
        rows += per
        print(f"{d:14s}{arm:11s}{len(per):>6d}{e_sumw:>13.2e}{e_sumd:>12.2e}{e_id:>15.2e}{n_cs:>9d}")

OUT = ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(OUT / f"wmsp_decomp{OUT_TAG}__{SLUG}.json", "w"))

# ---- mu_C (content-token mean NLL) vs the canonical all-token mean NLL -------------------------
print("\nANCHOR CHECK — content-token mean NLL vs canonical all-token mean NLL (per-response)")
print(f"{'dataset':14s}{'n':>6s}{'mean mu_C':>11s}{'mean ppl_all':>13s}{'mean diff':>11s}"
      f"{'max |diff|':>12s}{'% identical':>12s}")
base = [r for r in rows if r["arm"] == "lambda=0"]
for d in LONG:
    v = [r for r in base if r["dataset"] == d]
    if not v:
        continue
    a = np.array([r["mu_C"] for r in v]); b = np.array([r["ppl_all"] for r in v])
    print(f"{d:14s}{len(v):>6d}{a.mean():>11.4f}{b.mean():>13.4f}{(a-b).mean():>11.4f}"
          f"{np.abs(a-b).max():>12.4f}{100*np.mean(np.isclose(a,b,atol=1e-12)):>11.1f}%")
