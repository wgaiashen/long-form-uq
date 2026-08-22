#!/usr/bin/env python
"""Gate E: prove a clean-v2 GPU feature changed BECAUSE of the span and for no other reason.

THE PROBLEM THIS SOLVES. P(True) and Lookback cannot be produced by slicing a cached array, so they
are re-extracted. That raises the obvious worry: if a clean-v2 number differs from canonical, is it
the retained span, or is it the re-extraction? The pre-registered answer was "run the same path on
the raw population and reproduce canonical". On RCS that cannot pass, and not for any reason to do
with this correction: the canonical Llama P(True) and Lookback caches were produced on DoC, and the
same fp16 forward on a different GPU architecture accumulates differently. Measured on P(True),
layer 15: max abs deviation 1.17e-02, but per-row cosine similarity min 0.999992 and 0 of 1800 rows
below 0.999. Same computation, different floating-point accumulation.

THE FORM THAT DOES WORK. Compare clean-v2 against the RAW pass from the SAME job: same GPU, same
code, same session, same dtype. Then the only difference is the retained span, and the prediction is
sharp and falsifiable:

    rows whose span did NOT change  ->  the feature must be BIT-IDENTICAL
    rows whose span DID change      ->  the feature must differ

A single uncut row that moved would mean the extraction is not deterministic, or the two passes did
not see the same input. A single cut row that did not move would mean the truncation never reached
the feature. Either is a STOP.

    python scripts/checks/cleanv2_feature_span_attribution.py --feature ptrue_accurate
    python scripts/checks/cleanv2_feature_span_attribution.py --feature lookback --layer 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache                                             # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = cache._slug(MODEL)
DATASET = "med_quad"


def load_feats(ns, feature):
    p = ROOT / "cache" / ns / "features" / f"{SLUG}__{DATASET}__ID__{feature}.npz"
    if not p.exists():
        sys.exit(f"missing feature cache: {p}\n(has the extraction job finished?)")
    return np.load(p)["feats"], p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", required=True, help="ptrue_accurate | lookback")
    ap.add_argument("--layer", type=int, default=None,
                    help="plane to compare; default 15 for ptrue_accurate, 0 for lookback "
                         "(matching BASE_FEATS in probedriftlong)")
    args = ap.parse_args()
    layer = args.layer if args.layer is not None else (0 if args.feature == "lookback" else 15)

    raw, praw = load_feats("cleanv2_rawcheck", args.feature)
    cv, pcv = load_feats("cleanv2", args.feature)
    print(f"raw pass  : {praw}  {raw.shape}")
    print(f"clean-v2  : {pcv}  {cv.shape}")
    if raw.shape != cv.shape:
        sys.exit(f"shape mismatch {raw.shape} vs {cv.shape} -- the two passes are not comparable")

    canon = [json.loads(l) for l in
             open(ROOT / "cache" / "records" / f"{SLUG}__{DATASET}__ID.jsonl")]
    c2 = [json.loads(l) for l in
          open(ROOT / "cache" / "cleanv2" / "records" / f"{SLUG}__{DATASET}__ID.jsonl")]
    if not (len(canon) == len(c2) == raw.shape[0]):
        sys.exit(f"row mismatch: canonical {len(canon)}, cleanv2 {len(c2)}, feats {raw.shape[0]}")
    cut = np.array([len(b["gen_token_ids"]) < len(a["gen_token_ids"]) for a, b in zip(canon, c2)])

    a, b = raw[:, layer, :], cv[:, layer, :]
    d = np.abs(a - b).max(1)
    n_uncut_moved = int((d[~cut] != 0).sum())
    n_cut_static = int((d[cut] == 0).sum())

    print(f"\n  rows {len(cut)}   cut {int(cut.sum())}   uncut {int((~cut).sum())}   layer {layer}")
    print(f"  UNCUT rows must be identical : max|d| = {d[~cut].max():.3e}   "
          f"exactly equal on {int((d[~cut] == 0).sum())}/{int((~cut).sum())}")
    print(f"  CUT   rows must differ       : max|d| = {d[cut].max():.3e}   "
          f"median|d| = {np.median(d[cut]):.3e}   unchanged on {n_cut_static}")

    ok = (n_uncut_moved == 0) and (n_cut_static == 0)
    if n_uncut_moved:
        print(f"  **{n_uncut_moved} uncut rows MOVED** -- extraction is not deterministic, or the "
              f"two passes saw different inputs")
    if n_cut_static:
        print(f"  **{n_cut_static} cut rows did NOT move** -- the truncation is not reaching this feature")
    print(f"\n  GATE E ({args.feature}): "
          + ("PASS -- the feature changed on exactly the truncated rows and nowhere else"
             if ok else "**FAIL -- STOP**"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
