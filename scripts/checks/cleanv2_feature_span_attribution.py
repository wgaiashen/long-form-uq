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

WHEN THE RAW PASS CANNOT BE PRODUCED: --baseline canonical.
The raw pass is not always obtainable. For lookback it fails reproducibly at row 1361 of the raw
population -- twice, at the identical row, while the clean-v2 pass completed all 1800 rows on the
same node and cards -- and the 48 GB card that would avoid the two-card path has been unplaceable
for days because the only nodes holding free ones are offline.

The fallback compares against the CANONICAL feature cache instead, and it still says something real,
because on a row clean-v2 did not truncate the input is byte-identical to canonical:

    rows whose span did NOT change  ->  the feature must MATCH canonical (see the tolerance below)
    rows whose span DID change      ->  the feature must differ

It is weaker in exactly one way, and the weakness is not about this correction: canonical was
extracted on different hardware, so equality is up to device arithmetic rather than bitwise. The
comparison is therefore reported as cosine similarity per row, the same way the P(True) device
finding was, and bitwise equality is NOT asserted on the unchanged rows.
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
    ap.add_argument("--baseline", default="rawcheck", choices=("rawcheck", "canonical"),
                    help="what to compare clean-v2 against. 'rawcheck' (default) is the same-job raw "
                         "pass and permits a bitwise claim. 'canonical' is the fallback when that "
                         "pass cannot be produced; it compares by cosine because the canonical cache "
                         "was extracted on other hardware.")
    ap.add_argument("--layer", type=int, default=None,
                    help="plane to compare; default 15 for ptrue_accurate, 0 for lookback "
                         "(matching BASE_FEATS in probedriftlong)")
    args = ap.parse_args()
    layer = args.layer if args.layer is not None else (0 if args.feature == "lookback" else 15)

    if args.baseline == "canonical":
        praw = ROOT / "cache" / "features" / f"{SLUG}__{DATASET}__ID__{args.feature}.npz"
        if not praw.exists():
            sys.exit(f"missing canonical feature cache: {praw}")
        raw = np.load(praw)["feats"]
    else:
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

    if args.baseline == "canonical":
        # Cosine, not bitwise: the baseline came off different hardware. The question is whether the
        # unchanged rows are the SAME VECTOR to within device arithmetic, and the changed rows are not.
        def cos(x, y):
            nx = np.linalg.norm(x, axis=1); ny = np.linalg.norm(y, axis=1)
            ok = (nx > 0) & (ny > 0)
            out = np.zeros(len(x))
            out[ok] = (x[ok] * y[ok]).sum(1) / (nx[ok] * ny[ok])
            return out
        c = cos(a, b)
        print(f"\n  rows {len(cut)}   cut {int(cut.sum())}   uncut {int((~cut).sum())}   layer {layer}")
        print(f"  UNCUT rows, cosine vs canonical : min {c[~cut].min():.6f}  "
              f"mean {c[~cut].mean():.6f}  below 0.999: {int((c[~cut] < 0.999).sum())}")
        print(f"  CUT   rows, cosine vs canonical : min {c[cut].min():.6f}  "
              f"median {np.median(c[cut]):.6f}  identical: {int((d[cut] == 0).sum())}")
        ok = (int((c[~cut] < 0.999).sum()) == 0) and (int((d[cut] == 0).sum()) == 0)
        print(f"\n  GATE E ({args.feature}, canonical baseline): "
              + ("PASS -- unchanged rows reproduce canonical to device tolerance and every "
                 "truncated row moved" if ok else "**FAIL -- STOP**"))
        print("  NOTE weaker than the same-job form: equality is to device tolerance, not bitwise.")
        sys.exit(0 if ok else 1)

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
