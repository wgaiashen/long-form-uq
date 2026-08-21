#!/usr/bin/env python
"""Is a cell's PRR measuring uncertainty, or is it detecting malformed text? (the multi-model panel registration, deviation 6)

THE PROBLEM. `02_label_expertqa.py` and `02_label_factscore.py` apply a deterministic distrust rule:
a generation that `luq.degeneracy.is_severe()` flags is NEVER sent to the judge and is written
`factuality = 0.0, coherent = False, factuality_quarantined = True`; a judge verdict of
`coherent = false` is likewise forced to 0.0 after the call. For those rows the label is a function
of the TEXT, not a measurement of factuality.

That matters because any uncertainty score which happens to fire on malformed text earns PRR on
those rows for free. The cell's number is then part uncertainty estimate and part degeneracy
detector, and nothing in the headline PRR separates the two.

WHAT THIS REPORTS, per method per cell:

  1. `prr_full`      PRR on the whole eval set -- the number that currently gets quoted.
  2. `prr_clean`     PRR with every text-determined row removed. If the gain lives in the gap
                     between this and `prr_full`, the method was reading degeneracy.
  3. `prr_vs_quar`   PRR of the method scored against the QUARANTINE INDICATOR as the target,
                     i.e. how well the score alone predicts "this text is malformed". A high value
                     here is a direct measure of the confound, independent of (1) and (2).

ALIGNMENT IS ASSERTED, NOT ASSUMED. The sidecars carry no record index, so the mapping from sidecar
row to record is reconstructed through the SAME calls the ladder used (`records` -> finite-label
filter -> `xl_rungs.eval_split`). The reconstructed label vector is then compared element-by-element
against the sidecar's own `y`, and the script aborts on any mismatch. Silently mis-aligning two
vectors of the same length is exactly the failure that returns a plausible number for the wrong rows.

    python scripts/checks/degeneracy_prr_diagnostic.py --model google/gemma-2-9b --datasets factscore
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache                                   # noqa: E402
from luq.config import Config                           # noqa: E402
from luq.results import prr                             # noqa: E402
from xl_rungs import eval_split, label_of, CARVE        # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROMPT_REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}
RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]


def load_eval_rows(model, dataset):
    """(orig_idx, y, records) for the cell's EVAL rows, reproducing the ladder's own path."""
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    records = cache.load_records(cfg.cache_dir, cache.run_key(model, dataset, "ID"))
    field = label_of(dataset)
    y = np.array([r.get(field, np.nan) for r in records], dtype=float)
    split = np.array([r["split"] for r in records])
    orig = np.arange(len(records))
    finite = np.isfinite(y)
    if not finite.all():                                # the ladder drops unlabelled rows first
        keep = np.where(finite)[0]
        y, split, orig = y[keep], split[keep], orig[keep]
    _, te = eval_split(split)
    return orig[te], y[te], records, field


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-2-9b")
    ap.add_argument("--datasets", default="factscore",
                    help="only datasets whose labeller applies the distrust rule are meaningful "
                         "here: factscore, expertqa")
    ap.add_argument("--perex-dir", default=None, help="default results/perex_wmodels_lam3/<slug>")
    args = ap.parse_args()

    slug = args.model.replace("/", "_")
    pdir = Path(args.perex_dir) if args.perex_dir else ROOT / "results" / "perex_wmodels_lam3" / slug
    print(f"=== degeneracy-only PRR diagnostic ===\n  model {args.model}\n  perex {pdir}\n"
          f"  carve {CARVE}\n")

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        orig_te, y_te, records, field = load_eval_rows(args.model, ds)

        # The two text-determined label classes. `factuality_quarantined` means the judge was never
        # called; `coherent is False` means it was called and returned a collapse verdict. Both are
        # forced to 0.0 from the text, so both are confounded for this purpose.
        quar = np.array([bool(records[i].get("factuality_quarantined")) for i in orig_te])
        incoh = np.array([records[i].get("coherent") is False for i in orig_te])
        bad = quar | incoh

        print(f"--- {ds}  (label field '{field}', {len(y_te)} eval rows) ---")
        print(f"    quarantined (judge never called) {quar.sum():4d}"
              f"   coherent=false {incoh.sum():4d}"
              f"   text-determined total {bad.sum():4d}  ({100*bad.mean():.1f}%)")
        if bad.sum() == 0:
            print("    no text-determined rows in this eval split -> no confound to report\n")
            continue
        if bad.all():
            print("    EVERY eval row is text-determined -> prr_clean is undefined\n")
            continue

        for rung in RUNGS:
            p = pdir / f"{ds}__{rung}__{slug}.npz"
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=True)

            # HARD ALIGNMENT ASSERTION. Same length is not the same rows.
            if z["y"].shape != y_te.shape or not np.allclose(z["y"], y_te, atol=0, rtol=0):
                n_diff = (z["y"].shape != y_te.shape) or int((z["y"] != y_te).sum())
                print(f"    !!! {rung}: sidecar y does not match the reconstructed eval rows "
                      f"({z['y'].shape} vs {y_te.shape}, {n_diff} differing) -- ABORT, the mapping "
                      f"is wrong and any number from it would be for the wrong rows", file=sys.stderr)
                return 1

            methods = sorted(k[len("unc__"):] for k in z.keys() if k.startswith("unc__"))
            print(f"\n    [{rung}]  n={len(y_te)}  clean n={int((~bad).sum())}")
            print(f"      {'method':16s} {'prr_full':>9s} {'prr_clean':>10s} {'shift':>8s} {'prr_vs_quar':>12s}")
            for m in methods:
                u = z[f"unc__{m}"]
                if u.ndim > 1:                          # per-seed vectors -> average the PRRs, never the vectors
                    full = float(np.mean([prr(y_te, s) for s in u]))
                    clean = float(np.mean([prr(y_te[~bad], s[~bad]) for s in u]))
                    vsq = float(np.mean([prr(bad.astype(float), -s) for s in u]))
                else:
                    full, clean, vsq = prr(y_te, u), prr(y_te[~bad], u[~bad]), prr(bad.astype(float), -u)
                print(f"      {m:16s} {full:+9.4f} {clean:+10.4f} {clean-full:+8.4f} {vsq:+12.4f}")
        print()

    print("READING IT: `prr_vs_quar` is scored with the uncertainty NEGATED, so a positive value "
          "means\nhigh uncertainty coincides with malformed text -- the confound firing as expected. "
          "A large\nnegative `shift` means the method's headline PRR depends on the text-determined "
          "rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
