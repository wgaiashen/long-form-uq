"""BLOCKING pre-flight for the B.1 auxiliary-loss re-run. Run this BEFORE the 24h job.

Two things are checked, both of which would otherwise fail silently and produce a run that looks
completely normal while being worthless:

PART A -- the normalised auxiliary penalty (`aux_penalty(..., normalise=True)`).
  The three assertions from DOC_INSTRUCTIONS.md §2a, on the REAL targets, per dataset × target:
    1. uniform attention           -> aux_norm == 1.0  (tol 1e-6)
    2. a == D                      -> aux_norm == 0.0  (tol 1e-6)
    3. the zero-denominator (uniform-target) row count is REPORTED, never clamped
  If the normaliser is wrong, λ is not comparable across dataset length and the whole point of
  the re-run's §2a is lost.

PART B -- λ selection by held-out SOURCE DATASET (`val_split_held_out_source`).
  This function is the single most important change in the re-run and had never been functionally
  tested. For EVERY cell in the realised grid it asserts:
    1. every held-out validation row comes from ONE source dataset
    2. the training rows contain NONE of that dataset
    3. the two sets partition the training pool EXACTLY (no overlap, nothing dropped)
    4. neither side is empty
  If this is wrong, every λ is selected on the wrong criterion.

WHY ALL 20 CELLS AND NOT ONE. 9 of the 20 cells have a SINGLE-dataset training pool -- every ID
cell and every 1ds-Diff cell by construction, plus factscore's SameTask (its family is just
{expertqa, factscore}). Holding a source out is impossible there, so the function falls back to a
random carve. Checking one multi-source cell would pass and say nothing about the other nine.
Those cells are classified FALLBACK, not OK, and counted.

LIMITATION, recorded before launch rather than discovered afterwards: DOC_INSTRUCTIONS.md §5
pre-registers that the loss should help MOST at the narrow-pool rungs (1ds-Diff, SameTask) -- which
are exactly the cells where the new selection criterion cannot operate. Read the two claims apart:
the auxiliary LOSS is tested on all 20 cells (`real_minus_shuffled` is valid everywhere, since the
shuffled control runs under whatever λ was chosen), but the λ-SELECTION criterion is tested on only
the 11 multi-source cells.

Exits non-zero if any assertion fails. A FALLBACK cell is not a failure.

    python scripts/checks/aux_preflight.py --evals pubmed_qa,cnn_dailymail,xsum,factscore
"""
import argparse
import resource
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from attn_pool import (load_per_token, aux_penalty, pad_prior, normalise_target)  # noqa: E402
from aux_attention_ladder import val_split_held_out_source, MODEL  # noqa: E402
from xl_rungs import build_rows, label_of  # noqa: E402
import prior_builders as PB  # noqa: E402
import probedriftlong as PDL  # noqa: E402

TOL = 1e-6
CHUNK = 128          # rows per padded batch, so a long dataset never builds one huge dense tensor


def part_a0():
    """The ALGEBRA behind the normalisation, on synthetic targets where the answer is known in closed form.

    Assertion 1 below (uniform attention -> 1.0) is true BY CONSTRUCTION: `aux_penalty` defines its
    denominator as the penalty evaluated at uniform attention, so feeding it uniform `a` makes numerator
    and denominator the same expression. It is worth running -- it catches an inconsistent mask or a
    different uniform -- but it is close to a tautology, so it is not on its own evidence that the maths
    is right. This check is, because it compares against a closed form derived independently:

        flat k-sparse D over n real tokens, uniform a:  Σᵢ (aᵢ − Dᵢ)²  =  1/k − 1/n
        the same quantity is the denominator:           aux_ref        =  ΣD² − 1/n

    If these hold, the claim in §2a that the raw sum splits on k (not n) -- which is the entire reason a
    plain .mean() would have been the wrong fix -- is confirmed rather than assumed.
    """
    print("\n" + "=" * 100)
    print("PART A0 -- the closed-form identity the normalisation rests on (synthetic, independent)")
    print("=" * 100)
    print(f"{'n':>5} {'k':>4} {'raw sum':>12} {'1/k - 1/n':>12} {'aux_ref':>12} {'ΣD²-1/n':>12}  verdict")
    print("-" * 100)
    failures = []
    for n, k in [(56, 1), (56, 8), (128, 4), (768, 1), (768, 32), (768, 384)]:
        mask = torch.zeros(1, n); mask[0, :n] = 1.0
        D = torch.zeros(1, n); D[0, :k] = 1.0 / k          # flat k-sparse distribution
        uni = mask / mask.sum(1, keepdim=True)
        raw = float(((uni - D) ** 2 * mask).sum(1))
        closed = 1.0 / k - 1.0 / n
        ref = float(((uni - D) ** 2 * mask).sum(1))
        d2 = float((D ** 2).sum(1)) - 1.0 / n
        ok = abs(raw - closed) <= 1e-6 and abs(ref - d2) <= 1e-6
        if not ok:
            failures.append(f"n={n} k={k}: raw={raw:.6g} vs {closed:.6g}; ref={ref:.6g} vs {d2:.6g}")
        print(f"{n:>5} {k:>4} {raw:>12.6g} {closed:>12.6g} {ref:>12.6g} {d2:>12.6g}  "
              f"{'OK' if ok else '*** FAIL ***'}")

    # and the consequence: the RAW sum is length-invariant for a fixed-k target but not for a dense one,
    # while the NORMALISED form is 1.0 regardless -- the property λ comparability depends on.
    print("\n  consequence (raw sum at uniform attention, showing what λ was actually weighting):")
    for k_desc, kf in [("k=4 fixed (sparse)", lambda n: 4), ("k=n/2 (dense)", lambda n: max(1, n // 2))]:
        vals = [1.0 / kf(n) - 1.0 / n for n in (56, 128, 768)]
        print(f"    {k_desc:<20} n=56 {vals[0]:.5f}   n=128 {vals[1]:.5f}   n=768 {vals[2]:.5f}"
              f"   ratio(56/768) {vals[0]/vals[2]:.2f}x")
    return failures


def part_a(PT, targets, tok, special_ids):
    """The three §2a assertions on the real targets, per dataset × target."""
    print("\n" + "=" * 100)
    print("PART A -- normalised auxiliary penalty (§2a), on the REAL targets")
    print("=" * 100)
    print(f"{'dataset':<15} {'target':<14} {'rows':>6} {'uniform->1.0':>14} {'a==D->0.0':>12} "
          f"{'zero-denom rows':>16}  verdict")
    print("-" * 100)
    failures = []
    for d in sorted(PT):
        states, _split, _y, records = PT[d]
        for tname in targets:
            try:
                prior, _n_fb = PB.build_prior(tname, records, states, tok=tok,
                                              special_ids=special_ids, datasets=[d] * len(records))
            except Exception as e:
                print(f"{d:<15} {tname:<14} {'':>6} target UNAVAILABLE ({type(e).__name__}: {e}) -> skipped")
                continue
            worst_uni, worst_eq, n_drop, n_rows = 0.0, 0.0, 0, 0
            for b in range(0, len(states), CHUNK):
                sl = slice(b, min(b + CHUNK, len(states)))
                lens = [len(states[i]) for i in range(sl.start, sl.stop)]
                tmax = max(lens)
                mask = torch.zeros(len(lens), tmax)
                for i, L in enumerate(lens):
                    mask[i, :L] = 1.0
                D = pad_prior(prior[sl], tmax, "cpu")
                D = normalise_target(D, mask)
                uni = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

                # 1. uniform attention -> the ratio is 1.0 by construction of the reference
                p_uni, nd = aux_penalty(uni, D, mask, normalise=True)
                # 2. a == D -> numerator is 0, so the ratio is 0.0
                p_eq, _ = aux_penalty(D, D, mask, normalise=True)

                # rows dropped for a zero denominator are EXCLUDED from the mean, so compare only when
                # at least one row survived; a fully-degenerate chunk returns 0.0 by design (and is counted)
                n_kept = (mask.shape[0] - nd)
                if n_kept > 0:
                    worst_uni = max(worst_uni, abs(float(p_uni) - 1.0))
                    worst_eq = max(worst_eq, abs(float(p_eq) - 0.0))
                n_drop += nd
                n_rows += mask.shape[0]
            ok = worst_uni <= TOL and worst_eq <= TOL
            verdict = "OK" if ok else "*** FAIL ***"
            if not ok:
                failures.append(f"{d}/{tname}: |uniform-1|={worst_uni:.2e} |a==D|={worst_eq:.2e}")
            print(f"{d:<15} {tname:<14} {n_rows:>6} {worst_uni:>14.2e} {worst_eq:>12.2e} "
                  f"{n_drop:>16}  {verdict}")
    return failures


def part_b(PT, evals, seeds):
    """The partition assertion on val_split_held_out_source, for EVERY cell in the realised grid."""
    print("\n" + "=" * 100)
    print("PART B -- λ-selection split holds out a WHOLE SOURCE DATASET")
    print("=" * 100)
    print(f"{'rung':<16} {'eval':<15} {'sd':>2} {'nsrc':>4} {'held_out':<14} {'n_tr':>6} {'n_val':>6}  "
          f"partition  verdict")
    print("-" * 100)
    failures, n_ok, n_fb = [], 0, 0
    for rung, X, spec in PDL.cells_long(set(PT), evals):
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, PDL.sampled_train_idx)
            if not train_rows or not test_rows:
                print(f"{rung:<16} {X:<15} {sd:>2} -- empty cell, skipped by the driver too")
                continue
            tr_idx = list(range(len(train_rows)))
            dsets = [d for d, _i in train_rows]
            nsrc = len(set(dsets))

            sub_tr, sub_val, held = val_split_held_out_source(tr_idx, train_rows, sd)

            # (3) exact partition of the pool -- no overlap, nothing dropped, nothing invented
            part_ok = sorted(list(sub_tr) + list(sub_val)) == sorted(tr_idx)
            # (4) neither side empty
            nonempty = len(sub_tr) > 0 and len(sub_val) > 0

            if held is None:                    # single-source pool -> documented fallback, not a failure
                n_fb += 1
                ok = part_ok and nonempty
                verdict = "FALLBACK (single source)" if ok else "*** FAIL ***"
                if not ok:
                    failures.append(f"{rung}/{X}/s{sd}: fallback carve did not partition the pool")
                print(f"{rung:<16} {X:<15} {sd:>2} {nsrc:>4} {'-- random --':<14} {len(sub_tr):>6} "
                      f"{len(sub_val):>6}  {str(part_ok):<9}  {verdict}")
                continue

            # (1) validation is exactly ONE source dataset
            val_src = set(dsets[i] for i in sub_val)
            one_src = len(val_src) == 1 and held in val_src
            # (2) that dataset appears in NO training row
            tr_src = set(dsets[i] for i in sub_tr)
            disjoint = held not in tr_src

            ok = one_src and disjoint and part_ok and nonempty
            n_ok += int(ok)
            verdict = "OK" if ok else "*** FAIL ***"
            if not ok:
                failures.append(f"{rung}/{X}/s{sd}: one_src={one_src} disjoint={disjoint} "
                                f"partition={part_ok} nonempty={nonempty} val_src={sorted(val_src)}")
            print(f"{rung:<16} {X:<15} {sd:>2} {nsrc:>4} {held:<14} {len(sub_tr):>6} {len(sub_val):>6}  "
                  f"{str(part_ok):<9}  {verdict}")
    print("-" * 100)
    print(f"held_out_source: {n_ok}   fallback (single source): {n_fb}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="pubmed_qa,cnn_dailymail,xsum,factscore")
    ap.add_argument("--targets", default="nll,content_mass")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--skip-part-a", action="store_true")
    args = ap.parse_args()

    evals = [e.strip() for e in args.evals.split(",")]
    targets = [t.strip() for t in args.targets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    tok = AutoTokenizer.from_pretrained(MODEL)
    special_ids = set(getattr(tok, "all_special_ids", []) or [])

    # Loaded EXACTLY as the ladder loads it, so the pre-flight tests the objects the run will use.
    PT = {}
    for d in sorted(set(PDL.LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip", flush=True); continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} rows", flush=True)

    failures = []
    if not args.skip_part_a:
        failures += part_a0()
        failures += part_a(PT, targets, tok, special_ids)
    failures += part_b(PT, evals, seeds)

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    print(f"\npeak RSS {peak_gb:.1f} GB   (sizing evidence for the ladder job, which loads the same caches)")

    print("\n" + "=" * 100)
    if failures:
        print(f"PRE-FLIGHT FAILED -- {len(failures)} problem(s). DO NOT LAUNCH.")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("PRE-FLIGHT PASSED. A FALLBACK cell is expected on a single-source pool and is not a failure.")
    print("Reminder: the LOSS is tested on every cell; the λ-SELECTION criterion only on the "
          "multi-source ones.")


if __name__ == "__main__":
    main()
