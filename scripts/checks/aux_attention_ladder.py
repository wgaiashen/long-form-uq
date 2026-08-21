"""B.1 — screen auxiliary-loss attention supervision on the ProbeDriftLong population.

The method: add the AAAI-22 term to the LOSS, L = L_task + (λ/H)·Σ_i (a_i − d_i)², so the attention is
TRAINED to match a target distribution and moves away smoothly when the constraint lifts. This is a
different mechanism from the S3 prior arms, which modify the attention SCORE.

THE REPORTED QUANTITY IS (real target − SHUFFLED target), NOT (real − baseline). In the paper the
shuffled version performed worse than no supervision at all, so the shuffled arm is what separates "the
target carries useful information" from "constraining the attention regularises it". Reporting the raw
improvement would confuse the two.

HOW TO JUDGE THIS (decided in advance): on CONSISTENCY ACROSS CELLS, not on the size of the mean
improvement. Only the ID cells of the non-regenerating datasets are permanent — pubmed/asqa/expertqa/
factscore keep their generations, but their SameTask/LOO/DiffTask cells all train on pools containing
med_quad, samsum or cnn, so those move when v2 lands. "Helps on k of N cells, including the ID cells"
survives regeneration; "improves the mean by X" does not. Both are reported; the carry-forward decision
is on the count.

Pre-registered prediction: helps most on pubmed and med_quad (concentrated error signal, high
punctuation mass in the attention) and does little on xsum and cnn (already content-dominated).
UNIFORM IMPROVEMENT EVERYWHERE IS SUSPICIOUS and should be treated as a bug until checked.

    python scripts/checks/aux_attention_ladder.py --evals pubmed_qa,xsum --rungs ID --lambdas 0.2,1.0,1.8
    python scripts/checks/aux_attention_ladder.py --targets nll,content_mass --seeds 1,2,3
"""
import argparse
import os
import csv as _csv


# Column order is FIXED here rather than taken from `rows[0]`. Deriving it from the first row meant the
# header depended on whichever cell happened to finish first, and crashed outright on an empty run.
_FIELDS = ["rung", "eval", "seed", "target", "K", "J_supervised", "head_selection",
           "head_attn_corr", "head_corr_sup_free", "head_corr_within_sup", "head_corr_within_free",
           "best_lambda", "lambda_selection", "held_out_dataset", "n_pool_sources",
           "drop_epoch", "prr_baseline", "prr_real", "prr_shuffled", "real_minus_shuffled",
           "real_minus_baseline", "n_target_fallback", "n_test"]
# `lambda_selection` records WHICH criterion picked this row's λ: `held_out_source` (a whole source dataset
# was held out) or `same_dataset_slice` (a random carve, the only option when the pool has ONE source --
# every ID cell and every 1ds-Diff cell, by construction, plus factscore's SameTask whose family is just
# {expertqa, factscore}). 9 of the 20 cells in this run are single-source.
#
# READ THE TWO CLAIMS SEPARATELY. The auxiliary LOSS is tested on all 20 cells -- `real_minus_shuffled`
# is valid everywhere, because the shuffled control is run under whatever λ was chosen. The new
# λ-SELECTION criterion is tested on only the 11 multi-source cells. A fallback row is evidence about the
# loss, just not about the criterion, so never pool the two when reporting the criterion.
#
# LIMITATION FOUND BEFORE LAUNCH (2026-08-05): §5 pre-registers that the loss should help MOST at the
# narrow-pool rungs (1ds-Diff, SameTask) -- which are exactly the cells that cannot hold a source out.
# The rungs the prediction leans on are the ones the new criterion cannot reach. Recorded here rather
# than discovered in the results.
#
# ─────────────────────────────────────────────────────────────────────────────────────────────────────
# HEAD-SUBSET ARM: `head_selection` is ALWAYS "arbitrary_J", never "greedy". The paper picks the top J
# heads by supervising each individually and ranking them. That is NOT what runs here, and the honest
# label is more informative than the paper's would be:
#
#   Greedy per-head ranking was not implemented. At zero-init the heads are exchangeable, so first-J is
#   an arbitrary-J draw; B.2 established that the heads do not diverge under a shared training
#   condition, so a ranking would have nothing to sort.
#
# The paper's rationale does not transfer. Its 12 heads sit inside a FINE-TUNED BERT -- trained, and so
# specialised, which makes ranking them a ranking of genuinely different objects. Ours sit on a FROZEN
# Llama, are initialised identically, and B.2 measured them collapsing to 0.996-1.000 correlation
# whenever they share a training condition. Implementing greedy would sort objects already measured to
# be indistinguishable. Put this line in every caption for a K>1 row.
#
# THE GUARD (`head_corr_sup_free`). The arm exists to test the paper's STRUCTURAL claim: that leaving
# some heads free while supervising others beats supervising all of them. That comparison only means
# something if the free heads actually diverge from the supervised ones under this loss -- which B.2
# says they may not. So the supervised x free correlation block is reported ALONGSIDE the PRR.
#
#   PRE-REGISTERED READING RULE: if head_corr_sup_free comes back at ~0.99, this arm did NOT test the
#   paper's claim. It re-confirmed that the heads will not diverge, and must be written up that way --
#   NOT as "subset-of-heads does not help". Same numbers, two very different conclusions.
#
# Blank (not 0) when J==0 or J==K: there is no supervised/free split to measure, and a blank reads as
# not-measured while a 0 would read as measured-and-uncorrelated.


def _out_path(args):
    from pathlib import Path as _P
    return _P(args.out) if args.out else ROOT / "results" / f"aux_attention{regime_tag()}__{SLUG}.csv"


def _flush_rows(out, rows):
    """Atomic per-cell write (temp + os.replace).

    This driver ALSO wrote only at the very end until 2026-08-05. On that date nine 8-hour top-k jobs
    were killed at the walltime having written nothing, because `fixed_prior_ladder` had the same defect.
    All four ladder drivers were supposedly fixed on 3 August; this one and fixed_prior_ladder were both
    missed. Checked and fixed here BEFORE the auxiliary-loss re-run rather than after losing another run.
    """
    tmp = str(out) + ".partial"
    with open(tmp, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from luq import results  # noqa: E402
from aggregation_table import attn_unc, prr_from_conf  # noqa: E402
from attn_pool import (load_per_token, train_attn, select_temperature, regime_tag,  # noqa: E402
                       head_attention_correlation)  # noqa: E402
from xl_rungs import build_rows, eval_split, label_of  # noqa: E402
import prior_builders as PB  # noqa: E402
import probedriftlong as PDL  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
# the paper swept [0.2, 1.8] in steps of 0.2 and found 1.0 best for BERT, 0.8 for DeBERTa
LAMBDAS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
VAL_FRAC = 0.2


def val_split_held_out_source(tr_idx, train_rows, seed):
    """λ-selection split that holds out a WHOLE SOURCE DATASET, not a random slice.

    THIS IS THE MOST IMPORTANT CHANGE IN THE B.1 RE-RUN. The original screen carved a random 20% of the
    training pool, so the validation rows came from the SAME datasets as the training rows. A loss whose
    benefit is out-of-distribution robustness -- bought, as in the paper, at a small in-distribution cost --
    is invisible to that criterion: same-distribution validation sees only the cost. Selection duly chose
    λ=0 once it was offered, and that was recorded as evidence the method does nothing. It is equally
    consistent with the selection rule being unable to see the benefit.

    Here one source dataset is held out entirely (rotated by seed, so three seeds probe three different
    held-out sources), which makes the selection criterion a genuine shift -- the thing λ is meant to buy.

    Falls back to the random carve ONLY when the pool has a single source (every ID cell, by construction),
    and says so LOUDLY rather than silently degrading: an ID cell cannot pose an out-of-distribution
    question, so its λ is selected on the weaker criterion and must be read as such.

    Returns (sub_train_idx, sub_val_idx, held_out_name).
    """
    datasets = [d for d, _i in train_rows]
    uniq = sorted(set(datasets))
    if len(uniq) < 2:
        sub_tr, sub_val = _val_split_random(tr_idx, seed)
        return sub_tr, sub_val, None                      # caller prints the fallback
    held = uniq[seed % len(uniq)]
    sub_tr = [i for i in tr_idx if datasets[i] != held]
    sub_val = [i for i in tr_idx if datasets[i] == held]
    if not sub_tr or not sub_val:                          # cannot happen with >=2 sources; refuse to guess
        raise SystemExit(f"val_split_held_out_source: holding out {held} left {len(sub_tr)} train / {len(sub_val)} val")
    return sub_tr, sub_val, held


def _val_split_random(tr_idx, seed):
    """Carve a validation slice from TRAIN for λ selection. Never touches test."""
    perm = np.random.RandomState(seed).permutation(len(tr_idx))
    n_val = max(1, int(round(len(tr_idx) * VAL_FRAC)))
    return [tr_idx[i] for i in perm[n_val:]], [tr_idx[i] for i in perm[:n_val]]


def _blockmeans(M, J, K):
    """Split the K x K head-correlation matrix into supervised (0..J-1) x free (J..K-1) blocks.

    THE POINT OF THE ARM. Supervising a SUBSET only differs from supervising ALL if the free heads end up
    doing something different from the supervised ones. This returns the number that says whether they
    did. `head_attention_correlation` already computes the full matrix and the driver used to throw it
    away, so nothing new is measured here -- it is sliced.

    Returns (sup_x_free, within_sup, within_free), each None when that block does not exist. None ->
    written as BLANK, never 0: "no supervised/free split" and "split exists and is uncorrelated" must not
    look the same in the CSV.
    """
    import numpy as _np
    if M is None or J is None or not (0 < J < K):     # J=0 (all free) or J=K (all supervised): no split
        return None, None, None
    sup, free = list(range(J)), list(range(J, K))
    sup_free = float(_np.nanmean(M[_np.ix_(sup, free)]))

    def _within(g):                                   # off-diagonal mean within one group; needs >=2 heads
        if len(g) < 2:
            return None
        sub = M[_np.ix_(g, g)]
        off = ~_np.eye(len(g), dtype=bool)
        return float(_np.nanmean(sub[off]))
    return sup_free, _within(sup), _within(free)


def _r4(x):
    """Round for the CSV, but keep None/NaN as BLANK rather than letting them become a number."""
    return "" if x is None or x != x else round(x, 4)


def fit_and_score(states, y, tr, te, device, seed, best_T, return_model=False, **kw):
    m = train_attn(states, y, tr, device, seed=seed, temperature=best_T, **kw)
    prr = float(prr_from_conf(np.array([y[i] for i in te], float),
                              -np.asarray(attn_unc(m, states, te, device), float)))
    return (prr, m) if return_model else prr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default=",".join(PDL.LONG))
    ap.add_argument("--rungs", default=None, help="base rung names, e.g. ID,LOO. Default = all")
    ap.add_argument("--targets", default="nll,content_mass", help="nll|content_mass|orgad")
    ap.add_argument("--lambdas", default=",".join(str(x) for x in LAMBDAS))
    ap.add_argument("--drop-epoch", type=int, default=None,
                    help="strong-then-removed schedule: drop the aux term after N epochs (default: keep)")
    ap.add_argument("--n-query", type=int, default=1,
                    help="K attention heads (B.2). K=1 is the single-head pooler, unchanged.")
    ap.add_argument("--aux-heads", default=None,
                    help="B.2: comma-separated J values -- supervise only the first J of K heads, leaving "
                         "the rest free. The paper supervised 3 of 12 and found supervising ALL was worse "
                         "than a subset. J=0 is the unsupervised multi-head baseline. Requires --n-query>1.")
    ap.add_argument("--aux-normalise", action="store_true",
                    help="divide the auxiliary penalty by its value at UNIFORM attention (see "
                         "attn_pool.aux_penalty). Makes lambda comparable across dataset length AND target "
                         "density; without it a single lambda supervises a 56-token xsum answer far more "
                         "strongly than a 768-token med_quad one. Default off = the paper's raw sum, "
                         "byte-identical to the original B.1 screen.")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    evals = [e.strip() for e in args.evals.split(",")]
    lambdas = [float(x) for x in args.lambdas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    targets = [t.strip() for t in args.targets.split(",")]
    want_rungs = set(s.strip() for s in args.rungs.split(",")) if args.rungs else None
    # B.2: K heads with only J supervised. K=1 keeps the single-head path byte-identical (Js=[None],
    # mh_kw={} -> train_attn never sees n_query/aux_heads), so B.1's numbers are reproducible from here.
    if args.aux_heads is not None and args.n_query < 2:
        raise SystemExit("--aux-heads needs --n-query > 1; supervising J of 1 head is just B.1.")
    mh_kw = {"n_query": args.n_query, "n_head": args.n_query} if args.n_query > 1 else {}
    Js = ([int(j) for j in args.aux_heads.split(",")] if args.aux_heads
          else ([args.n_query] if args.n_query > 1 else [None]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} evals={evals} targets={targets} lambdas={lambdas} seeds={seeds}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    special_ids = set(getattr(tok, "all_special_ids", []) or [])

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

    rows = []
    for rung, X, spec in PDL.cells_long(set(PT), evals):
        if want_rungs is not None and rung.replace("-long", "") not in want_rungs:
            continue
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, PDL.sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows)
            tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            states = [PT[d][0][i] for d, i in allrows]
            recs = [PT[d][3][i] for d, i in allrows]
            dsets = [d for d, _ in allrows]
            best_T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
            base = fit_and_score(states, y, tr_idx, te_idx, device, sd, best_T)

            for tname in targets:
                try:
                    tgt, n_fb = PB.build_prior(tname, recs, states, tok=tok,
                                               special_ids=special_ids, datasets=dsets)
                except Exception as e:            # a missing prior is reported, never silently uniform
                    print(f"  [{rung}/{X}] target {tname}: UNAVAILABLE ({type(e).__name__}: {e}) -> cell "
                          "left BLANK", flush=True)
                    continue
                # λ selected by holding out a WHOLE SOURCE DATASET (see val_split_held_out_source). Same-dataset
                # validation structurally cannot see an OOD-robustness gain bought at an ID cost.
                sub_tr, sub_val, held_out = val_split_held_out_source(tr_idx, train_rows, sd)
                n_pool_sources = len(set(d for d, _i in train_rows))
                if held_out is None:
                    print(f"  [{rung}/{X}] λ-selection FELL BACK to a random carve (single-source pool: "
                          f"{sorted(set(d for d, _i in train_rows))}) — this row's λ comes from the WEAKER "
                          "criterion; `real_minus_shuffled` still tests the loss, but the row is not "
                          "evidence about held-out-source selection", flush=True)
                # λ is selected on the SAME architecture it will be used with (mh_kw threaded through),
                # otherwise K=4 runs would inherit a λ tuned on a 1-head model. Selection supervises all
                # K heads; J is varied afterwards, so λ is not tuned per-J -- stated rather than hidden,
                # and it is the conservative direction (J<K is not given its own tuned λ).
                curve = {lam: fit_and_score(states, y, sub_tr, sub_val, device, sd, best_T,
                                            aux_target=tgt, aux_lambda=lam,
                                            aux_drop_epoch=args.drop_epoch, aux_normalise=args.aux_normalise, **mh_kw)
                         for lam in lambdas}
                best_lam = max(curve, key=curve.get)
                for J in Js:
                    kwJ = dict(aux_lambda=best_lam, aux_drop_epoch=args.drop_epoch, aux_normalise=args.aux_normalise, **mh_kw)
                    if mh_kw:
                        kwJ["aux_heads"] = J
                    real, m_real = fit_and_score(states, y, tr_idx, te_idx, device, sd, best_T,
                                                 return_model=True, aux_target=tgt, **kwJ)
                    shuf = fit_and_score(states, y, tr_idx, te_idx, device, sd, best_T,
                                         aux_target=tgt, aux_shuffle=True, **kwJ)
                    # B.2 PRIMARY DIAGNOSTIC. Not PRR. The question is whether supervising a SUBSET of
                    # heads prevents the collapse we already measured at 0.996-1.000. If the heads still
                    # collapse, the PRR is uninformative -- a difference between arms with identical
                    # attention is a difference in the classifier, not the aggregation.
                    if args.n_query > 1:
                        corr, cmat = head_attention_correlation(m_real, states, te_idx, device)
                    else:
                        corr, cmat = float("nan"), None
                    # supervised x free block -- the guard that says whether this arm tested the paper's
                    # structural claim at all (see the header note on the pre-registered reading rule)
                    c_sf, c_ws, c_wf = _blockmeans(cmat, J, args.n_query)
                    rows.append({"rung": rung, "eval": X, "seed": sd, "target": tname,
                                 "K": args.n_query, "J_supervised": J,
                                 # never "greedy": the paper's ranking step is not implemented, and at
                                 # zero-init the heads are exchangeable so first-J is an arbitrary draw
                                 "head_selection": ("arbitrary_J" if args.n_query > 1 else ""),
                                 "head_attn_corr": _r4(corr),
                                 "head_corr_sup_free": _r4(c_sf),
                                 "head_corr_within_sup": _r4(c_ws),
                                 "head_corr_within_free": _r4(c_wf),
                                 "best_lambda": best_lam,
                                 "lambda_selection": ("same_dataset_slice" if held_out is None
                                                      else "held_out_source"),
                                 "held_out_dataset": (held_out or ""),
                                 "n_pool_sources": n_pool_sources,
                                 "drop_epoch": args.drop_epoch,
                                 "prr_baseline": round(base, 4), "prr_real": round(real, 4),
                                 "prr_shuffled": round(shuf, 4),
                                 "real_minus_shuffled": round(real - shuf, 4),
                                 "real_minus_baseline": round(real - base, 4),
                                 "n_target_fallback": int(n_fb), "n_test": len(te_idx)})
                    ctxt = f"corr {corr:+.4f}  " if corr == corr else ""
                    if c_sf is not None:
                        ctxt += f"sup|free {c_sf:+.4f}  "
                    print(f"  [{rung:14s}] {X:<13} {tname:<12} s{sd} K={args.n_query} J={J} "
                          f"λ={best_lam:.2f}  {ctxt}base {base:+.3f} real {real:+.3f} shuf {shuf:+.3f}  |  "
                          f"real-shuf {real-shuf:+.3f}", flush=True)

        # PER-CELL FLUSH -- see _flush_rows. A walltime kill now costs the cell in progress, not the run.
        if rows:
            _flush_rows(_out_path(args), rows)
            print(f"    [saved] {len(rows)} rows -> {_out_path(args).name}", flush=True)

    if not rows:
        raise SystemExit("no cells produced -- nothing to report")
    _flush_rows(_out_path(args), rows)

    print("\n=== SUMMARY (the decisive column is real_minus_shuffled) ===")
    for tname in targets:
        sub = [r for r in rows if r["target"] == tname]
        if not sub:
            continue
        rs_ = np.array([r["real_minus_shuffled"] for r in sub])
        rb = np.array([r["real_minus_baseline"] for r in sub])
        idc = [r for r in sub if r["rung"] == "ID"]
        print(f"{tname:<12} n={len(sub):>3}  real-shuf mean {rs_.mean():+.4f}, POSITIVE on "
              f"{int((rs_ > 0).sum())}/{len(rs_)}  |  real-base mean {rb.mean():+.4f}, "
              f"positive on {int((rb > 0).sum())}/{len(rb)}"
              + (f"  |  ID cells: {sum(1 for r in idc if r['real_minus_shuffled'] > 0)}/{len(idc)}"
                 if idc else ""))
    print("\nCarry-forward decision is on the COUNT (and the ID cells), not the mean -- only ID cells of "
          "the non-regenerating datasets are permanent.")
    print(f"wrote {_out_path(args)}")


if __name__ == "__main__":
    main()
