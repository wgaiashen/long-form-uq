"""B.1 — screen auxiliary-loss attention supervision on the ProbeDriftLong population.

The method: add the AAAI-22 term to the LOSS, L = L_task + (λ/H)·Σ_i (a_i − d_i)², so the attention is
TRAINED to match a target distribution and moves away smoothly when the constraint lifts. This is a
different mechanism from the S3 prior arms, which modify the attention SCORE.

⚠️ THE REPORTED QUANTITY IS (real target − SHUFFLED target), NOT (real − baseline). In the paper the
shuffled version performed worse than no supervision at all, so the shuffled arm is what separates "the
target carries useful information" from "constraining the attention regularises it". Reporting the raw
improvement would confuse the two.

⚠️ HOW TO JUDGE THIS (decided in advance): on CONSISTENCY ACROSS CELLS, not on the size of the mean
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
_FIELDS = ["rung", "eval", "seed", "target", "K", "J_supervised", "head_attn_corr", "best_lambda",
           "drop_epoch", "prr_baseline", "prr_real", "prr_shuffled", "real_minus_shuffled",
           "real_minus_baseline", "n_target_fallback", "n_test"]


def _out_path(args):
    from pathlib import Path as _P
    return _P(args.out) if args.out else ROOT / "results" / f"aux_attention{regime_tag()}__{SLUG}.csv"


def _flush_rows(out, rows):
    """Atomic per-cell write (temp + os.replace).

    ⚠️ This driver ALSO wrote only at the very end until 2026-08-05. On that date nine 8-hour top-k jobs
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


def val_split_lodo(tr_idx, train_rows, seed):
    """λ-selection split that holds out a WHOLE SOURCE DATASET, not a random slice.

    ⚠️ THIS IS THE MOST IMPORTANT CHANGE IN THE B.1 RE-RUN. The original screen carved a random 20% of the
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
        raise SystemExit(f"val_split_lodo: holding out {held} left {len(sub_tr)} train / {len(sub_val)} val")
    return sub_tr, sub_val, held


def _val_split_random(tr_idx, seed):
    """Carve a validation slice from TRAIN for λ selection. Never touches test."""
    perm = np.random.RandomState(seed).permutation(len(tr_idx))
    n_val = max(1, int(round(len(tr_idx) * VAL_FRAC)))
    return [tr_idx[i] for i in perm[n_val:]], [tr_idx[i] for i in perm[:n_val]]


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
                # λ selected by holding out a WHOLE SOURCE DATASET (see val_split_lodo). Same-dataset
                # validation structurally cannot see an OOD-robustness gain bought at an ID cost.
                sub_tr, sub_val, held_out = val_split_lodo(tr_idx, train_rows, sd)
                if held_out is None:
                    print(f"  [{rung}/{X}] λ-selection FELL BACK to a random carve (single-source pool; "
                          "an ID cell cannot pose an OOD question) — read this λ as the weaker criterion",
                          flush=True)
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
                    # ⭐ B.2 PRIMARY DIAGNOSTIC. Not PRR. The question is whether supervising a SUBSET of
                    # heads prevents the collapse we already measured at 0.996-1.000. If the heads still
                    # collapse, the PRR is uninformative -- a difference between arms with identical
                    # attention is a difference in the classifier, not the aggregation.
                    corr = (head_attention_correlation(m_real, states, te_idx, device)[0]
                            if args.n_query > 1 else float("nan"))
                    rows.append({"rung": rung, "eval": X, "seed": sd, "target": tname,
                                 "K": args.n_query, "J_supervised": J,
                                 "head_attn_corr": (round(corr, 4) if corr == corr else ""),
                                 "best_lambda": best_lam, "drop_epoch": args.drop_epoch,
                                 "prr_baseline": round(base, 4), "prr_real": round(real, 4),
                                 "prr_shuffled": round(shuf, 4),
                                 "real_minus_shuffled": round(real - shuf, 4),
                                 "real_minus_baseline": round(real - base, 4),
                                 "n_target_fallback": int(n_fb), "n_test": len(te_idx)})
                    ctxt = f"corr {corr:+.4f}  " if corr == corr else ""
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
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
