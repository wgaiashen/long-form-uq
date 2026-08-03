"""The contribution OOD ladder: weighted-MSP vs the attention pooler vs the MSP floor, across an
INCREMENTAL shift ladder (ID -> SameTask -> LOO -> DiffTask), all judge-labelled.

WHY A SEPARATE DRIVER (not ood_onegrid)
---------------------------------------
ood_onegrid.py is the verified BASELINE ladder (SAPLMA/linear/ptrue/lookback + poolers) and we leave
it untouched. This driver adds the two things the contribution needs and isolates their risk:
  * the two new methods -- weighted-MSP (Track 2) and the plain-MSP floor -- so a bug here cannot break
    the baseline ladder;
  * the SameTask rung (train on a same-task NEIGHBOUR dataset, e.g. med_quad for the QA evals), the mild
    shift step that ood_onegrid skips.

Every method here reads only the per-token states + the record logprobs, so a training SOURCE needs
only its pertok cache + records -- NOT the extra baseline features (lookback/ptrue). That is exactly
what lets med_quad (for which we extracted only generation + SAPLMA) slot in as a training source.

METHODS (all on the same per-token L15 states / same records, so the axis is aggregation only):
  uniform            frozen-query attention == mean-pool, torch linear head (the controlled baseline)
  attention          learned-query softmax attention, T selected on a val split
  weighted_msp_norm  learned per-token weight on NLL, softmax-normalised   (Track 2, the contribution)
  weighted_msp_unc   learned per-token weight on NLL, unconstrained
  msp_sum            plain MSP (unsupervised floor; shift-invariant)
  perplexity         length-normalised MSP (unsupervised floor)

RUNGS (get_training_spec-faithful, restricted to the sources we have a pertok cache for):
  ID        train on eval's own train split
  SameTask  OOD_ONE_DATASET_SAME_TASK  (QA -> med_quad)         <- the new mild rung
  LOO       OOD_LEAVE_ONE_OUT
  DiffTask  OOD_DIFF_TASK

PAIRED seeds + ID-diagonal gate, same discipline as ood_onegrid.

    python scripts/checks/contribution_ladder.py --seeds 1,2,3
    python scripts/checks/contribution_ladder.py --sources sciq,trivia_qa,pubmed_qa --seeds 1  # smoke
"""
import argparse
import os  # atomic replace in _flush_rows
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from luq import cache, msp, results, weighted_msp  # noqa: E402
from aggregation_table import load_per_token, build_arrays, attn_unc, paired_bootstrap  # noqa: E402
from attn_pool import train_attn, select_temperature  # noqa: E402
# Shared ProbeDriftXL rung machinery: makes med_quad/samsum/ExpertQA organic eval targets. `cells` dispatches
# keystone->get_training_spec (faithful), XL->family taxonomy; `eval_split` carves the XL eval test set;
# `label_of` gives the per-target label (ExpertQA=faithfulness); `different_label_projection` flags the ExpertQA OOD case.
from xl_rungs import cells as xl_cells, eval_split, label_of, different_label_projection, build_rows  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAB = "correctness"
EVALS = ["sciq", "trivia_qa", "pubmed_qa"]
# Candidate training sources; each included only if its pertok cache actually loads. (ExpertQA is EVAL-ONLY
# -- excluded as a source since its faithfulness label must not mix into the correctness training pool.)
# ASQA + ExpertQA included (author's decision 2026-07-22): they are ORDINARY training sources, not
# eval-only. Leaving them out here also silently starved canonical_ladder, which uses this as its
# --sources default. See the loading guard added there.
CANDIDATE_SOURCES = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum",
                     "expertqa", "asqa"]
# ID anchors (judge, from the aggregation table) the ID cells must reproduce.
ID_ANCHOR = {"sciq": {"uniform": 0.913, "attention": 0.932},
             "trivia_qa": {"uniform": 0.815, "attention": 0.844},
             "pubmed_qa": {"uniform": 0.683, "attention": 0.735}}
GATE_TOL = 0.03
# Rungs in ascending shift severity: SameTask (one same-family dataset) < LOO (everything-but-self,
# mixed) < OneDatasetDiffTask (one opposite-family dataset) < DiffTask (the whole opposite family).
# OneDatasetDiffTask self-skips until its single neighbour (samsum for QA evals) has a pertok cache.
SETTINGS = [("SameTask", "OOD_ONE_DATASET_SAME_TASK"),
            ("LOO", "OOD_LEAVE_ONE_OUT"),
            ("OneDatasetDiffTask", "OOD_ONE_DATASET_DIFF_TASK"),
            ("DiffTask", "OOD_DIFF_TASK")]
# Head-to-heads that get a paired test-set bootstrap CI (not just mean±std): does the contribution
# beat the OOD-robust floor, do the poolers, and does the contribution beat the pooler?
# The floor MUST be the FAIR floor = the best of the unsupervised baselines actually available, not the
# bare `msp_sum`. Hard-coding msp_sum understates the bar whenever the length-normalised floor is stronger,
# which is exactly the artefact that produced (and then killed) the cnn "win" -- see STOCKTAKE PART VI.
# Live example: on ASQA msp_sum=0.148 but perplexity=0.316, so every vs-msp_sum verdict was measured against
# less than half the honest bar (attention_vs_floor read +0.458 SIG at ID; against the fair floor it is
# +0.290, and at the two hardest OOD rungs EVERY supervised method is actually BELOW the floor).
# `fair_floor` is derived per cell below as the higher-PRR of {msp_sum, perplexity}. (Fixed 2026-07-22.)
FLOOR_CANDIDATES = ["msp_sum", "perplexity", "msp_min"]
COMPARISONS = [("wmsp_norm_vs_floor", "weighted_msp_norm", "fair_floor"),
               ("attention_vs_floor", "attention", "fair_floor"),
               ("wmsp_norm_vs_attention", "weighted_msp_norm", "attention")]


def sampled_train_idx(split, seed, cap):
    # Round-3 Task A (2026-07-27): eval-only sources (all split=="test") returned EMPTY here and silently
    # contributed 0 rows (V-A0). Source != eval is enforced by xl_cells, so draw from ALL rows when a source
    # has no dedicated train split. Core sets (with a real train split) are UNCHANGED.
    tr = np.where(split == "train")[0]
    if len(tr) == 0:
        tr = np.arange(len(split))
    if cap is None or cap >= len(tr):
        return tr
    return tr[np.random.RandomState(seed).permutation(len(tr))[:cap]]


CSV_FIELDS = ["rung", "eval", "train", "method", "prr_mean", "prr_std", "n_seeds",
              "different_label_projection", "eval_med_len", "train_med_len", "train_max_len",
              "ci_lo", "ci_hi", "boot_p", "significant"]


def _flush_rows(out, rows, fields):
    """Write everything accumulated SO FAR, atomically (temp + os.replace).

    CALLED PER CELL, NOT ONCE AT THE END (added 2026-08-03). This driver used to hold every row in
    memory and write the CSV only after the last cell, so a run killed at hour 15 of 16 -- walltime,
    OOM, or a node problem -- lost EVERYTHING and left nothing on disk saying which cells had already
    succeeded. A job was SIGTERM'd on cx3-14-9 the same day, and these ladders run 8-16 hours.

    temp-then-replace so a crash mid-write cannot leave a TRUNCATED csv, which would read as a
    short-but-valid grid. A half-written table is worse than no table: it looks complete.
    """
    tmp = Path(str(out) + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, out)


def main():
    global EVALS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--sources", default=",".join(CANDIDATE_SOURCES),
                    help="training sources to try; each used only if its pertok cache loads")
    ap.add_argument("--evals", default=",".join(EVALS),
                    help="eval targets (each needs a real ProbeDrift test split). ID_ANCHOR gate only "
                         "runs for anchored evals (sciq/trivia/pubmed); others just skip the gate.")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--length-normalise", default="yes", choices=["yes", "no"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # Resolved BEFORE the cell loop so the per-cell crash-safety flush has somewhere to write.
    out_path = Path(args.out) if args.out else (
        ROOT / "results" / f"contribution_ladder__{cache._slug(MODEL)}.csv")
    EVALS = args.evals.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    ln = args.length_normalise == "yes"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"device {device} | seeds {seeds} | length_normalise={ln}", flush=True)

    # Load per-token states + records for every SOURCE and every EVAL TARGET whose cache exists (tolerant).
    # Each dataset is scored on its own label (label_of): correctness for the core world, faithfulness for
    # ExpertQA. Partially-labelled sets (ExpertQA faithfulness has ~292 None) keep only their labelled rows.
    PT = {}
    for d in sorted(set(args.sources.split(",")) | set(EVALS)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip", flush=True)
            continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: fully unlabelled ({label_of(d)}) -> skip", flush=True)
            continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} rows loaded (label={label_of(d)}"
              f"{'' if finite.all() else f', {int(finite.sum())}/{len(finite)} labelled'})", flush=True)
    sources = set(PT)
    methods = ["uniform", "attention", "weighted_msp_norm", "weighted_msp_unc",
               "msp_sum", "perplexity", "msp_min"]

    out_rows = []
    for rung, X, spec in xl_cells(sources, EVALS):
        if X not in PT:
            continue
        # The eval target's FIXED train/test split: baked-in for the core datasets, a deterministic carve
        # for the split-less XL sets (so every method sees one stable ExpertQA/med_quad/samsum test set).
        X_tr, X_te = eval_split(PT[X][1])
        test_rows = [(X, int(i)) for i in X_te]
        if not test_rows:
            continue
        xlbl = different_label_projection(X)                 # ExpertQA OOD rungs are cross-label (correctness->faithfulness)
        per_method = {m: [] for m in methods}
        unc_acc = {m: [] for m in methods}   # per-seed per-example uncertainty vectors (for the bootstrap)
        yte_ref = None                        # test labels (identical across seeds; the bootstrap target)
        for sd in seeds:
            # Round-3 Task A (2026-07-27): use the SHARED build_rows (was an inline duplicate). It enforces
            # source != eval, lets eval-only sets draw all rows via the fixed sampler, and RAISES on a 0-row
            # named source (the silent-admission guard). test_rows it returns == test_rows above.
            train_rows, _ = build_rows(X, spec, PT, sd, sampled_train_idx)
            if not train_rows:
                continue
            n_tr = len(train_rows)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], dtype=float)
            yte = np.array([y[i] for i in te_idx], dtype=float)
            yte_ref = yte  # identical values every seed (test rows are fixed); kept for the bootstrap
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]

            # Each method emits a per-example uncertainty VECTOR over te_idx (higher = more uncertain);
            # we PRR it now and also stash it so the seed-averaged vector can feed a paired bootstrap.
            best_T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
            vecs = {}
            # poolers (uniform / attention) reuse the verified attn machinery
            vecs["uniform"] = np.asarray(attn_unc(
                train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True), states, te_idx, device), dtype=float)
            vecs["attention"] = np.asarray(attn_unc(
                train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T), states, te_idx, device), dtype=float)
            # weighted-MSP (contribution)
            vecs["weighted_msp_norm"] = np.asarray(weighted_msp.weighted_msp_unc(
                states, records, y, tr_idx, te_idx, device, weight_mode="normalised",
                length_normalise=ln, seed=sd), dtype=float)
            vecs["weighted_msp_unc"] = np.asarray(weighted_msp.weighted_msp_unc(
                states, records, y, tr_idx, te_idx, device, weight_mode="unconstrained",
                length_normalise=ln, seed=sd), dtype=float)
            # plain MSP floor (unsupervised -> identical across seeds/rungs, computed per cell for the table)
            vecs["msp_sum"] = np.asarray([msp.msp_uncertainty(records[i]["token_logprobs"], "sum")
                                          for i in te_idx], dtype=float)
            vecs["perplexity"] = np.asarray([msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity")
                                             for i in te_idx], dtype=float)
            # msp_min completes the standard floor trio. It is NOT optional: on pubmed_qa msp_min (+0.371)
            # is far the strongest floor, so a fair_floor built from {sum, perplexity} alone still
            # understates the bar there. See probedriftlong, which has used all three from the start.
            vecs["msp_min"] = np.asarray([msp.msp_uncertainty(records[i]["token_logprobs"], "min")
                                          for i in te_idx], dtype=float)
            for m, u in vecs.items():
                per_method[m].append(results.prr(yte, u))
                unc_acc[m].append(u)

        stats = {m: (float(np.mean(v)), float(np.std(v))) for m, v in per_method.items() if v}
        # HONEST LABEL (Round-3 Task A): build `train` from the REALISED per-source counts (last seed; counts
        # are seed-stable = min(cap, available)), NOT the requested caps -- so the label can never overstate
        # the pool (the V-A0 defect: `expertqa:360` while 0 were used). realised==labelled by construction.
        _real = {}
        for _d, _i in train_rows:
            _real[_d] = _real.get(_d, 0) + 1
        srcs = "+".join(f"{d}:{_real.get(d, 0)}" for d in dict.fromkeys(d for d, _c in spec))
        xflag = "  [CROSS-LABEL: train correctness -> test faithfulness]" if (xlbl and rung != "ID") else ""
        print(f"\n[{rung:9s}] eval={X} ({label_of(X)})  train={srcs}{xflag}", flush=True)
        for m in methods:
            if m in stats:
                print(f"    {m:18s} {stats[m][0]:+.3f} +/- {stats[m][1]:.3f}", flush=True)
        # ID-diagonal gate on the controlled poolers
        if rung == "ID" and X in ID_ANCHOR:
            for m in ("uniform", "attention"):
                d = abs(stats[m][0] - ID_ANCHOR[X][m])
                assert d < GATE_TOL, f"ID-GATE FAIL {X}/{m}: {stats[m][0]:.3f} vs {ID_ANCHOR[X][m]} (|d|={d:.3f})"
            print(f"    [ID-GATE OK] poolers reproduce anchors within {GATE_TOL}", flush=True)
        # LENGTH COLUMNS (Round-3 Task C): expose the two shifts "OOD" hides -- domain vs length. Computed over
        # the REALISED pool (last-seed train_rows; the widened Task-A pool should visibly raise train_max_len on
        # the long evals). Length = len(token_logprobs) (the canonical per-example gen length).
        _tl = [len(PT[d][3][i]["token_logprobs"]) for d, i in train_rows]
        _el = [len(PT[X][3][i]["token_logprobs"]) for _d, i in test_rows]
        _lens = {"eval_med_len": round(float(np.median(_el)), 1) if _el else "",
                 "train_med_len": round(float(np.median(_tl)), 1) if _tl else "",
                 "train_max_len": int(max(_tl)) if _tl else ""}
        for m in methods:
            if m in stats:
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": m,
                                 "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                                 "n_seeds": len(per_method[m]),
                                 "different_label_projection": (xlbl and rung != "ID"), **_lens})   # ExpertQA OOD = cross-label
        # Paired test-set bootstrap on the seed-averaged uncertainty vectors: turns the head-to-heads
        # (contribution vs floor, pooler vs floor, contribution vs pooler) into CI-backed verdicts.
        avg_unc = {m: np.mean(np.stack(unc_acc[m]), axis=0) for m in unc_acc if unc_acc[m]}
        # Derive the FAIR floor for this cell: whichever unsupervised baseline actually scores best. Both
        # its PRR row and its per-example vector are aliased, so the bootstrap compares against the real bar.
        avail = [f for f in FLOOR_CANDIDATES if f in stats and f in avg_unc]
        if avail:
            # PRE-REGISTERED primary bar = msp_min, FIXED across every dataset (2026-07-24 meeting). This
            # replaces the rejected max-of-three (Joe: "three shots for the baseline"). We still compute +
            # persist all three variants (methods list) so the per-dataset 3-variant table stays available,
            # and where a DIFFERENT variant is the strongest free score (cnn/samsum -> perplexity) we print a
            # DUAL-REPORT note so the honest "also clears the strongest free score" claim can be made.
            primary = f"msp_{msp.PRIMARY_FLOOR_AGG}"
            bar = primary if primary in avail else max(avail, key=lambda f: stats[f][0])
            strongest = max(avail, key=lambda f: stats[f][0])
            stats["fair_floor"] = stats[bar]
            avg_unc["fair_floor"] = avg_unc[bar]
            note = ("" if len(avail) == 1 else
                    "  [" + ", ".join(f"{f} {stats[f][0]:+.3f}" for f in avail) + "]")
            if strongest != bar:
                note += f"  (strongest free = {strongest} {stats[strongest][0]:+.3f}; DUAL-REPORT)"
            print(f"    primary_floor = {bar} ({stats[bar][0]:+.3f}){note}", flush=True)
            out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"fair_floor:{bar}",
                             "prr_mean": round(stats[bar][0], 4),
                             "prr_std": round(stats[bar][1], 4), "n_seeds": len(seeds)})
        for vk, a, b in COMPARISONS:
            if a in avg_unc and b in avg_unc and yte_ref is not None:
                mg, lo, hi, p, sig = paired_bootstrap(yte_ref, avg_unc[a], avg_unc[b])
                print(f"    [verdict] {vk:24s} margin {mg:+.3f}  95%CI [{lo:+.3f},{hi:+.3f}]  "
                      f"p={p:.3f} -> {'SIG' if sig else 'ns'}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"VERDICT:{vk}",
                                 "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                 "boot_p": round(p, 4), "significant": sig, "n_seeds": len(seeds)})
        # CRASH SAFETY: land this cell before starting the next. An interruption should cost the
        # current cell, never the whole run.
        _flush_rows(out_path, out_rows, CSV_FIELDS)
        print(f"    [saved] {len(out_rows)} rows -> {out_path.name}", flush=True)


    out = out_path
    _flush_rows(out, out_rows, CSV_FIELDS)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
