"""Track B (Phase 4): does the Blondel differentiable soft-rank loss beat the pairwise loss for
weighted-MSP, ID and OOD?

Focused comparison, DELIBERATELY separate from contribution_ladder.py (which the refresh job imports) so
nothing here can perturb that run. For each ladder cell (ID -> SameTask -> LOO -> DiffTask) it trains
weighted-MSP (weight_mode=normalised) with loss="pairwise" and loss="blondel" on the SAME rows/seeds and
reports PRR side by side, plus a paired test-set bootstrap (blondel - pairwise). The plain MSP floor
(fair_floor = best of msp_sum/perplexity/msp_min) is carried as the reference to beat OOD.

Reuses the verified cell machinery from contribution_ladder (same get_training_spec splits, same
per-token cache, same paired_bootstrap) -- only the loss differs. CPU only (weighted-MSP trains small
MLPs; torchsort.soft_rank runs on CPU). Needs torchsort installed (pbs/torchsort_setup.pbs).

    python scripts/checks/weighted_msp_blondel.py --seeds 1,2,3
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402

from luq import cache, msp, results, weighted_msp  # noqa: E402
from probe_drift.ood_settings import get_training_spec  # noqa: E402
from aggregation_table import load_per_token, paired_bootstrap  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAB = "correctness"
LAYER = 15
EVALS = ["sciq", "trivia_qa", "pubmed_qa"]
CANDIDATE_SOURCES = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "med_quad", "samsum"]
SETTINGS = [("SameTask", "OOD_ONE_DATASET_SAME_TASK"), ("LOO", "OOD_LEAVE_ONE_OUT"),
            ("OneDatasetDiffTask", "OOD_ONE_DATASET_DIFF_TASK"), ("DiffTask", "OOD_DIFF_TASK")]


def sampled_train_idx(split, seed, cap):
    tr = np.where(split == "train")[0]
    if cap is None or cap >= len(tr):
        return tr
    return tr[np.random.RandomState(seed).permutation(len(tr))[:cap]]


def cells(sources):
    out = [("ID", X, [(X, None)]) for X in EVALS]
    for X in EVALS:
        for tag, setting in SETTINGS:
            spec = [(s, n) for s, n in get_training_spec(X, setting) if s in sources and s != X]
            if spec:
                out.append((tag, X, spec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--sources", default=",".join(CANDIDATE_SOURCES))
    ap.add_argument("--length-normalise", default="yes", choices=["yes", "no"])
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ln = args.length_normalise == "yes"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not weighted_msp._HAVE_TORCHSORT:
        sys.exit("torchsort not installed -> cannot run the blondel arm. Run pbs/torchsort_setup.pbs first.")
    print(f"device {device} | seeds {seeds} | length_normalise={ln} | torchsort ok", flush=True)

    PT = {}
    # Load EVAL TARGETS as well as sources: loading only --sources means an eval target absent from that
    # list is never loaded, yielding zero cells while still exiting 0 (the canonical_ladder/asqa silent
    # failure). Harmless when evals are already a subset. (2026-07-23)
    for d in sorted(set(args.sources.split(",")) | set(EVALS)):
        loaded = load_per_token(MODEL, d, LAYER, LAB)
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip", flush=True); continue
        states, split, y, _, records = loaded
        if np.isnan(y).any():
            print(f"  {d}: unlabelled -> skip", flush=True); continue
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} rows", flush=True)
    sources = set(PT)

    out_rows = []
    for rung, X, spec in cells(sources):
        if X not in PT:
            continue
        # per-seed PRR for each loss + per-seed test uncertainty vectors (for the paired bootstrap)
        prr = {"pairwise": [], "blondel": [], "fair_floor": []}
        unc = {"pairwise": [], "blondel": []}
        yte_ref = None
        for sd in seeds:
            train_rows = [(d, i) for d, cap in spec for i in sampled_train_idx(PT[d][1], sd, cap)]
            test_rows = [(X, i) for i in np.where(PT[X][1] == "test")[0]]
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], dtype=float)
            yte = np.array([y[i] for i in te_idx], dtype=float)
            yte_ref = yte
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]
            for loss in ("pairwise", "blondel"):
                u = np.asarray(weighted_msp.weighted_msp_unc(
                    states, records, y, tr_idx, te_idx, device, weight_mode="normalised",
                    length_normalise=ln, seed=sd, loss=loss), dtype=float)
                prr[loss].append(results.prr(yte, u)); unc[loss].append(u)
            # FAIR FLOOR (fixed 2026-07-22): the honest unsupervised bar is the BEST of the three
            # standard floors, not the bare non-length-normalised `sum`. Comparing against `sum` alone
            # overstates every win -- e.g. pubmed_qa sum=+0.202 but min=+0.371, and ASQA sum=+0.148 but
            # perplexity=+0.316. That is the same artefact that produced and then killed the cnn
            # headline (the project's working notes). probedriftlong.py has always used all three.
            _cands = {k: np.asarray([msp.msp_uncertainty(records[i]["token_logprobs"], k)
                                     for i in te_idx], dtype=float)
                      for k in ("sum", "perplexity", "min")}
            # PRE-REGISTERED msp_min bar (2026-07-24 meeting) -- replaces the rejected max-of-three.
            floor = _cands[msp.PRIMARY_FLOOR_AGG]
            floor_name = f"msp_{msp.PRIMARY_FLOOR_AGG}"
            prr["fair_floor"].append(results.prr(yte, floor))

        if yte_ref is None:
            continue
        srcs = "+".join(f"{d}:{c}" if c else d for d, c in spec)
        m = {k: (float(np.mean(v)), float(np.std(v))) for k, v in prr.items() if v}
        print(f"\n[{rung:9s}] eval={X} train={srcs}", flush=True)
        for k in ("pairwise", "blondel", "fair_floor"):
            if k in m:
                print(f"    weighted_msp[{k:8s}] {m[k][0]:+.3f} +/- {m[k][1]:.3f}", flush=True)
        # paired bootstrap: blondel vs pairwise, and each vs the floor
        avg = {k: np.mean(np.stack(unc[k]), axis=0) for k in unc if unc[k]}
        # FAIR FLOOR for the BOOTSTRAP as well. This is a SECOND, INDEPENDENT floor computation from the
        # per-seed one further up -- fixing only that one moved the reported floor PRR (pubmed 0.202 ->
        # 0.371) while leaving EVERY verdict still compared against bare msp_sum, byte-identical to before.
        # Caught only by diffing the new verdicts against the old ones. Both sites must use the fair floor.
        _te = np.where(PT[X][1] == "test")[0]
        _fc = {k: np.asarray([msp.msp_uncertainty(PT[X][3][i]["token_logprobs"], k)
                              for i in _te], dtype=float) for k in ("sum", "perplexity", "min")}
        floor_vec = _fc[max(_fc, key=lambda k: results.prr(yte_ref, _fc[k]))]
        for tag, a, b, av, bv in [("blondel_vs_pairwise", "blondel", "pairwise", avg.get("blondel"), avg.get("pairwise")),
                                  ("blondel_vs_floor", "blondel", "fair_floor", avg.get("blondel"), floor_vec),
                                  ("pairwise_vs_floor", "pairwise", "fair_floor", avg.get("pairwise"), floor_vec)]:
            if av is not None and bv is not None:
                mg, lo, hi, p, sig = paired_bootstrap(yte_ref, av, bv)
                print(f"    [verdict] {tag:20s} margin {mg:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} "
                      f"{'SIG' if sig else 'ns'}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"VERDICT:{tag}",
                                 "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                 "boot_p": round(p, 4), "significant": sig})
        for k in ("pairwise", "blondel", "fair_floor"):
            if k in m:
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"weighted_msp_{k}",
                                 "prr_mean": round(m[k][0], 4), "prr_std": round(m[k][1], 4),
                                 "n_seeds": len(prr[k])})

    out = ROOT / "results" / f"weighted_msp_blondel__{cache._slug(MODEL)}.csv"
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "train", "method", "prr_mean", "prr_std",
                                           "n_seeds", "ci_lo", "ci_hi", "boot_p", "significant"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
