"""S7 — POOLING HEADS WITH DIFFERENT FIXED RECIPES (the ideas 3 + 4).

The design calls for several attention heads plus something to stop them converging. B.2 ran K heads sharing ONE
target and measured them collapsing to pairwise correlation 1.0000, so K heads only ever expressed two
behaviours ("has a target" / "has none"). This driver runs the version that CANNOT collapse: each frozen
head's attention IS a fixed recipe, so what makes the heads differ is not learned and no repulsion loss is
needed.

Arms per cell (all trained HERE, nothing inherited, so no cross-population comparison):

  floor_min          unsupervised floor, msp_min
  armA               single learned attention head              <- the bar being improved, and the
                                                                   reproduction gate against existing runs
  single_<recipe>    one frozen recipe, head-only training      <- each recipe alone (arm C equivalent)
  mh_diverse         3 DIFFERENT frozen recipes + 1 free head   <- THE METHOD
  mh_same            3 copies of ONE recipe + 1 free head       <- the mandatory control: isolates
                                                                   "more classifiers" from "different
                                                                   recipes". Same architecture, same
                                                                   parameter count, same free head.
  mh_diverse_nofree  3 different recipes, no free head          <- is the free head carrying the ensemble?

PRE-REGISTERED (prereg/multi_recipe_pooling_heads.md), thresholds fixed before running:
  PRIMARY   mh_diverse beats armA on the OOD rungs by > 0.02 (the aggregation-axis noise floor).
  SECONDARY mh_diverse beats mh_same by > 0.02. If not, any gain is "more classifiers", not "different
            recipes", and the line closes.
  DIAGNOSTIC head attention correlation, reported regardless. It must be LOW for mh_diverse by
            construction and ~1.0 for mh_same; if mh_diverse comes out near 1.0 the wiring is wrong.
  HANDICAP  B.2 measured multi-head 0.06-0.10 PRR BELOW single-head. Carried explicitly, not netted out.

The recipes and the control's recipe are DECLARED HERE, not chosen by looking at the results. Picking
the "best" recipe per dataset from the test PRR would be an oracle, which is the trap prereg/S4 named.

CPU/GPU job -> qsub, NEVER the login node (it trains poolers per cell).

    python scripts/checks/multi_recipe_heads.py --evals pubmed_qa --seeds 1,2,3
"""
import argparse
import csv as _csv
import hashlib
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, msp, results                                          # noqa: E402
from aggregation_table import attn_unc, paired_bootstrap, load_per_token     # noqa: E402
from attn_pool import train_attn, select_temperature, head_attention_correlation   # noqa: E402
from xl_rungs import build_rows, eval_split, label_of                        # noqa: E402
import probedriftlong as pdl                                                 # noqa: E402
from prior_builders import build_prior, OrgadCoverageError                   # noqa: E402
from transformers import AutoTokenizer                                       # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"

# The four datasets span the taxonomy: concentrated / no-signal-in-probabilities / spread / factuality.
DEFAULT_EVALS = ["pubmed_qa", "xsum", "cnn_dailymail", "factscore"]

# DECLARED IN ADVANCE. `nll` is every-token surprisal, `topk:25` was the strongest top-k arm on the OOD
# rungs of the S4 run, `content_mass` is the linguistic recipe. The control repeats CONTROL_RECIPE.
DEFAULT_RECIPES = ["nll", "topk:25", "content_mass"]
CONTROL_RECIPE = "nll"

_FIELDS = ["rung", "eval", "train", "method", "prr_mean", "prr_std", "n_seeds",
           "bar_msp_min", "head_attn_corr", "ci_lo", "ci_hi", "boot_p", "significant",
           "cluster", "env_hash", "commit", "seeds"]


def out_path(args):
    return Path(args.out) if args.out else (ROOT / "results" / f"multi_recipe_heads__{cache._slug(MODEL)}.csv")


def _flush_rows(out, rows):
    """Write everything accumulated SO FAR, atomically. CALLED PER CELL.

    Not optional: on 2026-08-05 nine jobs ran ~8 hours each and produced NO csv, because a sibling driver
    held every row in RAM until the end and the walltime arrived first. temp-then-replace so a crash
    mid-write cannot leave a TRUNCATED csv, which would read as a short-but-valid grid.
    """
    tmp = Path(str(out) + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--evals", default=",".join(DEFAULT_EVALS))
    ap.add_argument("--recipes", default=",".join(DEFAULT_RECIPES))
    ap.add_argument("--control-recipe", default=CONTROL_RECIPE)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--rungs", default="", help="base-rung filter (e.g. DiffTask,LOO); '' = all long OOD + ID")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    evals = [e for e in args.evals.split(",") if e]
    seeds = [int(s) for s in args.seeds.split(",")]
    recipes = [p for p in args.recipes.split(",") if p]
    want = set(r for r in args.rungs.split(",") if r)
    K = len(recipes) + 1                       # frozen recipe heads + one free head
    device = "cuda" if torch.cuda.is_available() else "cpu"
    host = socket.gethostname()
    cluster = ("RCS" if (host.startswith("login-") or "cx3" in host)
               else ("DoC" if ("cloud-vm" in host or host.startswith("gpu")) else host))
    env_hash = hashlib.sha1(f"{sys.version.split()[0]}|torch{torch.__version__}|"
                            f"np{np.__version__}".encode()).hexdigest()[:8]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    prov = {"cluster": cluster, "env_hash": env_hash, "commit": commit, "seeds": args.seeds}
    print(f"device {device} | host {host} | cluster {cluster} | env {env_hash} | commit {commit[:12]} | "
          f"recipes {recipes} | control {args.control_recipe} | K {K}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    special_ids = set(getattr(tok, "all_special_ids", []))
    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok -> skip", flush=True); continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: unlabelled -> skip", flush=True); continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} rows (label={label_of(d)})", flush=True)
    sources = set(PT)

    singles = [f"single_{p}" for p in recipes]
    multis = ["mh_diverse", "mh_same", "mh_diverse_nofree"]
    all_methods = ["floor_min", "armA"] + singles + multis
    out_rows = []
    skipped = set()

    for rung, X, spec in pdl.cells_long(sources, evals):
        base_rung = rung.replace("-long", "")
        if X not in PT or (want and base_rung not in want and rung != "ID"):
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        per = {m: [] for m in all_methods}
        acc = {m: [] for m in all_methods}
        corr = {m: [] for m in multis}
        mats = {}
        yte_ref = None
        train_desc = ""

        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows)
            tr_idx = list(range(n_tr))
            te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float)
            yte_ref = yte
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]
            cell_datasets = [d for d, _i in allrows]
            train_desc = f"{'+'.join(sorted({d for d, _ in train_rows}))}:{n_tr}"

            v = {}
            v["floor_min"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "min")
                                       for i in te_idx])
            best_T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
            v["armA"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd,
                                                       temperature=best_T),
                                            states, te_idx, device), float)

            # Build every recipe ONCE per (cell, seed). A recipe that cannot be built for this pool must
            # take the cell down loudly -- never be replaced by a uniform stand-in, which would silently
            # make a "different recipes" head identical to a free one.
            built = {}
            failed = None
            for p in set(recipes + [args.control_recipe]):
                try:
                    built[p], _nfb = build_prior(p, records, states, tok, special_ids,
                                                 datasets=cell_datasets)
                except OrgadCoverageError as e:
                    failed = (p, str(e))
                    break
            if failed is not None:
                if (rung, X, failed[0]) not in skipped:
                    print(f"    [{rung:14s} {X}] recipe '{failed[0]}' SKIPPED — {failed[1]} "
                          f"(LOUD; cell left BLANK, never silent-uniform)", flush=True)
                    skipped.add((rung, X, failed[0]))
                v = {}
                break

            for p in recipes:                    # each recipe alone, head-only training (arm C style)
                m = train_attn(states, y, tr_idx, device, seed=sd, prior_list=built[p],
                               frozen_prior=True)
                assert int(torch.count_nonzero(m.q)) == 0, f"single_{p}: query moved — freeze failed!"
                v[f"single_{p}"] = np.asarray(attn_unc(m, states, te_idx, device,
                                                       prior_list=built[p]), float)

            # The three multi-head arms differ ONLY in which recipes sit on the frozen heads.
            layouts = {
                "mh_diverse":        ([built[p] for p in recipes] + [None], tuple(range(len(recipes))), K),
                "mh_same":           ([built[args.control_recipe]] * len(recipes) + [None],
                                      tuple(range(len(recipes))), K),
                "mh_diverse_nofree": ([built[p] for p in recipes], tuple(range(len(recipes))), len(recipes)),
            }
            for name, (hp, frozen, kq) in layouts.items():
                m = train_attn(states, y, tr_idx, device, seed=sd, n_query=kq, n_head=kq,
                               head_priors=hp, frozen_heads=frozen)
                v[name] = np.asarray(attn_unc(m, states, te_idx, device, head_priors=hp), float)
                c, mat = head_attention_correlation(m, states, te_idx, device, head_priors=hp)
                corr[name].append(c)
                # Keep the FULL matrix, not just its mean. The mean off-diagonal mixes pair types: in
                # `mh_same` the three frozen heads are identical (1.0 with each other) while the free head
                # is near-zero with all of them, so the mean lands near 0.5 and NOT near 1.0. Without the
                # matrix that reads like a wiring fault when it is the design. Seed 1 only, to keep the log
                # short -- it is a structural property, not a noisy quantity.
                if sd == seeds[0]:
                    mats[name] = mat

            for m_ in v:
                per[m_].append(results.prr(yte, v[m_]))
                acc[m_].append(v[m_])

        if yte_ref is None or not per["armA"]:
            continue
        stats = {m: (float(np.mean(per[m])), float(np.std(per[m]))) for m in per if per[m]}
        bar = stats["floor_min"][0]

        # STAGE 1 — the MEASUREMENTS, written before anything that could fail.
        # The first smoke run computed this whole cell (~50 minutes) and then died unpacking
        # paired_bootstrap, losing every number because the flush sat after the verdicts. The bootstrap is
        # a summary OF the measurements, so it must never be able to destroy them: PRRs land on disk
        # first, verdicts are added second.
        cell_rows = {}
        for m in all_methods:
            if m not in stats:
                continue                          # not measured -> no row at all, never a zero
            cell_rows[m] = {"rung": rung, "eval": X, "train": train_desc, "method": m,
                            "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                            "n_seeds": len(per[m]), "bar_msp_min": round(bar, 4),
                            "head_attn_corr": (round(float(np.mean(corr[m])), 4)
                                               if m in corr and corr[m]
                                               and np.isfinite(np.mean(corr[m])) else ""),
                            **prov}
            out_rows.append(cell_rows[m])

        print(f"  [{rung:14s} {X}] " + "  ".join(
            f"{m}={stats[m][0]:+.3f}" for m in all_methods if m in stats), flush=True)
        for m in multis:
            if corr[m] and np.isfinite(np.mean(corr[m])):
                print(f"      head-corr {m:18s} mean-offdiag {np.mean(corr[m]):+.4f}", flush=True)
                if m in mats and np.isfinite(mats[m]).all():
                    for r_ in mats[m]:
                        print("          " + "  ".join(f"{x:+.3f}" for x in r_), flush=True)
        _flush_rows(out_path(args), out_rows)

        # STAGE 2 — the pre-registered contrasts: the method against the bar it must beat, and against
        # its own same-recipe control. Paired over TEST EXAMPLES, not over seed re-inits.
        if "mh_diverse" in stats:
            mean_unc = np.mean(np.stack(acc["mh_diverse"]), axis=0)
            for ref in ("armA", "mh_same"):
                if ref not in stats:
                    continue
                ref_unc = np.mean(np.stack(acc[ref]), axis=0)
                # paired_bootstrap returns FIVE values (margin, lo, hi, p, significant) -- unpacking
                # three is what crashed the smoke run. Use its own margin and verdict rather than
                # recomputing them, so this driver cannot disagree with the other ladders.
                mg, lo, hi, p_, is_sig = paired_bootstrap(yte_ref, mean_unc, ref_unc)
                sig = "SIG" if is_sig else "ns"
                print(f"    [verdict] mh_diverse_vs_{ref:9s} margin "
                      f"{mg:+.3f} CI[{lo:+.3f},{hi:+.3f}] {sig}", flush=True)
                if ref == "armA":
                    cell_rows["mh_diverse"].update({"ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                                    "boot_p": round(p_, 4), "significant": sig})
            _flush_rows(out_path(args), out_rows)
        print(f"    [saved] {len(out_rows)} rows -> {out_path(args).name}", flush=True)

    _flush_rows(out_path(args), out_rows)
    print(f"wrote {out_path(args)}  ({len(out_rows)} rows)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
