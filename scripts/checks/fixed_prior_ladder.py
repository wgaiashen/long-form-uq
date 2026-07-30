"""S3 — PRIOR-INIT POOLING driver: arms A/B/C/D on the long-form OOD rungs.

  A. incumbent  learned query (control)                     — the thing that dissolves
  B. uniform    frozen query -> mean-pool (reference floor) — NOT the success bar
  C. frozen prior   attention = renormalised prior, head-only training   (ours; ≈ Idea-2 replication)
  D. prior-init, learn away   scores = X@q + beta*log(prior), q trains    (Joe's proposal)

Priors (label-free, example-local): content_mass, nll (soft-Orgad is a SEPARATE RCS task, §3.6).

PRE-REGISTERED (see worklog 2026-07-28): expect **A and D over C** OOD (S1 showed the learned query beats
mean-pool; Idea-2 showed a frozen prior ≈ baseline). BAR = the LONG-ladder per-cell `msp_min` (NOT the
standard-ladder 0.284). Every arm trained in THIS run (no arm inherited); arm A must reproduce its §C.3 PRR
(the reproduction gate) before C/D are trusted. Correctness: arm C's query stays frozen (runtime assert);
flat-prior→arm B and example-local are covered by tests/test_prior_pool.py.

CPU/GPU job -> qsub / DoC, NOT the login node (it trains poolers per cell).

    python scripts/checks/fixed_prior_ladder.py --evals pubmed_qa,xsum,cnn_dailymail,med_quad,samsum,expertqa \
        --priors content_mass,nll --seeds 1,2,3
"""
import argparse
import csv as _csv
import hashlib
import socket
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, msp, results                                          # noqa: E402
from aggregation_table import attn_unc, paired_bootstrap, load_per_token     # noqa: E402
from attn_pool import train_attn, select_temperature                         # noqa: E402
from xl_rungs import build_rows, eval_split, label_of                        # noqa: E402
import probedriftlong as pdl                                                 # noqa: E402
from prior_builders import build_prior, OrgadCoverageError                   # noqa: E402
from transformers import AutoTokenizer                                       # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
DEFAULT_EVALS = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa"]
GATE_TOL = 0.05        # arm-A vs §C.3 attention reproduction tolerance (seed noise / env drift)


def c3_attention_prr(eval_, rung):
    """§C.3 attention prr_mean for (eval, -long rung) — the arm-A reproduction target."""
    f = ROOT / "results" / f"probedriftlong_{eval_}__{SLUG}.csv"
    if not f.exists():
        return None
    for r in _csv.DictReader(open(f)):
        if r["rung"] == rung and r["method"] == "attention":
            return float(r["prr_mean"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--evals", default=",".join(DEFAULT_EVALS))
    ap.add_argument("--priors", default="content_mass,nll")
    ap.add_argument("--beta", type=float, default=1.0, help="arm D annealed log-prior weight")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--rungs", default="", help="base-rung filter (e.g. DiffTask,LOO); '' = all long OOD + ID")
    ap.add_argument("--restricted-ood", action="store_true",
                    help="S3.6: instead of cells_long, build ID + a RESTRICTED-POOL OOD rung — train each eval on "
                         "the OTHER --evals only (e.g. pubmed <- {med_quad,expertqa}). A genuine OOD cell with "
                         "FULL Orgad coverage (the standard LOO/DiffTask pools pull in uncovered sources). Keeps "
                         "the ID cells as the ID-vs-OOD control.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    evals = args.evals.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    priors = [p for p in args.priors.split(",") if p]
    want = set(r for r in args.rungs.split(",") if r)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    host = socket.gethostname()
    cluster = "RCS" if (host.startswith("login-") or "cx3" in host) else ("DoC" if ("cloud-vm" in host or host.startswith("gpu")) else host)
    env_hash = hashlib.sha1(f"{sys.version.split()[0]}|torch{torch.__version__}|np{np.__version__}".encode()).hexdigest()[:8]
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    prov = {"cluster": cluster, "env_hash": env_hash, "commit": commit, "seeds": args.seeds}  # for the join's asserts
    print(f"device {device} | host {host} | cluster {cluster} | env {env_hash} | commit {commit[:12]} | "
          f"priors {priors} | beta {args.beta}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    special_ids = set(getattr(tok, "all_special_ids", []))
    PT = {}
    to_load = set(evals) if args.restricted_ood else (set(pdl.LONG_SRC) | set(evals))   # restricted = evals only
    for d in sorted(to_load):
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

    ARMS_PRIORLESS = ["floor_min", "armA", "armB"]
    ARMS_PRIOR = [f"{a}_{p}" for p in priors for a in ("armC", "armD")]
    all_methods = ARMS_PRIORLESS + ARMS_PRIOR
    out_rows = []; gate_rows = []; skipped = set()

    if args.restricted_ood:                          # S3.6: ID (control) + restricted-pool OOD (covered sets only)
        cell_iter = ([("ID", X, [(X, None)]) for X in evals if X in sources]
                     + [("RestrictedOOD-covered", X, [(o, 900) for o in evals if o != X and o in sources])
                        for X in evals if X in sources])
    else:
        cell_iter = pdl.cells_long(sources, evals)
    for rung, X, spec in cell_iter:
        base_rung = rung.replace("-long", "")
        if X not in PT or (want and base_rung not in want and rung != "ID"):
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        per = {m: [] for m in all_methods}; acc = {m: [] for m in all_methods}
        fb_count = {p: 0 for p in priors}; fb_total = {p: 0 for p in priors}
        yte_ref = None
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows); tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float); yte_ref = yte
            states = [PT[d][0][i] for d, i in allrows]; records = [PT[d][3][i] for d, i in allrows]
            v = {}
            v["floor_min"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "min") for i in te_idx])
            best_T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
            v["armA"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T),
                                            states, te_idx, device), float)
            v["armB"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True),
                                            states, te_idx, device), float)
            cell_datasets = [d for d, _i in allrows]
            for p in priors:
                try:
                    priors_cell, nfb = build_prior(p, records, states, tok, special_ids, datasets=cell_datasets)
                except OrgadCoverageError as e:
                    if (rung, X, p) not in skipped:
                        print(f"    [{rung:14s} {X}] prior '{p}' SKIPPED — {e} (LOUD; never silent-uniform)", flush=True)
                        skipped.add((rung, X, p))
                    continue
                fb_count[p] += nfb; fb_total[p] += len(priors_cell)
                pc = train_attn(states, y, tr_idx, device, seed=sd, prior_list=priors_cell, frozen_prior=True)
                assert int(torch.count_nonzero(pc.q)) == 0, "arm C query moved off init — freeze failed!"
                v[f"armC_{p}"] = np.asarray(attn_unc(pc, states, te_idx, device, prior_list=priors_cell), float)
                pd_ = train_attn(states, y, tr_idx, device, seed=sd, prior_list=priors_cell,
                                 frozen_prior=False, beta=args.beta)
                v[f"armD_{p}"] = np.asarray(attn_unc(pd_, states, te_idx, device, prior_list=priors_cell), float)
            for m in v:                      # only the methods actually computed (Orgad-skipped arms absent)
                per[m].append(results.prr(yte, v[m])); acc[m].append(v[m])
        if yte_ref is None:
            continue
        stats = {m: (float(np.mean(per[m])), float(np.std(per[m]))) for m in per if per[m]}
        avg = {m: np.mean(np.stack(acc[m]), 0) for m in acc if acc[m]}
        bar = stats["floor_min"][0]

        # arm-A reproduction gate vs §C.3 attention
        tgt = c3_attention_prr(X, rung)
        if tgt is not None:
            d_ = abs(stats["armA"][0] - tgt)
            gate_rows.append((X, rung, stats["armA"][0], tgt, d_, d_ < GATE_TOL))

        _real = {}
        for _d, _i in train_rows:
            _real[_d] = _real.get(_d, 0) + 1
        srcs = "+".join(f"{d}:{_real.get(d, 0)}" for d in dict.fromkeys(d for d, _c in spec))
        print(f"\n[{rung:14s}] eval={X} ({label_of(X)}) train={srcs}  BAR=msp_min {bar:+.3f}", flush=True)
        for p in priors:
            if fb_count[p]:
                print(f"    (prior {p}: {fb_count[p]}/{fb_total[p]} rows uniform-fallback — LOUD, counted)", flush=True)
        for m in all_methods:
            if m in stats:
                print(f"    {m:18s} {stats[m][0]:+.3f} +/- {stats[m][1]:.3f}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": m,
                                 "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                                 "n_seeds": len(per[m]), "bar_msp_min": round(bar, 4), **prov})
        # verdicts: each prior arm vs the floor bar, vs arm A, vs arm B
        for p in priors:
            for arm in (f"armC_{p}", f"armD_{p}"):
                if arm not in avg:            # prior skipped for this cell (e.g. Orgad coverage) -> no verdict
                    continue
                for vk, other in [(f"{arm}_vs_floor", "floor_min"), (f"{arm}_vs_armA", "armA"),
                                  (f"{arm}_vs_armB", "armB")]:
                    mg, lo, hi, pv, sig = paired_bootstrap(yte_ref, avg[arm], avg[other])
                    print(f"    [verdict] {vk:26s} margin {mg:+.3f} CI[{lo:+.3f},{hi:+.3f}] {'SIG' if sig else 'ns'}", flush=True)
                    out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"VERDICT:{vk}",
                                     "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                     "boot_p": round(pv, 4), "significant": bool(sig), "n_seeds": len(seeds), **prov})

    print("\n=== ARM-A REPRODUCTION GATE (arm A vs §C.3 attention, tol " + f"{GATE_TOL}) ===", flush=True)
    allok = True
    for X, rung, got, tgt, d_, ok in gate_rows:
        print(f"  {X:13s} {rung:14s} armA {got:+.3f} vs §C.3 {tgt:+.3f} |Δ|={d_:.3f} {'PASS' if ok else 'FAIL <== HALT'}", flush=True)
        allok = allok and ok
    print(f"GATE: {'ALL PASS' if allok else 'FAIL — do NOT trust C/D on failing cells'}", flush=True)

    # COVERED-SET CAPTION (S3.6): report exactly which (rung,eval) cells each prior actually ran on, and which
    # were skipped for coverage -- never a silent partial grid.
    if skipped:
        for p in sorted({s[2] for s in skipped}):
            sk = sorted(f"{s[1]}/{s[0]}" for s in skipped if s[2] == p)
            ran = sorted({(r["eval"], r["rung"]) for r in out_rows if r["method"] == f"armC_{p}"})
            print(f"\nPRIOR '{p}' COVERAGE: ran on {len(ran)} cells {sorted(f'{e}/{rg}' for e, rg in ran)}; "
                  f"SKIPPED {len(sk)} for coverage {sk}", flush=True)

    out = Path(args.out) if args.out else (ROOT / "results" / f"fixed_prior_ladder__{cache._slug(MODEL)}.csv")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "train", "method", "prr_mean", "prr_std", "n_seeds",
                                           "bar_msp_min", "ci_lo", "ci_hi", "boot_p", "significant",
                                           "cluster", "env_hash", "commit", "seeds"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
