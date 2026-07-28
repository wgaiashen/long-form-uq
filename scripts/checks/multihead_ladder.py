"""S6 — MULTI-HEAD attention (Joe idea 2): MH (K queries + K heads) + ABLATION (1 query + K heads) on the
long-form OOD rungs, vs the K=1 baseline (arm A, reused from the S3 fixed_prior_ladder CSV).

  MH        n_query=K, n_head=K  — K attention distributions -> K pooled -> K classifiers, ensembled
  ABLATION  n_query=1, n_head=K  — ONE attention, K classifiers, ensembled  (Joe's mandatory control:
                                   isolates "more classifier heads" from "attention diversity")
  K=1       n_query=1, n_head=1  — = arm A (reused from S3; not retrained)

PRE-REGISTERED (worklog 2026-07-28): MH beats K=1 modestly OOD, and the ABLATION captures most of that gain
(classifier ensembling, not attention diversity). Bar = per-cell long-ladder msp_min. K=4 fixed, no sweep,
no head-subset sampling (measure collapse first). MH/ABLATION at temperature=1 (no select_temperature) — the
K=1 baseline from S3 used best_T, so MH-vs-K1 carries a temperature caveat; MH-vs-ABLATION (both in-run, T=1)
is the clean paired control.

    python scripts/checks/multihead_ladder.py --evals pubmed_qa,xsum,cnn_dailymail,med_quad,samsum,expertqa \
        --K 4 --seeds 1,2,3

CPU/GPU job -> DoC (per the S6 plan). Reuses the same cells+seeds as S3 (paired).
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
from attn_pool import train_attn, pad_batch                                  # noqa: E402
from xl_rungs import build_rows, eval_split, label_of                        # noqa: E402
import probedriftlong as pdl                                                 # noqa: E402
from transformers import AutoTokenizer                                       # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
DEFAULT_EVALS = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa"]


def git_commit():
    """The exact code version this run used -- stamped on every row so a later join can ASSERT code-match."""
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def s3_armA_prr(eval_, rung):
    """arm A (K=1) PRR from the S3 fixed_prior_ladder CSV — the reused baseline (honors 'don't retrain arm A')."""
    f = ROOT / "results" / f"fixed_prior_ladder__{SLUG}.csv"
    if not f.exists():
        return None
    for r in _csv.DictReader(open(f)):
        if r["eval"] == eval_ and r["rung"] == rung and r["method"] == "armA":
            return float(r["prr_mean"])
    return None


def head_diversity(pooler, states, te_idx, device, bs=64):
    """Return (mean pairwise QUERY cosine, mean pairwise ATTENTION correlation) — Joe's 'do they converge?'.
    High = the heads collapsed to the same thing; low = diverse."""
    with torch.no_grad():
        q = pooler.q.unsqueeze(0) if pooler.q_rest is None else torch.cat([pooler.q.unsqueeze(0), pooler.q_rest], 0)
        K = q.shape[0]
        qn = q / q.norm(dim=1, keepdim=True).clamp(min=1e-9)
        cos = qn @ qn.t()                                              # (K,K)
        iu = torch.triu_indices(K, K, offset=1)
        q_cos = float(cos[iu[0], iu[1]].mean()) if K > 1 else float("nan")
        # attention distributions on the test set: stack all real-token weights -> (M, K)
        cols = []
        for b in range(0, len(te_idx), bs):
            idx = te_idx[b: b + bs]
            X, mask, pos = pad_batch([states[i] for i in idx], device)
            _, a = pooler(X, mask, pos)                                # (B,T,Q)
            if a.dim() == 2:
                a = a.unsqueeze(-1)
            m = mask.bool()
            for j in range(a.shape[0]):
                cols.append(a[j][m[j]].cpu().numpy())                  # (T_j, Q)
        A = np.concatenate(cols, axis=0)                              # (M, Q)
        if A.shape[1] > 1:
            C = np.corrcoef(A.T)                                       # (Q,Q)
            iu2 = np.triu_indices(A.shape[1], k=1)
            a_corr = float(np.nanmean(C[iu2]))
        else:
            a_corr = float("nan")
    return q_cos, a_corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--evals", default=",".join(DEFAULT_EVALS))
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--rungs", default="", help="base-rung filter; '' = all long OOD + ID")
    ap.add_argument("--defer-baseline", action="store_true",
                    help="do NOT read arm A from the S3 CSV as a precondition (S6 training is self-contained). "
                         "Leaves the K=1 comparison columns EMPTY + stamps baseline_deferred=True, so a blank is "
                         "never misread as a zero/loss; fill it later with join_arma_baseline.py. Unblocks S6 to "
                         "run in parallel with S3.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    evals = args.evals.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    K = args.K; want = set(r for r in args.rungs.split(",") if r)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    host = socket.gethostname()
    cluster = "RCS" if (host.startswith("login-") or "cx3" in host) else ("DoC" if ("cloud-vm" in host or host.startswith("gpu")) else host)
    env_hash = hashlib.sha1(f"{sys.version.split()[0]}|torch{torch.__version__}|np{np.__version__}".encode()).hexdigest()[:8]
    commit = git_commit()
    prov = {"cluster": cluster, "env_hash": env_hash, "commit": commit, "seeds": args.seeds,
            "baseline_deferred": bool(args.defer_baseline)}   # stamped on EVERY row for the later join's asserts
    print(f"device {device} | host {host} | cluster {cluster} | env {env_hash} | commit {commit[:12]} | "
          f"K={K} | defer_baseline={args.defer_baseline}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)  # noqa: F841 (kept for parity / future token diagnostics)
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

    methods = ["floor_min", "mh", "ablation"]
    out_rows = []
    for rung, X, spec in pdl.cells_long(sources, evals):
        base_rung = rung.replace("-long", "")
        if X not in PT or (want and base_rung not in want and rung != "ID"):
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        per = {m: [] for m in methods}; acc = {m: [] for m in methods}
        div_cos, div_corr = [], []
        yte_ref = None
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows); tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float); yte_ref = yte
            states = [PT[d][0][i] for d, i in allrows]
            v = {}
            v["floor_min"] = np.array([msp.msp_uncertainty(PT[d][3][i]["token_logprobs"], "min")
                                       for d, i in test_rows])
            mh = train_attn(states, y, tr_idx, device, seed=sd, temperature=1.0, n_query=K, n_head=K)
            v["mh"] = np.asarray(attn_unc(mh, states, te_idx, device), float)
            abl = train_attn(states, y, tr_idx, device, seed=sd, temperature=1.0, n_query=1, n_head=K)
            v["ablation"] = np.asarray(attn_unc(abl, states, te_idx, device), float)
            qc, ac = head_diversity(mh, states, te_idx, device)
            div_cos.append(qc); div_corr.append(ac)
            for m in methods:
                per[m].append(results.prr(yte, v[m])); acc[m].append(v[m])
        if yte_ref is None:
            continue
        stats = {m: (float(np.mean(per[m])), float(np.std(per[m]))) for m in per if per[m]}
        avg = {m: np.mean(np.stack(acc[m]), 0) for m in acc if acc[m]}
        bar = stats["floor_min"][0]
        # K=1 baseline: DEFERRED (self-contained S6) leaves it EMPTY + baseline_deferred=True; else reuse arm A.
        k1 = None if args.defer_baseline else s3_armA_prr(X, rung)
        _real = {}
        for _d, _i in train_rows:
            _real[_d] = _real.get(_d, 0) + 1
        srcs = "+".join(f"{d}:{_real.get(d, 0)}" for d in dict.fromkeys(d for d, _c in spec))
        k1s = "deferred" if args.defer_baseline else (f"{k1:+.3f}" if k1 is not None else "n/a")
        print(f"\n[{rung:14s}] eval={X} ({label_of(X)}) train={srcs}  BAR=msp_min {bar:+.3f}  K1(armA,S3)={k1s}", flush=True)
        print(f"    head-diversity: query-cosine {np.mean(div_cos):+.3f}  attn-corr {np.mean(div_corr):+.3f}"
              f"  (high = collapsed)", flush=True)
        for m in methods:
            print(f"    {m:12s} {stats[m][0]:+.3f} +/- {stats[m][1]:.3f}", flush=True)
            out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": m,
                             "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                             "n_seeds": len(per[m]), "bar_msp_min": round(bar, 4),
                             "k1_armA_s3": round(k1, 4) if k1 is not None else "",
                             "query_cosine": round(float(np.mean(div_cos)), 4),
                             "attn_corr": round(float(np.mean(div_corr)), 4), **prov})
        # paired verdicts: MH vs ABLATION (the control -- self-contained, NO S3), MH vs floor
        for vk, a, b in [("mh_vs_ablation", "mh", "ablation"), ("mh_vs_floor", "mh", "floor_min"),
                         ("ablation_vs_floor", "ablation", "floor_min")]:
            mg, lo, hi, pv, sig = paired_bootstrap(yte_ref, avg[a], avg[b])
            print(f"    [verdict] {vk:20s} margin {mg:+.3f} CI[{lo:+.3f},{hi:+.3f}] {'SIG' if sig else 'ns'}", flush=True)
            out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"VERDICT:{vk}",
                             "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                             "boot_p": round(pv, 4), "significant": bool(sig), "n_seeds": len(seeds), **prov})

    # headline paired count: MH beats K=1 (arm A from S3) on k/N cells (per-cell). Skipped when deferred.
    if args.defer_baseline:
        print("\nK=1 baseline DEFERRED -> fill with join_arma_baseline.py after S3 lands (columns left blank).", flush=True)
    else:
        mh_cells = [r for r in out_rows if r["method"] == "mh" and isinstance(r.get("k1_armA_s3"), float)]
        if mh_cells:
            wins = sum(1 for r in mh_cells if r["prr_mean"] > r["k1_armA_s3"])
            print(f"\nMH beats K=1 (arm A, S3) on {wins}/{len(mh_cells)} cells", flush=True)

    out = Path(args.out) if args.out else (ROOT / "results" / f"multihead_ladder__{cache._slug(MODEL)}.csv")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "train", "method", "prr_mean", "prr_std", "n_seeds",
                                           "bar_msp_min", "k1_armA_s3", "query_cosine", "attn_corr",
                                           "ci_lo", "ci_hi", "boot_p", "significant",
                                           "cluster", "env_hash", "commit", "seeds", "baseline_deferred"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
