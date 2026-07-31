"""PRIORITY-0 — certify the summarisation headline against the FAIR (length-normalised) floor, split by capping.

Two linked concerns (user, 2026-07-30):
  (1) The probe−floor gap on summarisation may be inflated because `msp_min` (a minimum over T token-probs) is an
      extreme-value statistic that falls with length REGARDLESS of correctness, and capped gens are the longest —
      so the FLOOR is handicapped on capped rows, not the probe. Report the CHANGE IN THE GAP (probe − best fair
      floor), full vs NOT-CAPPED, with the floor's gain shown ALONGSIDE the probe's.
  (2) The fair floor must be length-normalised: report msp_sum / perplexity / msp_min / msp_min_lennorm on the same
      cells (Item 1), even where it raises the bar against us.

Covers xsum/cnn/samsum at ID AND DiffTask-long (headlines are OOD; is_capped is rung-invariant so ID gives the
direction, DiffTask-long the quoted-cell magnitude). med_quad is UNINFORMATIVE (n_notcapped~16) — reported as such.
Also prints probe vs msp_min (the pre-registered bar) so the msp_min-vs-fair_floor discrepancy is explicit.

    python scripts/checks/truncation_confound.py --out results/truncation_confound__meta-llama_Meta-Llama-3.1-8B.csv
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, msp, probe, results                                   # noqa: E402
from aggregation_table import attn_unc, load_per_token                       # noqa: E402
from attn_pool import train_attn, select_temperature                         # noqa: E402
from xl_rungs import build_rows, eval_split, label_of                        # noqa: E402
from luq.data import MAX_NEW_TOKENS                                          # noqa: E402
import probedriftlong as pdl                                                 # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
EVALS = ["xsum", "cnn_dailymail", "samsum"]        # + med_quad printed as UNINFORMATIVE
RUNGS = ["ID", "DiffTask-long"]
SEED = 1


def floor_vectors(records, te_idx):
    """The four floor UNCERTAINTY vectors on the eval test rows (higher = more uncertain), label-free.
    msp_min_lennorm removes the length drift of the extreme: maxNLL − beta*log(T+1), beta = OLS slope."""
    lp = [np.asarray(records[i]["token_logprobs"], float) for i in te_idx]
    T = np.array([len(x) for x in lp], float)
    maxnll = np.array([(-x).max() for x in lp])
    sumnll = np.array([(-x).sum() for x in lp])
    ppl = np.array([(-x).mean() for x in lp])
    logT = np.log(T + 1)
    beta = np.polyfit(logT, maxnll, 1)[0] if len(set(T.tolist())) > 1 else 0.0
    return {"msp_sum": sumnll, "perplexity": ppl, "msp_min": maxnll, "msp_min_lennorm": maxnll - beta * logT}


def boot_prr_diff(y, ua, ub, n=2000, seed=1):
    """Paired bootstrap CI of PRR(ua) − PRR(ub) over the SAME rows (the gap)."""
    if len(y) < 20:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed); d = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        d.append(results.prr(y[idx], ua[idx]) - results.prr(y[idx], ub[idx]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / f"truncation_confound__{SLUG}.csv"))
    ap.add_argument("--layer", type=int, default=15)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[confound] device {device} | evals {EVALS}+med_quad | rungs {RUNGS} | seed {SEED}", flush=True)

    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(EVALS) | {"med_quad"}):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok -> skip", flush=True); continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]
        PT[d] = (states, split, y, records)
    sources = set(PT)

    rows = []
    for X in EVALS + ["med_quad"]:
        for rung, Xe, spec in pdl.cells_long(sources, [X]):
            if rung not in RUNGS or Xe != X or X not in PT:
                continue
            tr_rows, te_rows = build_rows(X, spec, PT, SEED, pdl.sampled_train_idx)
            if not tr_rows or not te_rows:
                continue
            ntr = len(tr_rows); tr = list(range(ntr)); te = list(range(ntr, ntr + len(te_rows)))
            allrows = tr_rows + te_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = y[te]
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]
            budget = MAX_NEW_TOKENS[X]
            capped = np.array([len(records[i]["gen_token_ids"]) >= budget - 1 for i in te])
            nc = ~capped
            # --- methods (per-example uncertainty on te) ---
            Xmean = np.stack([s.mean(0) for s in states])
            mlp = probe.train_probe_mlp(Xmean[tr], y[tr])
            sap = np.asarray(probe.uncertainty(mlp, Xmean[te]), float)
            best_T, _ = select_temperature(states, y, tr, device, SEED, False, False)
            armA = np.asarray(attn_unc(train_attn(states, y, tr, device, seed=SEED, temperature=best_T),
                                       states, te, device), float)
            floors = floor_vectors(records, te)

            def prr_on(u, mask=None):
                yy = yte if mask is None else yte[mask]
                uu = u if mask is None else u[mask]
                return results.prr(yy, uu) if len(yy) > 20 else float("nan")

            # best FAIR floor per subset (report which won)
            def best_floor(mask):
                p = {k: prr_on(v, mask) for k, v in floors.items()}
                w = max(p, key=lambda k: (p[k] if np.isfinite(p[k]) else -9))
                return w, p[w], p
            uninformative = nc.sum() < 50
            wf_full, pf_full, allf_full = best_floor(None)
            wf_nc, pf_nc, allf_nc = ("n/a", float("nan"), {}) if uninformative else best_floor(nc)
            sap_full, sap_ncv = prr_on(sap), (float("nan") if uninformative else prr_on(sap, nc))
            arm_full, arm_ncv = prr_on(armA), (float("nan") if uninformative else prr_on(armA, nc))
            mspmin_full = prr_on(floors["msp_min"]); mspmin_nc = float("nan") if uninformative else prr_on(floors["msp_min"], nc)
            # gap = probe − best fair floor, full and not-capped (SAPLMA is the headline probe)
            gap_full = sap_full - pf_full
            gap_nc = float("nan") if uninformative else sap_ncv - pf_nc
            print(f"\n== {X} / {rung}  (n {len(te)}, capped {100*capped.mean():.0f}%, n_notcapped {int(nc.sum())}"
                  f"{'  [UNINFORMATIVE]' if uninformative else ''}) ==", flush=True)
            print(f"   floors full: " + " ".join(f"{k} {v:+.3f}" for k, v in allf_full.items()) +
                  f"   -> best fair = {wf_full} {pf_full:+.3f}", flush=True)
            print(f"   SAPLMA {sap_full:+.3f}  armA {arm_full:+.3f}   (vs msp_min bar: SAPLMA−msp_min {sap_full-mspmin_full:+.3f})", flush=True)
            if not uninformative:
                print(f"   NOT-CAPPED: best fair = {wf_nc} {pf_nc:+.3f}  SAPLMA {sap_ncv:+.3f}  armA {arm_ncv:+.3f}", flush=True)
                print(f"   ⭐ SAPLMA−fairfloor GAP: full {gap_full:+.3f} -> not-capped {gap_nc:+.3f}  "
                      f"(floor gained {pf_nc-pf_full:+.3f}, SAPLMA gained {sap_ncv-sap_full:+.3f})", flush=True)
                clo, chi = boot_prr_diff(yte[nc], sap[nc], floors[wf_nc][nc])
                print(f"      not-capped gap CI[{clo:+.3f},{chi:+.3f}] {'CLEAN (probe>fairfloor)' if clo>0 else 'NOT certified — floor catches probe'}", flush=True)
            for k, v in {"SAPLMA": (sap_full, sap_ncv), "armA": (arm_full, arm_ncv),
                         **{f"floor_{fk}": (prr_on(fv), float('nan') if uninformative else prr_on(fv, nc)) for fk, fv in floors.items()}}.items():
                rows.append({"eval": X, "rung": rung, "method": k, "prr_full": round(v[0], 4),
                             "prr_notcapped": round(v[1], 4) if np.isfinite(v[1]) else "", "n_notcapped": int(nc.sum()),
                             "pct_capped": round(100*capped.mean(), 1), "best_fair_full": wf_full,
                             "uninformative": int(uninformative)})

    with open(args.out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["eval", "rung", "method", "prr_full", "prr_notcapped", "n_notcapped",
                                            "pct_capped", "best_fair_full", "uninformative"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[confound] wrote {args.out} ({len(rows)} rows)", flush=True)
    print("[confound] DECISION: per summ dataset/rung, the probe wins ONLY if SAPLMA/armA > best FAIR floor on the "
          "NOT-CAPPED rows. Pre-check expectation: xsum clean; cnn overturned by perplexity; samsum reduced.", flush=True)


if __name__ == "__main__":
    main()
