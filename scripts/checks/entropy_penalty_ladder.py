"""B.3 -- the ONE-SIDED entropy penalty: penalise over-sharp attention, leave broad attention alone.

⚠️ WHY THIS IS NOT A THIRD VARIANT OF B.1/B.2. Those supervised the attention toward a TARGET and both
found the target carries no information (real ~= shuffled on every cell). This constrains a PROPERTY of
the distribution and uses NO target, so the "a shuffled target does just as well" failure mode
structurally cannot arise. That is what makes it worth running after two nulls.

    L = L_task + lambda * mean( relu(tau - H_norm)^2 )

exactly zero for any example already broader than tau. `frac_bound` is reported per arm because a
penalty that binds everywhere is just the two-sided entropy control we already ran and found
PRR-neutral -- the one-sidedness has to be shown to be real rather than nominal.

PRIMARY SCREEN IS pubmed ID (pre-registered, prereg/B3_one_sided_entropy_penalty.md). pubmed is the only
dataset with genuinely concentrated ID attention (normalised entropy 0.61, ~half its mass on
punctuation). If a sharpness penalty does anything anywhere it does it there. Its OOD rungs are NOT
stable (their pools contain med_quad, regenerating) and xsum is undecided pending DoC's probe, so
everything except pubmed ID is labelled provisional.

Prediction: helps pubmed, no-op on xsum. A uniform effect is SUSPICIOUS -- check frac_bound on xsum.

    python scripts/checks/entropy_penalty_ladder.py --evals pubmed_qa,xsum --rungs ID
"""
import argparse, csv as _csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "checks"))
import torch  # noqa: E402
from aggregation_table import attn_unc, prr_from_conf  # noqa: E402
from attn_pool import (load_per_token, train_attn, select_temperature, regime_tag,  # noqa: E402
                       mean_attention_entropy, attention_entropies,  # noqa: E402
                       normalised_entropy, pad_batch)  # noqa: E402
from xl_rungs import build_rows, label_of  # noqa: E402
import probedriftlong as PDL  # noqa: E402

SLUG = "meta-llama_Meta-Llama-3.1-8B"
MODEL = "meta-llama/Meta-Llama-3.1-8B"
VAL_FRAC = 0.2


def val_split(tr_idx, seed):
    perm = np.random.RandomState(seed).permutation(len(tr_idx))
    n_val = max(1, int(round(len(tr_idx) * VAL_FRAC)))
    return [tr_idx[i] for i in perm[n_val:]], [tr_idx[i] for i in perm[:n_val]]


def frac_bound(model, states, idx, device, tau, bs=64):
    """Fraction of examples the hinge actually acts on. Near 1.0 = the penalty is NOT one-sided in
    practice and the arm must be read as the old two-sided entropy control."""
    model.eval(); n_b = n = 0
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            X, mask, pos = pad_batch([states[i] for i in idx[b:b + bs]], device)
            _l, a = model(X, mask, pos)
            if a.dim() == 3:
                a = a.mean(dim=2)
            Hn = normalised_entropy(a, mask)
            n_b += int((Hn < tau).sum()); n += Hn.shape[0]
    return n_b / max(n, 1)


def fit(states, y, tr, te, device, seed, T, **kw):
    m = train_attn(states, y, tr, device, seed=seed, temperature=T, **kw)
    prr = float(prr_from_conf(np.array([y[i] for i in te], float),
                              -np.asarray(attn_unc(m, states, te, device), float)))
    return prr, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="pubmed_qa,xsum")
    ap.add_argument("--rungs", default="ID")
    ap.add_argument("--taus", default="0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--lambdas", default="0.5,1.0,2.0")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    evals = [e.strip() for e in args.evals.split(",")]
    taus = [float(x) for x in args.taus.split(",")]
    lams = [float(x) for x in args.lambdas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    want = set(s.strip() for s in args.rungs.split(",")) if args.rungs else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} evals={evals} taus={taus} lambdas={lams}", flush=True)

    PT = {}
    for d in sorted(set(PDL.LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            continue
        st, sp, y, _, rec = loaded
        fin = np.isfinite(y)
        if not fin.any():
            continue
        if not fin.all():
            k = np.where(fin)[0]
            st = [st[i] for i in k]; rec = [rec[i] for i in k]; sp = sp[k]; y = y[k]
        PT[d] = (st, sp, y, rec)
        print(f"  {d}: {len(st)} rows", flush=True)

    rows = []
    for rung, X, spec in PDL.cells_long(set(PT), evals):
        if want is not None and rung.replace("-long", "") not in want:
            continue
        for sd in seeds:
            tr_rows, te_rows = build_rows(X, spec, PT, sd, PDL.sampled_train_idx)
            if not tr_rows or not te_rows:
                continue
            n_tr = len(tr_rows)
            tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(te_rows)))
            allr = tr_rows + te_rows
            y = np.array([PT[d][2][i] for d, i in allr], float)
            states = [PT[d][0][i] for d, i in allr]
            T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
            base, m0 = fit(states, y, tr_idx, te_idx, device, sd, T)
            # ⭐ THE LEFT TAIL. A one-sided penalty can only act on examples BELOW tau, so if there is no
            # mass down there it has nothing to act on -- and "no over-sharp subpopulation exists" is a
            # cleaner answer than "it acted and did not help". Reported before any penalty is applied.
            e0 = attention_entropies(m0, states, te_idx, device)
            q = {f"ent_p{p_}": round(float(np.percentile(e0, p_)), 4) for p_ in (5, 10, 25, 50)}
            ent0 = float(e0.mean())
            print(f"    baseline entropy distribution {X}: p05={q['ent_p5']:.3f} p10={q['ent_p10']:.3f} "
                  f"p25={q['ent_p25']:.3f} p50={q['ent_p50']:.3f} mean={ent0:.3f}", flush=True)
            # tau/lambda chosen on a validation slice carved from TRAIN, never on test
            sub_tr, sub_val = val_split(tr_idx, sd)
            grid = {(t, l): fit(states, y, sub_tr, sub_val, device, sd, T,
                                ent_lambda=l, ent_threshold=t)[0] for t in taus for l in lams}
            (bt, bl) = max(grid, key=grid.get)
            pen, mp = fit(states, y, tr_idx, te_idx, device, sd, T, ent_lambda=bl, ent_threshold=bt)
            ent1 = mean_attention_entropy(mp, states, te_idx, device)
            fb0 = frac_bound(m0, states, te_idx, device, bt)
            fb1 = frac_bound(mp, states, te_idx, device, bt)
            prov = "" if (X == "pubmed_qa" and rung == "ID") else "PROVISIONAL"
            rows.append({"rung": rung, "eval": X, "seed": sd, "tau": bt, "lambda": bl,
                         "prr_baseline": round(base, 4), "prr_penalty": round(pen, 4),
                         "delta": round(pen - base, 4),
                         "ent_before": round(ent0, 4), "ent_after": round(ent1, 4), **q,
                         "frac_bound_before": round(fb0, 3), "frac_bound_after": round(fb1, 3),
                         "stability": prov or "PERMANENT", "n_test": len(te_idx)})
            print(f"  [{rung:14s}] {X:<12} s{sd} tau={bt} lam={bl}  base {base:+.4f} pen {pen:+.4f} "
                  f"delta {pen-base:+.4f} | ent {ent0:.3f}->{ent1:.3f} | bound {fb0:.2f}->{fb1:.2f} "
                  f"{prov}", flush=True)

    if not rows:
        raise SystemExit("no cells produced")
    out = Path(args.out) if args.out else ROOT / "results" / f"entropy_penalty{regime_tag()}__{SLUG}.csv"
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\n=== READ IN THIS ORDER ===")
    print("1. frac_bound_before: if ~1.0 the penalty is NOT one-sided in practice -> read as the old")
    print("   two-sided entropy control, which was already PRR-neutral.")
    print("2. ent_before -> ent_after: did the constraint actually move the attention?")
    print("3. delta, and ONLY on pubmed ID as the permanent screen. Everything else is provisional.")
    for r in rows:
        print(f"   {r['eval']:<12} {r['rung']:<14} delta {r['delta']:+.4f}  ({r['stability']})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
