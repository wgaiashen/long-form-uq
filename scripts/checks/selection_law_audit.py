"""Audit the so-called "selection law": does a supervised probe's OOD advantage over the
training-free floor decline linearly with how strong that floor is?

The recorded claim regressed (SAPLMA_meanOOD - floor) on floor and found slope -0.758,
r^2 0.72, p 0.0082.  That regression is MATHEMATICALLY COUPLED: the regressor `floor`
appears inside the response with a minus sign, so measurement noise alone drives the slope
negative (the Oldham problem).  This script therefore makes the UNCOUPLED regression

    SAPLMA_meanOOD  =  a + b * floor

the primary analysis, and tests the scientific claim as `b < 1` (probe performance is
flatter across datasets than the floor is).  The coupled fit is still printed, purely as an
arithmetic identity check (its slope must equal b - 1), and is labelled as unquotable.

Populations are NEVER pooled: Llama-3.1-8B and Qwen2.5-14B are fitted separately, n = 8
datasets each.  The unit of analysis is the dataset; the four OOD rungs are averaged first.

Usage:
    python scripts/checks/selection_law_audit.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Pre-registered constants.  Fixed before any output was inspected.
# ---------------------------------------------------------------------------
PERM_SEED = 20260814
N_PERM = 10_000
OOD_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
EVALS = ["asqa", "cnn_dailymail", "expertqa", "factscore",
         "med_quad", "pubmed_qa", "samsum", "xsum"]

# The two masters use different method vocabularies for the same quantities.  The Qwen
# master already uses the canonical short names; the Llama master uses display names.
# We standardise everything onto the canonical short names.
LLAMA_RENAME = {
    "msp_min": "floor_min",
    "perplexity": "floor_ppl",
    "msp_sum": "floor_sum",
    "SAPLMA": "saplma",
    "armA(attention)": "attention",
    "armB(mean-pool)": "uniform",
    "wMSP-norm": "wmsp_norm",
    "wMSP-shrink@2": "wmsp_shrink2",
    "wMSP-shrink@10": "wmsp_shrink10",
}

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_llama():
    """Llama cells, one PRR per (eval, rung, method).

    The canonical master carries every method we need EXCEPT `fair_floor`, which is a
    derived max-of-three row.  `fair_floor` lives in the per-eval `pdl_fam_*` source files
    that the master was rolled up from, so we read those as well and assert that every
    method the two share agrees exactly.  That assertion is the provenance control: if the
    source files had drifted from the master we would rather crash than silently mix them.
    """
    master = pd.read_csv(f"{REPO}/results/pdl_master__meta-llama_Meta-Llama-3.1-8B.csv")
    master = master[master.method.isin(LLAMA_RENAME)].copy()
    master["method"] = master.method.map(LLAMA_RENAME)
    master = master.rename(columns={"prr": "prr_mean"})[
        ["eval", "rung", "method", "prr_mean", "n_seeds"]]

    fam = []
    for path in sorted(glob.glob(f"{REPO}/results/pdl_fam_*__meta-llama_Meta-Llama-3.1-8B.csv")):
        fam.append(pd.read_csv(path))
    fam = pd.concat(fam, ignore_index=True)
    fam = fam[fam["eval"].isin(EVALS) & fam.rung.isin(OOD_RUNGS + ["ID"])]
    # The `_base` / `_segsm` re-runs repeat the shared methods; they agree exactly, so
    # taking the first occurrence is safe.  Verified separately, and re-checked here.
    spread = fam.groupby(["eval", "rung", "method"])["prr_mean"].agg("nunique")
    assert spread.max() == 1, f"pdl_fam replicate files disagree: {spread[spread > 1]}"
    fam = fam.groupby(["eval", "rung", "method"], as_index=False).agg(
        prr_mean=("prr_mean", "first"), n_seeds=("n_seeds", "first"))

    check = master.merge(fam, on=["eval", "rung", "method"], suffixes=("_m", "_f"))
    delta = (check.prr_mean_m - check.prr_mean_f).abs().max()
    assert delta == 0.0, f"master and pdl_fam disagree by {delta}"

    keep = set(LLAMA_RENAME.values()) | {"fair_floor"}
    out = fam[fam.method.isin(keep)].copy()
    out["model"] = "meta-llama/Meta-Llama-3.1-8B"
    return out, len(check)


def load_qwen():
    d = pd.read_csv(f"{REPO}/results/pdl_master__Qwen_Qwen2.5-14B.csv")
    assert set(d.carve) == {"legacy"}, f"unexpected carve values: {set(d.carve)}"
    d = d[d["eval"].isin(EVALS)]
    return d[["eval", "rung", "method", "prr_mean", "n_seeds"]].assign(
        model="Qwen/Qwen2.5-14B"), len(d)


def mean_ood(df, method):
    """One number per dataset: the mean PRR over the four OOD rungs."""
    s = df[(df.method == method) & (df.rung.isin(OOD_RUNGS))]
    piv = s.pivot_table(index="eval", columns="rung", values="prr_mean")
    missing = [e for e in EVALS if e not in piv.index]
    assert not missing, f"{method}: missing evals {missing}"
    assert list(piv.columns) == sorted(OOD_RUNGS), f"{method}: rungs {list(piv.columns)}"
    return piv.loc[EVALS].mean(axis=1)


def ols(x, y):
    """Plain OLS of y on x with the pieces both verdict rules need.

    Returns the slope, its standard error, the intercept, r^2, the two-sided p for
    b != 0, and the ONE-SIDED p for b < 1 (which is the scientific claim here).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    res = stats.linregress(x, y)
    b, a, r, p2, se = res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr
    t_lt1 = (b - 1.0) / se
    p_lt1 = stats.t.cdf(t_lt1, df=n - 2)  # P(b < 1), one-sided
    return dict(a=a, b=b, se_b=se, r2=r ** 2, p_two_sided=p2,
                t_b_lt_1=t_lt1, p_one_sided_b_lt_1=p_lt1, n=n)


def permutation_r2(x, y, seed=PERM_SEED, n_perm=N_PERM):
    """Empirical p for the observed r^2 under random re-pairing of x and y.

    r^2 discards the sign of the relation, so the fraction of permuted r^2 at least as
    large as the observed one IS the two-sided p-value; no doubling is applied.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    obs = stats.linregress(x, y).rvalue ** 2
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        hits += (stats.linregress(x, rng.permutation(y)).rvalue ** 2) >= obs - 1e-15
    return obs, (hits + 1) / (n_perm + 1)


def lodo(x, y):
    """Leave-one-dataset-out: fit on 7, predict the 8th, check the SIGN of the advantage."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rows = []
    for i in range(len(x)):
        m = np.ones(len(x), bool)
        m[i] = False
        f = stats.linregress(x[m], y[m])
        pred = f.intercept + f.slope * x[i]
        pred_adv = pred - x[i]
        act_adv = y[i] - x[i]
        crossover = f.intercept / (1.0 - f.slope) if abs(1.0 - f.slope) > 1e-12 else np.nan
        rows.append(dict(held_out=EVALS[i], floor=x[i], b_fold=f.slope, a_fold=f.intercept,
                         pred_saplma=pred, actual_saplma=y[i],
                         pred_advantage=pred_adv, actual_advantage=act_adv,
                         abs_err=abs(pred - y[i]),
                         sign_correct=bool(np.sign(pred_adv) == np.sign(act_adv)),
                         crossover_floor_star=crossover))
    return pd.DataFrame(rows)


def fmt(d, keys=("a", "b", "se_b", "r2", "p_two_sided", "p_one_sided_b_lt_1")):
    return "  ".join(f"{k}={d[k]:+.4f}" if isinstance(d[k], float) else f"{k}={d[k]}"
                     for k in keys)


def run_model(name, df, out):
    print(f"\n{'=' * 78}\nMODEL: {name}   (n = {len(EVALS)} datasets, OOD = mean of 4 rungs)\n{'=' * 78}")
    res = {"model": name}

    floors = {}
    for f in ["floor_min", "floor_ppl", "fair_floor"]:
        floors[f] = mean_ood(df, f)
        # Floors are re-run under each rung's seed loop and do not depend on the training
        # pool, so their value must be identical across the four OOD rungs.  Confirm that
        # rather than assuming it.
        s = df[(df.method == f) & df.rung.isin(OOD_RUNGS)]
        rng_spread = s.pivot_table(index="eval", columns="rung", values="prr_mean").std(axis=1).max()
        print(f"[floor {f}] max across-rung SD within a dataset = {rng_spread:.2e} "
              f"({'constant, as expected' if rng_spread < 1e-9 else 'NOT CONSTANT'})")

    saplma = mean_ood(df, "saplma")
    floor = floors["floor_min"]

    # ---------------- A.3 PRIMARY ----------------
    prim = ols(floor, saplma)
    res["primary"] = prim
    verdict = "SUPPORTED" if prim["p_one_sided_b_lt_1"] < 0.05 else "NOT SUPPORTED"
    res["primary_verdict"] = verdict
    print(f"\n--- A.3 PRIMARY (uncoupled): saplma_meanOOD ~ a + b*floor_min ---")
    print("  " + fmt(prim))
    print(f"  one-sided p(b<1) = {prim['p_one_sided_b_lt_1']:.4f}  ->  VERDICT: {verdict}")
    print(f"  derived advantage slope b-1 = {prim['b'] - 1:+.4f}  "
          f"(ALGEBRAIC IDENTITY, not an independent measurement)")

    coupled = ols(floor, saplma - floor)
    res["coupled"] = coupled
    res["coupled_identity_gap"] = abs(coupled["b"] - (prim["b"] - 1))
    print(f"  COUPLED check: b={coupled['b']:+.6f}  r2={coupled['r2']:.4f}  "
          f"p={coupled['p_two_sided']:.4g}   "
          f"|b_coupled - (b-1)| = {res['coupled_identity_gap']:.2e}")
    print("  ^^ MATHEMATICALLY COUPLED — DO NOT QUOTE AS EVIDENCE")

    # ---------------- A.4.1 permutation ----------------
    obs_r2, p_perm = permutation_r2(floor, saplma)
    res["permutation"] = dict(obs_r2=obs_r2, p=p_perm, seed=PERM_SEED, n_perm=N_PERM)
    print(f"\n--- A.4.1 permutation (seed {PERM_SEED}, {N_PERM} shuffles) ---")
    print(f"  observed r2 = {obs_r2:.4f}   empirical two-sided p = {p_perm:.4f}")

    # ---------------- A.4.2 floor choice ----------------
    print("\n--- A.4.2 floor-choice sensitivity (dependent = saplma) ---")
    res["floor_sensitivity"] = {}
    for f, fv in floors.items():
        r = ols(fv, saplma)
        v = "b<1 SUPPORTED" if r["p_one_sided_b_lt_1"] < 0.05 else "b<1 NOT SUPPORTED"
        res["floor_sensitivity"][f] = dict(**r, verdict=v)
        print(f"  {f:11s} b={r['b']:+.4f} se={r['se_b']:.4f} r2={r['r2']:.4f} "
              f"p(b<1)={r['p_one_sided_b_lt_1']:.4f}  {v}")

    # ---------------- A.4.3 probe choice ----------------
    print("\n--- A.4.3 probe-choice sensitivity (floor = floor_min) ---")
    res["probe_sensitivity"] = {}
    for m in ["saplma", "attention", "uniform", "wmsp_shrink2"]:
        y = mean_ood(df, m)
        r = ols(floor, y)
        v = "b<1 SUPPORTED" if r["p_one_sided_b_lt_1"] < 0.05 else "b<1 NOT SUPPORTED"
        res["probe_sensitivity"][m] = dict(**r, verdict=v,
                                           macro_advantage=float((y - floor).mean()))
        print(f"  {m:13s} b={r['b']:+.4f} se={r['se_b']:.4f} r2={r['r2']:.4f} "
              f"p(b<1)={r['p_one_sided_b_lt_1']:.4f}  {v}   "
              f"macro adv={float((y - floor).mean()):+.4f}")

    # ---------------- A.4.4 rung sensitivity ----------------
    print("\n--- A.4.4 rung sensitivity (saplma at one rung ~ floor_min) ---")
    res["rung_sensitivity"] = {}
    for rung in OOD_RUNGS:
        s = df[(df.method == "saplma") & (df.rung == rung)].set_index("eval").loc[EVALS, "prr_mean"]
        r = ols(floor, s)
        v = "b<1 SUPPORTED" if r["p_one_sided_b_lt_1"] < 0.05 else "b<1 NOT SUPPORTED"
        res["rung_sensitivity"][rung] = dict(**r, verdict=v)
        print(f"  {rung:15s} b={r['b']:+.4f} se={r['se_b']:.4f} r2={r['r2']:.4f} "
              f"p(b<1)={r['p_one_sided_b_lt_1']:.4f}  {v}")

    # ---------------- A.5 LODO selection rule ----------------
    tab = lodo(floor, saplma)
    n_ok = int(tab.sign_correct.sum())
    b_range = (tab.b_fold.min(), tab.b_fold.max())
    sign_stable = bool(np.sign(tab.b_fold).nunique() == 1)
    useful = (n_ok >= 7) and sign_stable
    res["lodo"] = dict(table=tab.to_dict("records"), n_sign_correct=n_ok,
                       mae=float(tab.abs_err.mean()), b_min=float(b_range[0]),
                       b_max=float(b_range[1]), b_sign_stable=sign_stable,
                       crossover_min=float(tab.crossover_floor_star.min()),
                       crossover_max=float(tab.crossover_floor_star.max()),
                       verdict="USEFUL" if useful else "DESCRIPTIVE ONLY")
    print("\n--- A.5 leave-one-dataset-out selection rule ---")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"  signs correct: {n_ok}/8   MAE = {tab.abs_err.mean():.4f}   "
          f"b across folds [{b_range[0]:+.4f}, {b_range[1]:+.4f}]  sign stable={sign_stable}")
    print(f"  crossover floor* range [{tab.crossover_floor_star.min():+.4f}, "
          f"{tab.crossover_floor_star.max():+.4f}]")
    print(f"  VERDICT (>=7/8 signs AND b sign stable): {res['lodo']['verdict']}")

    # per-dataset raw values, for the write-up table
    res["per_dataset"] = pd.DataFrame({
        "eval": EVALS,
        "floor_min": floor.values,
        "floor_ppl": floors["floor_ppl"].values,
        "fair_floor": floors["fair_floor"].values,
        "saplma_meanOOD": saplma.values,
        "advantage": (saplma - floor).values,
    }).to_dict("records")
    out[name] = res
    print("\nper-dataset inputs:")
    print(pd.DataFrame(res["per_dataset"]).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


def run_xl(out):
    """A.6 — locate and re-derive the claimed XL-grid 'replication' at r = -0.895.

    Two things need saying about this grid.  First, the `xlcontrib_fam_*` family files
    genuinely contain NO SAPLMA, so a SAPLMA-based relation cannot come from them; but the
    `xlonegrid_fam_*` files rolled up into `xl_master` DO carry SAPLMA at full 10x5 coverage,
    and that is where the number comes from.  Second, XL adds two SHORT-form datasets (sciq,
    trivia_qa) whose floors sit far to the right of every long-form dataset, so we refit
    without them to see how much of the relation is leverage from those two points.
    """
    d = pd.read_csv(f"{REPO}/results/xl_master__meta-llama_Meta-Llama-3.1-8B.csv")
    ood = ["SameTask", "DiffTask", "LOO", "OneDatasetDiffTask"]
    all_ev = sorted(set(d["eval"]))
    long_ev = [e for e in all_ev if e not in ("sciq", "trivia_qa")]

    def mo(method, evs):
        p = d[(d.method == method) & d.rung.isin(ood)].pivot_table(
            index="eval", columns="rung", values="prr")
        return p.loc[evs].mean(axis=1).values

    print(f"\n{'=' * 78}\nA.6 XL GRID (meta-llama, xl_master; SEPARATE POPULATION from PDL — never pooled)\n{'=' * 78}")
    res = {}
    for label, evs in [("xl_all10", all_ev), ("xl_long8_short_dropped", long_ev)]:
        x, y = mo("msp_min", evs), mo("SAPLMA", evs)
        prim = ols(x, y)
        rc, pc = stats.pearsonr(x, y - x)
        res[label] = dict(n=len(x), evals=list(evs), primary=prim,
                          coupled_r=rc, coupled_p=pc,
                          floor_min=float(x.min()), floor_max=float(x.max()),
                          floor_sd=float(x.std(ddof=1)))
        print(f"--- {label} (n={len(x)}) ---")
        print(f"  PRIMARY (uncoupled): b={prim['b']:+.4f} se={prim['se_b']:.4f} "
              f"r2={prim['r2']:.4f} p2={prim['p_two_sided']:.4f} "
              f"p(b<1)={prim['p_one_sided_b_lt_1']:.4f}")
        print(f"  COUPLED: r={rc:+.4f} p={pc:.5f}   <-- MATHEMATICALLY COUPLED, DO NOT QUOTE")
        print(f"  floor range [{x.min():+.4f}, {x.max():+.4f}]  SD={x.std(ddof=1):.4f}")
    out["xl_grid"] = res


def main():
    out = {}
    llama, n_chk = load_llama()
    print(f"[provenance] Llama: pdl_fam dedup reconciled against canonical master on "
          f"{n_chk} shared cells, max |diff| = 0.0")
    qwen, n_q = load_qwen()
    print(f"[provenance] Qwen: canonical master, {n_q} rows, carve=legacy")
    run_model("meta-llama/Meta-Llama-3.1-8B", llama, out)
    run_model("Qwen/Qwen2.5-14B", qwen, out)
    run_xl(out)

    dest = f"{REPO}/results/analysis/selection_law_audit.json"
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    sys.exit(main())
