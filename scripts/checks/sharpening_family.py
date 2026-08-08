#!/usr/bin/env python
"""W1 -- THE TRAINING-FREE SHARPENING FAMILY between `perplexity` and `msp_min`.

Pre-registration: prereg/W1_sharpening_axis.md  (written and committed BEFORE this file was run).
Plan: ../PLAN_sharpening_axis.md   Results doc: ../STOCKTAKE_sharpening_axis.md

WHAT THIS IS
------------
`perplexity` and `msp_min` are treated in the literature as two separate baselines. They are the two
endpoints of ONE expression, `q = sum_t w_t * nll_t`, with `w` uniform for perplexity and
all-on-one-token for msp_min. This sweeps the parameter that connects them and asks whether a single
PRE-COMMITTED value, fixed across all datasets, beats both.

    z_t   = (nll_t - mean(nll)) / std(nll)     standardised WITHIN each answer
    w     = softmax(tau * z)
    score = sum_t w_t * nll_t                  scored with the RAW nll, not the standardised one

    tau = 0    -> uniform weights   -> score = mean(nll)  = PERPLEXITY exactly
    tau -> inf -> all mass on max   -> score = max(nll)   = MSP_MIN's RANKING

The asymmetry is deliberate. Standardise INSIDE the weighting but score with the RAW nll: that keeps
both endpoints exact while making tau dimensionless, so the same tau means the same thing on another
model later.

TWO MORE FAMILIES WITH THE SAME ENDPOINTS (the registered control, §6 of the prereg). If softmax-tau
works and these do not, the result is about the specific weighting function rather than the family:
    power mean   M_p = (mean(nll^p))^(1/p)                p=1 -> mean, p->inf -> max
    Lehmer       sum(w*nll)/sum(w) with w = nll^beta       beta=0 -> mean, beta->inf -> max

⚠️ ENDPOINT GATE (V1). Printed BEFORE any curve is readable, and the run STOPS if it fails. This is a
RANKING identity, not a value identity: `msp.msp_uncertainty(lp,"min")` returns `1 - min(exp(lp))`,
a monotone transform of `max(nll)` and NOT equal to it. PRR is rank-based so the PRRs match exactly
while the score vectors do not. A check written as "values agree to 1e-6" would fail for the wrong
reason. See prereg §7.

⚠️ THE UNIT OF ANALYSIS IS THE DATASET, n = 8, NOT 40 cells. A training-free score never sees the
training pool, so its PRR is identical at all five rungs. A free-vs-free interval computed over
"32 OOD cells" is 8 values counted four times and is roughly twice too narrow.

Reads cached `record["token_logprobs"]` ONLY. No GPU, no per-token hidden states, no training.

    python scripts/checks/sharpening_family.py                 # everything
    python scripts/checks/sharpening_family.py --gate-only     # just V1-V6, no sweep
"""
import argparse
import csv as _csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats as _st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, msp, results              # noqa: E402
from luq.config import Config                    # noqa: E402
from xl_rungs import eval_split, label_of        # noqa: E402  (shim; LUQ_CARVE defaults to "legacy")
from attn_pool import PROMPT_REGIME              # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"

# The 8 long evals ARE the population. sciq/trivia_qa are carried ONLY as the external-selection arm
# (A2) and are NEVER pooled into the n=8 statistics -- they are a different regime (~3-token answers).
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
SHORT = ["sciq", "trivia_qa"]
DATASETS = LONG + SHORT

# Published master-table floors -- the EXTERNAL half of the endpoint gate. Source:
# results/pdl_master__meta-llama_Meta-Llama-3.1-8B.csv (free methods, identical at all 5 rungs).
# sciq/trivia_qa from the §B.2 motivation table, as topk_floor_sweep.py uses them.
PUBLISHED = {
    "pubmed_qa":     {"min": +0.3710, "ppl": -0.1736},
    "med_quad":      {"min": +0.1492, "ppl": +0.0771},
    "asqa":          {"min": +0.2498, "ppl": +0.3161},
    "xsum":          {"min": -0.0149, "ppl": -0.1719},
    "cnn_dailymail": {"min": +0.1198, "ppl": +0.4096},
    "samsum":        {"min": -0.0243, "ppl": +0.1128},
    "expertqa":      {"min": +0.2054, "ppl": +0.0393},
    "factscore":     {"min": +0.4283, "ppl": +0.3260},
    "sciq":          {"min": +0.755,  "ppl": +0.541},
    "trivia_qa":     {"min": +0.747,  "ppl": +0.733},
}

# The registered grids (prereg §5 and §6). np.inf is the msp_min endpoint in every family.
TAUS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf]
POWERS = [1.0, 2.0, 4.0, 8.0, 16.0, np.inf]
BETAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, np.inf]

TAU_A0 = 1.0                 # the PRE-COMMITTED a-priori value (prereg §4, Q1)
GATE_EXT_TOL = 0.01          # my floor vs the published master value (CSV rounded to 4dp)
GATE_INT_TOL = 1e-9          # tau=0 vs perplexity, tau=inf vs msp_min (identical by construction)
STD_EPS = 1e-12              # below this an answer's NLLs are all equal -> uniform, NEVER divide

BAR_MARGIN = 0.010           # registered Q1 thresholds, fixed before the run
BAR_SIGNS = 6                # out of 8
BAR_P = 0.05

OUT = ROOT / "results" / f"sharpening_family__{cache._slug(MODEL)}.csv"


# ----------------------------------------------------------------------------------------------
# Loading: records only. Mirrors topk_floor_sweep.load_light so the population is provably the same.
# ----------------------------------------------------------------------------------------------
def load_light(dataset):
    """records + split + y ONLY. The family needs `token_logprobs`, nothing else.

    ⚠️ expertqa / asqa / factscore live under cache/<name>_rp12/, so a hard-coded `cache/` glob would
    silently see 5 of 8. Resolution goes through Config(prompt_regime=...), exactly as attn_pool does.
    """
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    records = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, dataset, "ID"))
    lf = label_of(dataset)
    split = np.array([r["split"] for r in records])
    y = np.array([r.get(lf, np.nan) for r in records], dtype=float)
    finite = np.isfinite(y)                       # IDENTICAL to probedriftlong's finite-label filter
    if not finite.all():
        keep = np.where(finite)[0]
        records = [records[k] for k in keep]
        split = split[keep]
        y = y[keep]
    return records, split, y, lf


# ----------------------------------------------------------------------------------------------
# The three families. Each takes one answer's NLL vector and returns one score (higher = more
# uncertain), and each has mean(nll) at one end and max(nll) at the other.
# ----------------------------------------------------------------------------------------------
def softmax_weights(nll, tau):
    """w = softmax(tau * z) with z standardised WITHIN this answer. Returns average-... no, sums to 1.

    tau = 0    -> uniform (so the weighted mean IS the plain mean)
    tau = inf  -> a one-hot on the largest NLL
    std ~ 0    -> uniform, EXPLICITLY. Never a division by near-zero (prereg V2).
    """
    n = len(nll)
    if n == 1:
        return np.ones(1)
    if tau == 0.0:
        return np.full(n, 1.0 / n)
    if not np.isfinite(tau):                      # tau -> inf: all mass on the worst token
        w = np.zeros(n)
        w[int(np.argmax(nll))] = 1.0
        return w
    sd = float(nll.std())
    if sd < STD_EPS:                              # every token equally surprising -> nothing to sharpen
        return np.full(n, 1.0 / n)
    z = (nll - nll.mean()) / sd
    a = tau * z
    a = a - a.max()                               # standard log-sum-exp shift, for numerical safety
    w = np.exp(a)
    return w / w.sum()


def score_softmax(nll, tau):
    """The registered score: weights from the STANDARDISED NLL, but summed against the RAW NLL."""
    return float((softmax_weights(nll, tau) * nll).sum())


def score_power(nll, p):
    """Power mean M_p = (mean(nll^p))^(1/p). p=1 -> mean, p->inf -> max.

    NLL is >= 0 always (it is -log of a probability), so the powers are well defined. We divide by
    the max first so that nll^p cannot overflow at p=16.
    """
    m = float(nll.max())
    if m <= 0.0:                                  # every token had probability 1; the mean IS the max
        return 0.0
    if not np.isfinite(p):
        return m
    u = nll / m
    return float(m * (np.mean(u ** p) ** (1.0 / p)))


def score_lehmer(nll, beta):
    """Lehmer mean: sum(w*nll)/sum(w) with w = nll^beta. beta=0 -> mean, beta->inf -> max.

    This is the family `weighting.beta_sharpen` already implements (w proportional to s^beta), applied
    to the NLL as its own source score. Same max-normalisation trick for stability.
    """
    m = float(nll.max())
    if m <= 0.0:
        return 0.0
    if not np.isfinite(beta):
        return m
    u = nll / m
    if beta == 0.0:
        return float(nll.mean())
    num = float(np.sum(u ** (beta + 1.0)))
    den = float(np.sum(u ** beta))
    return float(m * num / den)


FAMILIES = {
    "softmax_tau": (TAUS, score_softmax),
    "power_p":     (POWERS, score_power),
    "lehmer_beta": (BETAS, score_lehmer),
}


# ----------------------------------------------------------------------------------------------
# Selection arms. A1/A2 must report what the PROCEDURE scores, never what the best value scores.
# ----------------------------------------------------------------------------------------------
def one_se_pick(curves, train_ds, grid):
    """The one-standard-error rule over a set of TRAINING datasets.

    With 8 noisy units the raw argmax is optimistic, so we take the best training mean, widen it by
    one standard error across those datasets, and then -- among everything inside that band -- pick
    the value NEAREST AN ENDPOINT of the grid. "Nearest an endpoint" is measured in grid-index steps,
    min(i, last - i), with ties broken toward whichever endpoint has the higher training mean. That
    tie-break is stated here because it is a free choice and it must not be made after seeing results.

    Returns the index into `grid`.
    """
    M = np.array([[curves[d][i] for i in range(len(grid))] for d in train_ds], dtype=float)
    mean = M.mean(axis=0)
    se = M.std(axis=0, ddof=1) / np.sqrt(len(train_ds)) if len(train_ds) > 1 else np.zeros_like(mean)
    best = int(np.argmax(mean))
    band = np.where(mean >= mean[best] - se[best])[0]
    last = len(grid) - 1
    dist = np.array([min(i, last - i) for i in band])
    cands = band[dist == dist.min()]
    if len(cands) == 1:
        return int(cands[0])
    # tie: two candidates equidistant from opposite ends -> take the end with the better training mean
    return int(cands[int(np.argmax(mean[cands]))])


def wilcoxon(diffs):
    """Two-sided Wilcoxon signed-rank on the paired per-dataset differences. n=8 -> min p = 0.0078.

    scipy raises if every difference is zero, which cannot happen here but is handled so an edge case
    returns a number rather than killing the run.
    """
    d = np.asarray(diffs, dtype=float)
    if np.allclose(d, 0.0):
        return 1.0
    try:
        return float(_st.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-only", action="store_true", help="run V1-V6 and stop, no sweep")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    carve = os.environ.get("LUQ_CARVE", "legacy")
    print("=" * 100)
    print("W1 -- THE TRAINING-FREE SHARPENING FAMILY   (prereg: prereg/W1_sharpening_axis.md)")
    print(f"model={MODEL}  LUQ_CARVE={carve}  population=8 long evals (n=8 datasets, NOT 32 cells)")
    print("V3 NLL CONVENTION: cached `token_logprobs` are NATURAL-LOG LOGPROBS (negative), one per")
    print("   generated token (length G, no prompt anchor). nll = -logprob, computed here.")
    print("=" * 100)

    # ---------------- V4: availability, asserted, fail loud on a short load ----------------
    data = {}
    for d in DATASETS:
        try:
            records, split, y, lf = load_light(d)
        except Exception as e:                    # a missing cache must CRASH, never become an empty row
            raise SystemExit(f"V4 FAIL [{d}]: could not load records -- {type(e).__name__}: {e}")
        _, te = eval_split(split)                 # seed 0, legacy carve -> the master table's test rows
        if len(te) == 0:
            raise SystemExit(f"V4 FAIL [{d}]: empty test split")
        nlls = [-np.asarray(records[i]["token_logprobs"], dtype=float) for i in te]
        data[d] = {"nll": nlls, "y": y[te], "label": lf, "n": len(te)}
    missing_long = [d for d in LONG if d not in data]
    if missing_long:
        raise SystemExit(f"V4 FAIL: realised long-dataset count {len(LONG) - len(missing_long)} != 8, "
                         f"missing {missing_long}")
    print(f"\nV4 AVAILABILITY: PASS -- all {len(LONG)} long + {len(SHORT)} short datasets loaded.")

    # ---------------- V2: degenerate cases, counted and reported ----------------
    print("\nV2 DEGENERATE CASES (reported, never silently handled)")
    print(f"{'dataset':16s}{'n_eval':>8s}{'med_len':>9s}{'n_std0':>9s}{'n_len<5':>9s}{'label':>14s}")
    for d in DATASETS:
        nl = data[d]["nll"]
        n_std0 = sum(1 for a in nl if len(a) < 2 or a.std() < STD_EPS)
        n_short = sum(1 for a in nl if len(a) < 5)
        med = float(np.median([len(a) for a in nl]))
        data[d]["med_len"] = med
        print(f"{d:16s}{data[d]['n']:>8d}{med:>9.1f}{n_std0:>9d}{n_short:>9d}{data[d]['label']:>14s}")
    print("  (std0 answers fall back to UNIFORM weights explicitly -- no division by ~0.)")

    # ---------------- V1: the endpoint gate. The run STOPS if this fails. ----------------
    print("\n" + "=" * 100)
    print("V1 ENDPOINT GATE -- a RANKING identity (equal PRR), NOT a value identity.")
    print("   msp_min is 1-min(prob), a monotone transform of max(nll); PRR is rank-based so the")
    print("   PRRs match exactly while the score vectors do not.")
    print("=" * 100)
    gate_ok = True
    for d in DATASETS:
        nl, y = data[d]["nll"], data[d]["y"]
        v_min = np.array([msp.msp_uncertainty(-a, "min") for a in nl])          # back to logprobs
        v_ppl = np.array([msp.msp_uncertainty(-a, "perplexity") for a in nl])
        p_min, p_ppl = results.prr(y, v_min), results.prr(y, v_ppl)
        v_t0 = np.array([score_softmax(a, 0.0) for a in nl])
        v_ti = np.array([score_softmax(a, np.inf) for a in nl])
        p_t0, p_ti = results.prr(y, v_t0), results.prr(y, v_ti)
        pub = PUBLISHED[d]
        d_ext_min, d_ext_ppl = abs(p_min - pub["min"]), abs(p_ppl - pub["ppl"])
        d_int_0, d_int_i = abs(p_t0 - p_ppl), abs(p_ti - p_min)
        ok = (d_ext_min < GATE_EXT_TOL and d_ext_ppl < GATE_EXT_TOL
              and d_int_0 < GATE_INT_TOL and d_int_i < GATE_INT_TOL)
        gate_ok = gate_ok and ok
        data[d]["prr_min"], data[d]["prr_ppl"] = p_min, p_ppl
        print(f"[{d:14s}] msp_min me {p_min:+.4f} vs master {pub['min']:+.4f} d{d_ext_min:.4f} | "
              f"ppl me {p_ppl:+.4f} vs master {pub['ppl']:+.4f} d{d_ext_ppl:.4f} | "
              f"tau0==ppl d{d_int_0:.1e}  tauInf==min d{d_int_i:.1e}  "
              f"{'PASS' if ok else 'FAIL <=='}", flush=True)
    if not gate_ok:
        raise SystemExit("\nV1 FAIL -- the NLL convention or the row population differs from the master "
                         "table. NOTHING downstream is interpretable. Stopping, as pre-registered.")
    print("\nV1 GATE: PASS -- the family is on the master table's population.")
    if args.gate_only:
        return

    # ---------------- the sweeps ----------------
    rows = []                                     # (family, dataset, param, prr, n_eval, med_len, carve)
    curves = {f: {} for f in FAMILIES}            # family -> dataset -> [prr per grid point]
    for fam, (grid, fn) in FAMILIES.items():
        for d in DATASETS:
            nl, y = data[d]["nll"], data[d]["y"]
            cs = []
            for g in grid:
                v = np.array([fn(a, g) for a in nl])
                p = results.prr(y, v)
                cs.append(p)
                rows.append((fam, d, "inf" if not np.isfinite(g) else g, p,
                             data[d]["n"], data[d]["med_len"], carve))
            curves[fam][d] = cs

    # ---------------- §3.1 the full curve per dataset (the shape IS the evidence) ----------------
    for fam, (grid, _) in FAMILIES.items():
        print("\n" + "=" * 100)
        print(f"FULL CURVE -- family={fam}   (population: 8 long evals; short-form shown separately, "
              f"NEVER pooled)")
        print("=" * 100)
        hdr = "".join(f"{('inf' if not np.isfinite(g) else g):>9}" for g in grid)
        print(f"{'dataset':16s}{hdr}")
        for d in LONG:
            print(f"{d:16s}" + "".join(f"{p:>+9.3f}" for p in curves[fam][d]))
        m = [float(np.mean([curves[fam][d][i] for d in LONG])) for i in range(len(grid))]
        print(f"{'MEAN (n=8)':16s}" + "".join(f"{p:>+9.3f}" for p in m))
        print(f"{'-- short-form (A2 selection only, not pooled) --':16s}")
        for d in SHORT:
            print(f"{d:16s}" + "".join(f"{p:>+9.3f}" for p in curves[fam][d]))

    # ---------------- §3.2 the four arms, for the PRIMARY family ----------------
    fam = "softmax_tau"
    grid = TAUS
    i_a0 = grid.index(TAU_A0)
    base_min = np.array([data[d]["prr_min"] for d in LONG])
    base_ppl = np.array([data[d]["prr_ppl"] for d in LONG])

    def arm_report(name, per_ds_idx, note=""):
        """per_ds_idx: dataset -> index into `grid`. Reports what the PROCEDURE scores."""
        vals = np.array([curves[fam][d][per_ds_idx[d]] for d in LONG])
        dm = vals - base_min
        both = int(np.sum((vals > base_min) & (vals > base_ppl)))
        print(f"{name:34s}{vals.mean():>+10.4f}{dm.mean():>+12.4f}"
              f"{int(np.sum(dm > 0)):>10d}/8{wilcoxon(dm):>10.4f}{both:>10d}/8   {note}")
        return vals, dm, both

    print("\n" + "=" * 100)
    print("THE FOUR ARMS -- softmax-tau. NEVER POOLED. n = 8 DATASETS, not 32 cells.")
    print("=" * 100)
    print(f"{'arm':34s}{'mean PRR':>10s}{'vs msp_min':>12s}{'signs':>12s}{'Wilcoxon p':>10s}"
          f"{'beat BOTH':>13s}")

    a0_vals, a0_dm, _ = arm_report(f"A0 a priori tau={TAU_A0} (PRIMARY)",
                                   {d: i_a0 for d in LONG}, "committed before the run")

    # A1: leave-one-dataset-out over the 8 long evals, one-standard-error rule.
    a1_idx, a1_pick = {}, {}
    for d in LONG:
        tr = [o for o in LONG if o != d]
        j = one_se_pick(curves[fam], tr, grid)
        a1_idx[d] = j
        a1_pick[d] = grid[j]
    arm_report("A1 leave-one-dataset-out", a1_idx)

    # A2: selected on the two SHORT-FORM sets only -- touches no evaluation data at all.
    j2 = one_se_pick(curves[fam], SHORT, grid)
    arm_report(f"A2 external (sciq+trivia) tau={grid[j2]}", {d: j2 for d in LONG})

    # A3: ORACLE. Ceiling only. Never a result, never in a column with A0-A2.
    o_idx = {d: int(np.argmax(curves[fam][d])) for d in LONG}
    arm_report("A3 ORACLE tau (CEILING, NOT A RESULT)", o_idx, "<-- oracle, label-selected")

    print(f"{'msp_min (the pre-registered bar)':34s}{base_min.mean():>+10.4f}{0.0:>+12.4f}")
    print(f"{'perplexity':34s}{base_ppl.mean():>+10.4f}{(base_ppl-base_min).mean():>+12.4f}")
    print(f"{'ORACLE best-of-two-endpoints':34s}"
          f"{np.maximum(base_min, base_ppl).mean():>+10.4f}   <-- oracle, labelled, never a bar")

    print("\nA1 selected tau per fold (agreement across folds is the evidence that ONE global constant")
    print("exists; disagreement is a FINDING, not a nuisance):")
    print("  " + "  ".join(f"{d}={a1_pick[d]}" for d in LONG))
    picks = [a1_pick[d] for d in LONG]
    fin = [p for p in picks if np.isfinite(p)]
    print(f"  distinct values: {sorted(set(str(p) for p in picks))}   "
          f"(finite spread: {min(fin) if fin else float('nan')} to {max(fin) if fin else float('nan')})")

    # ---------------- §3.3 the Q1 verdict, against the bar fixed in the pre-registration -------
    margin = float(a0_dm.mean())
    signs = int(np.sum(a0_dm > 0))
    pval = wilcoxon(a0_dm)
    passed = (margin > BAR_MARGIN) and (signs >= BAR_SIGNS) and (pval < BAR_P)
    print("\n" + "=" * 100)
    print("Q1 VERDICT against the PRE-REGISTERED bar (all three parts required)")
    print("=" * 100)
    print(f"  margin vs msp_min  {margin:+.4f}   (bar: > +{BAR_MARGIN:.3f})   "
          f"{'PASS' if margin > BAR_MARGIN else 'FAIL'}")
    print(f"  sign count         {signs}/8        (bar: >= {BAR_SIGNS}/8)      "
          f"{'PASS' if signs >= BAR_SIGNS else 'FAIL'}")
    print(f"  Wilcoxon p         {pval:.4f}     (bar: < {BAR_P})        "
          f"{'PASS' if pval < BAR_P else 'FAIL'}")
    print(f"\n  Q1: {'YES' if passed else 'NO'}")
    if not passed and margin > BAR_MARGIN:
        print("  ⚠️ Margin passed but the test did not. Pre-registered reading: NOT ESTABLISHED.")
        print("     This is not re-described as directional support and no new threshold is invented.")

    # ---------------- §3.4 Q2: does best-tau order the datasets as predicted? ----------------
    HIGH = ["pubmed_qa", "factscore"]              # registered prediction: concentrated -> HIGH tau
    LOW = ["cnn_dailymail", "asqa", "samsum"]      # registered prediction: spread -> LOW tau
    rank = {d: int(np.argmax(curves[fam][d])) for d in LONG}
    pairs = [(h, l) for h in HIGH for l in LOW]
    correct = sum(1 for h, l in pairs if rank[h] > rank[l])
    print("\n" + "=" * 100)
    print("Q2 -- does best-tau order the datasets as PREDICTED (registered before looking)?")
    print("=" * 100)
    print("  prediction: pubmed_qa, factscore -> HIGH tau ; cnn_dailymail, asqa, samsum -> LOW tau")
    for d in LONG:
        print(f"    {d:16s} best tau = {grid[rank[d]]}")
    print(f"  cross-group pairs correct: {correct}/{len(pairs)}  "
          f"({'ordering supported' if correct == len(pairs) else 'ordering NOT supported'})")

    # ---------------- §3.5 the family control ----------------
    print("\n" + "=" * 100)
    print("FAMILY CONTROL -- same two endpoints, three different weightings.")
    print("If softmax-tau works and these do not, the result is about the WEIGHTING FUNCTION,")
    print("not about the family. (Registered in prereg §6, not added afterwards.)")
    print("=" * 100)
    print(f"{'family':16s}{'best fixed param':>20s}{'mean PRR (n=8)':>18s}{'vs msp_min':>13s}")
    for f2, (g2, _) in FAMILIES.items():
        means = [float(np.mean([curves[f2][d][i] for d in LONG])) for i in range(len(g2))]
        b = int(np.argmax(means))
        print(f"{f2:16s}{str('inf' if not np.isfinite(g2[b]) else g2[b]):>20s}"
              f"{means[b]:>+18.4f}{means[b] - base_min.mean():>+13.4f}")
    print("  ⚠️ softmax-tau standardises and is dimensionless; power/Lehmer act on raw NLL magnitudes")
    print("     and are not. They are like-for-like on ENDPOINTS only, not on scale invariance.")

    # ---------------- §3.6 diagnostics: concentration, length, ZGAP, and the cnn control -------
    print("\n" + "=" * 100)
    print("DIAGNOSTICS at tau = 1")
    print("  ESS = 1/sum(w^2) is the effective number of tokens the weight actually lands on.")
    print("  ⚠️ REGISTERED CONFOUND: the largest attainable z is bounded by about sqrt(n-1), so at a")
    print("     FIXED tau a long answer can concentrate far more than a short one. The family is")
    print("     therefore implicitly LENGTH-dependent. 'Length is not the axis' must be checked here,")
    print("     not asserted.")
    print("=" * 100)
    print(f"{'dataset':16s}{'ESS':>9s}{'ESS/len':>10s}{'med_len':>9s}{'ZGAP':>8s}"
          f"{'rho(ESS,len)':>14s}{'rho(ESS,ZGAP)':>15s}")
    for d in LONG:
        nl = data[d]["nll"]
        ess, lens, zg = [], [], []
        for a in nl:
            w = softmax_weights(a, TAU_A0)
            ess.append(1.0 / float((w ** 2).sum()))
            lens.append(len(a))
            sd = a.std()
            zg.append(float((a.max() - a.mean()) / sd) if sd > STD_EPS else 0.0)
        ess, lens, zg = np.array(ess), np.array(lens, float), np.array(zg)
        r_len = _st.spearmanr(ess, lens).statistic if lens.std() > 0 else float("nan")
        r_zg = _st.spearmanr(ess, zg).statistic if zg.std() > 0 else float("nan")
        print(f"{d:16s}{ess.mean():>9.2f}{(ess/np.maximum(lens,1)).mean():>10.3f}"
              f"{data[d]['med_len']:>9.1f}{zg.mean():>8.2f}{r_len:>14.3f}{r_zg:>15.3f}")
        rows.append(("diag", d, "ess_tau1", float(ess.mean()), data[d]["n"], data[d]["med_len"], carve))

    i_cnn = LONG.index("cnn_dailymail")
    d_cnn = float(a0_vals[i_cnn] - base_min[i_cnn])
    d_cnn_ppl = float(a0_vals[i_cnn] - base_ppl[i_cnn])
    print("\n" + "=" * 100)
    print("Q4 -- THE REGISTERED cnn_dailymail CONTROL")
    print("=" * 100)
    print("  R1 (prereg/R1_taxonomy_label_free.md) was FALSIFIED with cnn_dailymail, the most SPREAD")
    print("  dataset, carrying the HIGHEST ZGAP (3.93). So a fixed tau should concentrate cnn the most")
    print("  and push it toward msp_min -- the WRONG way, on the largest endpoint margin in the grid.")
    print(f"  cnn at tau=1: {a0_vals[i_cnn]:+.4f}   vs msp_min {d_cnn:+.4f}   vs perplexity {d_cnn_ppl:+.4f}")
    if passed and d_cnn_ppl > 0:
        print("  ⚠️⚠️ Q1 PASSED **and** cnn improved against perplexity. That CONTRADICTS R1.")
        print("     Pre-registered reading: treat as a SUSPECTED BUG and find the cause BEFORE")
        print("     reporting this as a result.")
    else:
        print("  (consistent with R1's direction)" if d_cnn_ppl <= 0 else "  (Q1 did not pass; no conflict)")

    # ---------------- write the CSV ----------------
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["family", "dataset", "param", "prr", "n_eval", "med_len", "carve"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {outp}  ({len(rows)} rows, carve={carve})")
    print("\nPopulation caption for every table above: widened cells_long, "
          "meta-llama/Llama-3.1-8B, judge label, legacy carve, n = 8 DATASETS.")


if __name__ == "__main__":
    main()
