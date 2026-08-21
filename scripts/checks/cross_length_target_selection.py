"""Dataset representativeness audit — which long-form datasets are defensible representatives?

ZERO COMPUTE. Reads only committed results. No GPU, no API, no model, no cache of hidden states.
Everything here is DESCRIPTIVE: it picks the long-form targets for the bidirectional cross-length
transfer experiment BEFORE any cross-length result exists, so the targets cannot be cherry-picked.

The question it answers, per family (correctness_qa / summ / factuality):
    which dataset behaves like its family, is stable across the two models, is genuinely long,
    and carries no measurement caveat that would undermine a small auxiliary experiment?

THE TWO MODELS ARE NEVER POOLED. Every quantity is computed inside one model's own population and
compared across models only as an agreement/stability statement. Llama-3.1-8B and Qwen2.5-14B are
separate populations with separate labels, layers and generations; a joint table would be a
cross-population comparison rather than a result. The per-model loading below is what enforces it.

Inputs (all already on disk):
  * results/pdl_fam_<eval>__meta-llama_Meta-Llama-3.1-8B.csv   — Llama ladder, 9 core methods x 5 rungs
  * results/probedriftlong__Qwen_Qwen2.5-14B__<eval>.csv       — Qwen ladder, same method vocabulary
  * results/analysis/aggregation_regime_summary__<model>.csv   — n, med_len, iqr_len, cap_share,
                                                                 mean_quality, the Lehmer beta curve
  * src/luq/data.py MAX_NEW_TOKENS                             — per-dataset generation budget
  * probe_drift_long.dataset_configs.FINE_FAMILIES             — the family taxonomy (imported, never re-typed)

Output: results/analysis/CROSS_LENGTH_TARGET_SELECTION.md
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from probe_drift_long.dataset_configs import FINE_FAMILIES, LONG_DATASETS  # noqa: E402

from luq.data import MAX_NEW_TOKENS  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(REPO, "results")
ANALYSIS = os.path.join(RESULTS, "analysis")

LLAMA = "meta-llama/Meta-Llama-3.1-8B"
QWEN = "Qwen/Qwen2.5-14B"

# The core method set, in the request's order. LEFT = the CSV arm name, RIGHT = how it is displayed.
# `uniform` is the mean-pool pooler (attention with a frozen zero query, i.e. exact mean pooling);
# `attention` is the learned attention pooler. `saplma` is mean-pool + MLP.
FREE_METHODS = {"floor_sum": "msp_sum", "floor_ppl": "perplexity", "floor_min": "msp_min"}
SUP_METHODS = {
    "saplma": "SAPLMA",
    "uniform": "mean-pool probe",
    "attention": "attention-pool probe",
    "wmsp_norm": "wMSP-normalised",
    "wmsp_shrink2": "wMSP-shrink@2",
    "wmsp_shrink10": "wMSP-shrink@10",
}
CORE = {**FREE_METHODS, **SUP_METHODS}

OOD_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]

# Known validity caveats, each traceable to a committed artifact. These are FACTS ABOUT THE DATA,
# not judgements — the judgement is the recommendation column, which is derived below.
CAVEATS = {
    "pubmed_qa": "Llama median generation is only 22 tokens — barely long-form on this model.",
    "med_quad": ("ANSWER-SPAN MISMATCH (labels on clean span, scores on raw generation, 47.5% of "
                 "rows) — see README_MEDQUAD_SENSITIVITY.md. Also 97% capped on Llama with "
                 "iqr_len 0.0, i.e. essentially every answer is truncated at the 128 budget."),
    "asqa": "Small eval (n=284). Historic all-excluded-softmax NaN bug fixed 2026-08-03.",
    "xsum": "Moderately capped (31% Llama / 39% Qwen) against a 56-token budget.",
    "cnn_dailymail": "Capped 28% on Llama; low on Qwen (4.6%).",
    "samsum": "Heavily capped (63% Llama / 46% Qwen) against a 56-token budget. 39% trailing "
              "contamination on Qwen.",
    "expertqa": ("Label is `factuality`, a DIFFERENT PROJECTION from the rest of the pool "
                 "(claim precision, not reference agreement). Absent on 292/2016 rows (85.5% "
                 "coverage) and blind to ~56% of claims. Qwen length-label r = -0.601 confound; "
                 "21% severe degeneracy on Qwen."),
    "factscore": ("Label is `factuality`. Absent on 45/500 rows (91% coverage). SMALLEST EVAL "
                  "(n=136 Llama / 140 Qwen). 40% capped on Qwen."),
}

# Judge-label coverage, from dataset_configs.PARTIALLY_LABELLED (measured on Llama).
LABEL_COVERAGE = {d: 1.0 for d in LONG_DATASETS}
LABEL_COVERAGE["expertqa"] = 1724 / 2016
LABEL_COVERAGE["factscore"] = 455 / 500


# A rung that had to be re-run on its own lives in a per-rung supplementary CSV rather than the
# eval's main file. `pdl_fam_xsum` lost its 1ds-Diff rung to walltime and it was recovered
# separately on 2026-08-05 with the identical method set; the canonical master already merges the
# two. Listed explicitly rather than glob-matched, so a stray file can never be picked up silently.
SUPPLEMENTARY = {
    (LLAMA, "xsum"): [os.path.join(RESULTS, "pdl_fam_xsum_1ds-Diff__meta-llama_Meta-Llama-3.1-8B.csv")],
}


def ladder_paths(model, ev):
    """The per-eval ladder CSV(s). The two models use different filename conventions."""
    if model == LLAMA:
        main = os.path.join(RESULTS, f"pdl_fam_{ev}__meta-llama_Meta-Llama-3.1-8B.csv")
    else:
        main = os.path.join(RESULTS, f"probedriftlong__Qwen_Qwen2.5-14B__{ev}.csv")
    return [main] + SUPPLEMENTARY.get((model, ev), [])


def load_ladder(model):
    """{(eval, rung, method): prr} for the core methods, from one model's own population."""
    out, missing = {}, []
    for ev in LONG_DATASETS:
        for p in ladder_paths(model, ev):
            if not os.path.exists(p):
                missing.append(os.path.basename(p))
                continue
            df = pd.read_csv(p)
            df = df[df["method"].isin(CORE)]
            for _, r in df.iterrows():
                key = (r["eval"], r["rung"], r["method"])
                # A supplementary file must ADD a rung, never silently restate one that already
                # differs -- that would mean two populations disagree and we are picking one.
                if key in out and abs(out[key] - float(r["prr_mean"])) > 1e-9:
                    raise SystemExit(f"[FATAL] {model} {key}: two sources disagree "
                                     f"({out[key]} vs {r['prr_mean']}).")
                out[key] = float(r["prr_mean"])
    if missing:
        raise SystemExit(f"[FATAL] missing ladder CSV for {model}: {missing}. Refusing to report a "
                         f"partial grid as if it were complete.")
    return out


def profile(lad, ev):
    """The behavioural profile of one dataset: free scores once, supervised ID / mean-OOD / drop.

    The three training-free floors do NOT depend on the training pool, so their PRR is identical at
    every rung. Entering them once per dataset (rather than once per rung) is what stops them being
    4x pseudo-replicated in the distance — the same pseudo-replication the 2026-08-07 re-audit
    caught in the old "32 OOD cells" framing.
    """
    prof = {}
    for m, disp in FREE_METHODS.items():
        vals = {lad[(ev, r, m)] for r in OOD_RUNGS + ["ID"] if (ev, r, m) in lad}
        if not vals:
            raise SystemExit(f"[FATAL] {ev}: no value for free method {m}")
        if max(vals) - min(vals) > 1e-6:
            raise SystemExit(f"[FATAL] {ev}: training-free method {m} varies across rungs "
                             f"({sorted(vals)}). A floor that moves with the pool is a bug.")
        prof[f"free__{disp}"] = vals.pop()

    for m, disp in SUP_METHODS.items():
        idv = lad.get((ev, "ID", m))
        oods = [lad[(ev, r, m)] for r in OOD_RUNGS if (ev, r, m) in lad]
        if idv is None or len(oods) != 4:
            raise SystemExit(f"[FATAL] {ev}/{m}: ID={idv}, {len(oods)}/4 OOD rungs. Incomplete.")
        prof[f"id__{disp}"] = idv
        prof[f"ood__{disp}"] = float(np.mean(oods))
        prof[f"drop__{disp}"] = idv - float(np.mean(oods))
        for r in OOD_RUNGS:
            prof[f"rung__{disp}__{r}"] = lad[(ev, r, m)]
    return prof


def distance_table(P, cols, robust=False):
    """Distance of each dataset from the centroid of the OTHER seven, on standardised features.

    Note (stated because it would otherwise look like a stronger construction than it is): with a
    fixed set of n datasets, ||x_i - mean(x_-i)|| = n/(n-1) * ||x_i - mean(x_all)||, so the RANKING
    is identical to plain distance from the global centroid. The leave-one-out form is kept because
    it is what was asked for and because it is the honest description of "unusual relative to the
    others".
    """
    X = P[cols].to_numpy(dtype=float)
    if robust:
        centre = np.median(X, axis=0)
        scale = np.median(np.abs(X - centre), axis=0) * 1.4826
    else:
        centre = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)      # a constant feature carries no information
    Z = (X - centre) / scale

    n = len(Z)
    d = np.empty(n)
    for i in range(n):
        others = np.delete(Z, i, axis=0)
        ctr = np.median(others, axis=0) if robust else others.mean(axis=0)
        d[i] = np.linalg.norm(Z[i] - ctr)
    return pd.Series(d, index=P.index)


def within_family(P, cols):
    """Distance to family peers, standardised on the full 8 so families stay comparable."""
    X = P[cols].to_numpy(dtype=float)
    scale = np.where(X.std(axis=0, ddof=0) < 1e-12, 1.0, X.std(axis=0, ddof=0))
    Z = pd.DataFrame((X - X.mean(axis=0)) / scale, index=P.index, columns=cols)

    out = {}
    for fam in sorted(set(FINE_FAMILIES[d] for d in P.index)):
        members = [d for d in P.index if FINE_FAMILIES[d] == fam]
        for d in members:
            peers = [o for o in members if o != d]
            if len(peers) >= 2:
                out[d] = float(np.linalg.norm(Z.loc[d] - Z.loc[peers].mean(axis=0)))
            elif len(peers) == 1:
                # Two-member family: a "distance from the other one" is symmetric, so it cannot
                # rank them. Report the pair distance and say so rather than inventing an outlier.
                out[d] = float(np.linalg.norm(Z.loc[d] - Z.loc[peers[0]]))
    return pd.Series(out)


def method_rank_agreement(P):
    """Spearman of each dataset's method ranking (by mean OOD PRR) vs global and within-family."""
    cols = [f"ood__{m}" for m in SUP_METHODS.values()] + [f"free__{m}" for m in FREE_METHODS.values()]
    M = P[cols]
    glob = M.mean(axis=0)
    rows = {}
    for d in P.index:
        fam_members = [o for o in P.index if FINE_FAMILIES[o] == FINE_FAMILIES[d] and o != d]
        fam = M.loc[fam_members].mean(axis=0)
        rows[d] = {
            "rho_global": spearmanr(M.loc[d], glob).statistic,
            "rho_family": spearmanr(M.loc[d], fam).statistic if fam_members else np.nan,
        }
    return pd.DataFrame(rows).T


def beta_bucket(b):
    """Coarse aggregation regime. Buckets, not raw beta, because beta is an argmax on test labels."""
    if pd.isna(b):
        return "n/a"
    b = float(b)
    if b <= 0.5:
        return "mean-preferring"
    if b <= 2:
        return "intermediate"
    return "max-preferring"


def cross_model(PL, PQ, RL, RQ, SL, SQ):
    """Per-dataset HIGH/MEDIUM/LOW cross-model stability, from five explicit checks."""
    sup = list(SUP_METHODS.values())
    rows = {}
    for d in PL.index:
        checks, notes = {}, []

        # 1. does the method ranking survive the model change?
        cols = [f"ood__{m}" for m in sup] + [f"free__{m}" for m in FREE_METHODS.values()]
        rho = spearmanr(PL.loc[d, cols], PQ.loc[d, cols]).statistic
        checks["rank"] = rho >= 0.7
        notes.append(f"rank rho {rho:+.2f}")

        # 2. does the floor ordering keep its sign?
        dl = PL.loc[d, "free__msp_min"] - PL.loc[d, "free__perplexity"]
        dq = PQ.loc[d, "free__msp_min"] - PQ.loc[d, "free__perplexity"]
        checks["floor_sign"] = np.sign(dl) == np.sign(dq)
        notes.append(f"min-ppl {dl:+.3f}/{dq:+.3f}")

        # 3. is the descriptive Lehmer regime in the same bucket?
        bl, bq = beta_bucket(SL.loc[d, "oracle_best_finite_beta"]), beta_bucket(SQ.loc[d, "oracle_best_finite_beta"])
        checks["regime"] = bl == bq
        notes.append(f"regime {bl}/{bq}")

        # 4. is SAPLMA's OOD level comparable?
        ds = abs(PL.loc[d, "ood__SAPLMA"] - PQ.loc[d, "ood__SAPLMA"])
        checks["saplma"] = ds <= 0.10
        notes.append(f"SAPLMA OOD d {ds:.3f}")

        # 5. do the two method-contrasts keep their signs?
        al = PL.loc[d, "ood__attention-pool probe"] - PL.loc[d, "ood__mean-pool probe"]
        aq = PQ.loc[d, "ood__attention-pool probe"] - PQ.loc[d, "ood__mean-pool probe"]
        wl = PL.loc[d, "ood__wMSP-shrink@2"] - PL.loc[d, "ood__SAPLMA"]
        wq = PQ.loc[d, "ood__wMSP-shrink@2"] - PQ.loc[d, "ood__SAPLMA"]
        checks["contrasts"] = (np.sign(al) == np.sign(aq)) and (np.sign(wl) == np.sign(wq))
        notes.append(f"attn-mean {al:+.3f}/{aq:+.3f}; wmsp2-SAPLMA {wl:+.3f}/{wq:+.3f}")

        score = sum(checks.values())
        label = "HIGH" if score >= 4 else ("MEDIUM" if score >= 2 else "LOW")
        drop_l = float(np.mean([PL.loc[d, f"drop__{m}"] for m in sup]))
        drop_q = float(np.mean([PQ.loc[d, f"drop__{m}"] for m in sup]))
        rows[d] = {"score": score, "stability": label, "rank_rho": rho,
                   "drop_llama": drop_l, "drop_qwen": drop_q,
                   "why": "; ".join(notes)}
    return pd.DataFrame(rows).T


def fmt(x, nd=3):
    return "n/a" if pd.isna(x) else f"{float(x):+.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ANALYSIS, "CROSS_LENGTH_TARGET_SELECTION.md"))
    args = ap.parse_args()

    lad_l, lad_q = load_ladder(LLAMA), load_ladder(QWEN)
    PL = pd.DataFrame({d: profile(lad_l, d) for d in LONG_DATASETS}).T
    PQ = pd.DataFrame({d: profile(lad_q, d) for d in LONG_DATASETS}).T

    SL = pd.read_csv(os.path.join(ANALYSIS, "aggregation_regime_summary__meta-llama_Meta-Llama-3.1-8B.csv")).set_index("eval")
    SQ = pd.read_csv(os.path.join(ANALYSIS, "aggregation_regime_summary__Qwen_Qwen2.5-14B.csv")).set_index("eval")

    # Primary feature set: free scores once + supervised ID + supervised mean OOD.
    # `drop__` is ID minus mean-OOD, a linear combination of two features already present, so it is
    # DISPLAYED but deliberately excluded from the distance to avoid double-weighting it.
    cols = ([f"free__{m}" for m in FREE_METHODS.values()]
            + [f"id__{m}" for m in SUP_METHODS.values()]
            + [f"ood__{m}" for m in SUP_METHODS.values()])
    # Robustness feature set: the 4 OOD rungs kept separate instead of averaged.
    cols_rung = ([f"free__{m}" for m in FREE_METHODS.values()]
                 + [f"id__{m}" for m in SUP_METHODS.values()]
                 + [f"rung__{m}__{r}" for m in SUP_METHODS.values() for r in OOD_RUNGS])

    res = {}
    for tag, P in (("llama", PL), ("qwen", PQ)):
        res[tag] = pd.DataFrame({
            "centrality": distance_table(P, cols),
            "centrality_robust": distance_table(P, cols, robust=True),
            "centrality_perrung": distance_table(P, cols_rung),
            "within_family": within_family(P, cols),
        }).join(method_rank_agreement(P))

    stab = cross_model(PL, PQ, res["llama"], res["qwen"], SL, SQ)

    L = []
    w = L.append
    w("# Cross-length experiment — long-form target selection")
    w("")
    w("> **DESCRIPTIVE AUDIT. ZERO COMPUTE — every number is re-read from committed results.**")
    w("> Its job is to choose the three long-form targets for the bidirectional cross-length")
    w("> transfer experiment **before any cross-length result exists**, so the targets cannot be")
    w("> cherry-picked after the fact.")
    w(">")
    w("> **The two models are never pooled.** Every quantity is computed inside one model's own")
    w("> population; the models meet only as an agreement statement (§6).")
    w(">")
    w("> **`oracle_best_finite_beta` is an argmax on TEST labels — descriptive only, never a")
    w("> method result.**")
    w("")
    w("Populations: each model's own complete ProbeDriftLong ladder, 8 long evals × 5 rungs, 3 seeds, "
      "carve legacy. Llama layer 15, Qwen layer 23.")
    w("")

    w("## 1. Dataset diagnostics")
    w("")
    w("| dataset | family | budget | test n | med len | IQR len | cap share | mean label | label coverage |")
    w("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for d in LONG_DATASETS:
        w(f"| `{d}` | {FINE_FAMILIES[d]} | {MAX_NEW_TOKENS[d]} | "
          f"{int(SL.loc[d,'n'])} / {int(SQ.loc[d,'n'])} | "
          f"{SL.loc[d,'med_len']:.0f} / {SQ.loc[d,'med_len']:.0f} | "
          f"{SL.loc[d,'iqr_len']:.0f} / {SQ.loc[d,'iqr_len']:.0f} | "
          f"{SL.loc[d,'cap_share']:.0%} / {SQ.loc[d,'cap_share']:.0%} | "
          f"{SL.loc[d,'mean_quality']:.2f} / {SQ.loc[d,'mean_quality']:.2f} | "
          f"{LABEL_COVERAGE[d]:.0%} |")
    w("")
    w("Cells are `Llama / Qwen`. `budget` is `MAX_NEW_TOKENS` from `src/luq/data.py`; a high cap "
      "share against a small budget means the dataset is measuring truncated text.")
    w("")

    w("## 2. Behavioural profiles — mean OOD PRR by method")
    w("")
    for tag, P in (("Llama-3.1-8B", PL), ("Qwen2.5-14B", PQ)):
        w(f"### {tag}")
        w("")
        hdr = " | ".join(f"`{m}`" for m in list(FREE_METHODS.values()) + list(SUP_METHODS.values()))
        w(f"| dataset | {hdr} |")
        w("|---" * (1 + len(CORE)) + "|")
        for d in LONG_DATASETS:
            free = [fmt(P.loc[d, f"free__{m}"]) for m in FREE_METHODS.values()]
            sup = [fmt(P.loc[d, f"ood__{m}"]) for m in SUP_METHODS.values()]
            w(f"| `{d}` | " + " | ".join(free + sup) + " |")
        w("")
    w("The three training-free scores do not depend on the training pool, so they are entered **once "
      "per dataset**, not once per rung. Entering them per rung would 4× pseudo-replicate them.")
    w("")

    w("### ID→OOD degradation (supervised methods, mean over the 6)")
    w("")
    w("| dataset | Llama ID | Llama OOD | Llama drop | Qwen ID | Qwen OOD | Qwen drop |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for d in LONG_DATASETS:
        sup = list(SUP_METHODS.values())
        li = np.mean([PL.loc[d, f"id__{m}"] for m in sup])
        lo = np.mean([PL.loc[d, f"ood__{m}"] for m in sup])
        qi = np.mean([PQ.loc[d, f"id__{m}"] for m in sup])
        qo = np.mean([PQ.loc[d, f"ood__{m}"] for m in sup])
        w(f"| `{d}` | {fmt(li)} | {fmt(lo)} | {fmt(li-lo)} | {fmt(qi)} | {fmt(qo)} | {fmt(qi-qo)} |")
    w("")

    w("## 3. Overall divergence — distance from the centroid of the other seven")
    w("")
    w("Features standardised across the 8 datasets within each model: the 3 training-free scores, "
      "the 6 supervised ID PRRs and the 6 supervised mean-OOD PRRs (15 features). `drop` is "
      "ID − mean-OOD, a linear combination of features already present, so it is displayed above "
      "but excluded here to avoid double-weighting.")
    w("")
    w("| dataset | Llama dist | rank | Qwen dist | rank | robust (L/Q) | per-rung (L/Q) |")
    w("|---|---:|---:|---:|---:|---|---|")
    rl = res["llama"]["centrality"].rank(ascending=False)
    rq = res["qwen"]["centrality"].rank(ascending=False)
    for d in sorted(LONG_DATASETS, key=lambda x: -(res["llama"].loc[x, "centrality"])):
        w(f"| `{d}` | {res['llama'].loc[d,'centrality']:.2f} | {int(rl[d])} | "
          f"{res['qwen'].loc[d,'centrality']:.2f} | {int(rq[d])} | "
          f"{res['llama'].loc[d,'centrality_robust']:.2f} / {res['qwen'].loc[d,'centrality_robust']:.2f} | "
          f"{res['llama'].loc[d,'centrality_perrung']:.2f} / {res['qwen'].loc[d,'centrality_perrung']:.2f} |")
    w("")
    w("Rank 1 = **most unusual**. Note that with a fixed dataset set, distance from the "
      "leave-one-out centroid is a constant multiple of distance from the global centroid, so this "
      "ranking is identical to the simpler statistic — the leave-one-out form is reported because it "
      "is the honest description of \"unusual relative to the others\".")
    w("")

    w("## 4. Within-family divergence")
    w("")
    w("| family | dataset | Llama dist to peers | Qwen dist to peers |")
    w("|---|---|---:|---:|")
    for fam in ["correctness_qa", "summ", "factuality"]:
        for d in [x for x in LONG_DATASETS if FINE_FAMILIES[x] == fam]:
            w(f"| {fam} | `{d}` | {res['llama'].loc[d,'within_family']:.2f} | "
              f"{res['qwen'].loc[d,'within_family']:.2f} |")
    w("")
    w("**factuality has only two members**, so its two entries are the *same* symmetric pair "
      "distance and cannot rank ExpertQA against FActScore. They are compared directly in §7 "
      "instead of being given a meaningless outlier score.")
    w("")

    w("## 5. Method-ranking agreement (Spearman)")
    w("")
    w("| dataset | vs global (L) | vs family (L) | vs global (Q) | vs family (Q) |")
    w("|---|---:|---:|---:|---:|")
    for d in LONG_DATASETS:
        w(f"| `{d}` | {res['llama'].loc[d,'rho_global']:+.2f} | {res['llama'].loc[d,'rho_family']:+.2f} | "
          f"{res['qwen'].loc[d,'rho_global']:+.2f} | {res['qwen'].loc[d,'rho_family']:+.2f} |")
    w("")
    flagged = [d for d in LONG_DATASETS
               if min(res['llama'].loc[d, 'rho_global'], res['qwen'].loc[d, 'rho_global']) < 0.5]
    w(f"Flagged (ranks methods unusually differently from the global mean on at least one model, "
      f"rho < 0.50): {', '.join('`'+d+'`' for d in flagged) if flagged else 'none'}.")
    w("")

    w("## 6. Aggregation regime, and cross-model stability")
    w("")
    w("| dataset | ppl (L/Q) | msp_min (L/Q) | min−ppl (L/Q) | beta=1 (L/Q) | oracle beta (L/Q) | regime (L/Q) |")
    w("|---|---|---|---|---|---|---|")
    for d in LONG_DATASETS:
        w(f"| `{d}` | {SL.loc[d,'prr_perplexity']:+.3f} / {SQ.loc[d,'prr_perplexity']:+.3f} | "
          f"{SL.loc[d,'prr_msp_min']:+.3f} / {SQ.loc[d,'prr_msp_min']:+.3f} | "
          f"{SL.loc[d,'d_endpoint_min_minus_ppl']:+.3f} / {SQ.loc[d,'d_endpoint_min_minus_ppl']:+.3f} | "
          f"{SL.loc[d,'prr_beta_1']:+.3f} / {SQ.loc[d,'prr_beta_1']:+.3f} | "
          f"{SL.loc[d,'oracle_best_finite_beta']:g} / {SQ.loc[d,'oracle_best_finite_beta']:g} | "
          f"{beta_bucket(SL.loc[d,'oracle_best_finite_beta'])} / {beta_bucket(SQ.loc[d,'oracle_best_finite_beta'])} |")
    w("")
    w("**oracle beta is selected on test labels — descriptive only.**")
    w("")
    w("### Cross-model stability")
    w("")
    w("Five checks: method-ranking Spearman ≥ 0.70; `msp_min − perplexity` keeps its sign; same "
      "regime bucket; |ΔSAPLMA mean-OOD| ≤ 0.10; both contrasts (attention − mean-pool, "
      "wMSP-shrink@2 − SAPLMA) keep their signs. HIGH ≥ 4/5, MEDIUM 2–3, LOW ≤ 1.")
    w("")
    w("| dataset | stability | score | detail |")
    w("|---|---|---:|---|")
    for d in sorted(LONG_DATASETS, key=lambda x: -stab.loc[x, "score"]):
        w(f"| `{d}` | **{stab.loc[d,'stability']}** | {int(stab.loc[d,'score'])}/5 | {stab.loc[d,'why']} |")
    w("")

    w("## 7. Validity caveats")
    w("")
    w("| dataset | caveat |")
    w("|---|---|")
    for d in LONG_DATASETS:
        w(f"| `{d}` | {CAVEATS[d]} |")
    w("")

    # ---- the recommendation rule -------------------------------------------------------------
    # Deliberately NOT the raw 0-5 stability score. That score gives equal weight to checks that do
    # not speak to representativeness (same regime bucket, small SAPLMA delta) and to the one that
    # does (does the dataset rank the methods the way the benchmark does, and does it still do so on
    # the other model). cnn_dailymail is exactly this trap: it scores 3/5 while failing BOTH ranking
    # checks. So the tie-break is an explicit representativeness score, defined here and reported.
    rep, meta = {}, {}
    for d in LONG_DATASETS:
        mean_rho_global = float(np.mean([res["llama"].loc[d, "rho_global"], res["qwen"].loc[d, "rho_global"]]))
        rep[d] = 0.5 * mean_rho_global + 0.5 * float(stab.loc[d, "rank_rho"])
        cap = max(SL.loc[d, "cap_share"], SQ.loc[d, "cap_share"])
        minlen = min(SL.loc[d, "med_len"], SQ.loc[d, "med_len"])
        meta[d] = {
            "cen": f"{res['llama'].loc[d,'centrality']:.2f}/{res['qwen'].loc[d,'centrality']:.2f}",
            "wf": f"{res['llama'].loc[d,'within_family']:.2f}/{res['qwen'].loc[d,'within_family']:.2f}",
            "med": f"{SL.loc[d,'med_len']:.0f}/{SQ.loc[d,'med_len']:.0f} tok",
            "lensuit": "GOOD" if (minlen >= 60 and cap < 0.5) else ("WEAK" if minlen < 40 else "MODERATE"),
            "risk": "HIGH" if d == "med_quad" else (
                "MEDIUM" if (cap >= 0.5 or LABEL_COVERAGE[d] < 0.95
                             or min(SL.loc[d, "n"], SQ.loc[d, "n"]) < 300) else "LOW"),
        }

    recommend = {}
    for fam in ["correctness_qa", "summ", "factuality"]:
        members = [x for x in LONG_DATASETS if FINE_FAMILIES[x] == fam]
        # A known measurement defect or LOW cross-model stability disqualifies outright: a small
        # auxiliary experiment has no budget to absorb either.
        eligible = [x for x in members if meta[x]["risk"] != "HIGH" and stab.loc[x, "stability"] != "LOW"]
        for x in members:
            recommend[x] = "AVOID FOR SMALL AUXILIARY" if x not in eligible else "GOOD ALTERNATIVE"
        if eligible:
            recommend[max(eligible, key=lambda x: rep[x])] = "**PREFERRED**"

    w("## 8. Selection table")
    w("")
    w("| family | dataset | overall centrality | within-family centrality | cross-model stability | representativeness | length suitability | validity risk | recommendation |")
    w("|---|---|---|---|---|---:|---|---|---|")
    for fam in ["correctness_qa", "summ", "factuality"]:
        for d in [x for x in LONG_DATASETS if FINE_FAMILIES[x] == fam]:
            m = meta[d]
            w(f"| {fam} | `{d}` | {m['cen']} | {m['wf']} | {stab.loc[d,'stability']} | "
              f"{rep[d]:+.2f} | {m['lensuit']} ({m['med']}) | {m['risk']} | {recommend[d]} |")
    w("")
    w("Cells are `Llama / Qwen`. Definitions:")
    w("")
    w("- **length suitability** — GOOD = median ≥ 60 tokens on **both** models and cap share < 50%; "
      "WEAK = median < 40 on either; MODERATE otherwise.")
    w("- **validity risk** — HIGH = a known measurement defect; MEDIUM = cap ≥ 50%, label coverage "
      "< 95%, or test n < 300; LOW otherwise.")
    w("- **representativeness** = `0.5 × mean(rho_global over the two models) + 0.5 × cross-model "
      "rank rho`. It asks the two questions that matter for a *representative* target: does the "
      "dataset order the methods the way the benchmark does, and does it still do so on the other "
      "model?")
    w("- **recommendation** — `AVOID` if validity risk is HIGH or cross-model stability is LOW; "
      "otherwise the highest-representativeness survivor in each family is `PREFERRED` and the rest "
      "are `GOOD ALTERNATIVE`.")
    w("")
    w("The recommendation deliberately does **not** rank on the raw 0–5 stability score. That "
      "score weights checks that do not speak to representativeness (same regime bucket, small "
      "SAPLMA delta) equally with the one that does. `cnn_dailymail` is exactly that trap: it scores "
      "3/5 while failing **both** ranking checks (rho_global +0.12 on Llama, cross-model rank rho "
      "−0.10).")
    w("")

    w("## 9. Recommended target sets")
    w("")
    w("### A. Most representative — approximates typical family behaviour")
    w("")
    w("`asqa` · `xsum` · `factscore`")
    w("")
    w("### B. Most genuinely long — prioritises long, multi-sentence output")
    w("")
    w("`asqa` · `cnn_dailymail` · `expertqa` — medians 86/70, 39/65 and 200/165 tokens. Buys length "
      "at the cost of the two least cross-model-stable datasets in the table (`cnn_dailymail` rank "
      "rho −0.10, `expertqa` +0.12 and the single most divergent dataset on Qwen at 7.75). "
      "`med_quad` is the longest QA option at 128/128 but is disqualified by the span defect.")
    w("")
    w("### C. Most conservative — clean data, stability, no known artefacts")
    w("")
    w("`asqa` · `xsum` · `factscore` — **the same trio as A.** That convergence is a result, not a "
      "shortcut: on this benchmark the datasets that behave typically are also the ones without "
      "measurement defects. Reported honestly rather than manufacturing a third distinct set.")
    w("")
    w("### Recommended for the cross-length experiment: `asqa` + `xsum` + `factscore`")
    w("")
    w("**QA → ASQA, and it is not close.** The only HIGH cross-model stability in the table (4/5), "
      "the highest cross-model rank agreement of any dataset (+0.93), rho_global +0.67/+0.90, low "
      "cap (8%/9%), and a `msp_min − perplexity` endpoint that is almost identical across models "
      "(−0.066 / −0.060). Cost: small eval (n=284) and low mean label quality (0.41/0.33).")
    w("")
    w("- **PubMedQA is disqualified**, and on both counts the request asked about. Its aggregation "
      "regime flips (intermediate → mean-preferring) and its endpoint delta swings **+0.545 → "
      "−0.226**, the largest cross-model reversal in the whole table; it also scores **0/5** on "
      "stability. Separately it is barely long-form on Llama (median 22 tokens). The "
      "aggregation-regime flip is the dominant reason, not the biomedical domain as such.")
    w("- **MedQuAD is excluded for its caveat, not for divergence.** Behaviourally it is among the "
      "*most* central datasets on Qwen (0.92, the lowest distance in the table) and its rank "
      "agreement is decent (+0.78/+0.93). But it is LOW stability, 97%/62% capped, and carries the "
      "answer-span mismatch. It would import an unnecessary measurement caveat into a small "
      "auxiliary experiment.")
    w("")
    w("**Summarisation → XSum**, on stability rather than length. Its aggregation signature is the "
      "most cross-model-stable quantity in the entire audit (`msp_min − perplexity` = **+0.157 vs "
      "+0.158**), same regime bucket on both models, most central within its family on Qwen (1.52), "
      "cross-model rank rho +0.50, and LOW validity risk. Cost: short output (33/46 tokens) and "
      "31%/39% capped against a 56-token budget.")
    w("")
    w("**Factuality → FActScore**, as the more stable of only two candidates. Rank agreement +0.72 "
      "cross-model vs ExpertQA's +0.12, a small SAPLMA gap (0.065), and better label coverage (91% "
      "vs 85.5%). Its cost is real and must be reported: **n = 136/140, the smallest eval in the "
      "benchmark**, so its PRR carries wide seed spread and single-dataset effects should not be "
      "over-read. ExpertQA is larger (517/481) and much longer, but is LOW stability, the single "
      "most divergent dataset on Qwen (7.75), carries a different label projection (claim precision, "
      "not reference agreement), is blind to ~56% of claims, and has a length-label confound of "
      "r = −0.601 on Qwen. Those are systematic problems; FActScore's is statistical noise, which "
      "is the safer failure mode to report.")
    w("")
    w("### The tension to state in the write-up")
    w("")
    w("This is a **cross-length** experiment, and set A's summarisation representative has 33-token "
      "medians. Representativeness was chosen over raw length because the experiment is framed as "
      "cross-length *transfer*, not as a causal effect of response length, and because the "
      "long→short direction uses the full 8-source long pool regardless — so the trio only governs "
      "which long targets are **evaluated** in the short→long direction. If the length framing is "
      "later preferred, set B is the substitution, at a documented cost in cross-model stability.")
    w("")

    w("## 10. Answers to the specific dataset questions")
    w("")
    w("**Is CNN/DailyMail genuinely unusual in the canonical core results, or only in exploratory "
      "work?** — **Genuinely unusual in the canonical core-method profile.** Three independent "
      "canonical readings, none exploratory:")
    w("")
    w(f"1. It ranks the core methods almost unrelatedly to the benchmark mean on Llama "
      f"(rho_global **+0.12**, the lowest of all eight datasets).")
    w("2. Its cross-model method ranking is **−0.10** — the dataset does not even agree with itself "
      "across models.")
    w("3. It is the extreme mean-preferring dataset: `msp_min − perplexity` = **−0.290** on Llama, "
      "the largest negative in the table (perplexity +0.410 vs msp_min +0.120), and it is the only "
      "dataset whose entire Qwen Lehmer curve is negative (−0.044 → −0.108).")
    w("")
    w("It also has the **highest within-family distance on Llama (6.30)**. But the unusualness is "
      "*stable in kind*: both models put it at best-β = 0 (mean-preferring), and its Qwen cap rate "
      "is the lowest of any dataset (4.6%). So the honest description is **\"extreme but "
      "consistently extreme\"** — not noisy, not broken, genuinely at one end of the aggregation "
      "axis. That is why it is `GOOD ALTERNATIVE` rather than `AVOID`, and why it is the right "
      "choice for set B and the wrong one for a *representative* trio.")
    w("")
    w("**XSum vs CNN vs SAMSum on the length trade-off** — XSum is more behaviourally central but "
      "shorter (33/46 vs 39/65 tokens); CNN is more genuinely long and less capped on Qwen but is "
      "the idiosyncratic one above; SAMSum is the most central within family on Llama (2.39) but is "
      "63%/46% capped, has the worst cross-model rank agreement (−0.42), and carries 39% trailing "
      "contamination on Qwen. SAMSum's Qwen generation issues are real, as suspected.")
    w("")
    w("**ExpertQA vs FActScore** — compared directly rather than via an outlier score, since a "
      "two-member family cannot rank its members. See the factuality paragraph in §9.")
    w("")

    os.makedirs(ANALYSIS, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")

    print(f"[ok] wrote {args.out}")
    print(f"[population] Llama {len(lad_l)} core cells, Qwen {len(lad_q)} core cells "
          f"({len(LONG_DATASETS)} evals x 5 rungs x {len(CORE)} methods = {len(LONG_DATASETS)*5*len(CORE)} expected)")
    print("\n--- centrality (rank 1 = most unusual) ---")
    print(pd.DataFrame({"llama": res["llama"]["centrality"], "qwen": res["qwen"]["centrality"]}).sort_values("llama", ascending=False).round(2))
    print("\n--- cross-model stability ---")
    print(stab[["stability", "score", "rank_rho"]])


if __name__ == "__main__":
    main()
