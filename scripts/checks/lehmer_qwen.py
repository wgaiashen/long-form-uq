#!/usr/bin/env python
"""W6 -- Lehmer beta = 1 on the Qwen2.5-14B grid, out-of-sample.

Pre-registration: prereg/W6_lehmer_qwen.md, written and committed 2026-08-09 BEFORE any Qwen
record was labelled. beta = 1 is fixed there from Llama data only (argmax of the Llama cross-dataset
mean, and the modal leave-one-dataset-out selection, 6 of 8 folds). Nothing is selected here.

WHY THIS IS A SEPARATE FILE, not a --model flag on sharpening_family.py. That file is the live
W-Sharpen driver and is being edited on the other cluster; a second agent adding a flag to it is the
collision the workstream split exists to avoid. The scoring function itself is IMPORTED from it, so
this is the same code path the prereg names, not a re-implementation (a re-implementation is exactly
what the project's "verify against the original, not a paraphrase" rule forbids).

WHAT IT COMPUTES

  Lehmer mean   q = sum(nll^(beta+1)) / sum(nll^beta)     beta = 0 -> perplexity
                                                          beta -> inf -> msp_min's ranking

  Q1  cross-dataset: mean margin over the 8 long evals vs Qwen's OWN msp_min.
      Bars, all three required: margin > +0.010, signs >= 6/8, two-sided Wilcoxon p < 0.05.
  Q2  family-level: per-example paired bootstrap on each of xsum / cnn_dailymail / samsum,
      2000 resamples, bar p < 0.0167 (Bonferroni over the 3). The registered DIRECTIONAL
      prediction -- no significant win on pubmed_qa / factscore -- is tested and printed too; a
      win there is a surprise to report, not a success to pool.
  Q3  secondary: Wilcoxon over the 16 pooled per-dataset differences (8 Llama + 8 Qwen).

Training-free, hence rung-invariant: it reads cached `token_logprobs` only. No probe, no per-token
hidden states, no ladder, no GPU.

    python scripts/checks/lehmer_qwen.py                       # the pre-registered run
    python scripts/checks/lehmer_qwen.py --model <other>       # explicit pin, never a glob
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache, msp, results                      # noqa: E402
from luq.config import Config                            # noqa: E402
from xl_rungs import eval_split, label_of                # noqa: E402  (shim over probe_drift_long)
from attn_pool import PROMPT_REGIME                      # noqa: E402
from sharpening_family import score_lehmer               # noqa: E402  (THE registered scorer)

MODEL_DEFAULT = "Qwen/Qwen2.5-14B"
BETA = 1.0                                    # PRE-COMMITTED. Not selected, not swept for a winner.
BETAS_RECORD = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, np.inf]   # the full curve, for the record only

LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
SUMM = ["xsum", "cnn_dailymail", "samsum"]                    # Q2's registered three
CONCENTRATED = ["pubmed_qa", "factscore"]                     # Q2's registered NO-win prediction

BAR_MARGIN = 0.010
BAR_SIGNS = 6
BAR_WILCOXON = 0.05
BAR_Q2 = 0.0167                               # 0.05 / 3, Bonferroni over the three summarisation sets

BOOT_B = 2000                                 # as registered
BOOT_SEED = 12345                             # same convention as aggregation_table.paired_bootstrap
GATE_TOL = 1e-9                               # the endpoint identity is exact, not approximate

# ⚠️ THE PARTIAL-LABELLING GUARD. The finite-label filter drops non-finite rows, which means a
# dataset that is still being judged scores CLEANLY on whatever fraction has landed and prints a
# number indistinguishable from a complete one. That is the project's recurring failure shape: an
# absence quietly becoming a plausible value. Judging is asynchronous here, so this is not
# hypothetical -- it is the state of the cache on the night this was written.
#
# But a missing label has TWO causes and only one of them is a problem:
#
#   DECLINED  the judge ran and returned no score. On expertqa / factscore the distrust rule
#             refuses a factuality score when nothing in the answer was covered by the reference
#             (the row carries `uncovered` / `coherent` but no `factuality`). That is a MEASURED
#             property of the data -- recorded in prereg/M4 appendix B -- and dropping those rows
#             is the same finite-filter the Llama grid applied, so the two models stay comparable.
#             Each model is scored on its OWN covered subset, which is what M3/M4 registered.
#   UNJUDGED  the judge has not reached the row yet: no label field, no sibling judge outputs.
#             This is the one that must block.
#
# The test is therefore STRUCTURAL (did the judge leave any trace on this row?), not a threshold on
# a count -- so it cannot go stale when a coverage number moves.
JUDGE_SIBLINGS = ("uncovered", "coherent")    # written by the dedicated expertqa/factscore labellers
MAX_UNJUDGED_FRAC = 0.01                      # >1% never seen by the judge = still labelling

# Llama's beta = 1 and beta = inf (= msp_min) PRR per dataset, for Q3's pooled arm ONLY.
# Source: STOCKTAKE_sharpening_axis.md appendix A1 (sharpening_family__meta-llama_...__round2.csv).
# Transcribed at the 3-dp the stocktake publishes -- Q3 is secondary and reported alongside Q1,
# never instead of it, so 3-dp is adequate; the sign of every difference is unambiguous at 3-dp.
LLAMA_A1 = {           # dataset: (lehmer beta=1, msp_min)
    "pubmed_qa":     (+0.365, +0.371),
    "med_quad":      (+0.185, +0.149),
    "asqa":          (+0.310, +0.250),
    "xsum":          (+0.015, -0.015),
    "cnn_dailymail": (+0.230, +0.120),
    "samsum":        (+0.187, -0.024),
    "expertqa":      (+0.208, +0.205),
    "factscore":     (+0.360, +0.428),
}


def load_light(model, dataset):
    """records -> (per-example NLL vectors on the TEST rows, labels, label field, n).

    Two traps this closes, both of them live in this project's history:

    1. MODEL-AGNOSTIC GLOB. `cache/records/*__{dataset}__ID.jsonl` matches any model, and glob[0]
       once silently returned a dropped dev model's cache. The run key here is built from the
       EXPLICIT model, and the resolved file is asserted to carry that model's slug.
    2. NAMESPACED SETS. expertqa / asqa / factscore live under cache/<name>_rp12/, so a hard-coded
       `cache/` path would silently see 5 of 8 datasets and report a 5-dataset mean as an 8-dataset
       one. Resolution goes through Config(prompt_regime=...), exactly as attn_pool does.
    """
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    key = cache.run_key(model, dataset, "ID")
    path = Path(cfg.cache_dir) / "records" / f"{key}.jsonl"
    slug = cache._slug(model)
    if not path.exists():
        raise SystemExit(f"V4 FAIL [{dataset}]: no record file at {path}")
    if slug not in path.name:
        raise SystemExit(f"V4 FAIL [{dataset}]: resolved {path.name} does not carry the pinned "
                         f"model slug {slug} -- refusing to score another model's cache")
    # Ambiguity check: more than one model's records for this dataset in the same directory is
    # fine on disk, but it must never be resolved by chance. We pinned it; say so out loud.
    siblings = sorted(p.name for p in path.parent.glob(f"*__{dataset}__ID.jsonl"))
    if len(siblings) > 1:
        print(f"    [{dataset}] {len(siblings)} models present in {path.parent.name}/; "
              f"pinned to {path.name}")

    records = cache.load_records(cfg.cache_dir, key)
    lf = label_of(dataset)
    split = np.array([r["split"] for r in records])
    y = np.array([r.get(lf, np.nan) for r in records], dtype=float)
    finite = np.isfinite(y)                       # IDENTICAL to probedriftlong's finite-label filter
    n_dropped = int((~finite).sum())

    # Split the dropped rows by cause (see JUDGE_SIBLINGS above): declined is data, unjudged is a
    # half-finished run. Only the second one may stop the analysis.
    touched = np.array([(lf in r) or any(s in r for s in JUDGE_SIBLINGS) for r in records])
    n_unjudged = int((~touched).sum())
    n_declined = int((~finite & touched).sum())
    frac = n_unjudged / max(len(y), 1)
    if frac > MAX_UNJUDGED_FRAC:
        raise SystemExit(
            f"LABELLING INCOMPLETE [{dataset}]: {n_unjudged}/{len(y)} rows ({frac:.1%}) were never "
            f"seen by the judge (no '{lf}', no {JUDGE_SIBLINGS}). Scoring now would report a partial "
            f"population as a complete one. Wait for the judge to finish, then re-run.")
    if n_dropped:
        keep = np.where(finite)[0]
        records = [records[k] for k in keep]
        split = split[keep]
        y = y[keep]
    _, te = eval_split(split)                     # W-Qwen's own split/carve, unmodified
    if len(te) == 0:
        raise SystemExit(f"V4 FAIL [{dataset}]: empty test split")
    # V3 convention: cached token_logprobs are natural-log logprobs (negative), one per generated
    # token (length G, no prompt anchor). nll = -logprob.
    nlls = [-np.asarray(records[i]["token_logprobs"], dtype=float) for i in te]
    return nlls, y[te], lf, n_declined, n_unjudged


def paired_bootstrap(y, unc_a, unc_b, b=BOOT_B, seed=BOOT_SEED):
    """Two-sided paired bootstrap over TEST EXAMPLES -- the real data variance, resampling WHICH
    examples were drawn. Same convention as aggregation_table.paired_bootstrap (shared so the two
    workstreams' p-values mean the same thing); B is 2000 here because W6 registers 2000."""
    y = np.asarray(y, dtype=float)
    ua, ub = np.asarray(unc_a, dtype=float), np.asarray(unc_b, dtype=float)
    n = len(y)
    margin = results.prr(y, ua) - results.prr(y, ub)
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(b):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if np.ptp(yb) < 1e-9:                     # a degenerate resample has no oracle -> skip
            continue
        deltas.append(results.prr(yb, ua[idx]) - results.prr(yb, ub[idx]))
    deltas = np.array(deltas)
    if len(deltas) < 2:
        return margin, float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = min(1.0, 2.0 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0))))
    return margin, float(lo), float(hi), float(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DEFAULT,
                    help="EXPLICIT model pin. There is no glob fallback and there must never be one.")
    ap.add_argument("--boot", type=int, default=BOOT_B, help="paired-bootstrap resamples (registered: 2000)")
    args = ap.parse_args()

    slug = cache._slug(args.model)
    out = ROOT / "results" / f"lehmer_qwen__{slug}.csv"

    print("=" * 100)
    print("W6 -- LEHMER beta = 1, OUT-OF-SAMPLE   (prereg: prereg/W6_lehmer_qwen.md)")
    print(f"model={args.model}   population = 8 long evals   unit of analysis = DATASET (n=8), NOT cell")
    print("beta = 1 is PRE-COMMITTED from Llama data only. No parameter is selected on Qwen.")
    print("V3 NLL CONVENTION: cached token_logprobs are natural-log logprobs (negative), one per")
    print("   generated token (length G, no prompt anchor). nll = -logprob, computed here.")
    print("Mask convention: ALL generated tokens. Lehmer uses no content mask, so the Llama-only")
    print("   `id >= 128000` special-token rule (the known Qwen port trap) is not touched by it.")
    print("=" * 100)

    # ---------------- load, fail loud, and report the population ----------------
    data = {}
    print(f"\n{'dataset':16s}{'n_test':>8s}{'med_len':>9s}{'label':>14s}{'declined':>10s}{'unjudged':>10s}")
    for d in LONG:
        try:
            nlls, y, lf, n_declined, n_unjudged = load_light(args.model, d)
        except SystemExit:
            raise
        except Exception as e:                    # a missing cache must CRASH, never become an empty row
            raise SystemExit(f"V4 FAIL [{d}]: could not load records -- {type(e).__name__}: {e}")
        med = float(np.median([len(a) for a in nlls]))
        data[d] = {"nll": nlls, "y": y, "label": lf, "n": len(y), "med_len": med,
                   "declined": n_declined}
        print(f"{d:16s}{len(y):>8d}{med:>9.1f}{lf:>14s}{n_declined:>10d}{n_unjudged:>10d}")
    print(f"V4 AVAILABILITY: PASS -- all {len(LONG)} long datasets loaded, none substituted.")
    print("  'declined' = the judge ran and returned no score (distrust rule, prereg/M4 app. B);")
    print("  those rows are dropped by the same finite-filter the Llama grid used. 'unjudged' = the")
    print("  judge never saw the row, and any nonzero count there would have stopped the run.")
    print("  ⚠️ n_test is POST-split; declined/unjudged are counted over the FULL population, before")
    print("  the split, because that is the population the judge ran on. They are not two columns of")
    print("  one total -- do not subtract them from n_test.")

    # ---------------- the endpoint gate ----------------
    # INTERNAL only. The Llama driver also checks against its published master table; Qwen has no
    # published table yet, so that external leg genuinely cannot run. Saying so is the point --
    # a gate that silently drops a leg is how a half-checked run gets read as a checked one.
    print("\n" + "=" * 100)
    print("ENDPOINT GATE -- internal only (no Qwen master table exists yet; the external leg of the")
    print("   Llama gate is NOT run here, and that absence is stated, not silently skipped).")
    print("   A RANKING identity, not a value identity: msp_min is a monotone transform of max(nll),")
    print("   and PRR is rank-based, so the PRRs match exactly while the score vectors do not.")
    print("=" * 100)
    gate_ok = True
    for d in LONG:
        nl, y = data[d]["nll"], data[d]["y"]
        v_min = np.array([msp.msp_uncertainty(-a, "min") for a in nl])          # back to logprobs
        v_ppl = np.array([msp.msp_uncertainty(-a, "perplexity") for a in nl])
        p_min, p_ppl = results.prr(y, v_min), results.prr(y, v_ppl)
        p_b0 = results.prr(y, np.array([score_lehmer(a, 0.0) for a in nl]))
        p_bi = results.prr(y, np.array([score_lehmer(a, np.inf) for a in nl]))
        ok = abs(p_b0 - p_ppl) < GATE_TOL and abs(p_bi - p_min) < GATE_TOL
        gate_ok = gate_ok and ok
        data[d].update(prr_min=p_min, prr_ppl=p_ppl, unc_min=v_min)
        print(f"[{d:14s}] msp_min {p_min:+.4f}  perplexity {p_ppl:+.4f}  "
              f"beta0==ppl d{abs(p_b0 - p_ppl):.1e}  betaInf==min d{abs(p_bi - p_min):.1e}  "
              f"{'PASS' if ok else 'FAIL <=='}", flush=True)
    if not gate_ok:
        raise SystemExit("\nGATE FAIL -- the NLL convention or the row population differs from what "
                         "the family assumes. Nothing downstream is interpretable. Stopping.")

    # ---------------- the full curve, for the record only ----------------
    print("\n" + "=" * 100)
    print("THE FULL LEHMER CURVE (for the record -- the CLAIM is beta = 1 and only beta = 1)")
    print("=" * 100)
    header = "".join(f"{b:>9.1f}" if np.isfinite(b) else f"{'inf':>9s}" for b in BETAS_RECORD)
    print(f"{'eval':16s}{header}")
    curve = {}
    for d in LONG:
        nl, y = data[d]["nll"], data[d]["y"]
        row = []
        for b in BETAS_RECORD:
            v = np.array([score_lehmer(a, b) for a in nl])
            row.append(results.prr(y, v))
            if b == BETA:
                data[d]["unc_lehmer"] = v
        curve[d] = row
        print(f"{d:16s}" + "".join(f"{p:>+9.3f}" for p in row))
    means = [float(np.mean([curve[d][i] for d in LONG])) for i in range(len(BETAS_RECORD))]
    print(f"{'mean':16s}" + "".join(f"{m:>+9.3f}" for m in means))

    # ---------------- Q1 ----------------
    diffs = np.array([results.prr(data[d]["y"], data[d]["unc_lehmer"]) - data[d]["prr_min"]
                      for d in LONG])
    margin = float(diffs.mean())
    signs = int((diffs > 0).sum())
    w_stat, w_p = wilcoxon(diffs, alternative="two-sided")

    print("\n" + "=" * 100)
    print("Q1 -- CROSS-DATASET (the claim that failed on Llama; genuinely out-of-sample here)")
    print("=" * 100)
    print(f"{'eval':16s}{'Lehmer b=1':>13s}{'msp_min':>10s}{'diff':>10s}")
    for d, df in zip(LONG, diffs):
        print(f"{d:16s}{results.prr(data[d]['y'], data[d]['unc_lehmer']):>+13.4f}"
              f"{data[d]['prr_min']:>+10.4f}{df:>+10.4f}")
    ok_margin, ok_signs, ok_wil = margin > BAR_MARGIN, signs >= BAR_SIGNS, w_p < BAR_WILCOXON
    print(f"\n  margin  {margin:+.4f}   bar > +{BAR_MARGIN:.3f}      {'PASS' if ok_margin else 'FAIL'}")
    print(f"  signs   {signs}/8        bar >= {BAR_SIGNS}/8        {'PASS' if ok_signs else 'FAIL'}")
    print(f"  Wilcoxon p {w_p:.4f}  bar < {BAR_WILCOXON}        {'PASS' if ok_wil else 'FAIL'}")
    q1 = ok_margin and ok_signs and ok_wil
    print(f"  ==> Q1 {'PASSES' if q1 else 'FAILS'} (all three bars required, as registered)")

    # ---------------- Q2 ----------------
    print("\n" + "=" * 100)
    print(f"Q2 -- FAMILY-LEVEL, per-example paired bootstrap ({args.boot} resamples), bar p < {BAR_Q2}")
    print("   The three summarisation sets are the CLAIM. pubmed_qa / factscore carry the registered")
    print("   directional prediction of NO win -- a win there is a surprise, not a success.")
    print("=" * 100)
    print(f"{'eval':16s}{'role':14s}{'margin':>10s}{'95% CI':>20s}{'boot p':>10s}{'verdict':>12s}")
    q2_pass, boot_rows = [], {}
    for d in SUMM + CONCENTRATED:
        m, lo, hi, p = paired_bootstrap(data[d]["y"], data[d]["unc_lehmer"], data[d]["unc_min"],
                                        b=args.boot)
        boot_rows[d] = (m, lo, hi, p)
        role = "CLAIM" if d in SUMM else "prediction"
        sig = p < BAR_Q2
        if d in SUMM:
            q2_pass.append(sig)
            verdict = "PASS" if sig else "fail"
        else:
            verdict = "SURPRISE" if sig else "as predicted"
        print(f"{d:16s}{role:14s}{m:>+10.4f}   [{lo:+.4f},{hi:+.4f}]{p:>10.4f}{verdict:>12s}")
    q2 = all(q2_pass)
    print(f"\n  ==> Q2 {'PASSES on all three' if q2 else f'partial/fails ({sum(q2_pass)}/3)'}")

    # ---------------- Q3 (secondary) ----------------
    llama_diffs = np.array([LLAMA_A1[d][0] - LLAMA_A1[d][1] for d in LONG])
    pooled = np.concatenate([llama_diffs, diffs])
    p_stat, p_p = wilcoxon(pooled, alternative="two-sided")
    print("\n" + "=" * 100)
    print("Q3 -- SECONDARY, pooled n = 16 (8 Llama + 8 Qwen). Reported ALONGSIDE Q1, never instead.")
    print("   Llama half transcribed from STOCKTAKE_sharpening_axis.md appendix A1 at 3 dp.")
    print("=" * 100)
    print(f"  Llama mean diff {llama_diffs.mean():+.4f} ({int((llama_diffs > 0).sum())}/8)   "
          f"Qwen mean diff {margin:+.4f} ({signs}/8)")
    print(f"  pooled mean {pooled.mean():+.4f}   signs {int((pooled > 0).sum())}/16   "
          f"Wilcoxon p {p_p:.4f}")

    # ---------------- the registered reading ----------------
    print("\n" + "=" * 100)
    print("REGISTERED FAILURE READING (prereg §4 -- fixed in advance, applied mechanically)")
    print("=" * 100)
    if q1 and q2:
        print("  Q1 and Q2 both pass: the cross-dataset claim replicates out-of-sample.")
    elif not q1 and q2:
        print("  Q1 fails, Q2 passes: the cross-dataset claim is DEAD; the family-level claim")
        print("  REPLICATES and is the reportable result (the regime map made actionable).")
    elif not q1 and not q2 and sum(q2_pass) == 0:
        print("  Q1 and Q2 both fail: the Llama summarisation wins were population-specific.")
        print("  THE LEHMER LINE CLOSES FULLY, and that closure is final.")
    elif not q1:
        print(f"  Q2 passes on only {sum(q2_pass)} of 3: PARTIAL replication, reported per dataset,")
        print("  no pooling across the three.")
    else:
        print(f"  Q1 passes, Q2 on {sum(q2_pass)}/3: report both, per dataset, no pooling.")

    # ---------------- persist ----------------
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "beta", "prr", "n_test", "med_len", "label"])
        for d in LONG:
            for b, p in zip(BETAS_RECORD, curve[d]):
                w.writerow([args.model, d, b, f"{p:.6f}", data[d]["n"],
                            f"{data[d]['med_len']:.1f}", data[d]["label"]])
        w.writerow([])
        w.writerow(["model", "dataset", "test", "margin_vs_msp_min", "ci_lo", "ci_hi", "boot_p"])
        for d, (m, lo, hi, p) in boot_rows.items():
            w.writerow([args.model, d, "Q2_paired_bootstrap", f"{m:.6f}", f"{lo:.6f}",
                        f"{hi:.6f}", f"{p:.6f}"])
        w.writerow([args.model, "ALL", "Q1_cross_dataset", f"{margin:.6f}", "", "", f"{w_p:.6f}"])
        w.writerow([args.model, "ALL", "Q3_pooled_n16", f"{pooled.mean():.6f}", "", "", f"{p_p:.6f}"])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
