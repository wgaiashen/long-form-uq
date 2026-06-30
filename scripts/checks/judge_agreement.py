"""Validate a cheaper judge against the GPT-5 labels we already paid for (Phase A #3).

Loads cached records (which carry the GPT-5 `correctness`), samples records spread
across the GPT-5 score range, runs a CANDIDATE judge on the SAME prompt, and reports a
graded-first agreement panel (with percentile-bootstrap CIs):
  GRADED (what PRR cares about):
    * Spearman rho  -- PRIMARY: PRR is ordering-only, so ranking agreement matters most.
    * Kendall tau-b -- rank agreement, robust to ties.
    * Krippendorff alpha (interval) -- chance-corrected agreement on the graded score.
  CALIBRATION (the soft label is also the training TARGET, a value not a rank):
    * Pearson r + MAE + mean bias (+ both judges' means).
  BINARY appendix (>= --binary-threshold; for comparability with Joe / SATMD):
    * % agreement, MCC/phi, Cohen kappa, Gwet AC1, and the kappa-AC1 gap (skew diagnostic
      -- under ~90%-positive labels kappa collapses but AC1 does not).
The PRR-stability headline (does the judge swap change method RANKING?) lives in the
sibling judge_prr_impact.py. Krippendorff/irrCAC are optional (panel degrades to n/a).
Saves the candidate scores so reruns are free.

Only long-form datasets have GPT-5 labels (sciq is string-match), so use pubmed_qa / xsum.

    # GPT-5-mini -- login node, no GPU, pennies:
    python scripts/checks/judge_agreement.py --dataset pubmed_qa --judge openai:gpt-5-mini --n 80
    python scripts/checks/judge_agreement.py --dataset xsum      --judge openai:gpt-5-mini --n 80
    # open-source -- needs a GPU:
    python scripts/checks/judge_agreement.py --dataset pubmed_qa --judge local:Qwen/Qwen2.5-7B-Instruct --n 80
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.data import SHORT_FORM  # noqa: E402


# --- agreement-metric helpers (graded-first panel; 2026-06-28 judge-panel design) ----
# The label is graded 0-1 and feeds PRR (rank) + soft-label training (magnitude), so the
# panel leads with rank/graded metrics; the binary block is for comparability with Joe /
# SATMD only. Cohen's kappa collapses under our ~90%-positive labels (prevalence paradox),
# so Gwet's AC1 is the trustworthy chance-corrected number; the kappa-AC1 gap is itself the
# skew diagnostic. krippendorff/irrCAC are optional — the panel degrades to n/a without them.

def _binarize(x, thr):
    return (np.asarray(x, dtype=float) >= thr).astype(int)


def _cohen_kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b))


def _mcc(a, b):
    from sklearn.metrics import matthews_corrcoef
    return float(matthews_corrcoef(a, b))


def _gwet_ac1(a, b):
    import pandas as pd
    from irrCAC.raw import CAC
    df = pd.DataFrame({"r1": list(a), "r2": list(b)})
    return float(CAC(df).gwet()["est"]["coefficient_value"])


def _kripp_alpha(g, c, level="interval"):
    import krippendorff
    return float(krippendorff.alpha(reliability_data=np.vstack([g, c]),
                                    level_of_measurement=level))


def _boot_ci(fn, *arrs, n_boot=1000, seed=1, alpha=0.05):
    """Percentile bootstrap CI for an agreement statistic, resampling the (g, c) pairs.
    n is small (~80-200) so a point estimate alone misleads. Skips undefined resamples."""
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(a) for a in arrs]
    n = len(arrs[0])
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = fn(*[a[idx] for a in arrs])
        except Exception:
            continue
        if v == v:  # drop NaN
            out.append(v)
    if not out:
        return float("nan"), float("nan")
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def stratified_sample(corr, n, seed=1, n_bins=5):
    """Pick n indices spread across the [0, 1] correctness range, so we test the
    partial-credit region (where judges disagree most), not just the 0/1 ends."""
    rng = np.random.default_rng(seed)
    corr = np.asarray(corr)
    bins = np.clip((corr * n_bins).astype(int), 0, n_bins - 1)  # 1.0 -> last bin
    per = max(1, n // n_bins)
    chosen = []
    for b in range(n_bins):
        idx = np.where(bins == b)[0]
        if len(idx):
            chosen.extend(rng.choice(idx, size=min(per, len(idx)), replace=False).tolist())
    # Top up randomly from the rest if some bins were sparse and we're short of n.
    if len(chosen) < n:
        rest = np.setdiff1d(np.arange(len(corr)), np.array(chosen, dtype=int))
        if len(rest):
            extra = rng.choice(rest, size=min(n - len(chosen), len(rest)), replace=False)
            chosen.extend(extra.tolist())
    return sorted(chosen[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pubmed_qa")
    ap.add_argument("--judge", required=True,
                    help="backend:model, e.g. openai:gpt-5-mini or "
                         "local:Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=80, help="records to sample")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name,
                    help="base model whose generations were judged (for the cache key)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--binary-threshold", type=float, default=0.5,
                    help="threshold to binarise the graded score for the binary appendix "
                         "(percent-agreement / MCC / Cohen kappa / Gwet AC1)")
    ap.add_argument("--coherent-only", action="store_true",
                    help="drop records whose generation is a few-shot-leak degeneration "
                         "(the dev 1.5B model's 'Yes + invented abstract' junk), so the judge "
                         "comparison is on real answers, not unscoreable garbage")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    # short-form is string-match by default, which is NOT a graded judge label to validate
    # a candidate judge against. Allow short-form ONLY when it has been promoted to the
    # judge (correctness = the gpt-5 judge score; see 02_label --promote-judge), detected
    # via the judge-model provenance stamp that step writes.
    if args.dataset in SHORT_FORM and not any(
        r.get("correctness_judge_model") or r.get("correctness_model") for r in records
    ):
        sys.exit(f"{args.dataset} is string-match labelled (no judge provenance) — run "
                 "02_label --judge-short-form --promote-judge first, or use a long-form "
                 "dataset (pubmed_qa, xsum).")

    # Keep only records that actually carry a GPT-5 label (skip judge failures / Nones).
    labelled = [r for r in records if isinstance(r.get("correctness"), (int, float))]
    if args.coherent_only:
        # The dev 1.5B model often degenerates on pubmed into "Yes" + an invented few-shot
        # example ("The following are abstract and question about them ..."). Scoring that
        # garbage is ill-defined, so BOTH judges (GPT-5 included) flail and disagree wildly,
        # which contaminates the agreement number. This marker flags those reliably without
        # catching legitimate answers that merely mention "the abstract describes ...".
        leak = "the following are abstract"
        before = len(labelled)
        labelled = [r for r in labelled if leak not in r["gen_text"].lower()]
        print(f"coherent-only: kept {len(labelled)}/{before} records "
              f"(dropped {before - len(labelled)} few-shot-leak generations)")
    if len(labelled) < args.n:
        print(f"warning: only {len(labelled)} labelled records; using all of them")
    gpt5 = np.array([r["correctness"] for r in labelled], dtype=float)
    pick = stratified_sample(gpt5, min(args.n, len(labelled)), seed=args.seed)
    sample = [labelled[i] for i in pick]
    gpt5_s = gpt5[pick]

    # --- backend dispatch: build the candidate scoring function ---
    backend, _, model_name = args.judge.partition(":")
    if not model_name:
        sys.exit("--judge must be backend:model, e.g. openai:gpt-5-mini")
    if backend == "openai":
        from luq.labels import llm_judge
        score_one = lambda r: llm_judge.judge(r, args.dataset, model=model_name)
    elif backend == "local":
        from luq.labels.local_judge import LocalJudge
        lj = LocalJudge(model_name)
        score_one = lambda r: lj.judge(r, args.dataset)
    else:
        sys.exit(f"unknown backend {backend!r} (use openai or local)")

    print(f"scoring {len(sample)} {args.dataset} records with {args.judge} ...")
    cand = []
    for j, r in enumerate(sample):
        cand.append(score_one(r))
        if (j + 1) % 10 == 0:
            print(f"  {j + 1}/{len(sample)}")
    cand = np.array([np.nan if c is None else c for c in cand], dtype=float)

    ok = ~np.isnan(cand)
    n_fail = int((~ok).sum())
    g, c = gpt5_s[ok], cand[ok]

    # --- agreement panel (graded-first; binary appendix for field comparability) ---
    from scipy.stats import spearmanr, kendalltau

    def _spear(a, b): return float(spearmanr(a, b)[0])
    def _kend(a, b): return float(kendalltau(a, b)[0])
    def _pear(a, b):
        return (float(np.corrcoef(a, b)[0, 1])
                if np.std(a) > 1e-9 and np.std(b) > 1e-9 else float("nan"))

    def _ci(fn, *arrs):
        lo, hi = _boot_ci(fn, *arrs)
        return f"[{lo:+.3f}, {hi:+.3f}]"

    thr = args.binary_threshold
    gb, cb = _binarize(g, thr), _binarize(c, thr)
    n_total = len(cand)

    print("\n=== judge agreement vs GPT-5 ===")
    print(f"dataset        : {args.dataset} ({args.ood})")
    print(f"candidate      : {args.judge}")
    print(f"n scored       : {len(g)}  (abstain/None: {n_fail}, "
          f"{100 * n_fail / max(n_total, 1):.1f}%)")
    print(f"both means     : GPT-5 {g.mean():.3f} | cand {c.mean():.3f}")

    print("\n-- GRADED (what PRR cares about) --")
    print(f"Spearman rho   : {_spear(g, c):+.3f}  {_ci(_spear, g, c)}   <- primary (PRR is ordering-only)")
    print(f"Kendall tau-b  : {_kend(g, c):+.3f}  {_ci(_kend, g, c)}")
    try:
        print(f"Krippendorff a : {_kripp_alpha(g, c):+.3f}  {_ci(_kripp_alpha, g, c)}   (interval)")
    except Exception as e:
        print(f"Krippendorff a : n/a  (pip install krippendorff)  [{type(e).__name__}]")

    print("\n-- CALIBRATION (soft-label training) --")
    print(f"Pearson r      : {_pear(g, c):+.3f}")
    print(f"MAE (0-1)      : {np.mean(np.abs(g - c)):.3f}")
    print(f"mean bias      : {np.mean(c - g):+.3f}   (cand - GPT-5; + = scores higher)")

    print(f"\n-- BINARY appendix (>= {thr}; for Joe/SATMD comparability) --")
    print(f"% agreement    : {float(np.mean(gb == cb)):.3f}")
    try:
        print(f"MCC / phi      : {_mcc(gb, cb):+.3f}")
    except Exception:
        print(f"MCC / phi      : n/a")
    try:
        kappa, ac1 = _cohen_kappa(gb, cb), _gwet_ac1(gb, cb)
        print(f"Cohen kappa    : {kappa:+.3f}  "
              f"{_ci(lambda a, b: _cohen_kappa(_binarize(a, thr), _binarize(b, thr)), g, c)}")
        print(f"Gwet AC1       : {ac1:+.3f}  "
              f"{_ci(lambda a, b: _gwet_ac1(_binarize(a, thr), _binarize(b, thr)), g, c)}")
        print(f"kappa-AC1 gap  : {kappa - ac1:+.3f}   (large gap = agreement is mostly the base rate)")
    except Exception as e:
        print(f"Cohen kappa/AC1: n/a  (pip install irrCAC)  [{type(e).__name__}]")

    # Worst disagreements to eyeball (is the cheap judge wrong, or arguably better?).
    order = np.argsort(-np.abs(g - c))
    print("\nworst disagreements (GPT-5 vs candidate):")
    for k in order[:5]:
        print(f"  gpt5={g[k]:.2f}  cand={c[k]:.2f}")

    # Save so reruns are free and the local-GPU run can be inspected later.
    out = Path(cfg.cache_dir) / "judge_agreement"
    out.mkdir(parents=True, exist_ok=True)
    tag = args.judge.replace("/", "_").replace(":", "_")
    if args.coherent_only:
        tag += "__coherent"
    path = out / f"{key}__{tag}.npz"
    np.savez(path, idx=np.array(pick), gpt5=gpt5_s, candidate=cand)
    print(f"\nsaved scores -> {path}")


if __name__ == "__main__":
    main()
