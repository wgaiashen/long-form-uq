"""Validate a cheaper judge against the GPT-5 labels we already paid for (Phase A #3).

Loads cached records (which carry the GPT-5 `correctness`), samples records spread
across the GPT-5 score range, runs a CANDIDATE judge on the SAME prompt, and reports
agreement:
  * Spearman rho  -- PRIMARY: PRR is ordering-only, so ranking agreement matters most.
  * Pearson r + MAE -- the soft label is also the training TARGET (a value, not a rank).
  * mean bias     -- does the candidate score systematically high/low vs GPT-5?
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
    ap.add_argument("--coherent-only", action="store_true",
                    help="drop records whose generation is a few-shot-leak degeneration "
                         "(the dev 1.5B model's 'Yes + invented abstract' junk), so the judge "
                         "comparison is on real answers, not unscoreable garbage")
    args = ap.parse_args()

    if args.dataset in SHORT_FORM:
        sys.exit(f"{args.dataset} uses string-match labels, not the GPT-5 judge — "
                 "nothing to validate against. Use a long-form dataset (pubmed_qa, xsum).")

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

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

    # --- agreement metrics ---
    from scipy.stats import spearmanr
    spear = float(spearmanr(g, c)[0]) if len(g) > 2 else float("nan")
    pear = float(np.corrcoef(g, c)[0, 1]) if np.std(c) > 1e-9 else float("nan")
    mae = float(np.mean(np.abs(g - c)))
    bias = float(np.mean(c - g))

    print("\n=== judge agreement vs GPT-5 ===")
    print(f"dataset       : {args.dataset} ({args.ood})")
    print(f"candidate     : {args.judge}")
    print(f"n scored      : {len(g)}  (failed/None: {n_fail})")
    print(f"Spearman rho  : {spear:.3f}   <- primary (PRR is ordering-only)")
    print(f"Pearson r     : {pear:.3f}")
    print(f"MAE (0-1)     : {mae:.3f}")
    print(f"mean bias     : {bias:+.3f}   (candidate - GPT-5; + = scores higher)")

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
