"""Step 1 of the A.1 pilot: how much does the judge disagree with ITSELF on unchanged text?

Why this is needed. The pilot asks whether regenerating samsum at a larger budget moves the judge
score. That question is unanswerable without knowing how much the score moves when NOTHING changes --
and `llm_judge._gpt_response` calls at `temperature=1, top_p=1`, so the judge is stochastic and repeat
calls on identical input disagree.

⚠️ Note on what this does and does not isolate. samsum's v1 labels were ALREADY produced by
gpt-5-mini (every row carries `correctness_model = gpt-5-mini`), so re-judging with gpt-5-mini is NOT
a judge-change comparison -- there is no judge change to isolate. It measures run-to-run VARIANCE of
the same judge. Pass a different --judge to measure a genuine judge change instead; the script reports
which case it is rather than letting the two be confused.

The number the pilot actually needs is the standard error of a DATASET-LEVEL MEAN under judge noise,
because the pilot compares mean scores over ~1800 rows -- not the per-item spread, which is much larger
and would make any real effect look invisible.

Costs £ (N calls). Runs on the LOGIN NODE: it is API-bound, needs internet, and uses no GPU.

    python scripts/checks/judge_noise_floor.py --dataset samsum --n 800
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.labels import llm_judge  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="samsum")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--judge", default="gpt-5-mini")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label-field", default="correctness")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config(model_name=MODEL, dataset=args.dataset, ood_setting="ID",
                 prompt_regime=REGIME.get(args.dataset, ""))
    recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, args.dataset, "ID"))
    scored = [r for r in recs if r.get(args.label_field) is not None]
    if not scored:
        raise SystemExit(f"{args.dataset}: no rows carry {args.label_field} -- nothing to re-judge")

    orig_models = {r.get(f"{args.label_field}_model") for r in scored}
    same_judge = orig_models == {args.judge}
    print(f"{args.dataset}: {len(scored)} labelled rows; v1 judge(s) = {sorted(map(str, orig_models))}")
    print("MEASURING: " + ("run-to-run VARIANCE of the same judge (no judge change to isolate)"
                           if same_judge else
                           f"a genuine JUDGE CHANGE ({sorted(map(str, orig_models))} -> {args.judge})"))

    idx = np.random.RandomState(args.seed).permutation(len(scored))[:args.n]
    old, new, failed = [], [], 0
    for j, i in enumerate(idx, 1):
        r = scored[i]
        s = llm_judge.judge(r, args.dataset, model=args.judge)
        if s is None:                       # never fabricate a score the judge refused to give
            failed += 1
            continue
        old.append(float(r[args.label_field])); new.append(float(s))
        if j % 100 == 0:
            print(f"  {j}/{len(idx)} re-judged", flush=True)

    old, new = np.asarray(old), np.asarray(new)
    d = new - old
    n = len(d)
    # the quantity the pilot needs: SE of a dataset-level mean under this much judge noise
    se_mean = float(d.std(ddof=1) / np.sqrt(n))
    res = {"dataset": args.dataset, "judge": args.judge, "same_judge_as_v1": bool(same_judge),
           "n_requested": int(args.n), "n_scored": int(n), "n_failed": int(failed),
           "mean_old": float(old.mean()), "mean_new": float(new.mean()),
           "mean_delta": float(d.mean()), "sd_delta": float(d.std(ddof=1)),
           "mean_abs_delta": float(np.abs(d).mean()),
           "frac_identical": float(np.mean(d == 0)),
           "pearson": float(np.corrcoef(old, new)[0, 1]),
           "se_of_dataset_mean": se_mean,
           "noise_floor_95pct_on_mean": float(1.96 * se_mean)}

    print(f"\n  n scored            {n}  ({failed} judge failures, excluded not zero-filled)")
    print(f"  mean v1 label       {res['mean_old']:+.4f}")
    print(f"  mean re-judged      {res['mean_new']:+.4f}")
    print(f"  mean Δ (bias)       {res['mean_delta']:+.4f}")
    print(f"  sd of Δ (per item)  {res['sd_delta']:.4f}")
    print(f"  mean |Δ|            {res['mean_abs_delta']:.4f}")
    print(f"  identical scores    {100*res['frac_identical']:.1f}%")
    print(f"  correlation         {res['pearson']:.4f}")
    print(f"\n  >>> NOISE FLOOR: a dataset-mean shift smaller than "
          f"{res['noise_floor_95pct_on_mean']:.4f} is INDISTINGUISHABLE from judge noise at n={n}.")
    print("      The pilot's 'judge score must not fall' gate must be read against this, not against 0.")

    out = Path(args.out) if args.out else ROOT / "results" / f"judge_noise_floor__{args.dataset}__{args.judge}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
