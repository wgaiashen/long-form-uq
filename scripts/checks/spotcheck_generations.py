"""Read the actual text before trusting a rate. Stratified spot-check of regenerated generations.

Why this exists: we were about to pay for ~1800 judge calls on med_quad text nobody had read, on the
strength of a 30.56% severe-degeneracy RATE. A rate is not a substitute for looking. The degeneracy
detector catches exact n-gram loops; it does NOT catch semantic repetition, so "30% severe" cannot
distinguish a thorough 768-token medical answer from the same claim restated six times.

Samples with a FIXED SEED and PRINTS THE INDICES, so the same 20 examples can be re-read after
labelling to check what the judge did with them.

  5 flagged SEVERE  |  5 at/near the cap  |  5 terminated naturally short  |  5 at random

For each: prompt tail, the full new generation, the reference answer, and the OLD generation at the
same index, side by side.

ALIGNMENT IS ASSERTED, NOT ASSUMED. The old and new caches are only comparable row-by-row if index i
is the same example in both. The prompts are compared at every sampled index and the run aborts on a
mismatch -- comparing two different questions side by side would look perfectly plausible.

    python scripts/checks/spotcheck_generations.py --dataset med_quad --new-regime v2_med_quad
    python scripts/checks/spotcheck_generations.py --dataset med_quad --new-regime v2_med_quad --with-labels
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luq import cache, data, degeneracy  # noqa: E402
from luq.config import Config  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"


def load(dataset, regime):
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID", prompt_regime=regime)
    return cache.load_records(cfg.cache_dir, cache.run_key(MODEL, dataset, "ID"))


def gold(r):
    t = r.get("target")
    return str(t[0] if isinstance(t, list) and t else t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="med_quad")
    ap.add_argument("--new-regime", required=True)
    ap.add_argument("--old-regime", default="")
    ap.add_argument("--budget", type=int, default=None, help="new budget; default = data.MAX_NEW_TOKENS")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-per-stratum", type=int, default=5)
    ap.add_argument("--label-field", default="correctness")
    ap.add_argument("--with-labels", action="store_true", help="second pass: show the judge's verdicts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    new = load(args.dataset, args.new_regime)
    old = load(args.dataset, args.old_regime)
    if len(new) != len(old):
        raise SystemExit(f"row counts differ: new {len(new)} vs old {len(old)} -- cannot align by index")
    budget = args.budget or data.MAX_NEW_TOKENS[args.dataset]

    glen = np.array([len(r["gen_token_ids"]) for r in new])
    sev = np.array([degeneracy.is_severe(r.get("gen_text") or "") for r in new])
    capped = glen >= budget - 1
    short = (~capped) & (glen < budget * 0.6)

    rs = np.random.RandomState(args.seed)
    def pick(mask, k, taken):
        pool = np.setdiff1d(np.where(mask)[0], taken)
        return rs.choice(pool, size=min(k, len(pool)), replace=False) if len(pool) else np.array([], int)

    k = args.n_per_stratum
    taken = np.array([], int)
    strata = {}
    for name, mask in [("SEVERE", sev), ("AT_CAP", capped & ~sev),
                       ("NATURAL_SHORT", short & ~sev), ("RANDOM", np.ones(len(new), bool))]:
        idx = pick(mask, k, taken); strata[name] = idx; taken = np.concatenate([taken, idx])

    lines = []
    def w(s=""):
        lines.append(s)

    w(f"SPOT-CHECK  dataset={args.dataset}  new_regime={args.new_regime!r}  old_regime={args.old_regime!r}")
    w(f"seed={args.seed}  budget={budget}  n={len(new)}")
    w(f"population rates: severe {100*sev.mean():.2f}%  capped {100*capped.mean():.2f}%  "
      f"median_len {np.median(glen):.0f}")
    w("SAMPLED INDICES (fixed seed -- re-read the SAME rows after labelling):")
    for nme, idx in strata.items():
        w(f"  {nme:<14} {sorted(int(i) for i in idx)}")
    w("=" * 100)

    for nme, idx in strata.items():
        for i in sorted(int(x) for x in idx):
            rn, ro = new[i], old[i]
            # alignment guard: same index must be the same example in both caches
            if (rn.get("prompt") or "")[-400:] != (ro.get("prompt") or "")[-400:]:
                raise SystemExit(f"FATAL idx {i}: prompts differ between old and new caches. The two are "
                                 "NOT index-aligned; a side-by-side read would compare different "
                                 "questions and look entirely plausible.")
            gn = rn.get("gen_text") or ""
            go = ro.get("gen_text") or ""
            cls = degeneracy.classify(gn)
            w(f"\n{'='*100}\n[{nme}] idx={i}   new_len={len(rn['gen_token_ids'])} tok"
              f"   old_len={len(ro['gen_token_ids'])} tok   severe={degeneracy.is_severe(gn)}")
            w(f"detector: {cls}")
            if args.with_labels:
                w(f"JUDGE  new={rn.get(args.label_field)}  old={ro.get(args.label_field)}"
                  f"   (new model={rn.get(args.label_field+'_model')})")
            w(f"\n--- PROMPT (tail 600 chars) ---\n...{(rn.get('prompt') or '')[-600:]}")
            w(f"\n--- REFERENCE ANSWER ---\n{gold(rn)[:1500]}")
            w(f"\n--- NEW GENERATION ({len(gn)} chars) ---\n{gn}")
            w(f"\n--- OLD GENERATION ({len(go)} chars) ---\n{go}")

    if args.with_labels:
        for tag, recs in (("OLD", old), ("NEW", new)):
            v = [r.get(args.label_field) for r in recs if r.get(args.label_field) is not None]
            w(f"\nLABEL BASE RATE {tag}: n={len(v)}  mean={np.mean(v):.4f}" if v else
              f"\nLABEL BASE RATE {tag}: none")

    out = Path(args.out) if args.out else ROOT / "results" / f"spotcheck_{args.dataset}_{args.new_regime}.txt"
    out.write_text("\n".join(lines))
    print("\n".join(lines[:12]))
    print(f"\nwrote {out}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()
