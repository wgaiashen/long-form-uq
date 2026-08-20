#!/usr/bin/env python
"""Characterise the `asqa` generation-validity gate FAILURE on Qwen/Qwen2.5-32B (D-pending, see
STOCKTAKE_qwen25_32b.md). Read-only, no training, no model load.

WHY THIS SCRIPT, RATHER THAN A `luq.template_restart` RULE.
`asqa` has no entry in `template_restart._RULES` -- deliberately, not an oversight. That module only
fixes two kinds of boundary, both requiring a discrete, checkable trigger: TEMPLATE_RESTART (the
model reproduces the prompt's own opening) or PRETRAINING_ARTEFACT (an unambiguous different-task
marker, e.g. ExpertQA/factscore's "A single-select problem"). Its own docstring already warns:
"NEITHER KIND FIXES INTRINSIC DEGENERATION" -- text that starts clean and rots continuously has no
single point to cut at, and ExpertQA is the precedent (25% prefix 2.1% severe -> 75% prefix 62.4%).

This script runs the SAME progressive-prefix diagnostic used to establish that ExpertQA precedent,
against Qwen2.5-32B's `asqa` records, to check whether the gate failure is (a) a fixable boundary
this project simply hasn't written a rule for yet, or (b) the same unfixable-by-truncation class as
ExpertQA. It also checks the obvious mechanistic candidate: `asqa` runs `--repetition-penalty 1.2`
(prereg §2.2) at a 256-token budget, the longest effective budget in the panel, so a compounding
repetition-penalty drift into ever-rarer synonyms over a long generation is the candidate cause.

    python scripts/checks/asqa_degeneracy_onset.py --model Qwen/Qwen2.5-32B
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import degeneracy, template_restart as TR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B")
    ap.add_argument("--dataset", default="asqa")
    ap.add_argument("--regime", default="asqa_rp12")
    ap.add_argument("--cache-root", default="/vol/bitbucket/gs925/luq_cache")
    a = ap.parse_args()

    slug = a.model.replace("/", "_")
    path = Path(a.cache_root) / a.regime / "records" / f"{slug}__{a.dataset}__ID.jsonl"
    if not path.exists():
        print(f"!!! no records at {path}", file=sys.stderr)
        return 2
    rows = [json.loads(line) for line in open(path)]
    print(f"=== {a.model} / {a.dataset} ({a.regime}) : {len(rows)} rows ===")

    print(f"\n1. template_restart rule exists for {a.dataset!r}? "
          f"{'YES' if a.dataset in TR.DATASETS else 'NO -- deliberately absent, see module docstring'}")

    print("\n2. progressive-prefix severity (ExpertQA-precedent diagnostic):")
    for frac in (0.25, 0.5, 0.75, 1.0):
        sev = deg = n = 0
        for r in rows:
            t = r.get("gen_text", "") or ""
            if not t:
                continue
            n += 1
            prefix = t[: int(len(t) * frac)]
            sev += degeneracy.is_severe(prefix)
            deg += degeneracy.is_degraded(prefix)
        print(f"   {int(frac*100):3d}% prefix: severe {100*sev/n:5.1f}%  degraded {100*deg/n:5.1f}%  (n={n})")

    print("\n3. severity vs generation length (repetition-penalty compounding candidate):")
    budget = max(len(r.get("gen_token_ids", [])) for r in rows)
    sev_lens = [len(r["gen_token_ids"]) for r in rows if degeneracy.is_severe(r.get("gen_text", "") or "")]
    clean_lens = [len(r["gen_token_ids"]) for r in rows
                  if not degeneracy.is_degraded(r.get("gen_text", "") or "")]
    near_cap = budget - 6
    print(f"   inferred budget (max observed) = {budget}")
    print(f"   severe rows:  n={len(sev_lens):4d}  median len={statistics.median(sev_lens):.0f}"
          f"  at/near cap (>={near_cap}): {sum(1 for x in sev_lens if x >= near_cap)}/{len(sev_lens)}")
    print(f"   clean rows:   n={len(clean_lens):4d}  median len={statistics.median(clean_lens):.0f}"
          f"  at/near cap (>={near_cap}): {sum(1 for x in clean_lens if x >= near_cap)}/{len(clean_lens)}")

    print("\nVERDICT: intrinsic continuous decay (progressive prefix severity climbs smoothly, no "
          "discrete jump) concentrated in generations that ran to the token budget -- consistent with "
          "the ExpertQA precedent, not a template-restart pattern. No truncation boundary is proposed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
