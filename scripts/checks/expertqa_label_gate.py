"""ExpertQA label-quality GATE — validate the three-state judge on a small sample BEFORE the full
2016 spend. Does quadruple duty on ~50 examples:

  1. UNCOVERED fraction  — the "measure the blind spot first" number. How much of the model's output
     does factuality-to-evidence not see? If small (~few %), uncovered can collapse to correct
     (justified option 1); if large (~40%), we need a coverage-extension workstream before ExpertQA
     can carry the OOD claim. This is the number the whole not-1-not-2 plan turns on.
  2. DERAILMENT rate     — luq.degeneracy on the same sample (should be ~0 on a clean regenerated set).
  3. mini-vs-gpt-5       — agreement on factuality/uncovered/coherent, on THIS prompt (the old
     r=0.83 was on the old gold-matching prompt; it does not transfer). Decides the judge model.
  4. AlignScore-vs-judge — EXPLICIT: if a similarity metric (AlignScore) and the factuality judge
     disagree a lot, that is evidence the projection choice matters (justifies not just using
     AlignScore). Low correlation here is a RESULT, not noise.

Applies the STAMPED distrust rule uniformly (luq.labels.expertqa_judge_draft): a SEVERE generation
(detector) is factuality=0.0 and is NOT sent to the judge; a coherent=false judge verdict forces
factuality=0.0. Both are distrust labels, kept not dropped.

Run (judge needs OPENAI_API_KEY + internet; --alignscore needs the GPU roberta model):
    python scripts/checks/expertqa_label_gate.py --n 50 --judges gpt-5-mini,gpt-5 --alignscore
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache, degeneracy, expertqa          # noqa: E402
from luq.config import Config                          # noqa: E402
from luq.labels import expertqa_judge_draft as judge   # noqa: E402
from luq.labels import llm_judge                        # noqa: E402


def load_eval_with_evidence():
    """The factual-core pool in file order (idx-aligned to the generation records), KEEPING each
    question's cited evidence so the judge reference = gold + citations. Mirrors
    expertqa.load_records()'s filter exactly."""
    out = []
    for line in open(expertqa.DEFAULT_JSONL, encoding="utf-8"):
        rec = json.loads(line)
        ans = next(iter(rec["answers"].values()))
        gold = (ans.get("revised_answer_string") or "").strip()
        if not gold:
            continue
        types = {t.strip() for t in rec["metadata"]["question_type"].split("|") if t.strip()}
        if not (types & expertqa.FACTUAL_CORE):
            continue
        # each claim's evidence is a list of "[n] URL\n snippet" strings; flatten + drop the URLs.
        ev = []
        for c in (ans.get("claims") or []):
            items = c.get("revised_evidence") or c.get("evidence") or []
            if isinstance(items, str):
                items = [items]
            for it in items:
                if isinstance(it, str):
                    t = re.sub(r"https?://\S+", "", it).strip()
                    if t:
                        ev.append(t)
        out.append({"gold": gold, "evidence": ev})
    return out


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--prompt-regime", default="expertqa_reppen")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--judges", default="gpt-5-mini,gpt-5",
                    help="comma list; first is primary for the uncovered/derailment reads")
    ap.add_argument("--alignscore", action="store_true", help="also compute AlignScore (GPU)")
    ap.add_argument("--clean-only", action="store_true",
                    help="sample only detector-clean generations (for a dry-run on the current, "
                         "partly-degraded set so the uncovered read isn't contaminated)")
    ap.add_argument("--out", default="results/expertqa/label_gate")
    args = ap.parse_args()
    judges = args.judges.split(",")

    cfg = Config(dataset="expertqa", ood_setting=args.ood, model_name=args.model,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(args.model, "expertqa", args.ood)
    recs = cache.load_records(cfg.cache_dir, key)
    raw = load_eval_with_evidence()
    assert len(raw) >= max(r["idx"] for r in recs) + 1, "raw pool / record idx misaligned"

    pool = [r for r in recs if not degeneracy.is_degraded(r["gen_text"])] if args.clean_only else recs
    rng = random.Random(1)
    sample = rng.sample(pool, min(args.n, len(pool)))
    print(f"gate on {len(sample)} examples; judges={judges}; alignscore={args.alignscore}", flush=True)

    rows = []
    for i, r in enumerate(sample):
        src = raw[r["idx"]]
        q = r["prompt"].split("Question:")[-1].split("\nAnswer:")[0].strip()
        reference = judge.build_reference(src["gold"], src["evidence"], max_ev_chars=4000)
        severe = degeneracy.is_severe(r["gen_text"])
        row = {"idx": r["idx"], "severe": severe, "gen": r["gen_text"], "gold": src["gold"]}
        for jm in judges:
            if severe:
                # STAMPED rule: quarantined-severe is a distrust label, not sent to the judge.
                row[jm] = {"factuality": 0.0, "uncovered": 0.0, "coherent": False, "quarantined": True}
                continue
            prompt = judge.fill(q, reference, r["gen_text"])
            raw_out = llm_judge._gpt_response(prompt, jm)
            parsed = judge.parse(raw_out)
            if parsed is None:
                row[jm] = {"factuality": None, "uncovered": None, "coherent": None, "parse_fail": True}
            else:
                # coherent=false forces the distrust label (belt-and-braces vs the prompt rule).
                if not parsed["coherent"]:
                    parsed["factuality"], parsed["uncovered"] = 0.0, 0.0
                row[jm] = parsed
        if args.alignscore:
            from luq.labels import alignscore
            row["alignscore"] = alignscore.score(r)
        rows.append(row)
        print(f"  {i+1}/{len(sample)} idx {r['idx']}{' [SEVERE]' if severe else ''}", flush=True)

    report(rows, judges, args)


def report(rows, judges, args):
    primary = judges[0]
    def col(jm, k):
        return [row[jm][k] for row in rows if row[jm].get(k) is not None]

    print("\n" + "=" * 70)
    print(f"LABEL-GATE READS  (n={len(rows)}, primary judge={primary})")
    print("=" * 70)

    # READ 1 — uncovered fraction (the blind-spot number)
    unc = col(primary, "uncovered")
    print("\n[1] UNCOVERED fraction (the benchmark blind spot):")
    print(f"    mean {np.mean(unc):.2f}  median {np.median(unc):.2f}  "
          f">0.3: {np.mean(np.array(unc) > 0.3):.0%}  >0.5: {np.mean(np.array(unc) > 0.5):.0%}")
    allunc = np.mean([row[primary].get("factuality") is None for row in rows])
    print(f"    ALL-uncovered (factuality undefined, no covered claims to score): {allunc:.0%}")
    print("    -> small => uncovered can collapse to correct; large => coverage-extension needed first.")

    # READ 2 — derailment on the sample
    sev = np.mean([row["severe"] for row in rows])
    incoh = np.mean([row[primary].get("coherent") is False and not row["severe"] for row in rows])
    print(f"\n[2] DERAILMENT: severe(detector) {sev:.0%} | coherent=false marginal(judge) {incoh:.0%}")

    # READ 3 — mini vs gpt-5 (on THIS prompt)
    if len(judges) >= 2:
        a = [row[judges[0]]["factuality"] for row in rows if row[judges[0]]["factuality"] is not None
             and row[judges[1]]["factuality"] is not None]
        b = [row[judges[1]]["factuality"] for row in rows if row[judges[0]]["factuality"] is not None
             and row[judges[1]]["factuality"] is not None]
        print(f"\n[3] {judges[0]} vs {judges[1]} on factuality (n={len(a)}):")
        print(f"    Pearson {pearson(a,b):.2f}  Spearman {spearman(a,b):.2f}  MAD {np.mean(np.abs(np.array(a)-np.array(b))):.3f}")
        print("    -> high => adopt the cheaper judge; low => the projection needs the stronger judge.")

    # READ 4 — AlignScore vs judge (EXPLICIT: low corr = projection matters)
    if args.alignscore:
        al = [row.get("alignscore") for row in rows]
        fa = [row[primary]["factuality"] for row in rows]
        pairs = [(x, y) for x, y in zip(al, fa) if x is not None and y is not None]
        if len(pairs) >= 3:
            ax, fy = zip(*pairs)
            print(f"\n[4] AlignScore vs {primary} factuality (n={len(pairs)}):")
            print(f"    Pearson {pearson(ax,fy):.2f}  Spearman {spearman(ax,fy):.2f}")
            print("    -> LOW correlation is a RESULT: the factuality projection differs from")
            print("       gold-similarity, justifying why we did not just use AlignScore.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1, default=str))
    print(f"\nwrote per-example rows -> {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
