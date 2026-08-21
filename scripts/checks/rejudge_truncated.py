"""Step 4 of the answer-span plan: does truncating to the answer span shift the JUDGE label?

If the model's trailing junk (fake Q/A, ramble, loops) was DEPRESSING the correctness label -- the
judge marking a good-answer-plus-junk generation down -- then re-judging the clean_text should raise
the label, and the raw labels we trained/evaluated on were biased by an artefact. If the label is
unchanged, the judge already scored the answer content and truncation is label-neutral.

Re-judges a SAMPLE of the CUT rows (rows where answer_span actually removed text -- the only rows a
label could move) on clean_text vs raw gen_text, with the SAME judge model (paired comparison).
For med_quad it additionally runs GPT-5 alongside the cheap judge on the same rows, to report the
cheap-vs-GPT-5 agreement (Spearman) and confirm the cheap judge tracks the expensive one here.

Runs on the LOGIN NODE (API, light). Costs £ (gpt-5-mini cheap; the med_quad GPT-5 pass is 2*N calls).

    python scripts/checks/rejudge_truncated.py --dataset xsum --n 300 --judge gpt-5-mini
    python scripts/checks/rejudge_truncated.py --dataset med_quad --n 80 --judge gpt-5-mini --also-gpt5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import answer_span as A, cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.labels import llm_judge  # noqa: E402

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B"
GPT5 = "gpt-5-2025-08-07"

# PORTED TO --model 2026-08-11 (Qwen2.5-14B). The default is the exact original string and the
# default --cut-source is still `answer_span`, so every pre-existing invocation is byte-identical.
#
# WHY A SECOND CUT SOURCE. `answer_span` has rules only for {med_quad, xsum, pubmed_qa} | SHORT_FORM.
# On samsum / cnn_dailymail / expertqa / factscore it returns "no-cut" for every row, so this script
# would sample ZERO rows there and silently report nothing — which reads as "no problem found".
# `luq.template_restart` supplies the frozen boundaries for those four, derived from the datasets'
# own prompt templates and committed BEFORE any of their PRR effects were computed. It is opt-in via
# --cut-source so nothing about the Llama path changes by accident.


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def judge_pair(rec, clean, dataset, model):
    """(raw_score, trunc_score) for one record under `model`. trunc swaps gen_text -> clean_text."""
    raw = llm_judge.judge(rec, dataset, model=model)
    trec = dict(rec); trec["gen_text"] = clean
    trunc = llm_judge.judge(trec, dataset, model=model)
    return raw, trunc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--judge", default="gpt-5-mini")
    ap.add_argument("--also-gpt5", action="store_true", help="also run GPT-5 (med_quad cross-check)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="EXPLICIT model pin. Default is the original string, so existing "
                         "invocations are byte-identical.")
    ap.add_argument("--cut-source", choices=["answer_span", "template_restart"],
                    default="answer_span",
                    help="where the clean span comes from. answer_span (default) is the promoted "
                         "Llama rule set; template_restart carries the frozen boundaries for the "
                         "four datasets answer_span has no rule for.")
    args = ap.parse_args()

    from attn_pool import PROMPT_REGIME  # noqa: E402  (the dataset -> cache-namespace map)
    from luq import template_restart as TR  # noqa: E402
    MODEL = args.model
    cfg = Config(model_name=MODEL, dataset=args.dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(args.dataset, ""))
    recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, args.dataset, "ID"))
    print(f"[{args.dataset}] model={MODEL} cut-source={args.cut_source} "
          f"namespace={PROMPT_REGIME.get(args.dataset, '') or '<base>'} rows={len(recs)}", flush=True)

    # only CUT rows can move the label; sample from those
    cut = []
    for r in recs:
        if args.cut_source == "answer_span":
            clean, _, rsn = A.answer_span(r["gen_text"], args.dataset, context=r.get("prompt"))
            fired = not rsn.startswith("no-cut")
        else:
            clean, ch, rsn, _st = TR.restart_cut(r["gen_text"], args.dataset)
            fired = ch is not None
        if fired and clean.strip() and clean.strip() != r["gen_text"].strip():
            cut.append((r, clean))
    rng = np.random.RandomState(args.seed)
    if len(cut) > args.n:
        cut = [cut[i] for i in rng.choice(len(cut), args.n, replace=False)]
    print(f"[{args.dataset}] {len(cut)} cut rows sampled (of the dataset's cut rows)", flush=True)

    raw_c, tr_c = [], []
    g5_raw, g5_tr = [], []
    for j, (rec, clean) in enumerate(cut):
        r, t = judge_pair(rec, clean, args.dataset, args.judge)
        if r is not None and t is not None:
            raw_c.append(r); tr_c.append(t)
        if args.also_gpt5:
            r5, t5 = judge_pair(rec, clean, args.dataset, GPT5)
            if r5 is not None and t5 is not None:
                g5_raw.append(r5); g5_tr.append(t5)
        if (j + 1) % 20 == 0:
            print(f"  {j+1}/{len(cut)} judged", flush=True)

    raw_c, tr_c = np.array(raw_c), np.array(tr_c)
    d = tr_c - raw_c
    print(f"\n=== {args.dataset}  judge={args.judge}  (n={len(raw_c)} cut rows) ===")
    print(f"  mean label  RAW={raw_c.mean():.3f}  TRUNC={tr_c.mean():.3f}  delta={d.mean():+.3f}")
    print(f"  |delta|>0.2: {100*np.mean(np.abs(d)>0.2):.1f}%   raw-vs-trunc spearman={spearman(raw_c, tr_c):.3f}")
    if args.also_gpt5 and g5_raw:
        g5_raw, g5_tr = np.array(g5_raw), np.array(g5_tr)
        print(f"  GPT-5  RAW={g5_raw.mean():.3f}  TRUNC={g5_tr.mean():.3f}  delta={ (g5_tr-g5_raw).mean():+.3f}")
        n = min(len(raw_c), len(g5_raw))
        print(f"  cheap-vs-GPT5 spearman: RAW={spearman(raw_c[:n], g5_raw[:n]):.3f}  "
              f"TRUNC={spearman(tr_c[:n], g5_tr[:n]):.3f}")

    out = ROOT / "results" / f"rejudge_truncated__{args.dataset}__{args.judge}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": args.dataset, "judge": args.judge,
                               "raw": raw_c.tolist(), "trunc": tr_c.tolist(),
                               "gpt5_raw": [float(x) for x in g5_raw] if args.also_gpt5 else [],
                               "gpt5_trunc": [float(x) for x in g5_tr] if args.also_gpt5 else []}, indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
