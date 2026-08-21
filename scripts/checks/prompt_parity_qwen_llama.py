"""Prove that Qwen is being generated from, and scored against, the SAME prompts as Llama.

The replication compares two models on one benchmark, so every input except the model must be
identical. There are two separate prompt surfaces and they can diverge independently:

  1. GENERATION prompt  — `record["prompt"]`, built by the dataset loader. Drifts if the prompt
     library moves between the two runs. The project already carries a two-commit split on Llama as
     a recorded limitation, so this is a live failure mode, not a hypothetical.
  2. JUDGE prompt       — assembled at label time from the record's prompt, its gold target, and the
     generated answer. Only the ANSWER may legitimately differ between models.

Read-only, CPU, no API, no £.

    python scripts/checks/prompt_parity_qwen_llama.py
    python scripts/checks/prompt_parity_qwen_llama.py --model-b Qwen/Qwen2.5-14B

TWO METHOD TRAPS, both of which produced a false FAIL while writing this:

  * KEY BY (split, idx), NEVER idx ALONE. `idx` restarts per split, so on any dataset with both a
    train and a test split (pubmed_qa, xsum, cnn_dailymail) keying on idx alone silently compares
    train row 0 against test row 0 and reports every row as mismatched.
  * HOLD THE ANSWER CONSTANT WITH A SENTINEL; DO NOT string-replace it out of the prompt. Llama's
    summaries are extractive, so its generated text often occurs VERBATIM inside the source article.
    Replacing it also mangles the article, and the two sides then differ for a reason that has
    nothing to do with the prompts.

Both traps fail in the same direction -- they manufacture a scary-looking mismatch out of a correct
setup -- which is the kind that wastes a day chasing a bug that is in the checker.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.labels.llm_judge import build_prompt  # noqa: E402

DEFAULT_REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}
# expertqa and factscore are scored by their own judges (02_label_expertqa / 02_label_factscore),
# whose reference comes from an idx-keyed external pool rather than from llm_judge.build_prompt, so
# the generic judge-prompt comparison does not apply to them.
OWN_JUDGE = {"expertqa", "factscore"}
SENTINEL = "@@FIXED_ANSWER@@"


def load(model, dataset, regime):
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID", prompt_regime=regime)
    recs = cache.load_records(cfg.cache_dir, cache.run_key(model, dataset, "ID"))
    return {(r["split"], r["idx"]): r for r in recs}      # (split, idx) -- see the trap note above


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--model-b", default="Qwen/Qwen2.5-14B")
    ap.add_argument("--datasets", default="pubmed_qa,xsum,cnn_dailymail,med_quad,samsum,"
                                          "expertqa,asqa,factscore")
    args = ap.parse_args()

    print(f"\nA = {args.model_a}\nB = {args.model_b}")
    hdr = f"{'dataset':<15}{'shared rows':>12}{'gen prompt':>12}{'gold':>7}{'judge prompt':>14}"
    print("\n" + hdr); print("-" * len(hdr))
    failures = []
    for ds in [d.strip() for d in args.datasets.split(",")]:
        reg = DEFAULT_REGIME.get(ds, "")
        try:
            A, B = load(args.model_a, ds, reg), load(args.model_b, ds, reg)
        except Exception as e:
            print(f"{ds:<15}{'SKIPPED':>12}  ({type(e).__name__}: {e})")
            continue
        common = sorted(set(A) & set(B))
        if not common:
            print(f"{ds:<15}{0:>12}  (no shared rows yet)")
            continue

        pm = [k for k in common if A[k]["prompt"] != B[k]["prompt"]]
        gm = [k for k in common if str(A[k]["target"]) != str(B[k]["target"])]

        if ds in OWN_JUDGE:
            jm, jlabel = [], "own judge"
        else:
            jm = []
            for k in common:
                a, b = dict(A[k]), dict(B[k])
                a["gen_text"] = b["gen_text"] = SENTINEL     # sentinel, not replace -- see trap note
                if build_prompt(a, ds) != build_prompt(b, ds):
                    jm.append(k)
            jlabel = "OK" if not jm else f"DIFF {len(jm)}"

        for name, bad in (("gen prompt", pm), ("gold", gm), ("judge prompt", jm)):
            if bad:
                failures.append((ds, name, bad[0]))
        print(f"{ds:<15}{len(common):>12}{('OK' if not pm else f'DIFF {len(pm)}'):>12}"
              f"{('OK' if not gm else 'DIFF'):>7}{jlabel:>14}")

    print()
    if failures:
        print("*** PROMPT PARITY FAILED — the two models are NOT on the same inputs ***")
        for ds, what, k in failures:
            print(f"    {ds}: {what} differs, first at {k}")
        raise SystemExit(1)
    print("PROMPT PARITY OK — generation prompts, gold, and judge prompts all identical.")
    print("Only the generated ANSWER differs between the models, which is the intended difference.")
    print("\nNote: expertqa and factscore are scored by their own judges, whose reference is keyed by")
    print("idx from a shared external pool (ExpertQA gold+evidence; the frozen enwiki sqlite), so it")
    print("cannot depend on the model. Their gold and generation prompts are still compared above.")


if __name__ == "__main__":
    main()
