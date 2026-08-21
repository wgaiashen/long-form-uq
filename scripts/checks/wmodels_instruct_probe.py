"""Predict the two instruct-checkpoint failure modes before a 12-hour run pays for them.

WHY THIS EXISTS
---------------
`google/gemma-2-9b-it` was withdrawn (prereg M5 deviation D1) after two failures that are properties
of INSTRUCTION TUNING, not of Gemma:

  1. WHITESPACE-FRONTING. An instruct model under raw few-shot prompting emits a leading newline.
     `pubmed_qa` generates with `--truncate-long` (the `generate_until=['\\n']`), so that newline
     truncates the entire answer away: 200/200 records came back as a bare "\\n". The job exits 0.
  2. ASSISTANT PERSONA. The model answers correctly and then keeps talking ("Let me know if you'd
     like me to analyze any other text!"). The judge scores the whole saved output, so the filler is
     graded as if it were the answer.

Any other instruct checkpoint is a candidate for both. This asks the question in ~5 minutes on an
abundant card instead of discovering it after 12 hours on a scarce one.

NOT A POPULATION. Writes nothing to any cache. Its numbers may never enter a results table. It is
allowed to run in bf16 on a small card even when the real population is fp32, because "is the first
emitted token whitespace?" is not a dtype-sensitive question.

It generates WITHOUT `truncate_at_newline` on purpose and inspects the first token itself. That
reproduces the CAUSE rather than the symptom, and it also shows what the untruncated answer would
have been -- which tells you whether the model is broken or merely mis-delimited.

    python scripts/checks/wmodels_instruct_probe.py --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import data, generate  # noqa: E402

_CHATTER = re.compile(
    r"(let me know|i hope (this|that) helps|hope this helps|would you like|feel free to|"
    r"anything else|if you'd like|shall i|do you want me to|is there anything)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", default="pubmed_qa,samsum")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device -- this needs a GPU")

    print(f"=== instruct probe: {args.model} (DIAGNOSTIC, writes nothing) ===", flush=True)
    model, tok = generate.load_model(args.model, attn_implementation="eager", dtype=torch.bfloat16)

    verdicts = {}
    for ds in [d for d in args.datasets.split(",") if d]:
        train_ds, _ = data.load(ds, "ID")
        prompts = list(train_ds.x)[:args.n]
        budget = data.MAX_NEW_TOKENS[ds]
        print(f"\n=== {ds} (budget {budget}, n={len(prompts)}) ===", flush=True)
        lead = chat = 0
        for i, p in enumerate(prompts):
            rec, _ = generate.generate(model, tok, p, max_new_tokens=budget)
            ids = rec["gen_token_ids"]
            first = tok.decode(ids[:1]) if len(ids) else ""
            txt = rec["gen_text"] or ""
            if first.strip() == "":
                lead += 1
            if _CHATTER.search(txt):
                chat += 1
            print(f"  [{i}] first={first!r} len={len(ids):3d} text={txt[:100]!r}", flush=True)
        print(f"  --> leading-whitespace {lead}/{len(prompts)}   chatter {chat}/{len(prompts)}",
              flush=True)
        verdicts[ds] = (lead, chat, len(prompts))

    print("\n=== verdict ===", flush=True)
    for ds, (lead, chat, n) in verdicts.items():
        notes = []
        if ds == "pubmed_qa" and lead:
            notes.append(f"pubmed_qa WOULD BE EMPTY under --truncate-long ({lead}/{n} lead with "
                         f"whitespace) -- this is the gemma-2-9b-it failure")
        if chat:
            notes.append(f"assistant chatter on {chat}/{n}")
        print(f"  {ds:14s} " + ("; ".join(notes) if notes else "clean on both mechanisms"),
              flush=True)
    print("\n  Diagnostic only. Nothing here is a population or a result.", flush=True)


if __name__ == "__main__":
    main()
