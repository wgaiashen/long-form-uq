"""Diagnostic: can we concentrate Gemma's yes/no verdict mass by fixing the CUE, not the question?

The verdict mass was low on Gemma because the top token at the verdict slot is whitespace
(' ', '\\n\\n') BEFORE ' Yes' — Gemma renders a leading space/newline, then the word. This tests
whether ending the cue differently (so the whitespace is already supplied) or using Gemma's chat
template moves the yes/no word to the immediate next position. Loads the model ONCE. GPU.

    srun --partition=a30 --gres=gpu:1 python scripts/checks/ptrue_cue_sweep.py
"""
import argparse, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
from luq import cache, generate
from luq.config import Config

Q = "Is the above response accurate?"   # the unified question; we vary only the CUE around it

# raw-concatenation cue variants (appended to prompt+answer token IDs, add_special_tokens=False)
RAW_CUES = {
    "A current  ':' ":      f"\n{Q} Answer (yes or no):",
    "B trailing space":     f"\n{Q} Answer (yes or no): ",
    "C trailing newline":   f"\n{Q} Answer (yes or no):\n",
    "D 'Answer:' only":     f"\n{Q} Answer:",
    "E newline+Answer:":    f"\n{Q}\nAnswer:",
}


def yesno_ids(tok):
    ids = set()
    for s in [" yes", " no", " Yes", " No", "yes", "no", "Yes", "No", " YES", " NO"]:
        t = tok(s, add_special_tokens=False).input_ids
        if len(t) == 1:
            ids.add(t[0])
    return sorted(ids)


def mass_and_top(model, tok, full_ids, yn):
    ids = torch.tensor(full_ids)[None].to(model.device)
    with torch.no_grad():
        probs = torch.softmax(model(ids).logits[0, -1].float(), dim=-1)
    top = torch.topk(probs, 5)
    shown = ", ".join(f"{tok.decode([t])!r}={p:.2f}" for t, p in zip(top.indices.tolist(), top.values.tolist()))
    return float(probs[yn].sum()), shown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--datasets", nargs="+", default=["sciq", "pubmed_qa"])
    args = ap.parse_args()

    model, tok = generate.load_model(args.model)
    yn = yesno_ids(tok)
    print("model:", args.model, flush=True)

    for ds in args.datasets:
        key = cache.run_key(args.model, ds, "ID")
        recs = [r for r in cache.load_records(Config.cache_dir, key) if r["split"] == "test"][:args.n]
        print(f"\n================ {ds} (n={len(recs)}) ================", flush=True)

        # raw cue variants
        for label, cue in RAW_CUES.items():
            sid = tok(cue, add_special_tokens=False).input_ids
            masses, sample = [], None
            for i, r in enumerate(recs):
                m, top = mass_and_top(model, tok, r["prompt_token_ids"] + r["gen_token_ids"] + sid, yn)
                masses.append(m)
                if i == 0:
                    sample = top
            print(f"  RAW {label:18s} mean yes/no {statistics.mean(masses):.3f} | ex top5: {sample}", flush=True)

        # chat-template variant: wrap [decoded prompt+answer + question] as a Gemma user turn
        masses, sample = [], None
        for i, r in enumerate(recs):
            body = tok.decode(r["prompt_token_ids"] + r["gen_token_ids"])
            msg = [{"role": "user", "content": f"{body}\n\n{Q} Answer yes or no."}]
            chat_ids = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=True)
            m, top = mass_and_top(model, tok, chat_ids, yn)
            masses.append(m)
            if i == 0:
                sample = top
        print(f"  CHAT-TEMPLATE         mean yes/no {statistics.mean(masses):.3f} | ex top5: {sample}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
