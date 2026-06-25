"""Diagnostic: yes/no verdict mass for several P(True) wordings, on THIS model.

The new unified wording ("Is the above response accurate?") failed the verdict-mass gate on Gemma
(sciq 0.47, pubmed 0.27) vs the old wording's ~0.96 — but that 0.96 was measured on Qwen-1.5B, never
on Gemma. This sweeps old vs new vs minimal variants on the SAME records so we can tell whether the
drop is the wording or the model, and find a task-agnostic wording that keeps the yes/no mass high.

Loads the model ONCE and evaluates every (wording x dataset). GPU.

    srun --partition=a40 --gres=gpu:1 python scripts/checks/ptrue_wording_sweep.py
"""
import argparse, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
from luq import cache, generate
from luq.config import Config

WORDINGS = {
    "OLD answer/true":        "\nIs the above answer true? Answer (yes or no):",
    "NEW response/accurate":  "\nIs the above response accurate? Answer (yes or no):",
    "response/true":          "\nIs the above response true? Answer (yes or no):",
    "answer/accurate":        "\nIs the above answer accurate? Answer (yes or no):",
    "response/correct":       "\nIs the above response correct? Answer (yes or no):",
    "answer/correct":         "\nIs the above answer correct? Answer (yes or no):",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--datasets", nargs="+", default=["sciq", "pubmed_qa"])
    args = ap.parse_args()

    model, tok = generate.load_model(args.model)
    yesno_ids = sorted({tok(s, add_special_tokens=False).input_ids[0]
                        for s in [" yes", " no", " Yes", " No", "yes", "no", "Yes", "No", " YES", " NO"]
                        if len(tok(s, add_special_tokens=False).input_ids) == 1})
    print("model:", args.model, "| yes/no ids:", yesno_ids, flush=True)

    print(f"\n{'dataset':10s} {'wording':24s} {'mean yes/no mass':>16s}", flush=True)
    for ds in args.datasets:
        key = cache.run_key(args.model, ds, "ID")
        recs = [r for r in cache.load_records(Config.cache_dir, key) if r["split"] == "test"][:args.n]
        for label, suffix in WORDINGS.items():
            sid = tok(suffix, add_special_tokens=False).input_ids
            masses = []
            for r in recs:
                full = r["prompt_token_ids"] + r["gen_token_ids"] + sid
                ids = torch.tensor(full)[None].to(model.device)
                with torch.no_grad():
                    probs = torch.softmax(model(ids).logits[0, -1].float(), dim=-1)
                masses.append(float(probs[yesno_ids].sum()))
            print(f"{ds:10s} {label:24s} {statistics.mean(masses):16.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
