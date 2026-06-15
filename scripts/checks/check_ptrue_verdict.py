"""Check 3: do "yes"/"no" dominate the next token at the P(True) verdict position? (GPU)

For ~20 records, builds [prompt + response + PTRUE_SUFFIX] (exactly as ptrue.ptrue_vector does),
runs one forward pass, and looks at the next-token distribution at the LAST position -- the slot
whose hidden state P(True) probes. Reports the top-5 next tokens and the combined probability on
yes/no variants, averaged over the records. If yes/no dominate, that position really encodes a
yes/no verdict (so the probed hidden state is a verdict, not noise).

    srun --partition=t4 --gres=gpu:1 python scripts/check_ptrue_verdict.py --dataset pubmed_qa
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from luq import cache, generate
from luq.config import Config
from luq.features.ptrue import PTRUE_SUFFIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pubmed_qa")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = [r for r in cache.load_records(cfg.cache_dir, key) if r["split"] == "test"][:args.n]

    model, tok = generate.load_model(cfg.model_name)
    suffix_ids = tok(PTRUE_SUFFIX, add_special_tokens=False).input_ids

    # Collect single-token ids for several yes/no surface forms (with and without a leading space).
    yesno_ids = set()
    for s in [" yes", " no", " Yes", " No", "yes", "no", "Yes", "No", " YES", " NO"]:
        ids = tok(s, add_special_tokens=False).input_ids
        if len(ids) == 1:
            yesno_ids.add(ids[0])
    yesno_ids = sorted(yesno_ids)
    print("yes/no single-token ids:", yesno_ids,
          "->", [repr(tok.decode([i])) for i in yesno_ids])

    masses = []
    for r in records:
        full = r["prompt_token_ids"] + r["gen_token_ids"] + suffix_ids
        ids = torch.tensor(full)[None].to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        mass = float(probs[yesno_ids].sum())
        masses.append(mass)
        top = torch.topk(probs, 5)
        shown = ", ".join(f"{repr(tok.decode([t]))}={p:.2f}"
                          for t, p in zip(top.indices.tolist(), top.values.tolist()))
        print(f"yes/no mass {mass:.3f} | top5: {shown}")

    print(f"\nmean yes/no probability mass over {len(masses)} records: "
          f"{statistics.mean(masses):.3f}")
    print("High mass / yes-no in the top-5 => the verdict position encodes a verdict (signal, not noise).")


if __name__ == "__main__":
    main()
