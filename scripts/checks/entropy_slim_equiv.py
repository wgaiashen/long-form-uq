"""Gate: does restricting the vocabulary projection to the scored window change the entropy?

WHY THIS EXISTS. scripts/01n_token_entropy.py needs the full predictive distribution at every
position it scores, so it cannot skip the projection the way the hidden-state extractor can. What it
CAN skip is the part it throws away: for a length-(P+G) sequence it asks for logits at all P+G
positions and then keeps only positions P-1 to P+G-2. Since those are the last G+1 positions, the
library can be asked for exactly that suffix instead. On a 5,866-token source with a 56-token
generation this is 29 MB rather than 3.0 GB, and on a model with a 256,000-entry vocabulary it is
58 MB rather than 6.0 GB.

The entropy definition is NOT changed. Both paths take the same positions, apply the same
log-softmax over the same full vocabulary, and sum the same quantity. The claim under test is only
that asking for a suffix returns the same numbers as asking for everything and slicing.

That claim is checked on the VECTORS elementwise, requiring exact equality, on records where the
full-logits path still fits. It also reports the measured peak memory of each path, because the whole
justification is a memory claim and this project does not accept unmeasured ones.

  python scripts/checks/entropy_slim_equiv.py --dataset xsum --n 8 --longest
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from luq import cache, generate                                                  # noqa: E402
from luq.config import Config                                                    # noqa: E402

_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def entropy_for(model, p_ids, g_ids, slim):
    """Exactly the two code paths in 01n_token_entropy, side by side and nothing else."""
    P, G = len(p_ids), len(g_ids)
    with torch.no_grad():
        ids = torch.tensor(p_ids + g_ids)[None].to(model.device)
        if slim:
            logits = model(ids, logits_to_keep=G + 1).logits[0]
            win = logits[:G].float()
        else:
            logits = model(ids).logits[0]
            win = logits[P - 1:P + G - 1].float()
        logp = torch.log_softmax(win, dim=-1)
        H = -(logp.exp() * logp).sum(-1)
        chosen = logp[torch.arange(G), torch.tensor(g_ids, device=logp.device)]
    return H.float().cpu(), chosen.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--dataset", default="xsum")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--prompt-regime", default="")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--longest", action="store_true",
                    help="take the longest records, where the saving is largest and where the "
                         "full-logits path is closest to not fitting")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--device-map", default="cuda")
    ap.add_argument("--max-memory", default="")
    args = ap.parse_args()

    max_memory = None
    if args.max_memory:
        max_memory = {}
        for item in args.max_memory.split(","):
            k, v = item.split("=")
            max_memory[int(k.strip())] = v.strip()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    usable = [r for r in records if len(r["gen_token_ids"]) > 0]
    if args.longest:
        usable.sort(key=lambda r: -(len(r["prompt_token_ids"]) + len(r["gen_token_ids"])))
    picked = usable[:args.n]

    model, tok = generate.load_model(cfg.model_name, attn_implementation=args.attn,
                                     dtype=_DTYPE[args.dtype],
                                     device_map=args.device_map, max_memory=max_memory)
    model.eval()
    placed = getattr(model, "hf_device_map", {})
    n_dev = len({v for v in placed.values() if isinstance(v, int)})
    print(f"{args.model} | {args.dataset} | {len(picked)} records | {n_dev or 1} GPU(s) hold weights",
          flush=True)

    # Model-weight footprint on its own, separately from anything the forward allocates. The
    # projection saving does not help with weights, and a model that does not fit will not be made to
    # fit by this change.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        for d in range(torch.cuda.device_count()):
            print(f"  weights resident on device {d}: "
                  f"{torch.cuda.memory_allocated(d) / 2 ** 30:.2f} GiB")

    peaks = {}
    worst_H, worst_lp, n_bad = 0.0, 0.0, 0
    for slim in (False, True):
        if torch.cuda.is_available():
            for d in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(d)
        res = []
        for r in picked:
            res.append(entropy_for(model, list(r["prompt_token_ids"]), list(r["gen_token_ids"]), slim))
        peaks[slim] = [torch.cuda.max_memory_allocated(d) / 2 ** 30
                       for d in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        if not slim:
            base = res
        else:
            for r, (Hb, cb), (Hs, cs) in zip(picked, base, res):
                n = len(r["gen_token_ids"])
                if Hb.shape != Hs.shape:
                    print(f"  FAIL shape {Hb.shape} vs {Hs.shape}")
                    n_bad += 1
                    continue
                dH = (Hb - Hs).abs().max().item()
                dl = (cb - cs).abs().max().item()
                worst_H, worst_lp = max(worst_H, dH), max(worst_lp, dl)
                if dH != 0.0 or dl != 0.0:
                    n_bad += 1
                    print(f"  record len {len(r['prompt_token_ids']) + n}: "
                          f"entropy max|d| {dH:.3e}, chosen-logprob max|d| {dl:.3e}")

    print(f"\nrecords {len(picked)} | entropy worst max|d| = {worst_H:.6e} | "
          f"chosen-logprob worst max|d| = {worst_lp:.6e} | vectors differing: {n_bad}")
    for slim in (False, True):
        tag = "suffix-only" if slim else "all positions"
        if peaks[slim]:
            print(f"PEAK allocated, {tag:<14}: "
                  + ", ".join(f"dev{d} {p:.2f} GiB" for d, p in enumerate(peaks[slim])))
    if n_bad == 0 and worst_H == 0.0 and worst_lp == 0.0:
        print("PASS: entropies are BIT-IDENTICAL. The suffix path may be used for production.")
        return 0
    print("FAIL: the suffix path changes the entropy. Do NOT use it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
