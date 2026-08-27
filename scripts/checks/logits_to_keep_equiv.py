"""Gate: does skipping the discarded vocabulary projection change any hidden state?

WHY THIS EXISTS. The teacher-forced pass in luq.generate.recompute_states reads only hidden states
and attentions, but the wrapped causal-language-model still projects every position to the full
128,256-entry vocabulary. On a 2,500-token sequence that is roughly 1.2 GB allocated and thrown away,
and on two 24 GB cards it is the allocation that sits between the extraction and the two longest
datasets in the grid. The library supports computing that projection at one position instead.

The argument that this cannot change a hidden state is easy to make and easy to get wrong: the head
consumes the final hidden state and nothing downstream feeds back, so the layers this function
returns are computed before the head runs at all. This project's convention is to check the vectors
rather than trust that reasoning, because a silent default that returns a plausible number is worse
than a crash, and because an equivalence claim tested on a rank statistic can hide a real difference.
So this compares the RETURNED TENSORS ELEMENTWISE and requires EXACT equality, not a tolerance.

It writes nothing into any cache. It loads a model, runs a handful of cached records both ways, and
reports. A non-zero exit means the substitution is not safe and must not be used.

  python scripts/checks/logits_to_keep_equiv.py --dataset cnn_dailymail --n 6 --layers 0,15,30
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from luq import cache, generate                                                  # noqa: E402
from luq.config import Config                                                    # noqa: E402

_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--dataset", default="cnn_dailymail")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--prompt-regime", default="")
    ap.add_argument("--layers", default="0,15,30")
    ap.add_argument("--n", type=int, default=6, help="records to compare")
    ap.add_argument("--longest", action="store_true",
                    help="take the longest records rather than the first, since length is what "
                         "makes the projection expensive and is where a difference would show")
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

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    seqs = [list(r["prompt_token_ids"]) + list(r["gen_token_ids"]) for r in records]
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i])) if args.longest else range(len(seqs))
    picked = list(order)[:args.n]

    model, tok = generate.load_model(cfg.model_name, attn_implementation=args.attn,
                                     dtype=_DTYPE[args.dtype],
                                     device_map=args.device_map, max_memory=max_memory)
    model.eval()
    if args.device_map == "auto":
        placed = getattr(model, "hf_device_map", {})
        n_dev = len({v for v in placed.values() if isinstance(v, int)})
        print(f"device map: {n_dev} GPU(s) hold weights", flush=True)

    print(f"{args.model} | {args.dataset} | layers {layers} | {len(picked)} records "
          f"| {'longest' if args.longest else 'first'}", flush=True)

    worst = 0.0
    n_mismatch = 0
    for i in picked:
        ids = seqs[i]
        with torch.no_grad():
            base = generate.recompute_states(model, tok, ids, layers)
            slim = generate.recompute_states(model, tok, ids, layers, logits_to_keep=1)
        for L, a, b in zip(layers, base, slim):
            if a.shape != b.shape:
                print(f"  FAIL record {i} layer {L}: shape {a.shape} vs {b.shape}")
                n_mismatch += 1
                continue
            d = (a - b).abs().max().item()
            worst = max(worst, d)
            if d != 0.0:
                n_mismatch += 1
                print(f"  record {i} (len {len(ids)}) layer {L}: max|d| = {d:.3e}")
        print(f"  record {i}: len {len(ids)}, checked {len(layers)} layers, "
              f"running worst {worst:.3e}", flush=True)

    print(f"\nrecords {len(picked)} | layers {layers} | worst max|d| = {worst:.6e} "
          f"| tensors differing: {n_mismatch}")
    if n_mismatch == 0 and worst == 0.0:
        print("PASS: hidden states are BIT-IDENTICAL with and without the projection.")
        return 0
    print("FAIL: the substitution changes returned states. Do NOT build caches with it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
