"""Locate WHERE teacher-forced log-probabilities diverge from the cached ones, position by position.

WHY. scripts/01n_token_entropy.py refuses to write an entropy cache unless the log-probability it
re-derives under teacher forcing matches the one stored at generation time. That check fails on five
of eight datasets for one model and on all eight for another, by 3 to 9 in log-probability, which is
a different distribution rather than rounding. Several explanations have already been ruled out by
checking rather than argument: the corrected answer span (every changed row is an exact token-level
prefix with an exact log-probability prefix), record structure, weight sharding across cards, and
model dtype.

WHAT THIS SEPARATES. The first generated token's probability depends on the prompt and nothing else.
So:

  divergence at position 0        -> the stored prompt is not the prompt the generation was produced
                                     under, and every later position inherits that.
  agreement at 0, divergence later -> the prompt is right and something about the continuation is
                                     not: a decoding setting, a truncation, or a record assembled
                                     from two different runs.

Reporting the FIRST diverging position per record, and the profile around it, distinguishes those
without guessing. Nothing is written.

  python scripts/checks/entropy_window_locate.py --dataset med_quad --prompt-regime cleanv2 --n 8
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prompt-regime", default="")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.02, help="the driver's own alignment tolerance")
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

    model, tok = generate.load_model(cfg.model_name, attn_implementation=args.attn,
                                     dtype=_DTYPE[args.dtype],
                                     device_map=args.device_map, max_memory=max_memory)
    model.eval()
    print(f"{args.model} | {args.dataset} (regime '{args.prompt_regime or 'canonical'}') "
          f"| {min(args.n, len(records))} records | tol {args.tol}", flush=True)

    first_positions, pos0_diffs, n_used = [], [], 0
    for i, r in enumerate(records[:args.n]):
        p_ids, g_ids = list(r["prompt_token_ids"]), list(r["gen_token_ids"])
        cached = r.get("token_logprobs")
        P, G = len(p_ids), len(g_ids)
        if not G or cached is None or len(cached) != G:
            continue
        with torch.no_grad():
            ids = torch.tensor(p_ids + g_ids)[None].to(model.device)
            logits = model(ids).logits[0]
            win = logits[P - 1:P + G - 1].float()
            logp = torch.log_softmax(win, dim=-1)
            got = logp[torch.arange(G), torch.tensor(g_ids, device=logp.device)].cpu().numpy()
        ref = np.asarray(cached, dtype=float)
        d = np.abs(got - ref)
        bad = np.where(d > args.tol)[0]
        first = int(bad[0]) if len(bad) else -1
        first_positions.append(first)
        pos0_diffs.append(float(d[0]))
        n_used += 1
        head = " ".join(f"{x:.4f}" for x in d[:6])
        print(f"  row {i}: G={G:<4} |d| at pos 0 = {d[0]:.6f}   first pos over tol = "
              f"{'none' if first < 0 else first}   max |d| = {d.max():.4f}", flush=True)
        print(f"          |d| for the first 6 positions: {head}", flush=True)

    if not n_used:
        sys.exit("no usable records")
    fp = np.array(first_positions)
    p0 = np.array(pos0_diffs)
    print(f"\n=== SUMMARY over {n_used} records ===")
    print(f"  records diverging at position 0            : {(fp == 0).sum()}/{n_used}")
    print(f"  records never diverging beyond tol         : {(fp < 0).sum()}/{n_used}")
    print(f"  records first diverging after position 0   : {((fp > 0)).sum()}/{n_used}")
    print(f"  |d| at position 0: median {np.median(p0):.6f}, max {p0.max():.6f}")
    if (fp == 0).sum() == n_used:
        print("\nREADING: every record disagrees at the FIRST generated token. That token's "
              "probability depends only on the prompt, so the stored prompt is not the one the "
              "generation was produced under.")
    elif (fp < 0).sum() == n_used:
        print("\nREADING: no record diverges beyond tolerance here.")
    else:
        print("\nREADING: divergence does not start uniformly at position 0, so a prompt mismatch "
              "alone does not explain it. Read the per-record profiles above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
