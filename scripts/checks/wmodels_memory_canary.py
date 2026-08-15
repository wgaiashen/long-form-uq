"""Memory canary for a W-Models replication population: does the LONGEST real input fit?

WHY THIS EXISTS, AND WHY `01_extract --limit N` IS NOT A SUBSTITUTE
------------------------------------------------------------------
`--limit N` takes the FIRST N rows by index. The rows that OOM are the LONGEST ones, and on xsum the
long tail sits nowhere near the front: the median prompt is ~395 tokens (Llama tokeniser) against a
maximum of ~3,801, and Qwen tokenises the same articles longer still (~5,908 measured on xsum for
Qwen2.5-14B). So a 40-row smoke can pass comfortably and the full run can still die hours in.

This runs the ACTUAL extraction path over the longest inputs only, and reports peak GPU memory. It is
an engineering gate, not an experiment: it writes no cache, produces no uncertainty score, and
nothing it prints may enter a results table.

The tokenisation is done with the TARGET model's own tokeniser, because "longest" is tokeniser
-dependent -- ranking by the Llama tokeniser would pick the wrong rows for Qwen or Gemma.

    python scripts/checks/wmodels_memory_canary.py --model Qwen/Qwen2.5-32B --dtype bf16 \
        --dataset xsum --top 3 --max-new-tokens 56
"""
import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import data, generate  # noqa: E402

_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="xsum")
    ap.add_argument("--dtype", required=True, choices=["fp32", "fp16", "bf16"],
                    help="never left to auto -- the point is to test the dtype the real run will use")
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--top", type=int, default=3, help="how many of the LONGEST prompts to run")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="defaults to the dataset's canonical budget")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device -- run this on a compute node, it is a GPU memory gate")

    budget = args.max_new_tokens or data.MAX_NEW_TOKENS[args.dataset]
    print(f"=== memory canary: {args.model} | {args.dataset} | {args.dtype} + {args.attn} "
          f"| budget {budget} ===", flush=True)

    # 1. Get the dataset's prompts (model-independent) and rank them by THIS model's tokeniser.
    #    data.load() is the project wrapper -- it routes asqa/factscore/samsum/med_quad/expertqa to
    #    their own loaders rather than straight to probe_drift.get_datasets.
    train_ds, eval_ds = data.load(args.dataset, "ID")
    prompts = list(train_ds.x) + list(eval_ds.x)
    print(f"  {len(prompts)} prompts in the ID pool", flush=True)

    model, tok = generate.load_model(args.model, attn_implementation=args.attn,
                                     dtype=_DTYPE[args.dtype])
    n_layers = model.config.num_hidden_layers
    print(f"  loaded: {n_layers} layers, hidden {model.config.hidden_size}, "
          f"fixed-rule layer {math.ceil(n_layers / 2) - 1}", flush=True)
    print(f"  weights on GPU: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    lens = [(len(tok(p, add_special_tokens=False)["input_ids"]), i) for i, p in enumerate(prompts)]
    lens.sort(reverse=True)
    print(f"  prompt tokens ({args.model} tokeniser): max {lens[0][0]}, "
          f"p50 {lens[len(lens) // 2][0]}, min {lens[-1][0]}", flush=True)

    # 2. Run the real generation path over the longest prompts, with hidden states requested --
    #    that is what the extraction actually does and what dominates memory.
    worst = 0.0
    for rank, (ntok, idx) in enumerate(lens[:args.top], 1):
        torch.cuda.reset_peak_memory_stats()
        try:
            generate.generate(model, tok, prompts[idx], max_new_tokens=budget)
        except torch.cuda.OutOfMemoryError as e:
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"  [{rank}/{args.top}] {ntok} tokens -> ❌ CUDA OOM at {peak:.1f} GiB peak",
                  flush=True)
            print(f"CANARY FAILED: {e}", flush=True)
            if "gemma" in args.model.lower():
                # ⛔ Do NOT suggest sdpa for Gemma-2. It soft-caps its attention logits and ONLY the
                # eager path applies that cap; SDPA silently skips it. Since the probe reads hidden
                # states, and those are computed FROM the attention, sdpa would give subtly wrong
                # internal features -- a quietly different method, not a memory optimisation.
                print("Gemma-2 REQUIRES eager (soft-capping), so switching to sdpa is NOT an option "
                      "here. Move this dataset to a larger card.", flush=True)
            else:
                print("Options, in order: switch --attn sdpa (eager materialises an n^2 buffer per "
                      "layer, and this panel does not use the attention-extraction path that needs "
                      "eager); or move this dataset to a larger card.", flush=True)
            print("⛔ Do NOT silently drop the dataset -- that leaves a hole in the panel and puts "
                  "this population on a different dataset set from every other one.", flush=True)
            sys.exit(1)
        peak = torch.cuda.max_memory_allocated() / 2**30
        worst = max(worst, peak)
        print(f"  [{rank}/{args.top}] {ntok} tokens -> peak {peak:.1f} GiB", flush=True)

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"  worst peak {worst:.1f} GiB of {total:.1f} GiB "
          f"({100 * worst / total:.0f}% of the card)", flush=True)
    print("CANARY PASSED", flush=True)


if __name__ == "__main__":
    main()
