"""Is the cached log-probability that of the EMITTED token, or of the model's top choice?

The divergence locator established three things about the datasets whose entropy cache cannot be
written: the prompt is right (the first generated token agrees to about 1e-05), most positions agree
exactly, and isolated positions disagree by 0.2 to 9 in log-probability. A repetition penalty was
ruled out: the diverging tokens are repeats less often than random positions are.

That pattern is what a record would look like if, at some positions, `token_logprobs` holds the
log-probability of the token the model ranked FIRST while `gen_token_ids` holds a different token
that was actually emitted. Teacher forcing scores the emitted token and gets a much lower value,
while every position where the emitted token WAS the top choice agrees exactly.

This tests that directly. At each position it reports the recomputed log-probability of the emitted
token, the recomputed maximum over the vocabulary, and which of the two the cached value matches.
Nothing is written.

  python scripts/checks/entropy_window_top1.py --dataset med_quad --prompt-regime cleanv2 --n 6
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
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tol", type=float, default=0.02)
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
    print(f"{args.model} | {args.dataset} (regime '{args.prompt_regime or 'canonical'}')", flush=True)

    n_match_emitted = n_match_top1 = n_match_neither = n_bad = 0
    n_emitted_is_top1 = n_pos = 0
    for i, r in enumerate(records[:args.n]):
        p_ids, g_ids = list(r["prompt_token_ids"]), list(r["gen_token_ids"])
        cached = r.get("token_logprobs")
        P, G = len(p_ids), len(g_ids)
        if not G or cached is None or len(cached) != G:
            continue
        with torch.no_grad():
            ids = torch.tensor(p_ids + g_ids)[None].to(model.device)
            logp = torch.log_softmax(model(ids).logits[0][P - 1:P + G - 1].float(), dim=-1)
            emitted = logp[torch.arange(G), torch.tensor(g_ids, device=logp.device)].cpu().numpy()
            top1v, top1i = logp.max(dim=-1)
            top1v = top1v.cpu().numpy(); top1i = top1i.cpu().numpy()
        ref = np.asarray(cached, dtype=float)
        bad = np.where(np.abs(emitted - ref) > args.tol)[0]
        n_bad += len(bad); n_pos += G
        n_emitted_is_top1 += int((top1i == np.array(g_ids)).sum())
        for pos in bad[:4]:
            de, dt = abs(ref[pos] - emitted[pos]), abs(ref[pos] - top1v[pos])
            which = ("EMITTED" if de <= args.tol else ("TOP-1" if dt <= args.tol else "NEITHER"))
            n_match_emitted += de <= args.tol
            n_match_top1 += (de > args.tol and dt <= args.tol)
            n_match_neither += (de > args.tol and dt > args.tol)
            print(f"  row {i} pos {pos:<4} cached {ref[pos]:8.4f} | emitted {emitted[pos]:8.4f} "
                  f"| top-1 {top1v[pos]:8.4f} | cached matches {which}"
                  f" | emitted==top1: {top1i[pos] == g_ids[pos]}", flush=True)

    print(f"\n=== SUMMARY ===")
    print(f"  positions checked overall           : {n_pos}")
    print(f"  positions over tolerance            : {n_bad}")
    print(f"  emitted token WAS the model's top-1 : {n_emitted_is_top1}/{n_pos} "
          f"({100*n_emitted_is_top1/max(n_pos,1):.1f}%)")
    print(f"  of the diverging ones sampled, cached matches the recomputed:")
    print(f"    EMITTED token : {n_match_emitted}")
    print(f"    TOP-1 token   : {n_match_top1}")
    print(f"    NEITHER       : {n_match_neither}")
    if n_match_top1 and not n_match_emitted:
        print("\nREADING: the cached value is the model's TOP-1 log-probability, not the emitted "
              "token's. The record stores the wrong quantity at positions where generation did not "
              "pick the top-ranked token.")
    elif n_match_neither and not n_match_top1:
        print("\nREADING: the cached value matches neither, so it was produced under a different "
              "distribution again. Neither hypothesis survives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
