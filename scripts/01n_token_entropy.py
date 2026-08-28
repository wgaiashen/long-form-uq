"""GPU step: cache the per-token PREDICTIVE ENTROPY of the generation, one float per generated token.

WHY THIS EXISTS
---------------
The published MSP-SATMD / MSP-SATRMD hybrids fit their meta-regressor on
[per-layer mean Mahalanobis distance, MSP, MEAN TOKEN ENTROPY]. The third feature is the entropy of
the model's FULL next-token distribution at each step, and the Tier-1 record deliberately stores only
the log-probability of the CHOSEN token, so it cannot be derived from what is already cached. Without
this cache the hybrid can only be run with a feature removed, which would make it a different method.

Entropy here is the same quantity the reference implementation computes:
    H_t = -sum_v p(v) log p(v)   over the full vocabulary at step t
evaluated at the positions that predict the generated tokens.

THE WINDOW, AND THE GATE THAT PROVES IT
---------------------------------------
In a teacher-forced pass over [prompt + generation], the distribution that produced generated token t
sits at position P-1+t, so the G generated tokens are predicted by logits[P-1 : P+G-1]. Getting this
off by one would still produce a plausible entropy vector, so the script does not assert the window in
a comment: it recomputes the chosen-token log-probability at those same positions and compares it to
the cached `token_logprobs`. A misaligned window fails that comparison immediately.

The cached log-probabilities came from generation with a KV cache, while this is a single teacher-forced
forward, so the two agree to roughly 1e-4 relative rather than exactly (the same inline-versus-
teacher-forced difference documented for the feature cache). The gate therefore checks agreement at a
loose absolute tolerance and REPORTS the observed maximum, which is the number that matters.

    python scripts/01n_token_entropy.py --model meta-llama/Meta-Llama-3.1-8B --dataset pubmed_qa

Writes cache/<regime>/entropy/<key>.npz with entropy (object array of (n_gen,) fp32), idx, and the
measured logprob agreement.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, generate  # noqa: E402
from luq.config import Config  # noqa: E402

_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--dtype", default="fp32", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--attn", default="eager", choices=["auto", "eager", "sdpa"])
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag; MUST match the one 01_extract used for this population")
    ap.add_argument("--tol", type=float, default=2e-2,
                    help="absolute tolerance for the chosen-token logprob agreement gate")
    ap.add_argument("--limit", type=int, default=0, help="debug only: stop after N records")
    ap.add_argument("--slim-logits", action="store_true",
                    help="ask the model for logits at only the last G+1 positions instead of all "
                         "P+G of them. Those last G+1 positions ARE the window this script uses, so "
                         "the entropy definition is untouched and every value is the same; what "
                         "changes is that the projection is not computed for the P-1 prompt "
                         "positions whose logits are sliced away and discarded. On a 5,866-token "
                         "source with a 56-token generation that is 29 MB instead of 3.0 GB.")
    ap.add_argument("--device-map", default="cuda",
                    help="passed to from_pretrained. 'cuda' (default) = one GPU, unchanged. "
                         "'auto' shards the weights across the visible GPUs, which is how a model "
                         "too large for one card is run on several smaller ones. Use with "
                         "--max-memory.")
    ap.add_argument("--max-memory", default="",
                    help="force a real split, e.g. '0=20GiB,1=20GiB'. accelerate fills GPU 0 first, "
                         "so --device-map auto on its own can silently place every layer on one "
                         "card; when this is set the split is asserted after loading rather than "
                         "assumed.")
    args = ap.parse_args()
    # Sharding the weights across several cards changes where the weights live and nothing else.
    # The dtype, the sequence handling and the values computed are identical to a single-card run.
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
    print(f"{args.model} / {args.dataset} / regime {args.prompt_regime or '(canonical)'}: "
          f"{len(records)} records", flush=True)

    dtype = None if args.dtype == "auto" else _DTYPE[args.dtype]
    attn = None if args.attn == "auto" else args.attn
    model, tok = generate.load_model(cfg.model_name, attn_implementation=attn, dtype=dtype,
                                     device_map=args.device_map, max_memory=max_memory)
    model.eval()
    # Prove the shard actually happened. A silent single-card placement would run out of memory part
    # way through the dataset, after hours of work, rather than here.
    if args.device_map == "auto":
        placed = getattr(model, "hf_device_map", {})
        n_dev = len({v for v in placed.values() if isinstance(v, int)})
        print(f"device map: {n_dev} GPU(s) hold weights", flush=True)
        if max_memory is not None and n_dev < 2:
            sys.exit("--max-memory asked for a split but every layer landed on one device; "
                     "the memory ceiling was too high or only one GPU is visible.")


    ents, idxs, worst, n_cmp = [], [], 0.0, 0
    todo = records[:args.limit] if args.limit else records
    for i, r in enumerate(todo):
        p_ids = list(r["prompt_token_ids"])
        g_ids = list(r["gen_token_ids"])
        P, G = len(p_ids), len(g_ids)
        if G == 0:
            # An empty generation has no token to score. Record it as an empty vector, never as a
            # zero-entropy row: a fabricated zero would read as "maximally confident" downstream.
            ents.append(np.zeros(0, dtype=np.float32))
            idxs.append(r.get("idx", i))
            continue
        with torch.no_grad():
            ids = torch.tensor(p_ids + g_ids)[None].to(model.device)
            if args.slim_logits:
                # The last G+1 positions of a length-(P+G) sequence are exactly P-1 .. P+G-1, so the
                # first G of them are P-1 .. P+G-2: the same positions the full-logits path slices
                # out below. Nothing is approximated and no position is dropped from the window.
                logits = model(ids, logits_to_keep=G + 1).logits[0]   # (G+1, vocab)
                win = logits[:G].float()
            else:
                logits = model(ids).logits[0]                # (P+G, vocab)
                win = logits[P - 1:P + G - 1].float()        # the positions that PREDICT the G gen tokens
            logp = torch.log_softmax(win, dim=-1)            # (G, vocab)
            H = -(logp.exp() * logp).sum(-1)                 # (G,) full-distribution entropy
            chosen = logp[torch.arange(G), torch.tensor(g_ids, device=logp.device)]
        ents.append(H.cpu().numpy().astype(np.float32).copy())
        idxs.append(r.get("idx", i))

        cached = r.get("token_logprobs")
        if cached is not None and len(cached) == G:
            d = float(np.max(np.abs(chosen.cpu().numpy() - np.asarray(cached, dtype=float))))
            worst = max(worst, d)
            n_cmp += 1
        # EARLY ABORT. The window gate below is the real check, but running a whole dataset before
        # discovering a one-position misalignment wastes a scarce GPU slot, so check as soon as there
        # is enough evidence and stop immediately if the window is already wrong.
        if n_cmp == 25 and worst > args.tol:
            sys.exit(f"FATAL after 25 records: chosen-token log-probabilities disagree by "
                     f"{worst:.6f} > {args.tol}. The entropy window is misaligned. Nothing written.")
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(todo)}  worst logprob |d| so far {worst:.6f}", flush=True)

    if n_cmp == 0:
        sys.exit("FATAL: no record carried a comparable token_logprobs vector, so the window could not "
                 "be verified. Refusing to write an unverified entropy cache.")
    print(f"\nWINDOW GATE: compared {n_cmp}/{len(todo)} records; "
          f"max |teacher-forced logprob - cached logprob| = {worst:.6f} (tol {args.tol})")
    if worst > args.tol:
        sys.exit(f"FATAL: chosen-token log-probabilities disagree by {worst:.6f} > {args.tol}. The "
                 f"entropy window is not aligned with the cached generation. Nothing written.")
    print("WINDOW GATE: PASS")
    # Report the measured peak rather than the arithmetic. The point of --slim-logits is a memory
    # claim, and a claim about memory that is never measured is exactly the kind of thing this
    # project does not accept elsewhere.
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            peak = torch.cuda.max_memory_allocated(d) / 2 ** 30
            resv = torch.cuda.max_memory_reserved(d) / 2 ** 30
            print(f"PEAK MEMORY device {d}: allocated {peak:.2f} GiB, reserved {resv:.2f} GiB "
                  f"(slim_logits={args.slim_logits})")

    out_dir = Path(cfg.cache_dir) / "entropy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{key}.npz"
    np.savez_compressed(out, entropy=np.array(ents, dtype=object), idx=np.array(idxs),
                        logprob_max_abs_dev=np.float64(worst))
    lens = np.array([len(e) for e in ents])
    print(f"wrote {out}\n  {len(ents)} rows | mean entropy "
          f"{np.mean([e.mean() for e in ents if len(e)]):.4f} | token count {int(lens.sum())}")


if __name__ == "__main__":
    main()
