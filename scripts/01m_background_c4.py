"""GPU step: build the C4 background population the relative-Mahalanobis baselines need.

WHY THIS EXISTS
---------------
The relative token Mahalanobis distance subtracts, from each token's distance to the training
centroid, that token's distance to a centroid estimated on a large general-purpose corpus:
    RMD(x) = MD(x) - MD_background(x)
The reference implementation draws that corpus from C4 and generates continuations from it with the
model under test, so the background describes generic activations of THIS model rather than of the
task data. Nothing in our caches supplies it, so it has to be generated once per model.

THE RECIPE, TAKEN FROM THE REFERENCE IMPLEMENTATION AND NOT FROM MEMORY
----------------------------------------------------------------------
  dataset          allenai/c4, data_files en/c4-train.00000-of-01024.json.gz, split train
  window           the first 100,000 rows are selected, then subsampled
  subsample        np.random.seed(seed); np.random.choice(100000, 2000, replace=False)
  text column      "text" (used as the prompt); the "url" column is a label and is unused here
  generation       the model's own greedy continuation, at the eval dataset's token budget

TWO PLACES THIS DELIBERATELY DIFFERS, BOTH RECORDED RATHER THAN SMOOTHED OVER
-----------------------------------------------------------------------------
1. TOKEN WINDOW. The reference takes the generated tokens only. Every per-token cache in this project
   stores the last prompt position plus the generated tokens, the same window the pooled features use.
   The background must be measured on the SAME window as the data it is compared against, or the
   distance is a comparison between two different things, so this script follows the project window.
2. BUDGET. The reference regenerates the background at each eval dataset's token budget. Greedy
   decoding makes a shorter generation an exact prefix of a longer one, so this generates ONCE at the
   maximum budget across the long-form grid and lets the per-budget background be taken as a prefix.
   That is an exact identity under greedy decoding, not an approximation, and `--verify-prefix`
   measures it rather than asserting it.

The newline position is recorded per row because the reference stops generation at the first newline.
Storing the position lets the newline-stopped background be recovered as a prefix slice of the same
run instead of costing a second pass over a contended GPU queue.

    python scripts/01m_background_c4.py --model meta-llama/Meta-Llama-3.1-8B --layer 15

Writes cache/background_c4/<slug>__L<layer>__b<budget>.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, generate  # noqa: E402

_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

C4_PATH = "allenai/c4"
C4_FILES = "en/c4-train.00000-of-01024.json.gz"
C4_WINDOW = 100_000          # the reference selects this many rows before subsampling
TEXT_COLUMN = "text"


def load_background_texts(n, seed):
    """The reference implementation's selection, reproduced step for step."""
    from datasets import load_dataset
    ds = load_dataset(C4_PATH, data_files=C4_FILES, split="train")
    if C4_WINDOW < len(ds):
        ds = ds.select(range(C4_WINDOW))
    np.random.seed(seed)
    if len(ds) < n:
        idx = np.arange(len(ds))
    else:
        idx = np.random.choice(len(ds), n, replace=False)
    return [ds[int(i)][TEXT_COLUMN] for i in idx], idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True, help="the model's middle layer")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=int, default=384,
                    help="max_new_tokens; must be >= the largest budget in the long-form grid")
    ap.add_argument("--max-prompt-tokens", type=int, default=2048,
                    help="cap on the C4 document length. Measured on this sample: median 234 tokens, "
                         "p95 1424, so 2048 truncates about 3 percent of documents.")
    ap.add_argument("--dtype", default="fp32", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--attn", default="eager", choices=["auto", "eager", "sdpa"])
    ap.add_argument("--verify-prefix", type=int, default=8,
                    help="regenerate this many rows at a shorter budget and check the prefix identity")
    ap.add_argument("--limit", type=int, default=0, help="debug only")
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

    slug = cache._slug(args.model)
    texts, c4_idx = load_background_texts(args.n, args.seed)
    print(f"{args.model} | layer {args.layer} | {len(texts)} C4 rows | budget {args.budget}", flush=True)

    model, tok = generate.load_model(args.model,
                                     attn_implementation=None if args.attn == "auto" else args.attn,
                                     dtype=None if args.dtype == "auto" else _DTYPE[args.dtype],
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

    n_layers = model.config.num_hidden_layers + 1
    if not 0 <= args.layer < n_layers:
        sys.exit(f"--layer {args.layer} out of range 0..{n_layers - 1}")

    # Cap the prompt by tokens. A handful of C4 documents run to tens of thousands of tokens, which
    # would both blow GPU memory and dominate the background with a few documents.
    capped = []
    n_trunc = 0
    for t in texts:
        ids = tok(t).input_ids
        if len(ids) > args.max_prompt_tokens:
            t = tok.decode(ids[:args.max_prompt_tokens], skip_special_tokens=True)
            n_trunc += 1
        capped.append(t)
    print(f"prompt cap {args.max_prompt_tokens}: truncated {n_trunc}/{len(capped)} documents", flush=True)

    todo = capped[:args.limit] if args.limit else capped
    states_out, gen_ids_out, plen_out, nl_out, keep_idx = [], [], [], [], []
    newline_id_cache = {}
    for i, prompt in enumerate(todo):
        rec, _ = generate.generate(model, tok, prompt, max_new_tokens=args.budget,
                                   truncate_at_newline=False)
        p_ids = list(rec["prompt_token_ids"])
        g_ids = list(rec["gen_token_ids"])
        P, G = len(p_ids), len(g_ids)
        if G == 0:
            # No generated token means no background token. Recorded as an empty row and skipped by
            # the consumer; never padded, which would inject a fabricated activation into the centroid.
            states_out.append(np.zeros((0, model.config.hidden_size), dtype=np.float32))
            gen_ids_out.append(np.zeros(0, dtype=np.int64)); plen_out.append(P)
            nl_out.append(-1); keep_idx.append(int(c4_idx[i]))
            continue
        st = generate.recompute_states(model, tok, p_ids + g_ids, [args.layer])
        # The project window: last prompt position P-1 through the final generated token.
        arr = st[0][P - 1:P + G].numpy().copy()
        if arr.shape[0] != G + 1:
            sys.exit(f"FATAL row {i}: window gave {arr.shape[0]} states for G={G} (expected G+1). "
                     "Refusing to cache a misaligned background.")
        states_out.append(arr.astype(np.float32))
        gen_ids_out.append(np.asarray(g_ids, dtype=np.int64))
        plen_out.append(P)
        # First newline among the generated tokens, so the newline-stopped variant is a prefix slice.
        nl = -1
        for t_i, tid in enumerate(g_ids):
            if tid not in newline_id_cache:
                newline_id_cache[tid] = "\n" in tok.decode([tid])
            if newline_id_cache[tid]:
                nl = t_i
                break
        nl_out.append(nl)
        keep_idx.append(int(c4_idx[i]))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(todo)} rows", flush=True)

    # PREFIX IDENTITY. Greedy decoding should make a short-budget generation an exact prefix of the
    # long-budget one. That is the assumption the per-budget reuse rests on, so it is measured.
    if args.verify_prefix:
        short = 56
        bad = 0
        for i in range(min(args.verify_prefix, len(todo))):
            rec_s, _ = generate.generate(model, tok, todo[i], max_new_tokens=short,
                                         truncate_at_newline=False)
            a = list(rec_s["gen_token_ids"])
            b = list(gen_ids_out[i][:len(a)])
            if a != b:
                bad += 1
                print(f"  PREFIX MISMATCH row {i}: short={len(a)} tokens differ from the long prefix")
        print(f"PREFIX GATE: {args.verify_prefix - bad}/{args.verify_prefix} rows exact at budget "
              f"{short} (greedy prefix identity)")
        if bad:
            sys.exit("FATAL: greedy prefix identity does not hold, so a per-budget background cannot "
                     "be taken as a prefix of this run. Nothing written.")

    out_dir = Path(__file__).resolve().parents[1] / "cache" / "background_c4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{slug}__L{args.layer}__b{args.budget}.npz"
    np.savez(out,
             states=np.array(states_out, dtype=object),
             gen_token_ids=np.array(gen_ids_out, dtype=object),
             prompt_len=np.array(plen_out), newline_pos=np.array(nl_out),
             c4_index=np.array(keep_idx), layer=np.int64(args.layer),
             budget=np.int64(args.budget), seed=np.int64(args.seed),
             n_prompt_truncated=np.int64(n_trunc),
             max_prompt_tokens=np.int64(args.max_prompt_tokens))
    tot = int(sum(s.shape[0] for s in states_out))
    print(f"\nwrote {out}\n  {len(states_out)} rows | {tot} background tokens | "
          f"{sum(1 for n in nl_out if n >= 0)} rows contain a newline")


if __name__ == "__main__":
    main()
