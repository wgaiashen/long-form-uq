"""GPU step: RE-POOL the SAPLMA features from cached token IDs, without regenerating.

Why: our original pooling excluded the last-prompt (pre-answer) hidden state, so for
short answers (trivia, mean 2.9 tokens) it pooled a single sub-word fragment. The fix
(generate.py) prepends the last-prompt position to match the SAPLMA masked-mean. This
script applies that fix to ALREADY-EXTRACTED data: it teacher-forces a forward over the
cached [prompt + gen] tokens (recompute_states), re-pools the window, and overwrites
ONLY the Tier-2 feature file. The Tier-1 records (and their gpt-5 labels) are untouched,
so NO re-labelling is needed (generations are unchanged).

the averaged window, in teacher-forced indexing (state[i] = state after token i, i.e.
the state that PREDICTS token i+1): positions P-1 .. P+G-2, i.e. the slice [P-1:P+G-1] =
the last-prompt state + the states predicting answer tokens gen[0..G-2]. (The final gen
token has no predicting-state in the generate()-based path, so it is dropped; we drop it
too for faithfulness.) Verified against compiled_features.py + full_seq_head_saplma.py.

    python scripts/01e_repool.py --model meta-llama/Meta-Llama-3.1-8B --dataset trivia_qa --ood ID

--truncate-long: also cut the cached gen tokens at the first newline (pubmed D1). NOTE: this
changes the answer text, so the gpt-5 labels NO LONGER match -> pubmed must be RE-LABELLED
after (02_label), unlike the no-truncation short-form re-pool. Off by default.

--cache-per-token: also dump the middle-layer per-token states over the pooled window (fp16)
to cache/pertok/, so future aggregation experiments (attention-pooling, Orgad filter, etc.)
are a free CPU re-pool instead of a GPU job each.
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


def _newline_cut(tok, g_ids):
    """Mirror generate.py's truncate_at_newline on cached gen token IDs."""
    for i, tid in enumerate(g_ids):
        if "\n" in tok.decode([tid]):
            return g_ids[: max(i, 1)]
    return g_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--prompt-regime", default="",
                    help="cache-namespace regime for regime-scoped sets (e.g. 'asqa_rp12', 'expertqa_rp12'); "
                         "'' = base cache/. Needed to repool the regime-namespaced XL/factuality sets.")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--dtype", default="fp32", choices=["auto", "fp32", "fp16", "bf16"],
                    help="must match the extraction dtype for identical states (keystone = fp32)")
    ap.add_argument("--attn", default="eager", choices=["auto", "eager", "sdpa"])
    ap.add_argument("--truncate-long", action="store_true",
                    help="cut cached gens at the first newline (pubmed D1) -> REQUIRES re-labelling after")
    ap.add_argument("--cache-per-token", action="store_true",
                    help="also dump middle-layer per-token states over the pooled window (fp16)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    print(f"repool: dataset={args.dataset} regime={args.prompt_regime or '(base)'} cache_dir={cfg.cache_dir}", flush=True)
    records = cache.load_records(cfg.cache_dir, key)
    if args.limit:
        records = records[: args.limit]

    dtype = None if args.dtype == "auto" else _DTYPE[args.dtype]
    attn = None if args.attn == "auto" else args.attn
    model, tok = generate.load_model(cfg.model_name, attn_implementation=attn, dtype=dtype)
    n_layers = model.config.num_hidden_layers + 1     # +1 for the embedding layer (index 0)
    layers = list(range(n_layers))
    mid = n_layers // 2

    feats = []                  # (n, n_layers, hidden), record order (03_probe aligns by order)
    pertok, pertok_idx = [], []  # optional middle-layer per-token window dumps
    short = 0
    for i, r in enumerate(records):
        p_ids = list(r["prompt_token_ids"])
        g_ids = list(r["gen_token_ids"])
        if args.truncate_long:
            g_ids = _newline_cut(tok, g_ids)
        P, G = len(p_ids), len(g_ids)
        seq = p_ids + g_ids
        states = generate.recompute_states(model, tok, seq, layers)  # list[(P+G, hidden)]
        # Window = last-prompt position P-1 PLUS all G answer-token states (positions P..P+G-1),
        # matching the masked-mean. Teacher-forcing gives every position, so unlike the
        # generate() path the last answer token (P+G-1) is available -- include it. slice [P-1:P+G].
        lo, hi = P - 1, P + G
        if G <= 1:
            short += 1
        pooled = torch.stack([states[l][lo:hi].mean(dim=0) for l in layers])  # (n_layers, hidden)
        feats.append(pooled.numpy())
        if args.cache_per_token:
            pertok.append(states[mid][lo:hi].to(torch.float16).numpy())
            pertok_idx.append(r.get("idx", i))
        if (i + 1) % 200 == 0:
            print(f"  re-pooled {i + 1}/{len(records)}", flush=True)

    feats = np.stack(feats)
    out = cache.save_features(feats, cfg.cache_dir, key, method="saplma")
    print(f"re-pooled {len(feats)} examples ({short} one-token) -> {out}  shape {feats.shape}")

    if args.cache_per_token:
        pdir = Path(cfg.cache_dir) / "pertok"
        pdir.mkdir(parents=True, exist_ok=True)
        ppath = pdir / f"{key}__L{mid}.npz"
        np.savez_compressed(ppath, states=np.array(pertok, dtype=object),
                            idx=np.array(pertok_idx), layer=mid)
        print(f"cached middle-layer (L{mid}) per-token window states -> {ppath}")


if __name__ == "__main__":
    main()
