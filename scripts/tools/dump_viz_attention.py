"""GPU step: dump per-token ATTENTION signals for the visualiser (Phase 1, Increment 2).

WHAT AND WHY
------------
The visualiser (scripts/tools/visualise_attention.py) colours each generated token by a
per-token signal. Increment 1 shipped the log-prob (surprisal) signal, which lives in the
record already. This step produces the two ATTENTION signals we cannot read off disk,
because attention is never cached -- it must be recomputed from the token IDs on a GPU:

  (a) learned AttnPool weights -- OUR method's per-token attention. A single learned query
      (scripts/checks/attn_pool.py, the model that "wins ID, collapses OOD") scores each
      token and a softmax gives the weights. This is the object whose ID-vs-OOD behaviour
      we most want to SEE. It needs the per-token hidden states (cache/pertok, from
      01h_pertoken.py) to train on and to weight.

  (b) raw model self-attention -- the base LLM's OWN attention onto the response tokens
      (same recompute path as Lookback: eager + fp32). This is what CSL will read in
      Phase 3, so the signal gets reused. We dump two views:
        - self_lastq: attention FROM the final position ONTO each response token (where the
          model "looks" when it has finished) -- the CSL-style query.
        - self_meanq: attention onto each response token AVERAGED over all response query
          rows (how much the rest of the generation attends back to that token).

The heavy pool-training forward is done once by 01h_pertoken (a prerequisite); this script
adds only the cheap pooler-weighting (CPU) and a small self-attention forward over the
subset of examples the visualiser actually shows.

OUTPUT (a sidecar the CPU renderer reads; the renderer stays GPU-free)
    cache/viz/<key>__attn.npz  with, aligned by record position:
      record_pos_all   : test record positions that have pool_w (all test examples)
      pool_w           : object[]  (G+1,) learned-pooler weights per test example
                         (row 0 = the last-prompt token; rows 1..G = generated tokens)
      record_pos_self  : test record positions that have self-attention (the render subset)
      self_lastq_block : object[]  (G,) self-attn from the final position, chosen block
      self_lastq_last  : object[]  (G,) self-attn from the final position, LAST block
      self_meanq_block : object[]  (G,) self-attn averaged over response query rows, chosen block
      block            : int   which attention block index was used (see note on indexing)
      layer            : int   the hidden layer the pooler used (15)

ALIGNMENT / INDEXING NOTES (the bugs this guards against)
  * pool_w spans the SAPLMA window [P-1 : P+G] -> G+1 values. Row 0 is the last prompt
    token (has a weight, but no generated token / logprob). The renderer drops row 0 to
    line pool_w up with the G generated tokens and reports row-0 mass separately.
  * self_* are length G (generated tokens only), already aligned to gen_token_ids.
  * "block index": HF returns one attention tensor per transformer BLOCK (n_layers of
    them); hidden_states has n_layers+1 (an embedding row first). So hidden-layer 15 and
    attention-block 15 are not the exact same depth. We use attention BLOCK 15 and say so;
    for a visual signal the exact depth match to the pooler is not critical, and we also
    dump the last block.

USAGE (GPU; run 01h_pertoken for the same dataset first)
    python scripts/tools/dump_viz_attention.py --dataset sciq --ood ID \
        --model meta-llama/Meta-Llama-3.1-8B --n-self 120
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))  # reuse the verified AttnPool code

from luq import cache, generate  # noqa: E402
from luq.config import Config  # noqa: E402
import attn_pool as ap  # noqa: E402  (scripts/checks/attn_pool.py)


def interestingness(unc_pct, correctness):
    """Same 'surface the interesting examples' score the renderer uses: a big value means
    the method's uncertainty rank disagrees with the truth (confident-but-wrong or
    uncertain-but-right). We rank the render subset by this so the sidecar covers exactly
    the examples worth eyeballing."""
    return abs(unc_pct - (1.0 - float(correctness)))


def self_attention_signals(model, tok, record, block, max_seq=2048):
    """Return (self_lastq_block, self_lastq_last, self_meanq_block), each (G,), for one
    example. One teacher-forced forward with attentions (eager + fp32), then reductions
    over the response-token columns. Long sources are bounded exactly like Lookback."""
    prompt_ids = list(record["prompt_token_ids"])
    gen_ids = list(record["gen_token_ids"])
    n_out = len(gen_ids)
    ctx_len = len(prompt_ids)
    if ctx_len + n_out > max_seq:                     # bound very long sources (OOM guard)
        keep = max_seq - n_out
        prompt_ids = prompt_ids[-keep:]
        ctx_len = keep
    full_ids = prompt_ids + gen_ids

    _, attentions = generate.recompute_states(model, tok, full_ids, layers=[],
                                              want_attentions=True)
    A = torch.stack(list(attentions))                # (L, H, seq, seq), causal
    L = A.shape[0]
    blk = min(block, L - 1)

    # Columns for the response tokens (keys), positions ctx_len .. ctx_len+n_out-1.
    resp_cols = A[:, :, :, ctx_len:ctx_len + n_out]  # (L, H, seq, n_out)
    # (a) from the final query position (last token of the whole sequence):
    lastq = resp_cols[:, :, -1, :]                   # (L, H, n_out)
    self_lastq_block = lastq[blk].mean(0).float().cpu().numpy()   # mean over heads
    self_lastq_last = lastq[-1].mean(0).float().cpu().numpy()
    # (b) averaged over the response query rows (how much the generation attends back):
    resp_rows = A[:, :, ctx_len:ctx_len + n_out, ctx_len:ctx_len + n_out]  # (L,H,q,k)
    meanq = resp_rows.mean(2)                         # mean over query rows -> (L,H,n_out)
    self_meanq_block = meanq[blk].mean(0).float().cpu().numpy()
    return self_lastq_block, self_lastq_last, self_meanq_block


def main():
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap_.add_argument("--dataset", default="sciq")
    ap_.add_argument("--ood", default="ID")
    ap_.add_argument("--layer", type=int, default=15,
                     help="hidden layer the pooler uses (must match the pertok cache).")
    ap_.add_argument("--block", type=int, default=15,
                     help="attention block index for the self-attention signals.")
    ap_.add_argument("--label-field", default="correctness",
                     help="correctness field used to rank the render subset and train the pooler.")
    ap_.add_argument("--n-self", type=int, default=120,
                     help="how many (most interesting) test examples get the self-attention "
                          "forward. Keep modest -- this is the GPU cost.")
    ap_.add_argument("--prompt-regime", default="")
    ap_.add_argument("--max-seq", type=int, default=2048,
                     help="bound the attention forward for long sources (OOM guard).")
    args = ap_.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  key: {key}  label-field: {args.label_field}")

    # ---- 1. per-token hidden states (from 01h_pertoken) + records ------------------
    loaded = ap.load_per_token(args.model, args.dataset, args.layer, args.label_field)
    if loaded is None:
        sys.exit(
            f"ERROR: no per-token cache for {args.dataset} at layer {args.layer}.\n"
            f"  Run it first (GPU):\n"
            f"    python scripts/01h_pertoken.py --model {args.model} "
            f"--dataset {args.dataset} --ood {args.ood} --layer {args.layer}"
            + (f" --prompt-regime {args.prompt_regime}" if args.prompt_regime else ""))
    states, split, y, layer, records = loaded
    if np.isnan(y).any():
        sys.exit(f"ERROR: {args.dataset} has missing '{args.label_field}' labels; label it first.")
    tr_idx = [i for i in range(len(states)) if split[i] == "train"]
    te_idx = [i for i in range(len(states)) if split[i] == "test"]
    print(f"  loaded per-token states: train {len(tr_idx)}, test {len(te_idx)} (layer {layer})")

    # ---- 2. train the learned pooler, then read its per-token weights (CPU-cheap) ---
    # Same recipe as attn_pool.diagnose_weights: pick the temperature on a validation slice
    # of train (never test), then train the plain softmax-attention pooler at that T.
    best_T, _ = ap.select_temperature(states, y, tr_idx, device, ap.SEED, False, False)
    pooler = ap.train_attn(states, y, tr_idx, device, seed=ap.SEED, temperature=best_T)
    pooler.eval()
    print(f"  trained AttnPool (plain softmax-attention, T*={best_T})")

    pool_w = []  # (G+1,) weights per TEST example, in te_idx order
    with torch.no_grad():
        for k in te_idx:
            X, mask, pos = ap.pad_batch([states[k]], device)
            _, a = pooler(X, mask, pos)
            pool_w.append(a[0, : states[k].shape[0]].float().cpu().numpy())

    # ---- 3. pick the render subset (most interesting test examples) -----------------
    # Rank by the same score the renderer uses, off the SAPLMA scores already on disk so the
    # sidecar covers what the renderer will show. Fall back to record order if unavailable.
    try:
        s = cache.load_scores(cfg.cache_dir, key, method="saplma")
        unc = s["unc"]
        order = np.argsort(np.argsort(unc))
        pct = dict(zip(te_idx, order / max(len(unc) - 1, 1)))
        ranked = sorted(te_idx, key=lambda i: interestingness(pct[i], y[i]), reverse=True)
        print("  render subset ranked by 'interesting' (SAPLMA)")
    except FileNotFoundError:
        ranked = list(te_idx)
        print("  no SAPLMA scores; render subset = record order")
    subset = ranked[: args.n_self]

    # ---- 4. self-attention forward over the subset (the GPU cost) -------------------
    print(f"  loading LLM (fp32, eager) for self-attention over {len(subset)} examples ...")
    model, tok = generate.load_model(cfg.model_name, attn_implementation="eager",
                                     dtype=torch.float32)
    self_lastq_block, self_lastq_last, self_meanq_block = [], [], []
    for n, k in enumerate(subset):
        a1, a2, a3 = self_attention_signals(model, tok, records[k], args.block, args.max_seq)
        self_lastq_block.append(a1)
        self_lastq_last.append(a2)
        self_meanq_block.append(a3)
        if (n + 1) % 20 == 0:
            print(f"    self-attn {n + 1}/{len(subset)}", flush=True)

    # Quick sanity echo: the top self-attended token of the first example, so a wrong
    # alignment is visible in the log rather than only in the HTML.
    if subset:
        k0 = subset[0]
        toks = tok.convert_ids_to_tokens(records[k0]["gen_token_ids"])
        top = int(np.argmax(self_lastq_block[0]))
        print(f"  [sanity] ex{k0}: final-position attention peaks on token "
              f"{top} = {toks[top]!r}")

    # ---- 5. write the sidecar -------------------------------------------------------
    outdir = Path(cfg.cache_dir) / "viz"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{key}__attn.npz"
    np.savez_compressed(
        out,
        record_pos_all=np.array(te_idx),
        pool_w=np.array(pool_w, dtype=object),
        record_pos_self=np.array(subset),
        self_lastq_block=np.array(self_lastq_block, dtype=object),
        self_lastq_last=np.array(self_lastq_last, dtype=object),
        self_meanq_block=np.array(self_meanq_block, dtype=object),
        block=args.block, layer=layer, best_T=float(best_T))
    print(f"wrote {out}  (pool_w for {len(te_idx)} test, self-attn for {len(subset)})")


if __name__ == "__main__":
    main()
