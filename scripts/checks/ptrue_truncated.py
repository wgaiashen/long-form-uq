"""Step 5 (append/P(True) arm): re-extract the P(True) verdict-position state over the CLEAN answer.

The append/P(True) probe reads the hidden state at an appended "is the above response accurate?"
question. That state sits RIGHT AFTER the generation, so any trailing junk (fake Q/A, ramble, loops)
between the real answer and the verdict question feeds directly into it -- the representation most
exposed to the junk. Unlike the mean/last re-pool (a slice of cached states), moving the verdict
question earlier CHANGES the forward pass, so this needs a GPU re-run.

For each record we run the verdict forward TWICE: raw (suffix after the full generation) and
truncated (suffix after gen_token_ids[:cut_tok], the answer span). We read L15 at the verdict
position both ways, train the SAPLMA probe on each, and report PRR raw vs truncated.

xsum's long sources OOM in fp32 on RCS -> run xsum on DoC; med_quad/sciq/trivia/pubmed fit L40S.

    python scripts/checks/ptrue_truncated.py --dataset med_quad --layer 15
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import answer_span as A, cache, generate, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import ptrue  # noqa: E402
from luq.repool import char_to_tok  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"


def verdict_state(model, tok, prompt_ids, gen_ids, suffix_ids, layer):
    """L15 hidden state at the appended verdict position for [prompt + gen + suffix]."""
    full = list(prompt_ids) + list(gen_ids) + list(suffix_ids)
    states = generate.recompute_states(model, tok, full, [layer])
    return states[0][-1].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="med_quad")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    args = ap.parse_args()

    cfg = Config(model_name=MODEL, dataset=args.dataset, ood_setting="ID")
    key = cache.run_key(MODEL, args.dataset, "ID")
    records = cache.load_records(cfg.cache_dir, key)
    _DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}
    model, tok = generate.load_model(MODEL, attn_implementation="eager", dtype=_DTYPE[args.dtype])
    suffix_ids = tok(ptrue.PTRUE_SUFFIX, add_special_tokens=False).input_ids

    n = len(records)
    raw = np.zeros((n, model.config.hidden_size), np.float32)
    trunc = np.zeros_like(raw)
    cut_toks = np.zeros(n, int)
    for k, r in enumerate(records):
        gen_ids = list(r["gen_token_ids"])
        _, cut_char, rsn = A.answer_span(r["gen_text"], args.dataset, context=r.get("prompt"))
        ct = char_to_tok(tok, gen_ids, cut_char)
        cut_toks[k] = ct
        raw[k] = verdict_state(model, tok, r["prompt_token_ids"], gen_ids, suffix_ids, args.layer)
        trunc[k] = verdict_state(model, tok, r["prompt_token_ids"], gen_ids[:ct], suffix_ids, args.layer)
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{n}", flush=True)

    split = np.array([r["split"] for r in records])
    y = np.array([r.get("correctness", np.nan) for r in records], float)
    tr, te = split == "train", split == "test"
    if np.isnan(y).any():
        raise SystemExit(f"{args.dataset}: records not fully labelled (need correctness for PRR)")

    print(f"\n=== P(True) verdict L{args.layer} RAW vs TRUNCATED ({args.dataset}, n={n}) ===", flush=True)
    for tag, X in [("raw", raw), ("trunc", trunc)]:
        vals = []
        for sd in (1, 2, 3):
            clf = probe.train_probe_mlp(X[tr], y[tr], seed=sd)
            vals.append(results.prr(y[te], 1.0 - clf.p_correct(X[te])))
        print(f"  {tag:5s}: PRR={np.mean(vals):+.3f}±{np.std(vals):.3f}", flush=True)

    out = ROOT / "cache" / "ptrue_trunc"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{cache._slug(MODEL)}__{args.dataset}__ID__L{args.layer}.npz"
    np.savez_compressed(p, raw=raw, trunc=trunc, cut_tok=cut_toks, split=split, y=y, layer=args.layer)
    print(f"  wrote {p}", flush=True)


if __name__ == "__main__":
    main()
