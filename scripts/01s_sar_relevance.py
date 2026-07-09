"""Compute + cache SAR/TokenSAR per-token relevance for a dataset (Track A).

SAR relevance is expensive (one cross-encoder pass per removed unit), so we compute it ONCE per dataset
and cache the normalised per-token relevance R~, exactly like a feature. Downstream (weighted-MSP + SAR
soft mask, and the TokenSAR scalar baseline) then reads the cache for free.

Granularity (the long-form adaptation):
  --granularity token     leave-one-token-out (faithful TokenSAR; use for SHORT-form sciq/trivia)
  --granularity sentence  leave-one-sentence-out (our long-form adaptation; use for pubmed/xsum/etc.)

Writes cache/sar/<key>__<granularity>.npz:
  relevance  (n,) object -- each is the length-G normalised relevance R~ over that record's gen tokens
  tokensar   (n,) float  -- the unsupervised TokenSAR scalar  E = sum_i (-log p_i) R~_i
  granularity, cross_encoder  (str)

Resumable: skips records already done if an existing cache is passed via --resume. Runs on GPU if
available (long-form), else CPU (fine for short-form). Cross-encoder must be downloaded first
(HF_HOME); ~1.3GB `cross-encoder/stsb-roberta-large`.

    python scripts/01s_sar_relevance.py --dataset sciq --granularity token
    python scripts/01s_sar_relevance.py --dataset pubmed_qa --granularity sentence
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import sar  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
CE_NAME = "cross-encoder/stsb-roberta-large"


def question_from_prompt(prompt, max_chars=800):
    """The last context block (question / abstract / text) to prepend to the cross-encoder pair --
    focuses the similarity on the actual question, not the few-shot preamble."""
    best = -1
    for marker in ("Question:", "Abstract:", "Text:", "Context:", "Story:"):
        best = max(best, prompt.rfind(marker))
    return (prompt[-max_chars:] if best == -1 else prompt[best: best + max_chars]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--granularity", default="token", choices=["token", "sentence"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--cross-encoder", default=CE_NAME)
    ap.add_argument("--limit", type=int, default=0, help="only first N records (smoke test)")
    ap.add_argument("--prompt-regime", default="")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(args.model, args.dataset, args.ood)
    records = cache.load_records(cfg.cache_dir, key)
    if args.limit:
        records = records[: args.limit]
    n = len(records)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[{args.dataset}] {n} records | granularity={args.granularity} | device={device} | "
          f"cross-encoder={args.cross_encoder}", flush=True)
    ce = sar.load_cross_encoder(args.cross_encoder, device=device)

    rel = np.empty(n, dtype=object)
    tsar = np.zeros(n, dtype=float)
    t0 = time.time()
    for i, r in enumerate(records):
        q = question_from_prompt(r["prompt"])
        _, Rn = sar.relevance(ce, tok, q, r["gen_token_ids"], gen_text=r.get("gen_text"),
                              granularity=args.granularity)
        rel[i] = Rn.astype(np.float32)
        tsar[i] = sar.tokensar_score(Rn, r["token_logprobs"])
        if (i + 1) % 100 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{n}  ({dt/(i+1):.2f}s/rec, eta {dt/(i+1)*(n-i-1)/60:.0f}m)", flush=True)

    out_dir = ROOT / "cache" / "sar"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{cache._slug(args.model)}__{args.dataset}__{args.ood}__{args.granularity}.npz"
    np.savez_compressed(out, relevance=rel, tokensar=tsar, granularity=args.granularity,
                        cross_encoder=args.cross_encoder)
    # sanity: relevance concentration (entropy/logG; 1=uniform, ->0 peaked) and tokensar spread
    ent = []
    for Rn in rel:
        Rn = np.asarray(Rn, float)
        if len(Rn) > 1:
            ent.append(float(-(Rn * np.log(Rn + 1e-12)).sum() / np.log(len(Rn))))
    print(f"[{args.dataset}] wrote {out}\n  relevance entropy/logG mean {np.mean(ent):.3f} "
          f"(1=uniform, lower=more selective) | tokensar mean {tsar.mean():.3f} std {tsar.std():.3f}",
          flush=True)


if __name__ == "__main__":
    main()
