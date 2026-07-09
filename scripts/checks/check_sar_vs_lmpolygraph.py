"""STEP 5: numerical parity of OUR token-level SAR relevance vs lm-polygraph's TokenSAR reference.

Same standard as check_lookback_vs_authors.py: reimplement the AUTHORS' (here lm-polygraph's) exact
computation inline and compare to our `sar.token_relevance` on identical inputs. The reference is
lm_polygraph/stat_calculators/cross_encoder_similarity.py (per-token sim) + estimators/token_sar.py
(R~ = (1-sim)/sum). NUMBERS decide faithfulness (target ~1e-6) -- no prose.

NOTE: this checks OUR mechanism vs lm-polygraph's token-id leave-one-out, which is itself a deviation
from the SAR authors' LITERAL code (SAR/src/get_tokenwise_importance.py uses substring-replace
`generated_text.replace(tokenizer.decode(token), '')`, not token-id leave-one-out). "Faithful to SAR"
is therefore only true w.r.t. lm-polygraph; that deviation is documented in sar.py + the report.

    python scripts/checks/check_sar_vs_lmpolygraph.py --n 8
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import sar  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"


def lmpolygraph_reference(ce, tok, question, gen_token_ids):
    """lm-polygraph's EXACT per-token relevance R~ (cross_encoder_similarity.py + token_sar.py),
    reimplemented inline as the reference."""
    tokens = list(gen_token_ids)
    if len(tokens) <= 1:
        ts = np.array([0.5] * len(tokens))
    else:
        special_ids = list(tok.added_tokens_decoder.keys())                 # lm-polygraph's special set
        is_special = np.isin(tokens, special_ids)
        cropped = list(itertools.combinations(tokens, len(tokens) - 1))[::-1]
        raw = question + " " + tok.decode(tokens, skip_special_tokens=True)
        batches = [(raw, question + " " + tok.decode(list(t), skip_special_tokens=True)) for t in cropped]
        ts = np.asarray(ce.predict(batches, batch_size=16)).reshape(-1)
        ts[is_special] = 1
    R = 1 - ts                                                              # NO clip (lm-polygraph)
    return R / R.sum() if R.sum() != 0 else np.full(len(R), 1.0 / max(len(R), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--datasets", nargs="+", default=["sciq", "trivia_qa"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    ce = sar.load_cross_encoder("cross-encoder/stsb-roberta-large", device=device)

    worst = 0.0
    for d in args.datasets:
        cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID")
        recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, d, "ID"))[: args.n]
        for r in recs:
            q = "".join(r["prompt"][r["prompt"].rfind("Question:"):][:400]) if "Question:" in r["prompt"] else ""
            _, mine = sar.token_relevance(ce, tok, q, r["gen_token_ids"])
            ref = lmpolygraph_reference(ce, tok, q, r["gen_token_ids"])
            n = min(len(mine), len(ref))
            md = float(np.abs(np.asarray(mine)[:n] - ref[:n]).max()) if n else 0.0
            worst = max(worst, md)
        print(f"  {d}: {len(recs)} records checked", flush=True)
    print(f"\nSAR token-relevance vs lm-polygraph TokenSAR: WORST |delta| = {worst:.2e}", flush=True)
    ok = worst < 1e-6
    print(f"PARITY {'PASS' if ok else 'FAIL'} (target < 1e-6)", flush=True)
    if not ok:
        print("  -> our sar.token_relevance deviates from lm-polygraph; likely the special-token set "
              "(all_special_ids vs added_tokens_decoder) or the clip(R,0). Fix sar.py to match, re-run.", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
