"""STEP 4 of the audit: is SAR's 'near-uniform relevance on long-form' an artifact of OUR context prefix?

Relevance R_i = 1 - sim(prefix+s, prefix+s\\{i}). With a LONG prefix (our <=800-char context block), removing
one unit changes the string negligibly -> sim ~1 -> R -> 0 everywhere (uniform). The authors prepend only
the short `question`. So we recompute relevance on a SAMPLE under three prefix settings and measure how
SELECTIVE R is (std of R~ across tokens, and entropy/logG; lower entropy / higher std = more selective):
  (a) context   -- our current <=800-char block
  (b) question  -- question-only (what the authors use)
  (c) none      -- no prefix
Long-form: pubmed_qa + med_quad (sentence-level, the granularity that came out uniform). Short-form
control: sciq + trivia_qa (token-level). If (b)/(c) make long-form R non-uniform, the SAR long-form
experiment is INVALID and its negative result is VOID until re-run.

    python scripts/checks/sar_prefix_diagnostic.py --n 80
"""
import argparse
import re
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
# (dataset, granularity, form)
SETS = [("pubmed_qa", "sentence", "long"), ("med_quad", "sentence", "long"),
        ("sciq", "token", "short"), ("trivia_qa", "token", "short")]
MARKERS = ("Question:", "Abstract:", "Text:", "Context:", "Story:")


def prefix_context(prompt, max_chars=800):
    best = max((prompt.rfind(m) for m in MARKERS), default=-1)
    return (prompt[-max_chars:] if best == -1 else prompt[best: best + max_chars]).strip()


def prefix_question(prompt):
    """Question-only: the last question/context marker line (authors' short prefix)."""
    best = max((prompt.rfind(m) for m in MARKERS), default=-1)
    if best == -1:
        return ""
    seg = prompt[best:]
    return seg.split("\n")[0].strip()


def selectivity(Rn):
    Rn = np.asarray(Rn, float)
    if len(Rn) < 2:
        return 0.0, 1.0
    ent = float(-(Rn * np.log(Rn + 1e-12)).sum() / np.log(len(Rn)))   # 1=uniform, ->0 peaked
    return float(Rn.std()), ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    ce = sar.load_cross_encoder("cross-encoder/stsb-roberta-large", device=device)
    print(f"device {device} | n={args.n} | (std higher = more selective; entropy 1=uniform, lower=selective)",
          flush=True)
    print(f"\n{'dataset':10s} {'form':5s} {'gran':8s} {'prefix':9s} {'pfx_chars':>9s} "
          f"{'mean_std':>9s} {'mean_ent':>9s}", flush=True)
    for d, gran, form in SETS:
        cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID")
        recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, d, "ID"))[: args.n]
        for pname, pfn in [("context", prefix_context), ("question", prefix_question), ("none", lambda p: "")]:
            stds, ents, plens = [], [], []
            for r in recs:
                q = pfn(r["prompt"])
                plens.append(len(q))
                _, Rn = sar.relevance(ce, tok, q, r["gen_token_ids"], gen_text=r.get("gen_text"),
                                      granularity=gran)
                s, e = selectivity(Rn)
                stds.append(s); ents.append(e)
            print(f"{d:10s} {form:5s} {gran:8s} {pname:9s} {np.mean(plens):9.0f} "
                  f"{np.mean(stds):9.4f} {np.mean(ents):9.3f}", flush=True)
    print("\nREAD: for the LONG sets, if entropy drops (std rises) going context->question->none, the "
          "uniform relevance was a PREFIX ARTIFACT and the SAR long-form negative is VOID.", flush=True)


if __name__ == "__main__":
    main()
