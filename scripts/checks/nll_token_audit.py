#!/usr/bin/env python
"""A5 — qualitative top-NLL token audit: a FIXED, DETERMINISTIC sample, 12 responses per dataset.

The researcher-facing question: when max-NLL works, is the extreme token actually answer-bearing /
fact-bearing? When it fails, is it a harmless rare name, formatting token, or lexical choice?

SAMPLING IS DETERMINISTIC AND PRE-COMMITTED: the dataset's scored TEST rows are split into quality quartiles, and within each quartile
3 examples are chosen by ascending sha1("<dataset>:<idx>:<position>") — no hand-picking. Any
example later quoted in the report must come from this audit and be labelled illustrative.

Rendering reuses viz_common (the project's one token-colouring engine). Signals per token:
  nll        the token's negative log-probability (the audit's primary signal);
  top_flags  1.0 / 0.66 / 0.33 for membership of the top-1 / top-5 / top-10% NLL sets, else 0.
Header scores per example: perplexity, msp_min, Lehmer beta=1, beta=2 (from the same records).
Token set: ALL generated tokens, raw logprobs (canonical floor policy).

    python scripts/checks/nll_token_audit.py                      # Llama, all 8, 12 each
    qsub -v LUQ_CMD="scripts/checks/nll_token_audit.py" pbs/audit_cpu.pbs
Output: results/viz/nll_token_audit__<slug>__<dataset>.html (one page per dataset).
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from luq import cache, msp, results                       # noqa: E402
from sharpening_family import score_lehmer                # noqa: E402
from aggregation_regime_rows import load_test_records     # noqa: E402  (same gated loader as A2)
import viz_common as V                                    # noqa: E402

MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
PER_QUARTILE = 3                                          # 4 quartiles x 3 = 12 per dataset


def pick(order_key, idxs, k):
    """k indices chosen deterministically by ascending sha1 hash — no hand-picking possible."""
    return sorted(idxs, key=order_key)[:k]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    args = ap.parse_args()
    slug = cache._slug(args.model)
    outdir = ROOT / "results" / "viz"
    outdir.mkdir(parents=True, exist_ok=True)
    tok = V.load_tokenizer(args.model)

    for d in LONG:
        recs, y, lf = load_test_records(args.model, d)
        nlls = [-np.asarray(r["token_logprobs"], dtype=float) for r in recs]
        scores = {
            "perplexity": np.array([msp.msp_uncertainty(-a, "perplexity") for a in nlls]),
            "msp_min": np.array([msp.msp_uncertainty(-a, "min") for a in nlls]),
            "lehmer_b1": np.array([score_lehmer(a, 1.0) for a in nlls]),
            "lehmer_b2": np.array([score_lehmer(a, 2.0) for a in nlls]),
        }
        methods = {m: {"at": dict(enumerate(v)),
                       "ranks": dict(enumerate(v.argsort().argsort() / max(len(v) - 1, 1)))}
                   for m, v in scores.items()}

        # quality quartiles over the test rows (rank-based, ties broken by position: deterministic)
        order = np.lexsort((np.arange(len(y)), y))
        qt = np.empty(len(y), dtype=int)
        qt[order] = (4 * np.arange(len(y)) // len(y))     # 0..3, low quality first
        chosen = []
        for q in range(4):
            idxs = [i for i in range(len(y)) if qt[i] == q]
            keyf = lambda i: hashlib.sha1(f"{d}:{recs[i]['idx']}:{i}".encode()).hexdigest()
            chosen += [(q, i) for i in pick(keyf, idxs, PER_QUARTILE)]

        blocks = []
        for q, i in chosen:
            r, a = recs[i], nlls[i]
            pieces = V.token_pieces(tok, r["gen_token_ids"])
            T = len(a)
            srt = np.argsort(-a)
            flags = np.zeros(T)
            flags[srt[:max(1, int(np.ceil(0.10 * T)))]] = 0.33
            flags[srt[:min(5, T)]] = 0.66
            flags[srt[0]] = 1.0
            rng = np.ptp(a) if np.ptp(a) > 0 else 1.0
            signals = {"nll": (a, (a - a.min()) / rng), "top_flags": (flags, flags)}
            meta = [f"nll {v:.3f} | rank {int(rk) + 1}/{T}"
                    for v, rk in zip(a, (-a).argsort().argsort())]
            blocks.append(f'<div class="quartile-tag">quality quartile Q{q + 1} '
                          f'(1 = worst)</div>'
                          + V.render_example(r, i, methods, signals, lf, pieces, token_meta=meta))

        out = outdir / f"nll_token_audit__{slug}__{d}.html"
        html_page = V.render_html(
            f"{slug}__{d}__ID", recs, methods, lf, ["nll", "top_flags"], "".join(blocks),
            subtitle=(f"A5 top-NLL audit — {d}: 12 deterministic test examples "
                      f"(3 per quality quartile, sha1-selected; no hand-picking). "
                      f"Token set: all generated tokens, raw logprobs."))
        out.write_text(html_page)
        print(f"[{d}] wrote {out}  (12 examples, quartiles balanced)")


if __name__ == "__main__":
    main()
