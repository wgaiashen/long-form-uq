"""Does prepending the question/source DILUTE SAR relevance on long-form?

Hypothesis: SAR relevance = 1 - sim(q+answer, q+answer_minus_unit). On long-form, removing one answer
unit is a small fraction of (question + answer), so sim stays ~1 and every unit looks equally
(un)important -> near-uniform. Prepending nothing (answer-only) should sharpen it.

This runs the cross-encoder on a sample of MULTI-sentence records (pubmed_qa + med_quad), computing
sentence-level relevance WITH the prepend (faithful) and WITHOUT (answer-only), and compares how
concentrated the per-sentence relevance is (max/min ratio + normalised entropy) and how saturated the
raw similarities are. If no-prepend concentrates the signal, the prepend is the diluter.

CPU is fine (a few hundred cross-encoder pairs).

    python scripts/checks/sar_prepend_ablation.py --n 40
"""
import argparse
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


def question_from_prompt(prompt, max_chars=800):
    best = -1
    for marker in ("Question:", "Abstract:", "Text:", "Context:", "Story:"):
        best = max(best, prompt.rfind(marker))
    return (prompt[-max_chars:] if best == -1 else prompt[best: best + max_chars]).strip()


def sentence_R(ce, tok, q, gen_ids, gen_text):
    """Return the DISTINCT per-sentence raw relevance R = 1 - sim, and the raw sims, for one record."""
    toks = list(gen_ids)
    sid, n_sent = sar._token_sentence_ids(tok, toks, gen_text or tok.decode(toks, skip_special_tokens=True))
    if n_sent <= 1:
        return None, None
    full = (q + " " + tok.decode(toks, skip_special_tokens=True)).strip()
    pairs = []
    for j in range(n_sent):
        kept = [t for t, s in zip(toks, sid) if s != j]
        pairs.append((full, (q + " " + tok.decode(kept, skip_special_tokens=True)).strip()))
    sims = np.asarray(ce.predict(pairs, batch_size=16)).reshape(-1)
    R = np.clip(1.0 - sims, 0.0, None)
    return R, sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pubmed_qa", "med_quad"])
    ap.add_argument("--n", type=int, default=40, help="multi-sentence records to sample per dataset")
    ap.add_argument("--min-sent", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    ce = sar.load_cross_encoder(device=device)
    print(f"device {device}\n", flush=True)

    for d in args.datasets:
        cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID")
        recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, d, "ID"))
        picked, i = [], 0
        rng = np.random.RandomState(0)
        order = rng.permutation(len(recs))
        for idx in order:
            r = recs[idx]
            _, n_sent = sar._token_sentence_ids(tok, list(r["gen_token_ids"]),
                                                r.get("gen_text") or tok.decode(r["gen_token_ids"], skip_special_tokens=True))
            if n_sent >= args.min_sent:
                picked.append(r)
            if len(picked) >= args.n:
                break

        stats = {"prepend": {"ratio": [], "ent": [], "sim": []}, "noprepend": {"ratio": [], "ent": [], "sim": []}}
        for r in picked:
            q = question_from_prompt(r["prompt"])
            for mode, qq in [("prepend", q), ("noprepend", "")]:
                R, sims = sentence_R(ce, tok, qq, r["gen_token_ids"], r.get("gen_text"))
                if R is None or R.sum() <= 0:
                    continue
                p = R / R.sum()
                ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(R)))
                ratio = float(R.max() / (R[R > 0].min() if (R > 0).any() else 1e-9))
                stats[mode]["ratio"].append(ratio)
                stats[mode]["ent"].append(ent)
                stats[mode]["sim"].append(float(np.mean(sims)))

        print(f"===== {d}  ({len(picked)} multi-sentence records, >={args.min_sent} sentences) =====")
        for mode in ("prepend", "noprepend"):
            s = stats[mode]
            if not s["ratio"]:
                print(f"  {mode:10s}: no valid records"); continue
            print(f"  {mode:10s}: per-sentence R max/min ratio mean={np.mean(s['ratio']):.2f}  "
                  f"norm-entropy mean={np.mean(s['ent']):.3f} (1=uniform)  "
                  f"mean raw sim={np.mean(s['sim']):.3f} (near 1 = saturated → tiny R)", flush=True)
        # verdict
        if stats["prepend"]["ratio"] and stats["noprepend"]["ratio"]:
            rp, rn = np.mean(stats["prepend"]["ratio"]), np.mean(stats["noprepend"]["ratio"])
            ep, en = np.mean(stats["prepend"]["ent"]), np.mean(stats["noprepend"]["ent"])
            print(f"  VERDICT: no-prepend sharpens the signal by {rn/rp:.2f}x on the max/min ratio "
                  f"(entropy {ep:.3f}->{en:.3f}). {'PREPEND IS DILUTING' if en < ep - 0.005 else 'prepend not clearly diluting'}\n", flush=True)


if __name__ == "__main__":
    main()
