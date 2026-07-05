"""Where does each attention signal's PEAK token land? (read-only viz-sidecar analysis)

For each dataset we ask a simple question of the visualiser's per-token signals: over the render
subset (the hardest examples the sidecar covers with self-attention), which token does each signal
weight MOST, and what KIND of token is it -- real content, punctuation/whitespace, or a special
token like the end-of-text marker? A signal that keeps peaking on punctuation or the end-of-text
"attention sink" is not really pointing at the content, which is the thing we care about.

Three signals come from the sidecar written by scripts/tools/dump_viz_attention.py:
  attnpool        our LEARNED pooler's per-token weights (row 0 = last-prompt token is dropped
                  so the weights line up with the generated tokens).
  selfattn_lastq  the base model's raw self-attention FROM the final position onto each token.
  selfattn_meanq  the base model's raw self-attention averaged over the response's own query rows.

THE SINK CAVEAT (why there are two columns per self-attention signal). Llama dumps a lot of
final-position attention onto the end-of-text / special tokens (a known "attention sink"), which
is a property of the model, not a fact about the response. That sink can win the argmax and hide
where the model actually looks among the real tokens. So for the two raw self-attention signals we
report the peak BOTH ways: "raw" (the true argmax) and "no-sink" (argmax after masking out special
tokens). The learned pooler is reported raw only -- if IT peaks on a sink that is a finding about
our method, not an artifact to hide.

Read-only and CPU-only: it reads the cached sidecar + records and the tokenizer (for the token
strings). Run it after the viz chain has produced cache/viz/<key>__attn.npz for each dataset.

    python scripts/checks/viz_peak_tokens.py --datasets sciq,trivia_qa,pubmed_qa,xsum
"""
import argparse
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402


def classify(tok_str: str) -> str:
    """content / punct-space / special, judged on the DECODED single-token string."""
    if tok_str.startswith("<|") and tok_str.endswith("|>"):
        return "special"               # <|end_of_text|>, <|begin_of_text|>, ...
    s = tok_str.strip()
    if s == "":
        return "punct/space"           # the token was pure whitespace
    if all(ch in string.punctuation for ch in s):
        return "punct/space"           # the token was pure punctuation, e.g. ',' or '.'
    return "content"


def peak_category(values, tok_strs, exclude_special=False):
    """Return the category of the argmax token of `values`. When exclude_special, mask out any
    special-token position first (set it to -inf) so the peak is taken among the real tokens."""
    v = np.asarray(values, dtype=float).copy()
    if exclude_special:
        for i, ts in enumerate(tok_strs):
            if classify(ts) == "special":
                v[i] = -np.inf
    if not np.isfinite(v).any():        # every position was special (degenerate)
        return None
    return classify(tok_strs[int(np.argmax(v))])


def pct(counter: Counter) -> str:
    """'content 94%, punct/space 5%, special 1%' -- most common first."""
    total = sum(counter.values()) or 1
    return ", ".join(f"{k} {100 * n / total:.0f}%" for k, n in counter.most_common())


def analyse_dataset(dataset, model, tok):
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID")
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    sidecar = Path(cfg.cache_dir) / "viz" / f"{key}__attn.npz"
    if not sidecar.exists():
        print(f"\n[{dataset}] no sidecar at {sidecar} -- run the viz chain first; skipping.")
        return None

    sc = np.load(sidecar, allow_pickle=True)
    pos_all = list(sc["record_pos_all"])                 # test positions that have pool_w
    pos_self = list(sc["record_pos_self"])               # the render subset (self-attn covered)
    pool_by_pos = dict(zip(pos_all, sc["pool_w"]))       # position -> (G+1,) weights
    self_lastq, self_meanq = sc["self_lastq_block"], sc["self_meanq_block"]

    records = cache.load_records(cfg.cache_dir, key)

    # One counter per (signal, sink-handling) we report.
    cats = {name: Counter() for name in
            ("pool", "lastq_raw", "lastq_nosink", "meanq_raw", "meanq_nosink")}
    peakmass, row0mass = [], []
    n = 0
    for m, k in enumerate(pos_self):
        gen_ids = records[k]["gen_token_ids"]
        G = len(gen_ids)
        if G == 0:
            continue
        tok_strs = [tok.decode([tid]) for tid in gen_ids]   # per-token strings, aligned to gen

        # learned pooler: drop row 0 (last-prompt token) to align to the G generated tokens
        pw = pool_by_pos.get(k)
        if pw is not None and len(pw) >= G + 1:
            c = peak_category(pw[1:1 + G], tok_strs)
            if c:
                cats["pool"][c] += 1

        # raw self-attention peaks (already length G), with and without the sink masked out
        for sig, store in (("lastq", self_lastq), ("meanq", self_meanq)):
            v = np.asarray(store[m], dtype=float)[:G]
            for suffix, excl in (("raw", False), ("nosink", True)):
                c = peak_category(v, tok_strs, exclude_special=excl)
                if c:
                    cats[f"{sig}_{suffix}"][c] += 1
        n += 1

    # concentration: how peaked is the pooler over the gen tokens, and how much mass sits on row 0?
    for k in pos_all:
        pw = pool_by_pos[k]
        if len(pw) >= 2:
            gen_w = pw[1:]
            s = gen_w.sum()
            if s > 0:
                peakmass.append(float(gen_w.max() / s))
        if pw.sum() > 0:
            row0mass.append(float(pw[0] / pw.sum()))

    print(f"\n[{dataset}] render subset {n} examples "
          f"(block {int(sc['block'])}, layer {int(sc['layer'])}, T*={float(sc['best_T']):.3g})")
    print(f"  learned pooler (attnpool)      : {pct(cats['pool'])}")
    print(f"  self-attn lastq  raw           : {pct(cats['lastq_raw'])}")
    print(f"  self-attn lastq  no-sink       : {pct(cats['lastq_nosink'])}")
    print(f"  self-attn meanq  raw           : {pct(cats['meanq_raw'])}")
    print(f"  self-attn meanq  no-sink       : {pct(cats['meanq_nosink'])}")
    if peakmass:
        pm, r0 = np.array(peakmass), np.array(row0mass)
        print(f"  pooler peak-weight fraction    : median {np.median(pm):.3f} "
              f"(uniform over ~{int(round(1 / np.median(pm)))} tokens); "
              f"row-0 mass median {np.median(r0):.3f}")
    return {"dataset": dataset, "n": n, "cats": cats}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="sciq,trivia_qa,pubmed_qa,xsum",
                    help="comma-separated ProbeDrift keys to analyse (ID only).")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    args = ap.parse_args()

    # Load the tokenizer once (shared across datasets -- same model). CPU-only.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        analyse_dataset(ds, args.model, tok)


if __name__ == "__main__":
    main()
