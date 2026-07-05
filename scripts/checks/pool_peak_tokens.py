"""Where does the learned attention pooler put its weight, by token type, per dataset.

Reproducible replacement for the earlier ad-hoc scratchpad analysis (the one whose
"punct/space 92%" number could not be re-derived). Everything here reads the COMMITTED viz
sidecars (cache/viz/<key>__attn.npz) written by scripts/tools/dump_viz_attention.py, so the
table is regenerable on demand.

For each test example it:
  1. takes the pooler's per-token softmax weights (length G+1: row 0 is the last PROMPT token,
     rows 1..G are the generated tokens),
  2. drops row 0 and finds the PEAK (argmax) generated token,
  3. decodes it and classifies it as content / punct_space / special / subword,
and then reports per dataset:
  - the peak token-TYPE distribution (what kind of token the pooler leans on),
  - the peak POSITION in the generation (0 = first gen token, 1 = last),
  - the most common peak token ids (decoded),
  - the weight on row 0 (last-prompt token) -- a boundary-artifact / bug check,
  - the pooler's concentration (mean weight-entropy / uniform-entropy; 1.0 = flat mean-pool).

Token classification is a documented heuristic (see classify()), not ground truth -- it is only
meant to separate "answer-bearing word" from "punctuation/whitespace" from "special/EOS".

Run (CPU only; the tokenizer import is slow on the login node but works -- or submit as a job):
    python scripts/checks/pool_peak_tokens.py --datasets sciq,trivia_qa,pubmed_qa,xsum
    python scripts/checks/pool_peak_tokens.py --datasets pubmed_qa --csv results/pool_peaks.csv
"""
import argparse
import os
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Keep the big cache off the over-quota home dir (matches the rest of the pipeline).
os.environ.setdefault("HF_HOME", "/vol/gpudata/gs925-msc_project/hf_cache")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luq import cache  # noqa: E402

# Punctuation/quote/space characters. A token that is ONLY these is "structural", not content.
PUNCT = set(string.punctuation) | {"…", "–", "—", "’", "“", "”"}


def classify(token_str: str, token_id: int, special_ids: set) -> str:
    """Bucket a decoded token. Heuristic, documented:
    - special      : a tokenizer special id (BOS/EOS/eot/pad).
    - punct_space  : decodes to whitespace only, or to punctuation only (',', '.', '):' ...).
    - content      : an alphabetic piece of >=3 chars (an answer-bearing word like ' energy').
    - subword      : everything else (digits, 1-2 char fragments, mixed pieces, ' is', ' of').
    """
    if token_id in special_ids:
        return "special"
    s = token_str.strip()
    if s == "":
        return "punct_space"
    if all(ch in PUNCT for ch in s):
        return "punct_space"
    if s.isalpha() and len(s) >= 3:
        return "content"
    return "subword"


def analyse(dataset: str, model: str, tok, cache_dir: Path):
    key = cache.run_key(model, dataset, "ID")
    sidecar = cache_dir / "viz" / f"{key}__attn.npz"
    if not sidecar.exists():
        print(f"\n#### {dataset}: no sidecar ({sidecar.name}) -- run dump_viz_attention first. skipped.")
        return None
    records = cache.load_records(cache_dir, key)
    z = np.load(sidecar, allow_pickle=True)
    pool = dict(zip(z["record_pos_all"].tolist(), z["pool_w"]))
    special_ids = set(getattr(tok, "all_special_ids", []))

    type_counts = Counter()
    peak_ids = Counter()
    frac_pos, row0_w, ent_ratio = [], [], []
    for i, w in pool.items():
        w = np.asarray(w, float)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        gids = records[i]["gen_token_ids"]
        G = len(gids)
        if len(w) != G + 1:          # alignment guard: G+1 (row0 = last-prompt token)
            continue
        row0_w.append(float(w[0]))
        gw = w[1:]                    # weights over the G generated tokens
        j = int(np.argmax(gw))        # the peak generated token
        tid = int(gids[j])
        type_counts[classify(tok.decode([tid]), tid, special_ids)] += 1
        peak_ids[tid] += 1
        frac_pos.append(j / max(G - 1, 1))
        ent_ratio.append(float(-(w * np.log(w + 1e-12)).sum() / np.log(len(w))))

    n = sum(type_counts.values())
    dist = {k: type_counts[k] / n for k in ("content", "subword", "punct_space", "special")}
    top = [(tid, c, repr(tok.decode([tid]))) for tid, c in peak_ids.most_common(6)]
    print(f"\n#### {dataset}  (n={n}) ####")
    print("  peak token-type:  " + "  ".join(f"{k} {dist[k]:.2f}" for k in dist))
    print(f"  peak position:    mean {np.mean(frac_pos):.2f}  (0=first gen tok, 1=last)")
    print(f"  concentration:    entropy/uniform {np.mean(ent_ratio):.2f}  (1.0=flat mean-pool)")
    print(f"  weight on row0:   mean {np.mean(row0_w):.3f}  (last-prompt token; ~0 => no boundary artifact)")
    print(f"  top peak ids:     {top}")
    return {"dataset": dataset, "n": n, **{f"peak_{k}": dist[k] for k in dist},
            "peak_pos_mean": float(np.mean(frac_pos)), "row0_w_mean": float(np.mean(row0_w)),
            "entropy_ratio": float(np.mean(ent_ratio))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sciq,trivia_qa,pubmed_qa,xsum")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--cache-dir", default=str(ROOT / "cache"))
    ap.add_argument("--csv", default="", help="optional path to also write the per-dataset table as CSV")
    args = ap.parse_args()

    from transformers import AutoTokenizer  # heavy import; do it after arg parsing
    tok = AutoTokenizer.from_pretrained(args.model)

    cache_dir = Path(args.cache_dir)
    rows = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        r = analyse(ds, args.model, tok, cache_dir)
        if r:
            rows.append(r)

    if args.csv and rows:
        import csv
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
