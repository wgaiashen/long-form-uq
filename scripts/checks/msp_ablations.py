"""P1.2 — MSP ablation battery: which tokens carry the plain-MSP signal?

the email asks us to understand *what drives MSP* by scoring it over token SUBSETS, so we know what a
good learned weighting (weighted-MSP, P1.1) should look like. Every ablation is also a cheap unsupervised
baseline. All from the cached per-token logprobs — NO GPU, NO training.

Ablations (each scored as MSP-sum AND perplexity = length-normalised mean-NLL, exactly msp.py's two
unsupervised aggregators, restricted to the kept tokens):
  full            all generated tokens (== plain MSP; the baseline every row is compared to)
  minus_stop      drop stop-word tokens (function words)
  first_of_word   keep only the FIRST sub-word token of each word
  last_of_word    keep only the LAST sub-word token of each word
  first_sentence  keep only the tokens of the first sentence (the long-form "is the signal up front?" test)
  content_word    keep only content-word tokens (== not stop-word, first sub-word of the word) — a stricter
                  "just the meaning-bearing tokens" subset

Unsupervised => rung-invariant, so ID (train/test same set) is the whole story; we PRR each subset on the
test split against the judge label. A per-dataset table is the deliverable.

WORD/SENTENCE GROUPING (the risky bit — audited): Llama-3's byte-level BPE marks a word start with a
leading space glyph in the HF piece ('Ġ' / '▁'). We group tokens into words on that marker, classify each
word (stop vs content), and split sentences on '.','!','?'. --audit dumps a few decoded examples with the
kept mask per ablation so the grouping can be eyeballed. CAVEAT flagged in the overnight log.

    python scripts/checks/msp_ablations.py --datasets sciq,trivia_qa,pubmed_qa,xsum,med_quad,samsum
    python scripts/checks/msp_ablations.py --datasets sciq --audit
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache, results  # noqa: E402
from luq.config import Config  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAB = "correctness"

# A standard English stop-word list (hardcoded — nltk's corpus isn't reliably available and compute nodes
# have no network). Function words whose token probability is unlikely to carry a correctness signal.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "of",
    "is", "are", "was", "were", "be", "been", "being", "am", "has", "have", "had", "having", "do",
    "does", "did", "doing", "would", "should", "could", "ought", "will", "shall", "can", "may", "might",
    "must", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "its", "our", "their", "this", "that", "these", "those", "who", "whom", "which", "what",
    "as", "so", "than", "too", "very", "just", "not", "no", "nor", "only", "own", "same", "such", "s",
    "t", "d", "ll", "m", "re", "ve", "there", "here", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "how", "why", "where",
}
SPACE_MARKERS = ("Ġ", "▁")   # HF byte-level / sentencepiece space glyphs
SENT_END = {".", "!", "?"}


def word_groups(pieces):
    """Group token positions into words. A word starts at a token whose HF piece begins with a space
    glyph (or the first token). Returns list of (positions, word_text) — word_text is the lowercased,
    glyph-stripped concatenation of its pieces."""
    groups = []
    cur_pos, cur_txt = [], ""
    for i, p in enumerate(pieces):
        starts_word = i == 0 or p.startswith(SPACE_MARKERS)
        clean = p
        for mk in SPACE_MARKERS:
            clean = clean.replace(mk, "")
        if starts_word and cur_pos:
            groups.append((cur_pos, cur_txt))
            cur_pos, cur_txt = [], ""
        cur_pos.append(i)
        cur_txt += clean
    if cur_pos:
        groups.append((cur_pos, cur_txt))
    return [(pos, txt.lower()) for pos, txt in groups]


def build_masks(pieces):
    """Return {ablation_name: boolean keep-mask over the tokens}. full = all True."""
    n = len(pieces)
    groups = word_groups(pieces)
    full = np.ones(n, dtype=bool)
    minus_stop = np.ones(n, dtype=bool)
    first_of_word = np.zeros(n, dtype=bool)
    last_of_word = np.zeros(n, dtype=bool)
    content_word = np.zeros(n, dtype=bool)
    # first sentence: find the group that ends the first sentence, keep up to and incl its tokens
    first_sentence = np.zeros(n, dtype=bool)
    sentence_closed = False
    for pos, txt in groups:
        core = txt.strip("".join(SENT_END) + ",;:\"'()")
        is_stop = core in STOPWORDS
        first_of_word[pos[0]] = True
        last_of_word[pos[-1]] = True
        if is_stop:
            for p in pos:
                minus_stop[p] = False
        else:
            content_word[pos[0]] = True
        if not sentence_closed:
            for p in pos:
                first_sentence[p] = True
            if any(txt.rstrip().endswith(e) for e in SENT_END):
                sentence_closed = True
    return {"full": full, "minus_stop": minus_stop, "first_of_word": first_of_word,
            "last_of_word": last_of_word, "content_word": content_word, "first_sentence": first_sentence}


def subset_scores(nll, mask):
    """(sum, perplexity) of NLL over the kept tokens. Empty subset -> fall back to full (a degenerate
    generation with no kept token should not become NaN; flagged by n_kept)."""
    kept = nll[mask]
    if kept.size == 0:
        kept = nll
    return float(kept.sum()), float(kept.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sciq,trivia_qa,pubmed_qa,xsum,med_quad,samsum")
    ap.add_argument("--audit", action="store_true", help="dump a few decoded examples + kept masks")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    ABLATIONS = ["full", "minus_stop", "first_of_word", "last_of_word", "content_word", "first_sentence"]
    out_rows = []
    for ds in args.datasets.split(","):
        cfg = Config(model_name=MODEL, dataset=ds, ood_setting="ID")
        try:
            recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, ds, "ID"))
        except Exception as e:
            print(f"[{ds}] no records ({e}) -> skip", flush=True)
            continue
        test = [r for r in recs if r.get("split") == "test" and r.get(LAB) is not None
                and np.isfinite(r.get(LAB, np.nan))]
        if len(test) < 20:
            print(f"[{ds}] only {len(test)} labelled test rows -> skip", flush=True)
            continue
        y = np.array([float(r[LAB]) for r in test])

        # per-record: NLL + masks
        scores = {a: {"sum": [], "ppl": []} for a in ABLATIONS}
        kept_frac = {a: [] for a in ABLATIONS}
        audited = 0
        for r in test:
            ids = list(r["gen_token_ids"])
            nll = np.array([-lp for lp in r["token_logprobs"]], dtype=float)
            n = min(len(ids), len(nll))            # guard any off-by-one; align to the shorter
            ids, nll = ids[:n], nll[:n]
            pieces = tok.convert_ids_to_tokens(ids)
            masks = build_masks(pieces)
            for a in ABLATIONS:
                s, p = subset_scores(nll, masks[a])
                scores[a]["sum"].append(s)
                scores[a]["ppl"].append(p)
                kept_frac[a].append(float(masks[a].mean()))
            if args.audit and audited < 3:
                dec = tok.convert_tokens_to_string(pieces)
                print(f"\n[AUDIT {ds}] y={float(r[LAB]):.2f}  gen={dec[:160]!r}", flush=True)
                for a in ABLATIONS:
                    kepttok = [pieces[i] for i in range(n) if masks[a][i]]
                    print(f"    {a:14s} keep {masks[a].mean():.0%}: {kepttok[:12]}", flush=True)
                audited += 1

        print(f"\n==== {ds} (n_test={len(test)}) ====", flush=True)
        for a in ABLATIONS:
            for agg in ("sum", "ppl"):
                unc = np.array(scores[a][agg])
                prr = results.prr(y, unc)
                kf = float(np.mean(kept_frac[a]))
                tag = "" if a != "full" else "  (= plain MSP)"
                print(f"    {a:14s} {agg:3s}  PRR {prr:+.3f}  (kept {kf:.0%}){tag}", flush=True)
                out_rows.append({"dataset": ds, "ablation": a, "aggregate": agg,
                                 "prr": round(prr, 4), "kept_frac": round(kf, 3), "n_test": len(test)})

    out = Path(args.out) if args.out else (ROOT / "results" / f"msp_ablations__{cache._slug(MODEL)}.csv")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["dataset", "ablation", "aggregate", "prr", "kept_frac", "n_test"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
