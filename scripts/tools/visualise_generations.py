"""Generation-QUALITY visualiser for a long-form run (gen vs gold, side by side).

WHY THIS EXISTS (distinct from visualise_attention.py)
------------------------------------------------------
`visualise_attention.py` shows where a per-token SIGNAL lands, to debug the UQ probe. This tool
answers a different question: *is the GENERATION itself poor (repetition, hitting
the token budget, format mismatch) in a way that could be depressing our long-form scores, rather
than the UQ method being at fault?* So it puts the model generation next to the gold answer and
flags quality problems, sorted so the worst cases are easy to eyeball.

It reads ONLY the cached records (`cache/records/<key>.jsonl`) -- no GPU, no model. MSP uncertainty
comes free from the record's `token_logprobs`. The judge `correctness` label is shown per example.

WHAT IT FLAGS PER EXAMPLE
-------------------------
  capped      generated length reached the dataset's max_new_tokens budget (summary cut off / ramble)
  soft-loop   some 4-gram repeats >= 3x -- catches semantic loops the exact-repeat ratio misses
              (e.g. "I'm not sure what the X is. I'm not sure what the Y is."), the xsum failure mode
  hi-rep      >30% of 4-grams are repeats (blatant verbatim repetition)
  junk        the generation re-emits a prompt marker (Question:/Answer:/Abstract:/Text:/Summary:),
              i.e. it ran past its answer into hallucinated continuation (long-form is not truncated)
  empty       empty generation

The header carries the aggregate rates so the HTML is self-summarising. Sections order examples:
capped, then degenerate (soft-loop/hi-rep/junk), then confident-but-wrong (low MSP uncertainty yet
low correctness) and uncertain-but-right, then the rest -- the diagnostic-interesting ones first.

USAGE
-----
    python scripts/tools/visualise_generations.py --dataset pubmed_qa --ood ID \
        --model meta-llama/Meta-Llama-3.1-8B
    python scripts/tools/visualise_generations.py --dataset xsum --ood ID

Writes results/viz/gen_quality__<key>.html (override with --out). Open in the browser / VS Code.
"""
import argparse
import html
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "checks"))

from luq import answer_span as A  # noqa: E402
from luq import cache, data, msp  # noqa: E402
from luq.config import Config  # noqa: E402
from attn_pool import PROMPT_REGIME  # noqa: E402  (regime-namespaced sets: expertqa/asqa/factscore)
from xl_rungs import label_of  # noqa: E402  (the dataset's ACTUAL label field, not bare correctness)

# Prompt markers whose reappearance in the generation means it ran past its answer into junk.
JUNK_MARKERS = ("Question:", "Answer:", "Abstract:", "Text:", "Summary:", "Story:", "Context:")


def four_gram_stats(text):
    """Return (repeat_ratio, max_count): fraction of 4-grams that are repeats, and the count of the
    most-repeated 4-gram. max_count >= 3 is the 'soft-loop' signal that catches semantic loops the
    ratio misses on long generations."""
    toks = text.split()
    if len(toks) < 5:
        return 0.0, 1
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    counts = Counter(grams)
    repeat_ratio = 1 - len(counts) / len(grams)
    return repeat_ratio, max(counts.values())


def flags_for(rec, max_new_tokens):
    """Quality flags for one record. `capped` uses the token budget for this dataset."""
    g = rec["gen_text"]
    glen = len(rec["gen_token_ids"])
    ratio, maxc = four_gram_stats(g)
    f = []
    if glen == 0 or not g.strip():
        f.append("empty")
    if glen >= max_new_tokens - 1:
        f.append("capped")
    if maxc >= 3:
        f.append("soft-loop")
    if ratio > 0.30:
        f.append("hi-rep")
    if any(m in g for m in JUNK_MARKERS):
        f.append("junk")
    return f, glen, ratio, maxc


def esc(s):
    return html.escape(str(s)).replace("\n", "<br>")


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
.wrap{max-width:1200px;margin:0 auto;padding:20px}
h1{font-size:20px} h2{font-size:16px;margin-top:28px;border-bottom:1px solid #333;padding-bottom:4px}
.summary{background:#171a21;border:1px solid #2a2f3a;border-radius:8px;padding:12px 16px;font-size:13px;line-height:1.6}
.card{background:#161922;border:1px solid #262b36;border-radius:8px;padding:12px 14px;margin:12px 0}
.meta{font-size:12px;color:#9aa4b2;margin-bottom:8px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.col h4{margin:0 0 4px;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#7f8a9a}
.gen{background:#12151c;border-left:3px solid #3b82f6;padding:8px 10px;border-radius:4px;font-size:14px;line-height:1.5}
.gold{background:#12151c;border-left:3px solid #22c55e;padding:8px 10px;border-radius:4px;font-size:14px;line-height:1.5}
.q{font-size:12px;color:#c9d1d9;background:#0d1017;padding:6px 10px;border-radius:4px;margin-bottom:8px;max-height:120px;overflow:auto}
.badge{display:inline-block;font-size:11px;padding:1px 7px;border-radius:10px;margin-right:5px;font-weight:600}
.b-capped{background:#78350f;color:#fed7aa} .b-softloop{background:#7f1d1d;color:#fecaca}
.b-hirep{background:#7f1d1d;color:#fecaca} .b-junk{background:#581c87;color:#e9d5ff}
.b-empty{background:#450a0a;color:#fca5a5}
.num{font-variant-numeric:tabular-nums} .good{color:#22c55e} .bad{color:#f87171} .mid{color:#fbbf24}
.cut{color:#5a6472;text-decoration:line-through;opacity:.75}
.b-cut{background:#1e3a5f;color:#bfdbfe}
"""


def badge_html(flags):
    m = {"capped": "b-capped", "soft-loop": "b-softloop", "hi-rep": "b-hirep",
         "junk": "b-junk", "empty": "b-empty"}
    return "".join(f'<span class="badge {m[x]}">{x}</span>' for x in flags)


def corr_class(c):
    if c is None:
        return "mid"
    return "good" if c >= 0.7 else ("bad" if c < 0.3 else "mid")


def card_html(rec, flags, glen, ratio, maxc, mspu, cut, reason, label_field="correctness"):
    c = float(rec[label_field]) if rec.get(label_field) is not None else None
    corr_txt = f"{c:.2f}" if c is not None else "&mdash;"  # em-dash = not labelled yet
    q = rec.get("prompt", "")
    # trim the few-shot prompt down to the last question/context block for readability
    for mk in ("Abstract:", "Text:", "Question:"):
        if mk in q:
            q = q[q.rfind(mk):]
            break
    # render the generation with the answer-span cut: KEPT text normal, discarded tail struck out
    g = rec["gen_text"]
    kept, tail = esc(g[:cut]), esc(g[cut:])
    gen_html = kept or "<i>(empty)</i>"
    if tail:
        gen_html += f'<span class="cut">{tail}</span>'
    cut_badge = (f'<span class="badge b-cut">cut: {esc(reason)}</span>'
                 if not reason.startswith("no-cut") else "")
    return f"""<div class="card">
  <div class="meta">corr <span class="num {corr_class(c)}">{corr_txt}</span> &nbsp;|&nbsp;
    MSP-unc <span class="num">{mspu:.2f}</span> &nbsp;|&nbsp; gen_len <span class="num">{glen}</span>
    &nbsp;|&nbsp; 4gram-rep <span class="num">{ratio:.2f}</span> (max&times;{maxc}) &nbsp; {badge_html(flags)}{cut_badge}</div>
  <div class="q">{esc(q)[:600]}</div>
  <div class="cols">
    <div class="col"><h4>Model generation <span style="color:#5a6472">(struck = answer-span cut)</span></h4><div class="gen">{gen_html}</div></div>
    <div class="col"><h4>Gold answer</h4><div class="gold">{esc(rec['target'])}</div></div>
  </div>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pubmed_qa")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--per-section", type=int, default=25, help="max cards per section")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=PROMPT_REGIME.get(args.dataset, ""))
    recs = cache.load_records(cfg.cache_dir, cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting))
    # A train-only neighbour (e.g. med_quad, an ID OOD source) has no 'test' split, so the default
    # --split test would select 0 records. Fall back to all records and note it in the header.
    split_note = ""
    if args.split != "all":
        sel = [r for r in recs if r["split"] == args.split]
        if not sel:
            split_note = f" (no '{args.split}' split — showing all {len(recs)} records)"
        else:
            recs = sel
    mnt = data.MAX_NEW_TOKENS[cfg.dataset]

    # per-record quality + MSP uncertainty (higher = more uncertain, from the cached logprobs)
    rows = []
    for r in recs:
        f, glen, ratio, maxc = flags_for(r, mnt)
        mspu = msp.msp_uncertainty(r["token_logprobs"], "sum")
        lf = label_of(cfg.dataset)
        corr = float(r[lf]) if r.get(lf) is not None else None
        _, cut, reason = A.answer_span(r["gen_text"], cfg.dataset, context=r.get("prompt"))
        rows.append({"rec": r, "flags": f, "glen": glen, "ratio": ratio, "maxc": maxc,
                     "msp": mspu, "corr": corr, "cut": cut, "reason": reason})

    # Pre-label sense-check: records generated but not yet judged carry no correctness field.
    labelled = any(x["corr"] is not None for x in rows)
    n = len(rows)
    def rate(pred):
        return 100 * sum(1 for x in rows if pred(x)) / n
    label_txt = (f"judge({data.TASK_OF[cfg.dataset]})" if labelled
                 else "UNLABELLED (pre-judge sense-check)")
    # expertqa/factscore: the judge can DECLINE a row (distrust rule) -> corr None on a labelled
    # dataset. Mean over the labelled subset only, and say how many rows have no label.
    lab_vals = [x["corr"] for x in rows if x["corr"] is not None]
    n_unlab = n - len(lab_vals)
    corr_txt = (f"mean_corr={np.mean(lab_vals):.3f}"
                + (f" ({n_unlab} rows unlabelled/declined)" if n_unlab else "")
                if labelled else "mean_corr=&mdash; (not labelled yet)")
    agg = (f"n={n}{split_note} &nbsp; label={label_txt} &nbsp; max_new_tokens={mnt} &nbsp; "
           f"{corr_txt}<br>"
           f"median gen_len={int(np.median([x['glen'] for x in rows]))} &nbsp; "
           f"CAPPED={rate(lambda x:'capped' in x['flags']):.1f}% &nbsp; "
           f"soft-loop={rate(lambda x:'soft-loop' in x['flags']):.1f}% &nbsp; "
           f"hi-rep={rate(lambda x:'hi-rep' in x['flags']):.1f}% &nbsp; "
           f"junk={rate(lambda x:'junk' in x['flags']):.1f}% &nbsp; "
           f"empty={rate(lambda x:'empty' in x['flags']):.1f}%<br>"
           f"answer-span CUT={rate(lambda x:x['cut']<len(x['rec']['gen_text'])):.1f}% "
           f"(struck-through text is removed before re-pooling)"
           + (f" &nbsp; ECHO-flagged={rate(lambda x:'ECHO-FLAG' in x['reason']):.1f}%"
              if cfg.dataset == 'pubmed_qa' else ""))

    # MSP uncertainty normalised to a rank in [0,1] so 'confident' = bottom third of uncertainty
    msp_sorted = np.argsort([x["msp"] for x in rows])
    msp_rank = np.empty(n); msp_rank[msp_sorted] = np.arange(n) / max(n - 1, 1)
    for i, x in enumerate(rows):
        x["mrank"] = msp_rank[i]

    degen = lambda x: any(t in x["flags"] for t in ("soft-loop", "hi-rep", "junk", "empty"))
    sections = [
        ("Capped at the token budget (summary cut off / ran to the cap)",
         [x for x in rows if "capped" in x["flags"] and not degen(x)],
         lambda x: -x["glen"]),
        ("Degenerate (soft-loop / hi-rep / junk / empty)",
         [x for x in rows if degen(x)], lambda x: -x["maxc"]),
    ]
    if labelled:
        # correctness-driven diagnostic sections (need judge labels)
        sections += [
            ("Confident but WRONG (low MSP uncertainty, low correctness)",
             [x for x in rows if x["mrank"] < 0.33 and x["corr"] is not None
              and x["corr"] < 0.3 and not degen(x)],
             lambda x: x["mrank"]),
            ("Uncertain but RIGHT (high MSP uncertainty, high correctness)",
             [x for x in rows if x["mrank"] > 0.67 and x["corr"] is not None
              and x["corr"] > 0.7 and not degen(x)],
             lambda x: -x["mrank"]),
        ]
    else:
        # pre-label: no correctness to sort by, so show an evenly-spread sample of clean
        # generations so you can eyeball whether the content actually answers the gold.
        clean = [x for x in rows if not degen(x) and "capped" not in x["flags"]]
        step = max(len(clean) // max(args.per_section, 1), 1)
        sections.append(
            ("Sample — sense-check content vs gold (evenly spread, clean generations)",
             clean[::step], lambda x: x["rec"]["idx"]))

    parts = [f"<style>{CSS}</style><div class='wrap'>",
             f"<h1>Generation quality — {cfg.dataset} / {cfg.ood_setting}</h1>",
             f"<div class='summary'>{agg}</div>"]
    for title, items, key in sections:
        items = sorted(items, key=key)[:args.per_section]
        parts.append(f"<h2>{title} &nbsp;<span style='color:#7f8a9a;font-size:12px'>({len(items)} shown)</span></h2>")
        if not items:
            parts.append("<div class='meta'>none</div>")
        for x in items:
            parts.append(card_html(x["rec"], x["flags"], x["glen"], x["ratio"], x["maxc"],
                                   x["msp"], x["cut"], x["reason"],
                                   label_field=label_of(cfg.dataset)))
    parts.append("</div>")

    out = Path(args.out) if args.out else (cfg.results_dir / "viz" /
          f"gen_quality__{cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print(f"wrote {out}  ({n} records; {sum(len(sorted(i,key=k)[:args.per_section]) for _,i,k in sections)} cards)")


if __name__ == "__main__":
    main()
