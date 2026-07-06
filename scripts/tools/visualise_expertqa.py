"""ExpertQA generation spot-check visualiser.

Emits ONE self-contained HTML file (no server, no external deps) that shows, per example:
  - the QUESTION (from ExpertQA),
  - OUR Llama-3.1-8B generation, with length + a CAPPED badge + a degeneracy flag,
  - the GOLD revised answer (what the judge scores against),
  - collapsibly: the ORIGINAL ExpertQA answer (pre-revision) and the gold per-claim evidence
    with the expert correctness Likert code.
It's for eyeballing whether the generations make sense, are far off, or are the wrong format.
It needs NO judge labels (works before labelling); if a `correctness` field is present it is shown.

The generation records are joined to the raw ExpertQA source BY idx. The record idx indexes into
the SAME factual-core-filtered pool that `expertqa.load_records()` builds, so we rebuild that pool
here in file order (reusing the module's own filter constants so it can't drift) and assert the
counts match before trusting the join.

Run (reads only the cached records + the ExpertQA jsonl; no GPU, no judge):
    python scripts/tools/visualise_expertqa.py --ood ID --prompt-regime expertqa_reppen
    # subset for a quick look:
    python scripts/tools/visualise_expertqa.py --field "Healthcare / Medicine" --limit 100
Opens at results/expertqa/generations_view.html (override with --out).
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import expertqa                 # noqa: E402  (filter constants + prompt live here)
from luq.config import Config            # noqa: E402
from luq import cache                    # noqa: E402

CAP = expertqa.MAX_NEW_TOKENS            # 384 — a generation this long ran the whole budget


def load_raw_aligned():
    """Rebuild the factual-core pool in file order, KEEPING the raw answer object, so that
    raw[idx] lines up with the generation record whose 'idx' == idx. Mirrors
    expertqa.load_records() exactly (same drop-empty-gold + FACTUAL_CORE filter, same order)."""
    out = []
    for line in open(expertqa.DEFAULT_JSONL, encoding="utf-8"):
        rec = json.loads(line)
        ans = next(iter(rec["answers"].values()))
        gold = (ans.get("revised_answer_string") or "").strip()
        if not gold:
            continue
        types = {t.strip() for t in rec["metadata"]["question_type"].split("|") if t.strip()}
        if not (types & expertqa.FACTUAL_CORE):
            continue
        field = rec["metadata"]["field"]
        out.append({
            "question": rec["question"],
            "gold": gold,
            "original": (ans.get("answer_string") or "").strip(),   # pre-revision model answer
            "field": field,
            "specific_field": rec["metadata"].get("specific_field", ""),
            "cluster": expertqa.CLUSTER.get(field, "other"),
            "question_types": sorted(types),
            "claims": [
                {"text": (c.get("claim_string") or "").strip(),
                 "correctness": c.get("correctness"),
                 "support": c.get("support")}
                for c in (ans.get("claims") or [])
            ],
        })
    return out


def distinct_n(tokens, n):
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def max_sentence_repeat(text):
    sents = [re.sub(r"\s+", " ", s).strip().lower() for s in re.split(r"[.!?\n]+", text)]
    sents = [s for s in sents if len(s) > 3]
    if not sents:
        return 0
    counts = {}
    for s in sents:
        counts[s] = counts.get(s, 0) + 1
    return max(counts.values())


def build_rows(recs, raw, field=None, limit=None):
    """One dict per example for the HTML, joining generation <-> raw ExpertQA by idx."""
    rows = []
    for r in recs:
        src = raw[r["idx"]]
        if field and src["field"] != field:
            continue
        toks = r["gen_text"].split()
        n_gen = len(r["gen_token_ids"])
        d3 = distinct_n(toks, 3)
        srep = max_sentence_repeat(r["gen_text"])
        degenerate = (d3 < 0.5) or (srep >= 3)
        rows.append({
            "idx": r["idx"],
            "field": src["field"],
            "specific_field": src["specific_field"],
            "cluster": src["cluster"],
            "types": src["question_types"],
            "question": src["question"],
            "generation": r["gen_text"],
            "gold": src["gold"],
            "original": src["original"],
            "claims": src["claims"],
            "n_tokens": n_gen,
            "capped": bool(n_gen >= CAP),
            "degenerate": bool(degenerate),
            "distinct3": round(d3, 2),
            "max_sent_repeat": int(srep),
            # judge label if labelling has already run (else None)
            "correctness": r.get("correctness"),
            "correctness_model": r.get("correctness_model"),
        })
        if limit and len(rows) >= limit:
            break
    return rows


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>ExpertQA generations — spot check</title>
<style>
 body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 18px;z-index:10;
   box-shadow:0 1px 4px rgba(0,0,0,.06)}
 header h1{margin:0 0 6px;font-size:16px}
 .stats{color:#555;font-size:13px;margin-bottom:8px}
 .controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
 .controls input,.controls select{padding:5px 7px;border:1px solid #ccc;border-radius:5px;font-size:13px}
 .controls label{font-size:13px;color:#333}
 #list{padding:16px;max-width:1100px;margin:0 auto}
 .card{background:#fff;border-radius:8px;margin-bottom:14px;padding:14px 16px;border-left:5px solid #4caf50;
   box-shadow:0 1px 3px rgba(0,0,0,.08)}
 .card.capped{border-left-color:#ff9800}
 .card.degenerate{border-left-color:#e53935}
 .meta{display:flex;gap:8px;flex-wrap:wrap;align-items:center;font-size:12px;color:#555;margin-bottom:8px}
 .badge{padding:2px 7px;border-radius:10px;background:#eceff1;color:#37474f;font-size:11px;font-weight:600}
 .badge.warn{background:#fff3e0;color:#e65100}
 .badge.bad{background:#ffebee;color:#b71c1c}
 .badge.ok{background:#e8f5e9;color:#1b5e20}
 .q{font-weight:600;margin:6px 0 10px}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}
 .col h4{margin:0 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#666}
 .gen{background:#f1f8ff;border:1px solid #d6e4f0;border-radius:6px;padding:8px 10px;white-space:pre-wrap}
 .gold{background:#f6fff2;border:1px solid #d9ecd0;border-radius:6px;padding:8px 10px;white-space:pre-wrap}
 details{margin-top:10px;font-size:13px}
 details summary{cursor:pointer;color:#1565c0;font-weight:600}
 .orig{white-space:pre-wrap;background:#fafafa;border:1px solid #eee;border-radius:6px;padding:8px 10px;margin-top:6px}
 .claim{border-bottom:1px solid #eee;padding:6px 0}
 .claim .cc{font-weight:600}
 @media(max-width:720px){.cols{grid-template-columns:1fr}}
</style></head><body>
<header>
 <h1>ExpertQA generations — spot check</h1>
 <div class="stats" id="stats"></div>
 <div class="controls">
  <input id="search" placeholder="search question / generation / gold…" size="34">
  <select id="fieldsel"></select>
  <label><input type="checkbox" id="cappedonly"> capped only</label>
  <label><input type="checkbox" id="degenonly"> degenerate only</label>
  <select id="sortsel">
   <option value="idx">sort: idx</option>
   <option value="lenA">sort: length ↑</option>
   <option value="lenD">sort: length ↓</option>
  </select>
  <span id="count"></span>
 </div>
</header>
<div id="list"></div>
<script>
const DATA = __DATA__;
const CAP = __CAP__;
const list = document.getElementById('list');
const el = (t,c,txt)=>{const e=document.createElement(t); if(c)e.className=c; if(txt!=null)e.textContent=txt; return e;};

// field dropdown
const fields = [...new Set(DATA.map(d=>d.field))].sort();
const fsel = document.getElementById('fieldsel');
fsel.appendChild(new Option('all fields',''));
fields.forEach(f=>fsel.appendChild(new Option(f,f)));

function badge(txt,cls){return `<span class="badge ${cls||''}">${txt}</span>`;}

function card(d){
 const c = el('div','card'+(d.degenerate?' degenerate':(d.capped?' capped':'')));
 const meta = el('div','meta');
 let m = badge('#'+d.idx)+badge(d.field)+(d.specific_field?badge(d.specific_field):'')+badge(d.cluster);
 d.types.forEach(t=>m+=badge(t));
 m += badge(d.n_tokens+' tok', d.capped?'warn':'ok');
 if(d.capped) m += badge('⚠ CAPPED (hit '+CAP+')','warn');
 if(d.degenerate) m += badge('⚠ degenerate (d3='+d.distinct3+', sent×'+d.max_sent_repeat+')','bad');
 if(d.correctness!=null) m += badge('judge '+d.correctness+' ('+(d.correctness_model||'?')+')');
 meta.innerHTML = m;
 c.appendChild(meta);
 c.appendChild(el('div','q','Q: '+d.question));
 const cols = el('div','cols');
 const g1 = el('div','col'); g1.appendChild(el('h4',null,'Our Llama generation'));
 g1.appendChild(el('div','gen',d.generation));
 const g2 = el('div','col'); g2.appendChild(el('h4',null,'Gold (revised answer)'));
 g2.appendChild(el('div','gold',d.gold));
 cols.appendChild(g1); cols.appendChild(g2); c.appendChild(cols);
 if(d.original){
  const det = el('details'); det.appendChild(el('summary',null,'Original ExpertQA answer (pre-revision)'));
  det.appendChild(el('div','orig',d.original)); c.appendChild(det);
 }
 if(d.claims && d.claims.length){
  const det = el('details'); det.appendChild(el('summary',null,'Gold claims + expert correctness ('+d.claims.length+')'));
  d.claims.forEach(cl=>{
   const cd = el('div','claim');
   cd.appendChild(el('span','cc',(cl.correctness||'—')+': '));
   cd.appendChild(document.createTextNode(cl.text));
   det.appendChild(cd);
  });
  c.appendChild(det);
 }
 return c;
}

function render(){
 const q = document.getElementById('search').value.toLowerCase();
 const f = fsel.value;
 const cap = document.getElementById('cappedonly').checked;
 const deg = document.getElementById('degenonly').checked;
 const sort = document.getElementById('sortsel').value;
 let rows = DATA.filter(d=>
   (!f||d.field===f) && (!cap||d.capped) && (!deg||d.degenerate) &&
   (!q || (d.question+' '+d.generation+' '+d.gold).toLowerCase().includes(q)));
 if(sort==='lenA') rows.sort((a,b)=>a.n_tokens-b.n_tokens);
 else if(sort==='lenD') rows.sort((a,b)=>b.n_tokens-a.n_tokens);
 else rows.sort((a,b)=>a.idx-b.idx);
 list.innerHTML='';
 rows.slice(0,400).forEach(d=>list.appendChild(card(d)));
 document.getElementById('count').textContent = rows.length+' shown'+(rows.length>400?' (first 400)':'');
}
['search','fieldsel','cappedonly','degenonly','sortsel'].forEach(id=>{
 const e=document.getElementById(id); e.addEventListener(e.tagName==='INPUT'&&e.type==='text'?'input':'change',render);
});
document.getElementById('stats').textContent =
  DATA.length+' generations | capped '+DATA.filter(d=>d.capped).length+
  ' | degenerate '+DATA.filter(d=>d.degenerate).length+
  ' | median length '+DATA.map(d=>d.n_tokens).sort((a,b)=>a-b)[Math.floor(DATA.length/2)]+' tok';
render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="expertqa")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--prompt-regime", default="expertqa_reppen")
    ap.add_argument("--field", default=None, help="restrict to one ExpertQA field")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of examples rendered")
    ap.add_argument("--out", default="results/expertqa/generations_view.html")
    args = ap.parse_args()

    cfg = Config(dataset=args.dataset, ood_setting=args.ood, model_name=args.model,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(args.model, args.dataset, args.ood)
    recs = cache.load_records(cfg.cache_dir, key)
    raw = load_raw_aligned()
    # guard the join: the generation pool and the rebuilt raw pool must be the same length
    assert len(raw) >= (max(r["idx"] for r in recs) + 1), \
        f"raw pool ({len(raw)}) smaller than max record idx — filter drifted, join would be wrong"
    print(f"loaded {len(recs)} generations; joined to {len(raw)} raw ExpertQA records by idx")

    rows = build_rows(recs, raw, field=args.field, limit=args.limit)
    data_json = json.dumps(rows).replace("</", "<\\/")  # keep </script> from breaking out
    page = PAGE.replace("__DATA__", data_json).replace("__CAP__", str(CAP))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    n_cap = sum(r["capped"] for r in rows)
    n_deg = sum(r["degenerate"] for r in rows)
    print(f"wrote {len(rows)} cards -> {out}  ({out.stat().st_size/1e6:.1f} MB; "
          f"{n_cap} capped, {n_deg} degenerate)")


if __name__ == "__main__":
    main()
