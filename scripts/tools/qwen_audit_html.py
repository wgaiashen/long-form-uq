#!/usr/bin/env python
"""§2 of the Qwen validity audit: the self-contained browsable visualiser.

WHY THIS EXISTS AT ALL. On Llama, an aggressive `no_repeat_ngram_size=3` fix removed literal
repetition and produced non-repetitive word salad instead — and the automated repetition scan
returned CLEAN. The problem was caught only by a human reading generations. So this is not a
convenience; visual inspection is the instrument, and the automated flags are only the index into it.

WHAT IS INCLUDED (and why not everything). All 18,464 generations with their prompts and source
articles would be a ~200 MB page no browser handles. Instead, per dataset:
  · the deterministic stratified sample (§3)                      -- unbiased coverage
  · EVERY flagged example, up to a stated per-dataset cap         -- no failure mode goes unshown
  · a deterministic random pool                                   -- unbiased browsing
Any flagged example dropped by the cap is COUNTED AND REPORTED in the page header, because a silent
truncation is how "we looked at everything" becomes false.

⚠️ FIELDS THAT DO NOT EXIST ARE SHOWN AS ABSENT, NEVER FABRICATED. In particular there are no
per-claim judge states, no SUPPORTED/CONTRADICTED counts and no judge rationale anywhere on disk —
expertqa/factscore records carry only `factuality`, `uncovered`, `coherent`, `factuality_quarantined`.
The page says so rather than leaving a blank that reads as zero.

    python scripts/tools/qwen_audit_html.py
"""
import argparse
import csv
import hashlib
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, degeneracy, msp                    # noqa: E402
from luq.config import Config                             # noqa: E402
from attn_pool import PROMPT_REGIME                       # noqa: E402
from xl_rungs import label_of                             # noqa: E402
from sharpening_family import score_lehmer                # noqa: E402

QWEN = "Qwen/Qwen2.5-14B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
BUDGET = {"pubmed_qa": 128, "med_quad": 128, "asqa": 256, "xsum": 56,
          "cnn_dailymail": 128, "samsum": 56, "expertqa": 384, "factscore": 256}
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}
BLEED = re.compile(
    r"^\s*(?:Question:|Answer:|Available choices:|\(\d+\)\.|Summary:|Article:|Story:|Abstract:"
    r"|Text:|Document:|Dialogue:|Conversation:|A single-select problem|Is the question answered)",
    re.MULTILINE)
# Per-dataset flagged cap. expertqa and factscore get NO cap: the entire audit question for those
# two is what the flagged population looks like, so capping them would hide the evidence being
# sought. The others are capped at 250 with the overflow counted and stated in the page header --
# a stated cap is a limitation, a silent one is a false claim of coverage.
FLAG_CAP = {"expertqa": 10**9, "factscore": 10**9}
FLAG_CAP_DEFAULT = 250
POOL = 60
PROMPT_HEAD, PROMPT_TAIL = 1200, 1200


def h(d, i):
    return int(hashlib.sha256(f"{d}:{i}".encode()).hexdigest()[:12], 16)


def esc(s):
    return html.escape(s if isinstance(s, str) else str(s))


def clip(s):
    if len(s) <= PROMPT_HEAD + PROMPT_TAIL:
        return esc(s), False
    return (esc(s[:PROMPT_HEAD]) + "\n\n<i>… [" + f"{len(s)-PROMPT_HEAD-PROMPT_TAIL:,}"
            + " chars elided] …</i>\n\n" + esc(s[-PROMPT_TAIL:]), True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN)
    ap.add_argument("--out", default="results/analysis/qwen_generation_audit.html")
    args = ap.parse_args()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    sample_csv = ROOT / "results/analysis/qwen_generation_audit_sample.csv"
    sample = {}
    if sample_csv.exists():
        for r in csv.DictReader(open(sample_csv)):
            sample[(r["dataset"], int(r["example_id"]))] = r["selection_strata"]

    cards, notes, stats = [], [], []
    for d in LONG:
        cfg = Config(model_name=args.model, dataset=d, ood_setting="ID",
                     prompt_regime=PROMPT_REGIME.get(d, ""))
        p = Path(cfg.cache_dir) / "records" / f"{cache.run_key(args.model, d, 'ID')}.jsonl"
        if not p.exists():
            notes.append(f"{d}: RECORDS MISSING at {p}")
            continue
        recs = [json.loads(l) for l in open(p)]
        bud, lf = BUDGET[d], label_of(d)
        chosen, why = {}, {}
        flag_ids, all_ids = [], []
        meta = {}
        for k, r in enumerate(recs):
            i = r.get("idx", k)
            t = r.get("gen_text", "") or ""
            c = degeneracy.classify(t)
            bl = bool(BLEED.search(t))
            n_gen = len(r["gen_token_ids"])
            meta[i] = (k, c, bl, n_gen)
            all_ids.append(i)
            if c["severe"] or c["degraded"] or bl:
                flag_ids.append(i)
        # 1. the deterministic sample
        for i in all_ids:
            if (d, i) in sample:
                chosen[i] = True; why[i] = "sample:" + sample[(d, i)]
        # 2. every flagged example, deterministic order, capped and COUNTED
        fl = sorted(flag_ids, key=lambda z: h(d, z))
        cap = FLAG_CAP.get(d, FLAG_CAP_DEFAULT)
        for i in fl[:cap]:
            chosen[i] = True; why.setdefault(i, "flagged")
        dropped = max(0, len(fl) - cap)
        # 3. a deterministic pool
        for i in sorted(all_ids, key=lambda z: h(d, z))[:POOL]:
            chosen[i] = True; why.setdefault(i, "pool")
        stats.append((d, len(recs), len(chosen), len(fl), dropped))
        if dropped:
            notes.append(f"{d}: {len(fl):,} flagged, showing {cap} "
                         f"(<b>{dropped:,} not rendered</b> — stated, not hidden)")

        for i in sorted(chosen, key=lambda z: h(d, z)):
            k, c, bl, n_gen = meta[i]
            r = recs[k]
            t = r.get("gen_text", "") or ""
            nll = -np.asarray(r["token_logprobs"], dtype=float)
            y = r.get(lf, None)
            capped = n_gen >= bud
            lbin = ("none" if y is None else "low" if y < 0.34 else "mid" if y < 0.67 else "high")
            fl_cls = "sev" if c["severe"] else "deg" if c["degraded"] else "bleed" if bl else "ok"
            pr, elided = clip(r.get("prompt", ""))
            extra = ""
            for fld, lbl in [("uncovered", "uncovered"), ("coherent", "coherent"),
                             ("factuality_quarantined", "quarantined")]:
                if fld in r:
                    extra += f'<span class="b">{lbl} {esc(r[fld])}</span>'
            cards.append(f"""
<div class="card" data-ds="{d}" data-cap="{int(capped)}" data-flag="{fl_cls}"
     data-lbin="{lbin}" data-src="{why[i].split(':')[0]}" data-len="{n_gen}">
 <div class="hdr"><b>{d}</b> · id {i}
  <span class="b">{n_gen}/{bud} tok</span>
  {'<span class="b cap">CAPPED</span>' if capped else ''}
  <span class="b {fl_cls}">{fl_cls.upper()}</span>
  <span class="b">{lf} {'—' if y is None else f'{float(y):.3f}'}</span>
  <span class="b">{esc(r.get(lf + '_model', r.get('correctness_model', '—')))}</span>
  {extra}
  <span class="b">run {c['max_content_run']} · code {c['code_density']} · ws {c['max_whitespace_gap']}</span>
  <span class="b">ppl {msp.msp_uncertainty(r['token_logprobs'],'perplexity'):.3f} ·
       min {msp.msp_uncertainty(r['token_logprobs'],'min'):.3f} ·
       sum {msp.msp_uncertainty(r['token_logprobs'],'sum'):.1f} ·
       lehmer1 {score_lehmer(nll,1.0):.3f}</span>
  <span class="b why">{esc(why[i])}</span>
 </div>
 <details><summary>prompt ({len(r.get('prompt','')):,} chars{', clipped' if elided else ''})</summary>
   <pre class="pr">{pr}</pre></details>
 <div class="lab">GENERATION</div><pre class="gen">{esc(t)}</pre>
 <details><summary>gold / reference</summary><pre class="pr">{esc(str(r.get('target','—'))[:4000])}</pre></details>
</div>""")

    hdr_rows = "".join(f"<tr><td>{d}</td><td>{n:,}</td><td>{c:,}</td><td>{f:,}</td>"
                       f"<td>{dr:,}</td></tr>" for d, n, c, f, dr in stats)
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Qwen2.5-14B generation audit</title><style>
:root{{--bg:#fff;--fg:#111;--mut:#666;--line:#ddd;--card:#fafafa;--pre:#f4f4f6}}
@media(prefers-color-scheme:dark){{:root{{--bg:#15161a;--fg:#e8e8ea;--mut:#9a9aa2;--line:#33343a;--card:#1c1d22;--pre:#232429}}}}
*{{box-sizing:border-box}} body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}}
h1{{font-size:19px;margin:0 0 4px}} .sub{{color:var(--mut);margin-bottom:12px}}
table{{border-collapse:collapse;margin:8px 0}} td,th{{border:1px solid var(--line);padding:3px 9px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}
#bar{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
padding:8px 0;margin-bottom:12px;z-index:9;display:flex;gap:14px;flex-wrap:wrap;align-items:center}}
select,input{{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:3px 6px}}
.card{{border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:12px;background:var(--card)}}
.hdr{{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:7px}}
.b{{font-size:11px;background:var(--pre);border:1px solid var(--line);border-radius:4px;padding:1px 6px;color:var(--mut)}}
.b.cap{{color:#b45309;border-color:#b45309}} .b.sev{{color:#b91c1c;border-color:#b91c1c;font-weight:700}}
.b.deg{{color:#a16207;border-color:#a16207}} .b.bleed{{color:#6d28d9;border-color:#6d28d9}}
.b.ok{{color:#15803d;border-color:#15803d}} .b.why{{font-style:italic}}
pre{{white-space:pre-wrap;word-wrap:break-word;background:var(--pre);padding:8px;border-radius:6px;
margin:4px 0;font:12.5px/1.55 ui-monospace,Menlo,Consolas,monospace;overflow-x:auto}}
pre.gen{{border-left:3px solid #6d28d9}} .lab{{font-size:11px;color:var(--mut);letter-spacing:.06em}}
summary{{cursor:pointer;color:var(--mut);font-size:12px;padding:2px 0}}
.note{{background:var(--pre);border-left:3px solid #b45309;padding:8px;border-radius:5px;margin:8px 0}}
</style></head><body>
<h1>Qwen2.5-14B — generation validity audit</h1>
<div class="sub">Read-only. Population: ProbeDriftLong, 8 long evals, fp32 + eager, base checkpoint
(no chat template — verified against the cached token ids). Flags come from
<code>luq.degeneracy</code> unchanged, plus the pre-fixed bleed pattern.
<b>Automated flags are an index into the text, not a verdict — read the generations.</b></div>
<div class="note"><b>Fields that do not exist on disk, and are therefore not shown:</b>
per-claim judge states, SUPPORTED/CONTRADICTED counts, judge rationale. expertqa/factscore records
carry only <code>factuality</code>, <code>uncovered</code>, <code>coherent</code>,
<code>factuality_quarantined</code>. Nothing here is fabricated to fill a gap.</div>
<table><tr><th>dataset</th><th>records</th><th>shown</th><th>flagged</th><th>flagged not shown</th></tr>
{hdr_rows}</table>
{''.join(f'<div class="note">{n}</div>' for n in notes)}
<div id="bar">
 <label>dataset <select id="fds"><option value="">all</option>
  {''.join(f'<option>{d}</option>' for d in LONG)}</select></label>
 <label>set <select id="fsrc"><option value="">all</option><option value="sample">sample only</option>
  <option value="flagged">flagged only</option><option value="pool">pool only</option></select></label>
 <label>flag <select id="fflag"><option value="">all</option><option value="sev">severe</option>
  <option value="deg">degraded</option><option value="bleed">bleed</option><option value="ok">clean</option></select></label>
 <label>capped <select id="fcap"><option value="">all</option><option value="1">capped</option>
  <option value="0">not capped</option></select></label>
 <label>label <select id="flab"><option value="">all</option><option value="low">low</option>
  <option value="mid">mid</option><option value="high">high</option><option value="none">unlabelled</option></select></label>
 <label>min len <input id="flen" type="number" value="0" style="width:70px"></label>
 <span id="cnt" class="b"></span></div>
{''.join(cards)}
<script>
const S=['fds','fsrc','fflag','fcap','flab','flen'].map(i=>document.getElementById(i));
function ap(){{let n=0;document.querySelectorAll('.card').forEach(c=>{{
 const ok=(!S[0].value||c.dataset.ds==S[0].value)&&(!S[1].value||c.dataset.src==S[1].value)
  &&(!S[2].value||c.dataset.flag==S[2].value)&&(!S[3].value||c.dataset.cap==S[3].value)
  &&(!S[4].value||c.dataset.lbin==S[4].value)&&(+c.dataset.len>=(+S[5].value||0));
 c.style.display=ok?'':'none';if(ok)n++;}});
 document.getElementById('cnt').textContent=n+' shown';}}
S.forEach(s=>s.addEventListener('input',ap));ap();
</script></body></html>"""
    out.write_text(doc)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB, {len(cards):,} examples)")
    for d, n, c, f, dr in stats:
        print(f"  {d:15s} records {n:>5,}  shown {c:>5,}  flagged {f:>5,}"
              + (f"  ⚠️ {dr:,} flagged NOT shown" if dr else ""))


if __name__ == "__main__":
    main()
