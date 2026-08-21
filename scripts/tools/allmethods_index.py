#!/usr/bin/env python
"""Build results/viz/ALLMETHODS_index.html linking every generated ALLMETHODS_<dataset>.html, with the
track legend + the three eyeball checks + the taxonomy-family grouping. Light (no pertok); scans the dir."""
import glob
import html
import re
from pathlib import Path

VIZ = Path(__file__).resolve().parents[2] / "results" / "viz"
FAMILY = {"pubmed_qa": "QA / factuality", "med_quad": "QA / factuality", "asqa": "QA / factuality",
          "expertqa": "factuality", "factscore": "factuality",
          "xsum": "summarisation", "cnn_dailymail": "summarisation", "samsum": "summarisation"}
TAXONOMY = {"pubmed_qa": "concentrated (k=1 floor)", "cnn_dailymail": "spread (k=all floor)",
            "xsum": "no-signal (probe-only)"}


def main():
    files = sorted(glob.glob(str(VIZ / "ALLMETHODS_*.html")))
    files = [f for f in files if "index" not in f]
    rows = []
    for f in files:
        ds = Path(f).stem.replace("ALLMETHODS_", "")
        txt = Path(f).read_text()
        ntoggles = len(re.findall(r"recolour\('", txt))
        nabsent = len(re.findall(r'class="absent"', txt))
        nex = len(re.findall(r'class="ex ', txt))
        tax = f' &middot; <b>{TAXONOMY[ds]}</b>' if ds in TAXONOMY else ""
        rows.append(f'<tr><td><a href="ALLMETHODS_{html.escape(ds)}.html">{html.escape(ds)}</a></td>'
                    f'<td>{FAMILY.get(ds,"?")}{tax}</td><td>{nex}</td><td>{ntoggles} active / {nabsent} absent</td></tr>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>ALL-METHODS token overlays</title>
<style>body{{font-family:system-ui,sans-serif;margin:28px;color:#111;max-width:1000px}}
table{{border-collapse:collapse;width:100%;margin:16px 0}} td,th{{border:1px solid #ddd;padding:6px 10px;text-align:left}}
th{{background:#f5f5f5}} code{{background:#f3f3f3;padding:1px 4px}} .note{{color:#555;font-size:14px}}</style></head><body>
<h1>All-methods token overlays</h1>
<p class="note">One HTML per dataset: the same generations with every weighting method as a toggleable per-token
track. <b>Hypothesis generation only</b> &mdash; any claim these suggest needs a full-population statistic before
it enters the project record. Examples are the pooler <b>test set</b>, stratified 4-way (correct/incorrect &times;
confident/uncertain by the msp_min-floor percentile, fixed seed). Every G+1 track has row&nbsp;0 (prompt anchor)
dropped; alignment asserted per example.</p>
<table><tr><th>dataset</th><th>family / taxonomy</th><th>examples</th><th>tracks</th></tr>
{''.join(rows)}
</table>
<h3>Tracks</h3>
<p class="note"><b>Model-side:</b> <code>surprisal</code> (per-token NLL), <code>msp_min(argmin)</code> (one-hot on
the lowest-logprob token &mdash; the pre-registered floor's pick), <code>perplexity/uniform</code> (flat).
<b>Probe-side:</b> <code>attnpool_ID</code> vs <code>attnpool_DiffTask-long</code> (the <b>dissolution contrast</b>
&mdash; same generation, the learned pooler trained ID vs under task shift), <code>MultiMax(pick)</code> (one-hot
on argmax W&middot;x), <code>content_mass</code> / <code>NLL_prior</code> / <code>soft_orgad</code> priors.
<b>Absent (greyed toggles, with reason):</b> wMSP variants (per-dataset retrain), self-attention (GPU pass),
multi-head heads &amp; armD (poolers not persisted). soft_orgad is present only on pubmed_qa/med_quad/expertqa.</p>
<h3>Three things to check by eye</h3>
<ol class="note">
<li><b>pubmed &mdash; probe peak vs error signal.</b> Toggle <code>attnpool_ID</code> then <code>surprisal</code>/
<code>msp_min</code>: does the probe's peak token sit near the error signal? Statistically it does NOT
(AUROC 0.41 at rel-pos ~0.05).</li>
<li><b>Do the taxonomy families look different?</b> Compare <code>attnpool_ID</code> across pubmed (concentrated)
vs cnn_dailymail (spread) vs xsum (no-signal).</li>
<li><b>What did soft-Orgad select?</b> Toggle <code>soft_orgad</code> on pubmed/med_quad/expertqa; the verbatim
prompt that produced the selection is printed in each page's Method-reference panel.</li>
</ol></body></html>"""
    out = VIZ / "ALLMETHODS_index.html"
    out.write_text(page)
    print(f"index: {len(files)} datasets linked -> {out}")
    for f in files:
        print("  ", Path(f).name)


if __name__ == "__main__":
    main()
