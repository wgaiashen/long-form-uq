# AI assistance: the rendering and layout code in this file was drafted with Claude Code
# (Anthropic), then reviewed, corrected and tested by the author. The quantities displayed
# and their interpretation are the author's own. See ACKNOWLEDGEMENTS.md.
"""Shared render engine for the per-token visualisers (attention family + weighted-MSP family).

WHAT THIS IS (in plain words)
-----------------------------
Both visualisers -- `visualise_attention.py` (attention family: the learned pooler and the model's
own self-attention) and `visualise_token_weights.py` (weighted-MSP family: the learned MSP weights,
Orgad's answer mask, TokenSAR relevance) -- render exactly the same thing: one HTML page per dataset
where each generated token is coloured by a chosen per-token signal, with a row of toggle buttons to
switch the signal. The ONLY thing that differs between the two families is *which signals* they
compute. So the whole page-building machinery lives here, once, and each family script just builds
its own `{signal_name: (raw_values, normalised_values)}` dict and hands it to `render_example`.

This module was lifted out of `visualise_attention.py` unchanged (same HTML, same toggle JS, same
"interesting" sort) so the two families look and behave identically. The one deliberate cleanup:
`render_example` now takes the decoded token `pieces` as an argument instead of reaching into a
module-global cache, so it has no hidden state and both callers pass their own pieces.

Nothing here needs a GPU: it only reads cached records / scores and the tokenizer.
"""
import html
import sys
from pathlib import Path

import numpy as np

# Make `import luq` work when a caller runs from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache, msp  # noqa: E402

# The supervised methods that 03_probe/04_eval may have scored for a run. We show whichever ones
# actually have a scores file on disk; the rest are silently skipped. Both families show these in
# each example's header so you can see what every probe predicted for the same generation.
KNOWN_METHODS = ["saplma", "linear", "ptrue", "ptrue_accurate", "lookback"]


# --------------------------------------------------------------------------------------
# Data loading and joining
# --------------------------------------------------------------------------------------

def load_tokenizer(model_name: str):
    """Load just the tokenizer (CPU, no model weights). We need it to turn the cached
    `gen_token_ids` back into readable per-token text pieces. If this fails it is almost
    always because HF_HOME is not pointing at the volume that holds the cached model --
    the message says so rather than dumping a stack trace."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:  # noqa: BLE001 -- we want a friendly, actionable message
        sys.exit(
            f"ERROR: could not load the tokenizer for '{model_name}'.\n"
            f"  Reason: {e}\n"
            f"  Fix: make sure HF_HOME points at the volume with the cached model, e.g.\n"
            f"       export HF_HOME=<your hf_cache dir>\n"
            f"  (No GPU or download is needed -- the tokenizer files are tiny and already cached.)"
        )


def token_pieces(tok, gen_token_ids):
    """Decode each generated token id to the text it contributes, one piece per id.

    We decode ids one at a time so each piece maps to exactly one logprob. Single-token
    decoding keeps the leading space of word-initial tokens, which we render verbatim
    (CSS `white-space: pre-wrap`) so the reader sees the real spacing and newlines."""
    return [tok.decode([tid]) for tid in gen_token_ids]


def method_scores(cache_dir, key, test_positions):
    """Load every available method's test-set uncertainty and map it back to record
    positions, exactly the way scripts/04_eval.py does it: the scores array is in
    test-record order, so score j belongs to records[test_positions[j]]."""
    out = {}  # method -> {"at": {record_pos: unc}, "layer": int, "ranks": {record_pos: pct}}
    for m in KNOWN_METHODS:
        try:
            s = cache.load_scores(cache_dir, key, method=m)
        except FileNotFoundError:
            continue
        unc = s["unc"]
        if len(unc) != len(test_positions):
            # Out of step with the records (e.g. a partial re-extract); skip rather than
            # mis-align silently. 04_eval would hard-fail here; the viewer is best-effort.
            print(f"  (skipping method '{m}': {len(unc)} scores vs {len(test_positions)} test records)")
            continue
        at = dict(zip(test_positions, unc))
        # Percentile rank of each score among the test set (0 = most confident, 1 = most
        # uncertain). Used to label an example confident/uncertain and to sort.
        order = np.argsort(np.argsort(unc))  # rank of each element
        pct = order / max(len(unc) - 1, 1)
        ranks = dict(zip(test_positions, pct))
        out[m] = {"at": at, "layer": int(s["layer"]), "ranks": ranks}
    return out


def minmax(xs):
    """Scale a list to [0, 1] by its own min/max. A flat list maps to all-zeros (no
    misleading contrast). Kept tiny and dependency-light on purpose."""
    if len(xs) == 0:
        return []
    lo, hi = float(np.min(xs)), float(np.max(xs))
    if hi - lo < 1e-12:
        return [0.0 for _ in xs]
    return [(float(x) - lo) / (hi - lo) for x in xs]


# --------------------------------------------------------------------------------------
# Sorting: surface the interesting examples
# --------------------------------------------------------------------------------------

def interestingness(record_pos, correctness, methods, primary):
    """A big value = the primary method disagrees with the truth -> worth eyeballing.

    We want to find (a) confident-but-wrong: low predicted uncertainty yet low
    correctness, and (b) uncertain-but-right: high predicted uncertainty yet high
    correctness. Both are captured by |uncertainty_percentile - (1 - correctness)|:
    a well-behaved score has high uncertainty exactly when correctness is low, so this
    gap is near 0; a mismatch pushes it toward 1."""
    if primary is None or record_pos not in methods.get(primary, {}).get("ranks", {}):
        return -1.0  # no score (e.g. a train row) -> sort to the bottom of the test view
    unc_pct = methods[primary]["ranks"][record_pos]
    return abs(unc_pct - (1.0 - float(correctness)))


# --------------------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------------------

def tail_prompt(prompt, n=400):
    """Few-shot prompts are long; the actual question is at the end. Show the tail as a
    preview (full prompt is in a collapsible block)."""
    p = prompt.strip()
    return ("..." + p[-n:]) if len(p) > n else p


def render_token_spans(pieces, signals, token_meta=None):
    """One <span> per generated token, carrying the normalised value of every available
    signal as a data-attribute. JS recolours on toggle; here we just set the default
    (the first signal) inline so the file looks right even with JS disabled.
    `token_meta` (optional, additive): list length == pieces of extra hover strings per token
    (e.g. 'logprob -2.3 | punct') appended to the title. None = unchanged behaviour."""
    names = list(signals.keys())
    default = names[0]
    spans = []
    for i, piece in enumerate(pieces):
        data_attrs = " ".join(
            f'data-{name}="{signals[name][1][i]:.4f}"' for name in names
        )
        title_bits = "; ".join(
            f"{name} {signals[name][0][i]:.3f}" for name in names
        )
        if token_meta is not None and i < len(token_meta):
            title_bits += "  ||  " + token_meta[i]
        v = signals[default][1][i]
        style = f"background: rgba(220,38,38,{v:.3f});"  # red, alpha = normalised value
        spans.append(
            f'<span class="tok" {data_attrs} title="{html.escape(title_bits)}" '
            f'style="{style}">{html.escape(piece)}</span>'
        )
    return "".join(spans)


def render_example(record, record_pos, methods, signals, label_field, pieces, token_meta=None):
    # label may be explicitly None (e.g. ExpertQA's faithfulness on an unlabelled row), not just missing --
    # coerce both to NaN so the render doesn't crash (the row shows a blank/NaN correctness, which is honest).
    _lab = record.get(label_field, float("nan"))
    correctness = float(_lab) if _lab is not None else float("nan")
    # Correct/incorrect badge is only a coarse colour cue; we always show the graded value.
    verdict = "correct" if correctness >= 0.5 else "wrong"

    # Per-method uncertainty line: value + confident/uncertain label from its percentile.
    method_bits = []
    for m, info in methods.items():
        if record_pos in info["at"]:
            unc = float(info["at"][record_pos])
            pct = info["ranks"][record_pos]
            tag = "uncertain" if pct >= 0.5 else "confident"
            method_bits.append(
                f'<span class="m"><b>{m}</b> {unc:.3f} '
                f'<span class="pct {tag}">{tag} (p{pct*100:.0f})</span></span>'
            )
    # MSP mean is free from the same logprobs -- always show it as the unsupervised anchor.
    msp_mean = msp.msp_uncertainty(record["token_logprobs"], "mean")
    method_bits.append(f'<span class="m"><b>msp_mean</b> {msp_mean:.3f}</span>')

    target = record.get("target", "")
    if isinstance(target, list):  # trivia gives an alias list
        target = " | ".join(map(str, target))

    return f"""
    <div class="ex {verdict}">
      <div class="ex-head">
        <span class="idx">#{record.get('idx', record_pos)} ({record['split']})</span>
        <span class="corr {verdict}">correctness {correctness:.3f}</span>
        {' '.join(method_bits)}
      </div>
      <div class="q"><b>Q (prompt tail):</b> {html.escape(tail_prompt(record['prompt']))}</div>
      <div class="gold"><b>gold:</b> {html.escape(str(target))}</div>
      <div class="gen"><b>generation (coloured by <span class="sig-name">{list(signals.keys())[0]}</span>):</b><br>
        <div class="toks">{render_token_spans(pieces, signals, token_meta)}</div>
      </div>
      <details><summary>full prompt</summary><pre>{html.escape(record['prompt'])}</pre></details>
    </div>"""


def render_html(key, records, methods, label_field, signal_names, examples_html, subtitle="",
                method_docs=None, absent=None):
    toggle_buttons = "".join(
        f'<button onclick="recolour(\'{n}\')">{n}</button>' for n in signal_names
    )
    # `absent` (optional, additive): {track_name: reason} for methods with NO data on this dataset -- rendered as
    # disabled greyed buttons carrying the reason, so a missing track is EXPLICIT (never silently omitted, never
    # a uniform fallback). None = unchanged behaviour.
    if absent:
        toggle_buttons += "".join(
            f'<button disabled class="absent" title="{html.escape(reason)}">{html.escape(n)} — absent</button>'
            for n, reason in absent.items()
        )
    # Optional "Method reference" panel: one short technical blurb per toggle actually shown.
    # method_docs maps signal_name -> a short HTML string (kept trusted; we author it, no user input).
    # We only list signals present in signal_names, in that order, so a dataset missing orgad/SAR does
    # not show docs for toggles it has no button for.
    ref_html = ""
    if method_docs:
        items = "".join(
            f'<div class="mref-item"><span class="mref-name">{html.escape(n)}</span>'
            f'<span class="mref-desc">{method_docs[n]}</span></div>'
            for n in signal_names if n in method_docs
        )
        if items:
            ref_html = (
                '<details open class="mref"><summary>Method reference &mdash; what each toggle is, '
                'and how it is computed</summary>'
                f'<div class="mref-grid">{items}</div></details>'
            )
    # A tiny bit of JS: recolour every token span from a chosen signal's data-attribute.
    script = """
    function recolour(name) {
      document.querySelectorAll('.tok').forEach(function (el) {
        var v = parseFloat(el.getAttribute('data-' + name));
        if (isNaN(v)) v = 0;
        el.style.background = 'rgba(220,38,38,' + v.toFixed(3) + ')';
      });
      document.querySelectorAll('.sig-name').forEach(function (el) { el.textContent = name; });
    }
    """
    subtitle_html = (f'<div style="font-size:12px;color:#666;margin-top:4px;">{subtitle}</div>'
                     if subtitle else "")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>viz {html.escape(key)}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #111; }}
  h1 {{ font-size: 18px; }}
  .controls {{ position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid #ddd; }}
  .controls button {{ margin-right: 6px; padding: 4px 10px; cursor: pointer; }}
  .controls button.absent {{ color: #999; background: #f3f3f3; border-style: dashed; cursor: not-allowed; font-style: italic; }}
  .ex {{ border: 1px solid #e5e5e5; border-radius: 8px; padding: 12px; margin: 14px 0; }}
  .ex.wrong {{ border-left: 4px solid #dc2626; }}
  .ex.correct {{ border-left: 4px solid #16a34a; }}
  .ex-head {{ display: flex; flex-wrap: wrap; gap: 12px; font-size: 13px; margin-bottom: 8px; align-items: baseline; }}
  .idx {{ color: #666; }}
  .corr.wrong {{ color: #dc2626; font-weight: 600; }}
  .corr.correct {{ color: #16a34a; font-weight: 600; }}
  .m {{ font-size: 12px; color: #333; }}
  .pct.uncertain {{ color: #dc2626; }}
  .pct.confident {{ color: #2563eb; }}
  .q, .gold {{ font-size: 13px; color: #333; margin: 4px 0; }}
  .gen {{ margin-top: 8px; font-size: 14px; }}
  .toks {{ white-space: pre-wrap; line-height: 1.9; border: 1px solid #eee; padding: 8px; border-radius: 6px; }}
  .tok {{ border-radius: 3px; padding: 0 1px; }}
  details {{ margin-top: 8px; }}
  pre {{ white-space: pre-wrap; font-size: 12px; color: #444; background: #fafafa; padding: 8px; border-radius: 6px; }}
  .mref {{ margin: 12px 0 6px; border: 1px solid #e5e5e5; border-radius: 8px; background: #fafbfc; }}
  .mref > summary {{ cursor: pointer; padding: 8px 12px; font-weight: 600; font-size: 13px; }}
  .mref-grid {{ padding: 4px 12px 12px; display: grid; gap: 8px; }}
  .mref-item {{ display: grid; grid-template-columns: 150px 1fr; gap: 12px; align-items: baseline;
    font-size: 12.5px; border-top: 1px solid #eee; padding-top: 8px; }}
  .mref-name {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-weight: 700; color: #b91c1c; }}
  .mref-desc {{ color: #333; line-height: 1.5; }}
  .mref-desc code {{ font-family: ui-monospace, Menlo, Consolas, monospace; background: #eef; padding: 0 3px; border-radius: 3px; }}
  @media (max-width: 640px) {{ .mref-item {{ grid-template-columns: 1fr; gap: 2px; }} }}
</style></head>
<body>
  <h1>{html.escape(key)}</h1>
  <div class="controls">
    colour tokens by: {toggle_buttons}
    &nbsp;|&nbsp; label field: <b>{html.escape(label_field)}</b>
    &nbsp;|&nbsp; {len(records)} records, showing the sorted view below
    <div style="font-size:12px;color:#666;margin-top:4px;">
      deeper red = higher value. Sorted so confident-but-wrong and uncertain-but-right
      float to the top. Hover a token for its raw value.
    </div>
    {subtitle_html}
  </div>
  {ref_html}
  {examples_html}
  <script>{script}</script>
</body></html>"""
