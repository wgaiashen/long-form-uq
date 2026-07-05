"""Per-example visualiser for a cached run (Phase 1 of the post-3-July plan).

WHAT THIS IS (in plain words)
-----------------------------
Lihu asked for an HTML viewer that shows, example by example, where a signal lands
on the generated tokens, so we can *see* why a probe succeeds or fails instead of
guessing. This script builds that viewer from data we already have on disk -- no GPU.

It reads one run's cached records (`cache/records/<key>.jsonl`) and the per-method
uncertainty scores (`cache/scores/<key>__<method>.npz`), then writes a single, self-
contained HTML file where each generated token is coloured by a chosen per-token
signal. For each example it also shows the question, the generation, the gold answer,
the graded correctness label, and every method's predicted uncertainty.

INCREMENT 1 (this file, CPU-only): the only per-token signal available is the token
**surprisal** (-logprob), which is exactly the MSP signal and lives in the record
already (`token_logprobs`). No model forward pass is needed.

INCREMENT 2 (next, needs one GPU job): add an **attention** signal (the learned
AttnPool weights and/or the model's own self-attention). Those are not cached, so a
small GPU step will dump a per-token sidecar that this renderer reads. The colour
toggle and the per-token data layout below are already built to accept extra signals,
so wiring attention in later is a small change, not a rewrite.

USAGE
-----
    python scripts/tools/visualise_attention.py --dataset sciq --ood ID \
        --model meta-llama/Meta-Llama-3.1-8B

Writes results/viz/<key>.html (override with --out). Open it in the browser / VS Code.

ALIGNMENT (the classic bug this guards against)
-----------------------------------------------
`gen_token_ids[k]` is the k-th generated token and `token_logprobs[k]` is its logprob,
so they are the same length G and line up one-to-one. We assert that. (The per-token
attention rows in Increment 2 are length G+1 -- an extra row 0 for the last prompt
token -- so that alignment gets its own assert when we add it.)
"""
import argparse
import html
import sys
from pathlib import Path

import numpy as np

# Make `import luq` work when this file is run directly from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache, msp  # noqa: E402
from luq.config import Config  # noqa: E402

# The supervised methods that 03_probe/04_eval may have scored for a run. We show
# whichever ones actually have a scores file on disk; the rest are silently skipped.
KNOWN_METHODS = ["saplma", "linear", "ptrue", "ptrue_accurate", "lookback",
                 "uhead", "uhead_v2"]


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


# --------------------------------------------------------------------------------------
# Per-token signal values (Increment 1: surprisal only)
# --------------------------------------------------------------------------------------

def load_viz_sidecar(cache_dir, key):
    """Load the attention sidecar written by dump_viz_attention.py, if it exists. Returns
    per-record-position lookups for each attention signal, or None if Increment 2 has not
    been run yet (in which case the renderer just shows the surprisal signal)."""
    path = Path(cache_dir) / "viz" / f"{key}__attn.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    pool = dict(zip(z["record_pos_all"].tolist(), z["pool_w"]))
    self_pos = z["record_pos_self"].tolist()
    return {
        "pool_w": pool,  # pos -> (G+1,) weights (row 0 = last-prompt token)
        "self_lastq": dict(zip(self_pos, z["self_lastq_block"])),
        "self_meanq": dict(zip(self_pos, z["self_meanq_block"])),
        "self_lastq_last": dict(zip(self_pos, z["self_lastq_last"])),
        "block": int(z["block"]), "layer": int(z["layer"]),
    }


def per_token_signals(record, record_pos, sidecar):
    """Return a dict {signal_name: (values, norm)} of the per-token signals AVAILABLE for
    this example. `values` is the raw per-token number (shown on hover); `norm` is that
    value scaled to [0, 1] *within this example* so the standout tokens show regardless of
    the example's overall scale. Deeper colour = higher `norm`.

    Always present: `surprisal` = -logprob (higher = the model found this token less
    likely = the MSP signal). When the Increment-2 sidecar exists, we add the attention
    signals for the examples it covers:
      attnpool         our learned pooler's weights (row 0 = last-prompt token is dropped
                       so the G values line up with the G generated tokens)
      selfattn_lastq   raw model self-attention from the final position onto each token
      selfattn_meanq   raw self-attention averaged over the response's own query rows
    """
    g = len(record["gen_token_ids"])
    surprisal = [-lp for lp in record["token_logprobs"]]
    signals = {"surprisal": (surprisal, _minmax(surprisal))}
    if not sidecar:
        return signals

    if record_pos in sidecar["pool_w"]:
        w = list(sidecar["pool_w"][record_pos])
        # pool weights span the SAPLMA window (G+1): drop row 0 (last-prompt token) to
        # align with the G generated tokens. Guard the length so a mismatch is skipped,
        # not silently mis-coloured.
        if len(w) == g + 1:
            aligned = w[1:]
            signals["attnpool"] = (aligned, _minmax(aligned))
    for name, store in (("selfattn_lastq", "self_lastq"),
                        ("selfattn_meanq", "self_meanq")):
        if record_pos in sidecar[store]:
            v = list(sidecar[store][record_pos])
            if len(v) == g:
                signals[name] = (v, _minmax(v))
    return signals


def _minmax(xs):
    """Scale a list to [0, 1] by its own min/max. A flat list maps to all-zeros (no
    misleading contrast). Kept tiny and dependency-light on purpose."""
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


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


def render_token_spans(pieces, signals):
    """One <span> per generated token, carrying the normalised value of every available
    signal as a data-attribute. JS recolours on toggle; here we just set the default
    (the first signal) inline so the file looks right even with JS disabled."""
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
        v = signals[default][1][i]
        style = f"background: rgba(220,38,38,{v:.3f});"  # red, alpha = normalised value
        spans.append(
            f'<span class="tok" {data_attrs} title="{html.escape(title_bits)}" '
            f'style="{style}">{html.escape(piece)}</span>'
        )
    return "".join(spans)


def render_example(record, record_pos, methods, signals, label_field):
    correctness = float(record.get(label_field, float("nan")))
    pieces = token_pieces_cache[record_pos]
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
        <div class="toks">{render_token_spans(pieces, signals)}</div>
      </div>
      <details><summary>full prompt</summary><pre>{html.escape(record['prompt'])}</pre></details>
    </div>"""


def render_html(key, records, methods, label_field, signal_names, examples_html):
    toggle_buttons = "".join(
        f'<button onclick="recolour(\'{n}\')">{n}</button>' for n in signal_names
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
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>viz {html.escape(key)}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #111; }}
  h1 {{ font-size: 18px; }}
  .controls {{ position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid #ddd; }}
  .controls button {{ margin-right: 6px; padding: 4px 10px; cursor: pointer; }}
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
  </div>
  {examples_html}
  <script>{script}</script>
</body></html>"""


# Module-level cache so render_example can reach the decoded pieces without re-decoding.
token_pieces_cache = {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag (e.g. 'pdnew'); must match the run's cache.")
    ap.add_argument("--label-field", default="correctness",
                    help="which correctness field to show / sort by.")
    ap.add_argument("--primary-method", default="saplma",
                    help="method used to rank 'interesting' examples; falls back to any "
                         "available method, then to unsorted if none are scored.")
    ap.add_argument("--max-examples", type=int, default=60,
                    help="cap the number of examples rendered (keeps the HTML light).")
    ap.add_argument("--split", default="test", choices=["test", "train", "all"],
                    help="which split to show. 'test' has method scores; 'train' does not.")
    ap.add_argument("--sort", default="interesting", choices=["interesting", "index"],
                    help="'interesting' surfaces confident-wrong / uncertain-right first.")
    ap.add_argument("--out", default="",
                    help="output HTML path (default: results/viz/<key>.html).")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    print(f"loading run: {key}")
    records = cache.load_records(cfg.cache_dir, key)

    # --- the alignment guard the plan insists on: one logprob per generated token ---
    for i, r in enumerate(records):
        assert len(r["gen_token_ids"]) == len(r["token_logprobs"]), (
            f"record {i}: {len(r['gen_token_ids'])} gen tokens vs "
            f"{len(r['token_logprobs'])} logprobs -- tokenisation/logprob mismatch")

    test_positions = [i for i, r in enumerate(records) if r["split"] == "test"]
    methods = method_scores(cfg.cache_dir, key, test_positions)
    print(f"  methods with scores: {list(methods.keys()) or '(none)'}")

    sidecar = load_viz_sidecar(cfg.cache_dir, key)
    if sidecar:
        print(f"  attention sidecar found: pool_w for {len(sidecar['pool_w'])} test, "
              f"self-attn for {len(sidecar['self_lastq'])} (block {sidecar['block']})")
    else:
        print("  no attention sidecar (run dump_viz_attention.py for the attn signals); "
              "showing surprisal only")

    # Decode tokens once (module cache) so rendering is cheap.
    tok = load_tokenizer(cfg.model_name)
    for i, r in enumerate(records):
        token_pieces_cache[i] = token_pieces(tok, r["gen_token_ids"])

    # Which records to show.
    if args.split == "all":
        candidates = list(range(len(records)))
    else:
        candidates = [i for i, r in enumerate(records) if r["split"] == args.split]

    # Pick the primary method for sorting: the requested one if scored, else any scored.
    primary = args.primary_method if args.primary_method in methods else (
        next(iter(methods), None))
    if args.sort == "interesting" and primary is not None:
        candidates.sort(
            key=lambda i: interestingness(i, records[i].get(args.label_field, 0.0),
                                          methods, primary),
            reverse=True)
        print(f"  sorted by 'interesting' using primary method: {primary}")
    else:
        print("  showing in record order")

    shown = candidates[:args.max_examples]

    # Compute each shown example's available signals, and the UNION of signal names across
    # them (the toggle buttons). Examples missing a signal simply keep their default colour
    # when that toggle is picked -- the JS treats an absent value as 0.
    per_example = {i: per_token_signals(records[i], i, sidecar) for i in shown}
    signal_names = []
    for i in shown:
        for name in per_example[i]:
            if name not in signal_names:
                signal_names.append(name)
    if not signal_names:
        signal_names = ["surprisal"]

    ex_html = []
    for i in shown:
        ex_html.append(render_example(records[i], i, methods, per_example[i], args.label_field))

    out = Path(args.out) if args.out else (cfg.results_dir / "viz" / f"{key}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(key, records, methods, args.label_field,
                               signal_names, "\n".join(ex_html)))
    print(f"wrote {out}  ({len(shown)} of {len(candidates)} {args.split} examples)")


if __name__ == "__main__":
    main()
