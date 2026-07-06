"""ExpertQA decoding-config sweep: find a config that stops the degeneration WITHOUT trading it
for fluent-off-topic, before regenerating all 2016.

The reppen full-set used repetition_penalty=1.3 + no_repeat_ngram_size=3, which INDUCED word-salad
(forbidden from repeating any 3-gram, the base model spews novel incoherent tokens). This sweep
regenerates a stratified 50-example subset (currently-derailed + currently-clean) under several
configs and scores each with the VALIDATED list-aware detector (luq.degeneracy) — NOT the old
repetition scan, so a config that merely suppresses lists can't score well for the wrong reason.

Two things the detector cannot see, handled separately:
  - Relevance ("does it still answer the question?"): a high repetition_penalty can push the model
    off the correct technical term into fluent-but-off-topic text, which the detector passes. This
    script emits a side-by-side HTML (question | gold | each config) so that can be eyeballed, and
    the label-gate's judge scores it later. The detector rate is necessary, not sufficient.
  - The precommitted bar (set BEFORE reading results): SEVERE (salad/code/enum) <= 2%,
    DEGRADED (incl. function-word-dropped rambling) <= 8%. A config that only beats the broken
    26% baseline is NOT a pass.

Run (GPU):
    python scripts/tools/sweep_decoding.py --n 50 --out results/expertqa/decoding_sweep
"""
import argparse
import html
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache, generate, degeneracy   # noqa: E402
from luq.config import Config                  # noqa: E402
from luq import expertqa                        # noqa: E402

# (label, repetition_penalty, no_repeat_ngram_size). None => arg not passed (HF no-op default).
CONFIGS = [
    ("current_1.3_ng3", 1.3, 3),    # the broken reference (reproduce it, to calibrate the delta)
    ("rp1.2_only",      1.2, None),  # rep-penalty only, gentle  (tests: is no_repeat the culprit?)
    ("rp1.3_only",      1.3, None),  # rep-penalty only, current strength
    ("rp1.2_ng4",       1.2, 4),     # relaxed n-gram
    ("rp1.15_ng4",      1.15, 4),    # gentle both
]
CAP = expertqa.MAX_NEW_TOKENS


def pick_subset(records, n, seed=1):
    """Stratified: ~70% currently-derailed (where configs differ most) + ~30% currently-clean
    (to catch a config that BREAKS good generations). Deterministic under seed."""
    derailed, clean = [], []
    for r in records:
        (derailed if degeneracy.is_degraded(r["gen_text"]) else clean).append(r)
    rng = random.Random(seed)
    rng.shuffle(derailed); rng.shuffle(clean)
    n_der = min(len(derailed), int(round(n * 0.7)))
    picked = derailed[:n_der] + clean[:n - n_der]
    rng.shuffle(picked)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="results/expertqa/decoding_sweep")
    args = ap.parse_args()

    key = cache.run_key(args.model, "expertqa", "ID")
    recs = cache.load_records(Config(prompt_regime="expertqa_reppen").cache_dir, key)
    subset = pick_subset(recs, args.n)
    print(f"sweep on {len(subset)} examples x {len(CONFIGS)} configs", flush=True)

    model, tok = generate.load_model(args.model, attn_implementation="eager")  # fp32 default path
    import torch
    model = model.to(torch.float32)

    # rows[idx] = {question, gold, gens: {config: {text, signals}}}
    rows = []
    for j, r in enumerate(subset):
        row = {"idx": r["idx"], "prompt": r["prompt"], "gens": {}}
        for label, rp, ng in CONFIGS:
            rec, _ = generate.generate(model, tok, r["prompt"], CAP,
                                       repetition_penalty=rp, no_repeat_ngram_size=ng)
            sig = degeneracy.classify(rec["gen_text"])
            sig["n_tokens"] = len(rec["gen_token_ids"])
            row["gens"][label] = {"text": rec["gen_text"], "sig": sig}
        rows.append(row)
        print(f"  {j+1}/{len(subset)} done (idx {r['idx']})", flush=True)

    # per-config summary against the precommitted bar
    summary = {}
    for label, _, _ in CONFIGS:
        sigs = [row["gens"][label]["sig"] for row in rows]
        n = len(sigs)
        summary[label] = {
            "severe_pct": round(100 * sum(s["severe"] for s in sigs) / n, 1),
            "degraded_pct": round(100 * sum(s["degraded"] for s in sigs) / n, 1),
            "median_tokens": sorted(s["n_tokens"] for s in sigs)[n // 2],
            "capped_pct": round(100 * sum(s["n_tokens"] >= CAP for s in sigs) / n, 1),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".json")).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    write_html(rows, summary, out.with_suffix(".html"))

    print("\n=== SWEEP SUMMARY (bar: SEVERE<=2%, DEGRADED<=8%) ===")
    print(f"{'config':<18}{'severe%':>9}{'degraded%':>11}{'capped%':>9}{'med_tok':>9}")
    for label, s in summary.items():
        print(f"{label:<18}{s['severe_pct']:>9}{s['degraded_pct']:>11}{s['capped_pct']:>9}{s['median_tokens']:>9}")
    print(f"\nEyeball relevance (question vs each config): {out.with_suffix('.html')}")


def write_html(rows, summary, path):
    def esc(s): return html.escape(s or "")
    cfgs = [c[0] for c in CONFIGS]
    cards = []
    for row in rows:
        cols = ""
        for c in cfgs:
            g = row["gens"][c]; sig = g["sig"]
            flag = "severe" if sig["severe"] else ("degraded" if sig["degraded"] else "ok")
            cols += (f"<div class='cfg {flag}'><b>{c}</b> "
                     f"<span class=t>{sig['n_tokens']}tok run{sig['max_content_run']} [{flag}]</span>"
                     f"<div class=g>{esc(g['text'])}</div></div>")
        cards.append(f"<div class=card><div class=q>#{row['idx']} {esc(row['prompt'].split('Question:')[-1].split('Answer:')[0].strip())}</div>{cols}</div>")
    srows = "".join(f"<tr><td>{k}</td><td>{v['severe_pct']}</td><td>{v['degraded_pct']}</td>"
                    f"<td>{v['capped_pct']}</td><td>{v['median_tokens']}</td></tr>" for k, v in summary.items())
    path.write_text(f"""<!doctype html><meta charset=utf-8><title>decoding sweep</title><style>
body{{font:13px sans-serif;margin:16px;background:#f4f5f7}}table{{border-collapse:collapse;margin-bottom:16px}}
td,th{{border:1px solid #ccc;padding:4px 8px}}.card{{background:#fff;border-radius:6px;padding:10px;margin-bottom:12px}}
.q{{font-weight:600;margin-bottom:6px}}.cfg{{border-left:4px solid #4caf50;padding:4px 8px;margin:4px 0}}
.cfg.degraded{{border-color:#ff9800}}.cfg.severe{{border-color:#e53935}}.t{{color:#888;font-size:11px}}
.g{{white-space:pre-wrap;margin-top:3px}}</style>
<h3>Decoding sweep — bar: SEVERE&le;2%, DEGRADED&le;8%</h3>
<table><tr><th>config</th><th>severe%</th><th>degraded%</th><th>capped%</th><th>med_tok</th></tr>{srows}</table>
{''.join(cards)}""", encoding="utf-8")


if __name__ == "__main__":
    main()
