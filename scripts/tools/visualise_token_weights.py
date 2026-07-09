"""Visualise WHICH TOKENS each method focuses on, side by side, for a few example generations.

For each example we render the generated tokens once per signal, each token shaded by that signal's
weight (darker = higher). This makes the methods concrete:
  surprisal   -log p(token)         the raw MSP signal (how unconfident the model was on this token)
  weighted_MSP  learned weight       what the weighted-MSP MLP decides each token is worth (softmax)
  attention    learned attention     what the attention POOLER weights each token's hidden state by
  orgad        0/1 answer-span mask   the tokens Orgad keeps (the gold answer span); 0 elsewhere
  SAR          relevance R~           how much removing the token changes the answer's meaning

Shows short-form (sciq: token-level, all signals meaningful) and long-form (pubmed: SAR sentence-level).
Writes a self-contained HTML.

    python scripts/tools/visualise_token_weights.py --out results/viz/token_weights.html
"""
import argparse
import html
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from luq import cache, weighted_msp  # noqa: E402
from luq.config import Config  # noqa: E402
from aggregation_table import load_per_token, pad_batch  # noqa: E402
from attn_pool import train_attn  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"


def norm01(v):
    v = np.asarray(v, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def load_sar(d, gran):
    p = ROOT / "cache" / "sar" / f"{cache._slug(MODEL)}__{d}__ID__{gran}.npz"
    if not p.exists():
        return None
    return list(np.load(p, allow_pickle=True)["relevance"])


def row_html(pieces, weights, label):
    cells = []
    w01 = norm01(weights)
    for p, w in zip(pieces, w01):
        txt = html.escape(p.replace("Ġ", " ").replace("Ċ", "\\n"))
        cells.append(f'<span class="tok" style="background:rgba(15,118,110,{0.08 + 0.85 * w:.2f})" '
                     f'title="{w:.2f}">{txt}</span>')
    return f'<div class="row"><span class="lab">{label}</span><span class="toks">{"".join(cells)}</span></div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "viz" / "token_weights.html"))
    ap.add_argument("--per", type=int, default=4, help="examples per dataset")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)

    blocks = []
    for d, gran in [("sciq", "token"), ("trivia_qa", "token"),
                    ("pubmed_qa", "sentence"), ("xsum", "sentence")]:
        loaded = load_per_token(MODEL, d, 15, "correctness")
        if loaded is None:
            continue
        states, split, y, _, records = loaded
        tr = [i for i in range(len(states)) if split[i] == "train"]
        te = [i for i in range(len(states)) if split[i] == "test"]
        # train the weighting models once per dataset: weighted-MSP (pairwise + Blondel), attention
        # pooler (learned) and uniform (frozen-query = mean-pool)
        wm = weighted_msp.train_weighted_msp(states, records, y, tr, device, weight_mode="normalised",
                                             length_normalise=True, seed=1, loss="pairwise")
        wm_bl = (weighted_msp.train_weighted_msp(states, records, y, tr, device, weight_mode="normalised",
                 length_normalise=True, seed=1, loss="blondel") if weighted_msp._HAVE_TORCHSORT else None)
        attn = train_attn(states, y, tr, device, seed=1, temperature=0.25)
        unif = train_attn(states, y, tr, device, seed=1, freeze_query=True)
        masks, _ = weighted_msp.build_answer_masks(tok, [records[i] for i in te[:args.per]])
        sar_rel = load_sar(d, gran)
        # pick a couple correct + a couple incorrect test examples for contrast
        te_sorted = sorted(te, key=lambda i: y[i])
        picks = te_sorted[:args.per // 2] + te_sorted[-(args.per - args.per // 2):]
        for k, i in enumerate(picks):
            g = records[i]["gen_token_ids"]
            pieces = tok.convert_ids_to_tokens(g)
            nll = np.array([-lp for lp in records[i]["token_logprobs"]], float)[: len(pieces)]
            # weighted-MSP weights = softmax(MLP(answer_states)) over the G tokens
            with torch.no_grad():
                asx = torch.from_numpy(weighted_msp.answer_states(states[i])).to(device)
                wmsp_w = torch.softmax(wm(asx), dim=0).cpu().numpy()[: len(pieces)]
                wmsp_bl = (torch.softmax(wm_bl(asx), dim=0).cpu().numpy()[: len(pieces)]
                           if wm_bl is not None else None)
                X, m, pos = pad_batch([states[i]], device)
                _, a = attn(X, m, pos)
                _, au = unif(X, m, pos)
                att_w = a[0, : states[i].shape[0]].cpu().numpy()[1: len(pieces) + 1]   # drop anchor row 0
                unif_w = au[0, : states[i].shape[0]].cpu().numpy()[1: len(pieces) + 1]
            om, _ = weighted_msp.build_answer_masks(tok, [records[i]])
            rows = [row_html(pieces, nll, "surprisal (MSP)"),
                    row_html(pieces, wmsp_w, "wMSP pairwise")]
            if wmsp_bl is not None:
                rows.append(row_html(pieces, wmsp_bl, "wMSP Blondel"))
            rows += [row_html(pieces, att_w, "attention pool"),
                     row_html(pieces, unif_w, "uniform (mean)"),
                     row_html(pieces, om[0][: len(pieces)], "orgad (0/1)")]
            if sar_rel is not None and i < len(sar_rel):
                rows.append(row_html(pieces, np.asarray(sar_rel[i])[: len(pieces)], f"SAR ({gran})"))
            gold = records[i]["target"]
            tag = "correct" if y[i] >= 0.5 else "WRONG"
            blocks.append(f'<div class="ex"><div class="meta"><b>{d}</b> · judge={y[i]:.2f} '
                          f'<span class="{("ok" if y[i]>=0.5 else "no")}">{tag}</span> · gold: '
                          f'{html.escape(str(gold)[:80])}</div>{"".join(rows)}</div>')

    css = """<style>body{font-family:system-ui;margin:24px;background:#0e1217;color:#e6ebf1}
    h1{font-size:1.3rem}.ex{margin:22px 0;border:1px solid #242c36;border-radius:8px;padding:12px}
    .meta{font-size:.82rem;color:#9aa6b2;margin-bottom:8px}.ok{color:#57c79a}.no{color:#e08a76}
    .row{display:flex;gap:8px;align-items:baseline;margin:3px 0;white-space:nowrap;overflow-x:auto}
    .lab{font-family:monospace;font-size:.7rem;color:#9aa6b2;min-width:96px;text-align:right}
    .tok{padding:1px 1px;border-radius:2px;font-size:.82rem;white-space:pre}
    .toks{white-space:nowrap}</style>"""
    doc = (f"<!doctype html><meta charset=utf8><title>token weights</title>{css}"
           f"<h1>What tokens each method focuses on (darker = higher weight)</h1>"
           f"<p style='color:#9aa6b2;font-size:.85rem'>surprisal = raw MSP signal · weighted_MSP/attention "
           f"= LEARNED weights · orgad = 0/1 answer-span mask · SAR = relevance. Each row normalised 0-1 "
           f"for visibility.</p>{''.join(blocks)}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
