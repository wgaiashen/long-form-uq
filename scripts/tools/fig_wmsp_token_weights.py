#!/usr/bin/env python

# AI assistance: the plotting code in this file was drafted with Claude Code (Anthropic),
# then reviewed, corrected and tested by the author. The figure design, the quantities
# plotted and their interpretation are the author's own. See ACKNOWLEDGEMENTS.md.
"""Static report figure: learned wMSP token weights beside token surprisal, for three responses.

Population discipline: the three responses are drawn ONLY from the deterministic A5 audit set
(12 sha1-selected examples per dataset, scripts/checks/nll_token_audit.py) intersected with the
pre-existing per-response weight dump (cache/viz/*__wmsp_weights.npz, written by
visualise_token_weights.py --dump-weights through the production `_weights_from_raw`). No example
outside that population was considered, and selection used STRUCTURAL criteria only -- weight
concentration, weight dispersion, and whether the max-weight and max-NLL tokens differ -- never any
method's score.

Two aligned rows per response, colour = intensity, tokens on the shared x axis:
    row 1   learned wMSP weight w_t   (averages 1 over content tokens; special tokens are 0)
    row 2   token NLL  -log p_t       (the quantity the weights multiply)
Each row is normalised WITHIN its own response, so colours compare positions inside one response,
never across responses -- the per-row maximum is printed in the row label so the scale stays visible.

    python scripts/tools/fig_wmsp_token_weights.py
Output: results/analysis/fig_wmsp_token_weights.png (+ .pdf), ~7.5in wide, 300 dpi.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.colors import Normalize                      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from luq import cache                                        # noqa: E402
from luq.config import Config                                # noqa: E402
from luq.weighted_msp import content_keep, per_token_nll     # noqa: E402
from attn_pool import PROMPT_REGIME                          # noqa: E402
import viz_common as V                                       # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = cache._slug(MODEL)

# (dataset, record idx, caption). Fixed here so the figure is reproducible and cannot drift.
PANELS = [
    ("cnn_dailymail", 480,
     "concentrated on a few content positions"),
    ("samsum", 642,
     "weight spread across the response"),
    ("pubmed_qa", 1264,
     "almost all weight on the verdict token"),
]


def _short(piece, n=9):
    """Axis label for one token. Long BPE pieces are truncated because a rotated label longer than
    ~9 characters runs into the panel below and gets clipped at the figure edge (which silently ate
    the first characters of 'protein', 'reflect', 'overall' in an earlier render). The full token
    strings are reported in the accompanying table, not read off the figure."""
    s = piece.replace("\n", "\\n").strip()
    if s.startswith("<|") and s.endswith("|>"):
        return "<eot>"
    return s if len(s) <= n else s[: n - 1] + "…"


def load_one(dataset, idx, tok):
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, dataset, "ID"))
    z = np.load(ROOT / "cache" / "viz" / f"{SLUG}__{dataset}__ID__wmsp_weights.npz", allow_pickle=True)
    pos = z["record_pos"].tolist()
    hit = [p for p in pos if recs[p].get("idx") == idx]
    if not hit:
        raise SystemExit(f"{dataset} #{idx} is not in the weight dump -- refusing to invent it")
    p = hit[0]
    rec = recs[p]
    w = np.asarray(z["wMSP_pairwise"][pos.index(p)], dtype=float)
    nll = per_token_nll(rec).astype(float)
    keep = content_keep(rec).astype(bool)
    if not (len(w) == len(nll) == len(rec["gen_token_ids"])):
        raise SystemExit(f"{dataset} #{idx}: length mismatch, refusing to plot")
    pieces = V.token_pieces(tok, rec["gen_token_ids"])
    return rec, pieces, w, nll, keep


def main():
    tok = V.load_tokenizer(MODEL)
    data = [(d, i, cap) + load_one(d, i, tok) for d, i, cap in PANELS]
    widths = [len(x[4]) for x in data]                       # token counts

    maxT = max(widths)          # shared cell pitch: every panel uses the same physical cell width,
                                # so "concentrated" vs "spread" is a fair visual comparison
    cmap = matplotlib.colormaps["viridis"].with_extremes(bad="0.85")

    fig = plt.figure(figsize=(7.5, 2.25 * len(data)), dpi=300)
    gs = fig.add_gridspec(len(data), 1, hspace=0.78)

    for k, (ds, idx, cap, rec, pieces, w, nll, keep) in enumerate(data):
        T = len(pieces)
        ax = fig.add_subplot(gs[k])
        wr = np.where(keep, w, np.nan)                       # specials: no weight -> grey cell
        rows = np.vstack([wr / np.nanmax(wr), nll / nll.max()])
        ax.imshow(rows, aspect="auto", cmap=cmap, norm=Normalize(0, 1),
                  extent=[-0.5, T - 0.5, 1.5, -0.5], interpolation="nearest")
        ax.set_xlim(-0.5, maxT - 0.5)
        # grid between cells
        for x in np.arange(-0.5, T, 1.0):
            ax.axvline(x, color="white", lw=0.4)
        ax.axhline(0.5, color="white", lw=1.2)

        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"wMSP weight\n(max {np.nanmax(wr):.1f}$\\times$)",
                            f"token NLL\n(max {nll.max():.2f})"], fontsize=6.5)
        ax.set_xticks(range(T))
        ax.set_xticklabels([_short(p) for p in pieces],
                           rotation=90, fontsize=5.6, family="monospace")
        ax.tick_params(axis="both", length=0, pad=1.5)
        for s in ax.spines.values():
            s.set_visible(False)

        jw = int(np.nanargmax(wr)); jn = int(np.argmax(nll))
        for j, r_ in ((jw, 0), (jn, 1)):                     # ring the two maxima
            ax.add_patch(plt.Rectangle((j - 0.5, r_ - 0.5), 1, 1, fill=False,
                                       edgecolor="crimson", lw=1.1))
        q = rec.get("correctness", rec.get("factuality"))
        ax.set_title(f"{ds}  #{idx}   quality {q:.2f}   {T} tokens   —   {cap}",
                     fontsize=7.4, pad=4, loc="left")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    cb = fig.colorbar(sm, ax=fig.axes, orientation="horizontal",
                      fraction=0.030, pad=0.16, aspect=55)
    cb.set_label("intensity, normalised within each row (red outline = row maximum; "
                 "grey = special token, excluded from the weighting)", fontsize=6.2)
    cb.ax.tick_params(labelsize=6)

    out = ROOT / "results" / "analysis" / "fig_wmsp_token_weights"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
    print(f"wrote {out}.png and {out}.pdf")
    for ds, idx, _c, rec, pieces, w, nll, keep in data:
        wr = np.where(keep, w, np.nan)
        print(f"  {ds} #{idx}: maxW={pieces[int(np.nanargmax(wr))]!r} "
              f"maxNLL={pieces[int(np.argmax(nll))]!r} "
              f"coincide={int(np.nanargmax(wr)) == int(np.argmax(nll))}")


if __name__ == "__main__":
    main()
