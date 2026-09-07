#!/usr/bin/env python
"""A6 — do the learned wMSP token weights agree with raw NLL ranking, or point elsewhere?

EXPLORATORY MECHANISM ANALYSIS (already-read Llama test labels; nothing here selects anything).
Tests whether the hidden-state weighting signal is merely rediscovering high-surprisal tokens or
is genuinely different information — the question that decides how much headroom a
hidden-state-driven aggregation gate could have.

Per dataset: train the CANONICAL normalised wMSP exactly as the ladder/visualiser do
(train_weighted_msp defaults: pairwise soft-rank loss, AdamW lr=1e-3, 5 epochs, batch 32, seed 1,
special tokens excluded), populations ladder-identical (finite-label filter FIRST, then
xl_rungs.eval_split — the §3.1a order). Then, on each TEST response, compare the weight vector w
with the NLL vector.

MATCHED-MASK SENSITIVITY: the canonical floors read ALL generated
tokens while wMSP masks specials (content_keep), so every agreement statistic is computed under
BOTH token populations:
  floor-policy   all G tokens (w is 0 on masked positions by construction);
  wmsp-policy    content tokens only (both vectors restricted to keep == 1).
A disagreement that appears only under one policy is a token-population artefact, not a finding.

Per-response statistics (each under both policies):
  argmax_agree            argmax(w) == argmax(nll)
  rank_maxw_in_nll        1-based rank of the max-weight token in the NLL ordering (1 = largest)
  rank_maxnll_in_w        1-based rank of the max-NLL token in the weight ordering
  spearman_w_nll          within-response Spearman(w, nll)
  weighted_nll_pctile     sum_t (w_t / sum w) * percentile_rank(nll_t)  (0..1; 1 = mass on the
                          most surprising tokens, 0.5 = length-uniform)
  topk_overlap_{1,5,10pct} |top_k(w) ∩ top_k(nll)| / k

Aggregated per dataset and by quality quartile. Llama-only (per-token L15 caches; Workstream A).

    python scripts/checks/wmsp_weight_agreement.py
    qsub -v LUQ_CMD="scripts/checks/wmsp_weight_agreement.py" pbs/audit_cpu.pbs
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache                                     # noqa: E402
from luq import weighted_msp as wm                        # noqa: E402
from xl_rungs import eval_split, label_of                 # noqa: E402
from attn_pool import load_per_token                      # noqa: E402

MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
POLICIES = ("floor", "wmsp")


def pctile_rank(v):
    """percentile rank in [0, 1] of each element within its own vector (average ties)."""
    order = v.argsort().argsort().astype(float)
    return (order + 0.5) / len(v)


def agree_stats(w, nll):
    """The per-response agreement dict for one (weights, nll) pair on ONE token population."""
    T = len(nll)
    if T < 2 or np.ptp(w) < 1e-12 or np.ptp(nll) < 1e-12:
        return None                                       # degenerate: stated by the caller, not 0-filled
    nll_rank_desc = (-nll).argsort().argsort() + 1        # 1 = largest nll
    w_rank_desc = (-w).argsort().argsort() + 1
    rho = spearmanr(w, nll).statistic
    stats = {
        "argmax_agree": int(np.argmax(w) == np.argmax(nll)),
        "rank_maxw_in_nll": int(nll_rank_desc[np.argmax(w)]),
        "rank_maxnll_in_w": int(w_rank_desc[np.argmax(nll)]),
        "spearman_w_nll": float(rho),
        "weighted_nll_pctile": float((w / w.sum() * pctile_rank(nll)).sum()) if w.sum() > 0 else np.nan,
    }
    for name, k in (("topk_overlap_1", 1), ("topk_overlap_5", min(5, T)),
                    ("topk_overlap_10pct", max(1, int(np.ceil(0.10 * T))))):
        top_w = set(np.argsort(-w)[:k]); top_n = set(np.argsort(-nll)[:k])
        stats[name] = len(top_w & top_n) / k
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--out", default=None,
                    help="output CSV path. Give one when running under a cache-root override, so the "
                         "two population arms of a paired comparison cannot overwrite each other.")
    args = ap.parse_args()
    slug = cache._slug(args.model)
    out_csv = (Path(args.out) if args.out
               else ROOT / "results" / "analysis" / f"wmsp_weight_agreement__{slug}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    all_rows = []
    print("=" * 110)
    print(f"A6 WEIGHT-vs-NLL AGREEMENT (EXPLORATORY)  model={args.model}  layer={LAYER}")
    print("=" * 110)
    for d in LONG:
        loaded = load_per_token(args.model, d, LAYER, label_of(d))
        if loaded is None:
            raise SystemExit(f"A1-gate FAIL [{d}]: no per-token cache at L{LAYER}")
        states, split, y, _, records = loaded
        finite = np.isfinite(y)                            # ladder order: filter FIRST, then carve
        keepi = np.where(finite)[0]
        states = [states[i] for i in keepi]
        records = [records[i] for i in keepi]
        split, y = split[keepi], y[keepi]
        tr, te = eval_split(split)
        print(f"[{d}] n={len(y)} train={len(tr)} test={len(te)} — training canonical wMSP "
              f"(normalised, pairwise, seed 1)...", flush=True)
        model = wm.train_weighted_msp(states, records, y, list(map(int, tr)), device,
                                      weight_mode="normalised", length_normalise=True,
                                      seed=1, loss="pairwise")
        model.eval()
        n_degen = {p: 0 for p in POLICIES}
        with torch.no_grad():
            for i in map(int, te):
                nll = wm.per_token_nll(records[i])
                raw = model(torch.from_numpy(wm.answer_states(states[i])).to(device))
                keep = wm.content_keep(records[i])
                w = wm._weights_from_raw(raw, "normalised",
                                         keep=torch.from_numpy(keep).to(device)).cpu().numpy()
                if len(w) != len(nll):
                    raise SystemExit(f"G-align FAIL [{d}] pos={i}: {len(w)} weights vs "
                                     f"{len(nll)} nll — the G+1/G convention broke")
                for policy in POLICIES:
                    if policy == "floor":
                        wv, nv = w, nll
                    else:
                        sel = keep.astype(bool)
                        if sel.sum() < 2:
                            n_degen[policy] += 1
                            continue
                        wv, nv = w[sel], nll[sel]
                    s = agree_stats(np.asarray(wv, dtype=float), np.asarray(nv, dtype=float))
                    if s is None:
                        n_degen[policy] += 1
                        continue
                    all_rows.append({"eval": d, "example_pos": i, "policy": policy,
                                     "quality_label": float(y[i]), "n_tokens": len(nll),
                                     "n_content": int(keep.sum()), **s})
        if any(n_degen.values()):
            print(f"    degenerate rows skipped (stated, not zero-filled): {n_degen}")

    with open(out_csv, "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w_.writeheader()
        for r in all_rows:
            w_.writerow({k: ("" if isinstance(v, float) and not np.isfinite(v) else v)
                         for k, v in r.items()})
    print(f"\nwrote {out_csv}  ({len(all_rows)} response × policy rows)")

    # ---------------- aggregates ----------------
    import pandas as pd
    df = pd.DataFrame(all_rows)
    num = ["argmax_agree", "rank_maxw_in_nll", "rank_maxnll_in_w", "spearman_w_nll",
           "weighted_nll_pctile", "topk_overlap_1", "topk_overlap_5", "topk_overlap_10pct"]
    print("\nPER-DATASET MEDIANS (argmax_agree/topk are MEANS), by policy:")
    hdr = f"{'eval':15s}{'policy':>7s}" + "".join(f"{c[:12]:>14s}" for c in num)
    print(hdr)
    for d in LONG:
        for p in POLICIES:
            g = df[(df["eval"] == d) & (df["policy"] == p)]
            if not len(g):
                continue
            vals = [g[c].mean() if c.startswith(("argmax", "topk")) else g[c].median() for c in num]
            print(f"{d:15s}{p:>7s}" + "".join(f"{v:>14.3f}" for v in vals))
    print("\nBY QUALITY QUARTILE (pooled over datasets within each policy — mechanism view only):")
    df["quartile"] = df.groupby(["eval"])["quality_label"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=[1, 2, 3, 4]))
    for p in POLICIES:
        for q in (1, 2, 3, 4):
            g = df[(df["policy"] == p) & (df["quartile"] == q)]
            vals = [g[c].mean() if c.startswith(("argmax", "topk")) else g[c].median() for c in num]
            print(f"{'Q' + str(q):15s}{p:>7s}" + "".join(f"{v:>14.3f}" for v in vals))


if __name__ == "__main__":
    main()
