"""Track 2 / Idea 2: UNSUPERVISED weights -> pool the hidden states -> SAPLMA probe.

The project's core question, feature side: does an *unsupervised* aggregation (Orgad important-tokens,
SAR relevance) transfer OOD where the *learned* attention pooler collapses? For each weight source we pool
the answer-token L15 hidden states weighted by that source into one vector, train the SAPLMA MLP on it, and
score PRR on ID + the full 5-rung ladder -- the same harness as weighted-MSP, so it is apples-to-apples.

Weight sources (all pool the SAME G answer-token states; weights average to 1 over the G tokens):
  uniform   w = 1/G  -> mean-pool over the answer tokens  (the GROUNDING baseline; == SAPLMA mean-pool)
  orgad     w uniform on the model's OWN answer span (leak-free LLM locate), 0 elsewhere -> mean over the
            answer tokens only  (Orgad important-token pooling)
  sar       w proportional to SAR relevance R~ (beta-sharpened) -> relevance-weighted pooling
Plus, for reference in every cell: the LEARNED attention pooler and the uniform frozen-q pooler
(attn_pool), so we can see unsupervised-vs-learned directly.

GROUNDING: with uniform weights the pooled vector == the mean-pool SAPLMA feature (to <1e-5), so the
`uniform` row must match the mean-pool row -- a wiring check printed at startup.

CPU only. Reuses the cached L15 per-token states, the Orgad LLM caches, and the SAR caches.

    python scripts/checks/idea2_weighted_probe.py --seeds 1,2,3
"""
import argparse
import csv as _csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from luq import cache, probe, results, weighted_msp, weighting  # noqa: E402
from luq.features import orgad_llm  # noqa: E402
from aggregation_table import load_per_token, attn_unc  # noqa: E402
from attn_pool import train_attn, select_temperature  # noqa: E402
from weighted_msp_all_variants import EVALS, CANDIDATES, cells, sampled  # reuse the exact ladder

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAB = "correctness"
LAYER = 15


# ---- weight sources: return a per-token weight over the G answer tokens (or None if unavailable) ----

def load_orgad(model, dataset, variant="exact"):
    """Orgad important-token cache. variant='broad' reads the refined long-form-QA span cache (__broad)."""
    suffix = "__broad" if variant == "broad" else ""
    p = ROOT / "cache" / "orgad_llm" / f"{cache._slug(model)}__{dataset}__ID{suffix}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_sar_rel(model, dataset):
    for gran in ("token", "sentence"):
        for suf in ("", "__noprepend"):
            p = ROOT / "cache" / "sar" / f"{cache._slug(model)}__{dataset}__ID__{gran}{suf}.npz"
            if p.exists():
                return list(np.load(p, allow_pickle=True)["relevance"]), f"{gran}{suf}"
    return None, None


def orgad_w(tok, record, orgad_json, g, floor=0.0):
    """SOFT two-tier weight over the G answer tokens: located Orgad-span tokens = 1.0, background = `floor`
    (before avg-1 normalisation). floor=0.0 -> the hard 0/1 mask (old behaviour); floor large -> approaches
    uniform. Handles both the exact-answer str cache and the broad span-LIST cache (via locate_important_rows).
    Returns the soft mask, or None only when nothing is located AND floor==0 (so it falls back to uniform)."""
    ex = orgad_json.get(f"{record['split']}:{record['idx']}") if orgad_json else None
    m = np.full(g, float(floor), dtype=float)
    rows, found = orgad_llm.locate_important_rows(tok, record["gen_token_ids"], ex) if ex else ([], False)
    for r in rows:
        j = r - 1
        if 0 <= j < g:
            m[j] = 1.0
    if not found and floor == 0.0:
        return None
    return m


def normalise_avg1(w):
    """avg-1 weight over G tokens (sum = G); uniform if degenerate."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    s = w.sum()
    g = len(w)
    return (w / s * g) if s > 0 else np.ones(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--with-pooler", action="store_true",
                    help="also compute the learned attention pooler + uniform frozen-q per cell (SLOW: "
                         "per-cell temperature selection). Off by default -- those numbers already exist in "
                         "contribution_ladder/canonical_ladder for the same cells.")
    ap.add_argument("--orgad-variant", default="exact", choices=["exact", "broad"],
                    help="broad = the refined long-form-QA claim-span cache (__broad).")
    ap.add_argument("--orgad-floor", type=float, default=0.0,
                    help="soft-tier background weight tau (0 = hard mask; larger -> closer to uniform).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"device {device} | seeds {seeds}", flush=True)

    PT, ORG, SAR = {}, {}, {}
    for d in CANDIDATES:
        loaded = load_per_token(MODEL, d, LAYER, LAB)
        if loaded is None:
            print(f"  {d}: no pertok -> skip"); continue
        states, split, y, _, records = loaded
        if np.isnan(np.asarray(y, float)).any():
            print(f"  {d}: unlabelled -> skip"); continue
        PT[d] = (states, split, y, records)
        ORG[d] = load_orgad(MODEL, d, args.orgad_variant)
        rel, tag = load_sar_rel(MODEL, d)
        SAR[d] = rel
        print(f"  {d}: {len(states)} rows | orgad {'yes' if ORG[d] else 'no'} | sar {tag or 'no'}", flush=True)
    sources = set(PT)

    # SAR weight (indexed by full-record position within a dataset) -- built per cell from SAR[d].
    def build_pooled(rows, source):
        """rows: list of (dataset, idx). Pool each with `source`."""
        vecs = []
        for d, i in rows:
            st = PT[d][0][i]; r = PT[d][3][i]
            A = weighted_msp.answer_states(st); g = A.shape[0]
            if source == "uniform":
                w = np.ones(g)
            elif source == "orgad":
                w = orgad_w(tok, r, ORG[d], g, floor=args.orgad_floor) if ORG[d] else None
                w = normalise_avg1(w) if w is not None else np.ones(g)
            elif source == "sar":
                rel = SAR[d][i] if (SAR[d] is not None and i < len(SAR[d])) else None
                rel = np.asarray(rel, float)[:g] if rel is not None else None
                w = normalise_avg1(rel) if (rel is not None and len(rel) == g) else np.ones(g)
            else:
                w = np.ones(g)
            vecs.append((A * (w[:, None] / g)).sum(axis=0))
        return np.stack(vecs)

    SRC = ["uniform", "orgad", "sar"]
    out_rows = []
    for rung, X, spec in cells(sources):
        spec = [(d, c) for d, c in spec if d in PT]
        if not spec:
            continue
        te0 = np.where(PT[X][1] == "test")[0]
        if len(te0) == 0:
            continue
        yte = np.array([PT[X][2][i] for i in te0], float)
        test_rows = [(X, i) for i in te0]

        # per source: probe on pooled(train) -> PRR(test), averaged over seeds
        per = {s: [] for s in SRC}
        attn_prr, unif_prr = [], []
        for sd in seeds:
            train_rows = [(d, i) for d, cap in spec for i in sampled(PT[d][1], sd, cap)]
            if not train_rows:
                continue
            ytr = np.array([PT[d][2][i] for d, i in train_rows], float)
            for s in SRC:
                Xtr = build_pooled(train_rows, s); Xte = build_pooled(test_rows, s)
                clf = probe.train_probe_mlp(Xtr, ytr, seed=sd)
                per[s].append(results.prr(yte, probe.uncertainty(clf, Xte)))
            # reference: learned attention pooler + uniform frozen-q (on the full window, like the ladder)
            if args.with_pooler:
                allst = [PT[d][0][i] for d, i in train_rows + test_rows]
                yall = np.concatenate([ytr, yte])
                tr_i = list(range(len(train_rows))); te_i = list(range(len(train_rows), len(train_rows) + len(te0)))
                bestT, _ = select_temperature(allst, yall, tr_i, device, sd, False, False)
                attn_prr.append(results.prr(yte, attn_unc(
                    train_attn(allst, yall, tr_i, device, seed=sd, temperature=bestT), allst, te_i, device)))
                unif_prr.append(results.prr(yte, attn_unc(
                    train_attn(allst, yall, tr_i, device, seed=sd, freeze_query=True), allst, te_i, device)))

        row = {"eval": X, "rung": rung}
        line = f"[{rung:18s}] {X:14s}"
        for s in SRC:
            if per[s]:
                m = float(np.mean(per[s])); row[f"{s}_prr"] = round(m, 4); line += f"  {s} {m:+.3f}"
        if attn_prr:
            row["attn_pooler_prr"] = round(float(np.mean(attn_prr)), 4)
            row["uniform_pooler_prr"] = round(float(np.mean(unif_prr)), 4)
            line += f"  | attn {np.mean(attn_prr):+.3f} unifpool {np.mean(unif_prr):+.3f}"
        print(line, flush=True)
        out_rows.append(row)

    # grounding: uniform-source (probe on mean-pool) should ~equal the uniform frozen-q pooler PRR
    out = Path(args.out) if args.out else (ROOT / "results" / f"idea2_weighted_probe__{cache._slug(MODEL)}.csv")
    cols = ["eval", "rung", "uniform_prr", "orgad_prr", "sar_prr", "attn_pooler_prr", "uniform_pooler_prr"]
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nwrote {out} ({len(out_rows)} cells)", flush=True)


if __name__ == "__main__":
    main()
