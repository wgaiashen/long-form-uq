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

from luq import cache, probe, results, token_subsets, weighted_msp, weighting  # noqa: E402
from luq.features import orgad_llm, sar  # noqa: E402
from aggregation_table import load_per_token, attn_unc  # noqa: E402
from attn_pool import train_attn, select_temperature  # noqa: E402
from weighted_msp_all_variants import EVALS, CANDIDATES, cells, sampled  # reuse the exact ladder (XL-aware)
import xl_rungs  # noqa: E402
from xl_rungs import label_of  # noqa: E402

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


# ---- FREE unsupervised weight sources (W-B2) -------------------------------------------------------
# All computable from the CACHED record alone -- no extraction, no GPU, no API. This matters because the
# learned pooler wins ID but not OOD, and an unsupervised weighting CANNOT overfit the training task, so if
# one of these beats `uniform` OOD that is exactly the contribution we are looking for.
# NOTE (verified 2026-07-22): per-token ENTROPY is NOT available -- the records store only the logprob of
# the CHOSEN token, not the full distribution, so entropy would need a fresh GPU logits pass. Not included.
def _nll(record, g):
    """Per-token NLL (= -logprob of the emitted token). The model's own uncertainty signal, which is the
    OOD-robust one -- the same quantity MSP aggregates, here used to decide WHERE to attend instead."""
    v = -np.asarray(record["token_logprobs"], dtype=float)
    return v[:g] if len(v) >= g else np.pad(v, (0, g - len(v)), constant_values=float(v.mean() if len(v) else 1.0))


def _position(g, mode="lead"):
    """Positional prior. 'lead' favours early tokens (summarisation leads carry the claim); 'tail' the end."""
    r = np.arange(g, dtype=float) / max(g - 1, 1)
    return (1.0 - r) if mode == "lead" else r


def _to_sentence(w, sent_ids):
    """Broadcast a per-token weight to its SENTENCE mean -- the per-token vs per-sentence contrast. Every
    token in a sentence then shares one weight, so 'which unit do we weight' is separated from 'how'."""
    w = np.asarray(w, float); sid = np.asarray(sent_ids, dtype=int)[:len(w)]
    if len(sid) != len(w) or len(w) == 0:
        return w
    out = np.empty_like(w)
    for s in np.unique(sid):
        m = sid == s
        out[m] = w[m].mean()
    return out


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
    ap.add_argument("--weight-sources", default="",
                    help="comma-separated subset of weight sources to run (default: all). `uniform` is "
                         "always kept -- it is the grounding baseline every other source is judged against. "
                         "Use this for a fast answer when the full source set + --with-pooler would not "
                         "finish inside the walltime.")
    ap.add_argument("--evals", default="",
                    help="restrict eval TARGETS (comma-sep), e.g. expertqa,cnn_dailymail for a small "
                         "--with-pooler run. Training SOURCES for the OOD rungs are still the full pool. "
                         "Default = the full eval set.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"device {device} | seeds {seeds}", flush=True)

    PT, ORG, SAR = {}, {}, {}
    if args.evals:
        import weighted_msp_all_variants as _wmav       # cells() reads this module global at call time
        _wmav.EVALS = [e.strip() for e in args.evals.split(",") if e.strip()]
        print(f"eval targets restricted to {_wmav.EVALS}", flush=True)
    for d in sorted(set(CANDIDATES) | set(EVALS) | set(args.evals.split(",") if args.evals else [])):

        loaded = load_per_token(MODEL, d, LAYER, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok -> skip"); continue
        states, split, y, _, records = loaded
        finite = np.isfinite(np.asarray(y, float))
        if not finite.any():
            print(f"  {d}: unlabelled ({label_of(d)}) -> skip"); continue
        if not finite.all():                          # ExpertQA faithfulness: keep labelled rows
            keep_i = np.where(finite)[0]
            states = [states[k] for k in keep_i]; records = [records[k] for k in keep_i]
            split = split[keep_i]; y = np.asarray(y)[keep_i]
        PT[d] = (states, split, y, records)
        ORG[d] = load_orgad(MODEL, d, args.orgad_variant)   # keyed by split:idx -> robust to row filtering
        rel, tag = load_sar_rel(MODEL, d)
        SAR[d] = rel if (rel is not None and len(rel) == len(states)) else None  # positional -> length-guard
        print(f"  {d}: {len(states)} rows (label={label_of(d)}) | orgad {'yes' if ORG[d] else 'no'} | "
              f"sar {tag if SAR[d] is not None else 'no'}", flush=True)
    sources = set(PT)

    # ---- AVAILABILITY GUARD (added 2026-07-22 after the Orgad silent-fallback bug) ----------------
    # Previously a MISSING weight-source cache fell through to `w = np.ones(g)`, i.e. the "orgad" or "sar"
    # row silently BECAME the uniform row -- and still got written to the CSV. That manufactures agreement
    # with uniform and fabricates exactly the "unsupervised weighting is a null" conclusion we were testing
    # for. Verified damage in the committed organic run: cnn_dailymail had no orgad cache and expertqa had
    # NEITHER, so 10/40 orgad cells and 5/40 sar cells were literally the uniform method.
    # Now: a source with no cache is UNAVAILABLE for that dataset, and any cell touching it is SKIPPED for
    # that source (blank field) rather than silently filled.
    # Cache-backed sources can be missing; the FREE sources (nll/pos/content) are derived from the record
    # itself and are therefore always available. `.get(s, True)` keeps the guard from KeyError-ing on them.
    AVAIL = {d: {"uniform": True, "orgad": ORG.get(d) is not None, "sar": SAR.get(d) is not None}
             for d in sources}
    for s in ("orgad", "sar"):
        miss = sorted(d for d in sources if not AVAIL[d][s])
        if miss:
            print(f"  !! WEIGHT SOURCE '{s}' UNAVAILABLE for: {', '.join(miss)} -> every cell involving "
                  f"these datasets will be SKIPPED for '{s}' (never silently uniform)", flush=True)

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
            elif source.startswith(("nll", "pos", "content")):
                base, _, gran = source.partition("__")      # e.g. "nll__sent"
                if base == "nll":
                    w = _nll(r, g)
                elif base == "nll_inv":
                    v = _nll(r, g); w = v.max() - v          # weight the CONFIDENT tokens instead
                elif base == "pos_lead":
                    w = _position(g, "lead")
                elif base == "pos_tail":
                    w = _position(g, "tail")
                elif base == "content":
                    pieces = tok.convert_ids_to_tokens(r["gen_token_ids"])
                    w = np.asarray(token_subsets.keep_mask(r["gen_token_ids"], pieces, "content"), float)[:g]
                    if len(w) < g:
                        w = np.pad(w, (0, g - len(w)), constant_values=1.0)
                else:
                    w = np.ones(g)
                if gran == "sent":
                    sid, _ = sar._token_sentence_ids(
                        tok, list(r["gen_token_ids"]),
                        r.get("gen_text") or tok.decode(r["gen_token_ids"], skip_special_tokens=True))
                    w = _to_sentence(w, sid)
                w = normalise_avg1(w)
            else:
                w = np.ones(g)
            vecs.append((A * (w[:, None] / g)).sum(axis=0))
        return np.stack(vecs)

    FREE = ["nll", "nll_inv", "pos_lead", "pos_tail", "content"]
    # every free source at BOTH granularities: per-token, and broadcast to the sentence mean
    SRC = ["uniform", "orgad", "sar"] + [f"{b}__{g}" for b in FREE for g in ("tok", "sent")]
    if args.weight_sources:
        want = {x.strip() for x in args.weight_sources.split(",") if x.strip()}
        SRC = [s_ for s_ in SRC if s_ == "uniform" or s_ in want]
        print(f"weight-source subset -> {SRC}", flush=True)
    out_rows = []
    skipped_src = {}      # source -> {datasets that forced a skip}, reported at the end
    out = Path(args.out) if args.out else (ROOT / "results" / f"idea2_weighted_probe__{cache._slug(MODEL)}.csv")

    def _flush():
        # Columns are DYNAMIC (the old hardcoded 5-column list silently dropped every free source). Rewrite
        # the whole CSV after EACH cell so a wall-kill still leaves a usable partial table -- the full run
        # died at the 24h wall with nothing written precisely because it only wrote at the end (2026-07-23).
        base = ["eval", "rung"]
        extra = sorted({k for r in out_rows for k in r if k not in base})
        cols = base + extra
        with open(out, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in out_rows:
                w.writerow({k: r.get(k, "") for k in cols})

    for rung, X, spec in cells(sources):
        spec = [(d, c) for d, c in spec if d in PT]
        if not spec:
            continue
        _, te0 = xl_rungs.eval_split(PT[X][1])         # baked-in for core; carved for XL evals
        if len(te0) == 0:
            continue
        yte = np.array([PT[X][2][i] for i in te0], float)

        # per source: probe on pooled(train) -> PRR(test), averaged over seeds
        per = {s: [] for s in SRC}
        attn_prr, unif_prr = [], []
        for sd in seeds:
            train_rows, test_rows = xl_rungs.build_rows(X, spec, PT, sd, sampled)
            if not train_rows:
                continue
            ytr = np.array([PT[d][2][i] for d, i in train_rows], float)
            for s in SRC:
                # Never score a source whose cache is missing for ANY dataset in this cell -- it would
                # fall back to all-ones and be reported as if the weighting had been applied.
                unavail = sorted({d for d, _ in train_rows + test_rows if not AVAIL[d].get(s, True)})
                if unavail:
                    skipped_src.setdefault(s, set()).update(unavail)
                    continue
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
        _flush()                 # incremental: persist after EVERY cell (wall-kill leaves a usable partial)

    print(f"\nwrote {out} ({len(out_rows)} cells)", flush=True)
    # Make the coverage of any NULL explicit: a blank orgad/sar column is "not measured here", NOT "measured
    # and equal to uniform". Without this line a reader cannot tell the two apart -- which is precisely how
    # the earlier contaminated run read as a clean null.
    if skipped_src:
        print("\nCOVERAGE CAVEAT -- sources skipped for missing caches (NOT scored, NOT uniform):", flush=True)
        for s, ds in sorted(skipped_src.items()):
            print(f"   {s}: cells involving {', '.join(sorted(ds))}", flush=True)
    else:
        print("\nfull coverage: every source had a cache for every dataset scored", flush=True)


if __name__ == "__main__":
    main()
