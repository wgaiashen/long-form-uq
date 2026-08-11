#!/usr/bin/env python
"""Build the med_quad CLEAN-SPAN shadow regime root: cache/med_quad_clean/{records,pertok,meta}.

WHY (2026-08-11 span-mismatch finding): canonical med_quad LABELS describe the answer_span-cut
text (the clean protocol, promoted 2026-07-07) while canonical UQ SCORES read the full raw
generation — 47.5% of which carries a next-`Question:` continuation. This script builds the
representation half of the clean protocol as a SEPARATE regime root, so the EXISTING ladder can
be run under `LUQ_REGIME="med_quad=med_quad_clean"` as a shadow sensitivity. NOTHING canonical is
touched: the canonical records, pertok cache, and pdl_master stay byte-identical, and the
regime-tagged outputs cannot collide with them by construction (attn_pool.regime_tag).

WHAT IT WRITES, per record (1800 rows):
  * records/<key>.jsonl — a copy with `gen_token_ids[:cut_tok]`, `token_logprobs[:cut_tok]`,
    `gen_text` = the clean span, plus `span_cut_tok`, `span_reason`, `span_raw_len` provenance
    fields. Every label field is carried unchanged (canonical `correctness` is ALREADY the
    clean-span judge label — this script changes representation only, never labels).
  * pertok/<key>__L15.npz — each per-token state array sliced to rows [0 : cut_tok+1]
    (row 0 = the last-prompt anchor, kept: the SAPLMA window convention, as repool.truncated_pool).
  * meta/<key>.prompthash — copied verbatim (same prompts; only the generation tail is cut).

EDGE POLICY (predeclared): cut_tok == 0 (empty clean span) keeps 1 token — an empty sequence
breaks every scorer — and the count is printed loudly. cut_tok is computed by repool.char_to_tok
(prefix-decode binary search; skip_special_tokens=True is load-bearing, see repool.py).

GATES (fail loud BEFORE anything is written):
  G-window   len(states_i) == len(gen_token_ids_i) + 1 on every raw row (the G+1 invariant);
  G-logprob  len(token_logprobs_i) == len(gen_token_ids_i) on every raw row;
  G-pool     max|raw window-mean − cached SAPLMA L15 feature| — WARN above 1e-4, FAIL above 0.05.
             med_quad is a documented GATE_TOLERANT set (~0.009: the feature cache is
             generation-inline while pertok is teacher-forced fp32 — capture numerics, not
             corruption; see repool_truncated.py). The measured value is printed either way.
  G-slice    len(states_clean_i) == cut_tok_i + 1 == len(gen_ids_clean_i) + 1 after slicing.

IMMEDIATE DELIVERABLE printed at the end: the clean-span med_quad floors (msp_min / perplexity /
msp_sum on the eval-split test rows) next to the canonical raw floors from the published master —
the training-free half of the sensitivity, available before any ladder job runs.

    python scripts/checks/make_med_quad_clean_regime.py
    qsub -v LUQ_CMD="scripts/checks/make_med_quad_clean_regime.py" pbs/audit_cpu.pbs
"""
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from luq import answer_span as A          # noqa: E402
from luq import cache, msp, repool, results  # noqa: E402
from luq.config import CACHE_DIR          # noqa: E402
import viz_common as V                    # noqa: E402  (load_tokenizer)

MODEL = "meta-llama/Meta-Llama-3.1-8B"
DATASET = "med_quad"
LAYER = 15
POOL_WARN, POOL_FAIL = 1e-4, 0.05         # med_quad is GATE_TOLERANT (~0.009 documented)
OUT_ROOT = CACHE_DIR / "med_quad_clean"
MASTER = ROOT / "results" / "pdl_master__meta-llama_Meta-Llama-3.1-8B.csv"


def master_medquad_floors():
    out = {}
    for r in csv.DictReader(open(MASTER)):
        if (r.get("seed_regime") == "3seed" and r["eval"] == DATASET and r["rung"] == "ID"
                and r["method"] in ("msp_min", "perplexity", "msp_sum")):
            out[r["method"]] = float(r["prr"])
    return out


def main():
    import time
    t0 = time.time()

    def tick(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    key = cache.run_key(MODEL, DATASET, "ID")
    slug = cache._slug(MODEL)
    records = cache.load_records(CACHE_DIR, key)
    tick(f"records loaded ({len(records)})")
    z = np.load(CACHE_DIR / "pertok" / f"{key}__L{LAYER}.npz", allow_pickle=True)
    states = [np.asarray(z["states"][k], dtype=np.float32) for k in range(len(records))]
    tick("pertok states loaded")
    if len(states) != len(records):
        raise SystemExit(f"G-window FAIL: {len(states)} states vs {len(records)} records")
    feats = np.ascontiguousarray(cache.load_features(CACHE_DIR, key, "saplma")[:, LAYER, :])
    tick("saplma feature plane loaded")
    if len(feats) != len(records):
        raise SystemExit(f"G-pool FAIL: feature cache {len(feats)} rows vs {len(records)} records")
    tok = V.load_tokenizer(MODEL)
    tick("tokenizer loaded")

    # ---------------- gates on the RAW cache ----------------
    pool_diffs = []
    for i, (r, st) in enumerate(zip(records, states)):
        g = len(r["gen_token_ids"])
        if st.shape[0] != g + 1:
            raise SystemExit(f"G-window FAIL row {i}: {st.shape[0]} state rows vs G={g}")
        if len(r["token_logprobs"]) != g:
            raise SystemExit(f"G-logprob FAIL row {i}: {len(r['token_logprobs'])} logprobs vs G={g}")
        pool_diffs.append(float(np.max(np.abs(st.mean(axis=0) - feats[i]))))
    mx = max(pool_diffs)
    if mx > POOL_FAIL:
        raise SystemExit(f"G-pool FAIL: max|window-mean − cached feature| = {mx:.4g} > {POOL_FAIL}")
    print(f"G-window/G-logprob PASS on {len(records)} rows; G-pool max|Δ| = {mx:.4g} "
          f"({'within the documented med_quad tolerance' if mx > POOL_WARN else 'tight'})")

    tick(f"raw gates done (G-pool max|Δ| {mx:.4g})")

    # ---------------- the cut ----------------
    n_cut = n_empty = 0
    clean_records, clean_states, cut_toks = [], [], []
    for _i, (r, st) in enumerate(zip(records, states)):
        if _i and _i % 300 == 0:
            tick(f"cut loop {_i}/{len(records)}")
        clean, cut_char, reason = A.answer_span(r["gen_text"], DATASET, context=r.get("prompt"))
        ct = repool.char_to_tok(tok, r["gen_token_ids"], cut_char)
        if ct == 0:                                      # predeclared edge policy: keep 1 token
            ct = 1
            n_empty += 1
        if ct < len(r["gen_token_ids"]):
            n_cut += 1
        r2 = dict(r)
        r2["gen_token_ids"] = list(r["gen_token_ids"][:ct])
        r2["token_logprobs"] = list(r["token_logprobs"][:ct])
        r2["gen_text"] = clean if ct < len(r["gen_token_ids"]) else r["gen_text"]
        r2["span_cut_tok"] = int(ct)
        r2["span_raw_len"] = int(len(r["gen_token_ids"]))
        r2["span_reason"] = reason
        st2 = np.ascontiguousarray(st[: ct + 1])
        if st2.shape[0] != ct + 1 or len(r2["gen_token_ids"]) != ct:
            raise SystemExit(f"G-slice FAIL idx={r['idx']}")
        clean_records.append(r2)
        clean_states.append(st2)
        cut_toks.append(ct)
    print(f"cut rows: {n_cut}/{len(records)} ({100*n_cut/len(records):.1f}%); "
          f"empty-span rows kept at 1 token: {n_empty}")

    # ---------------- write the shadow root ----------------
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cache.save_records(clean_records, OUT_ROOT, key)
    ptdir = OUT_ROOT / "pertok"
    ptdir.mkdir(parents=True, exist_ok=True)
    tick("cut loop done; writing shadow root (UNCOMPRESSED npz — compression measured too slow)")
    st_obj = np.empty(len(clean_states), dtype=object)
    for k, s in enumerate(clean_states):
        st_obj[k] = s
    np.savez(ptdir / f"{key}__L{LAYER}.npz", states=st_obj, layer=LAYER)
    tick("pertok written")
    (OUT_ROOT / "meta").mkdir(parents=True, exist_ok=True)
    src_hash = CACHE_DIR / "meta" / f"{key}.prompthash"
    if src_hash.exists():
        shutil.copy2(src_hash, OUT_ROOT / "meta" / src_hash.name)
    print(f"wrote {OUT_ROOT}/records/{key}.jsonl ({len(clean_records)} rows), "
          f"pertok ({sum(s.nbytes for s in clean_states)/1e9:.2f} GB), meta")

    # ---------------- immediate deliverable: clean-span floors ----------------
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from xl_rungs import eval_split, label_of            # noqa: E402
    lf = label_of(DATASET)
    y = np.array([r.get(lf, np.nan) for r in clean_records], dtype=float)
    finite = np.isfinite(y)
    keep = np.where(finite)[0]
    split = np.array([clean_records[k]["split"] for k in keep])
    _, te = eval_split(split)
    rows = [keep[i] for i in te]
    yte = y[rows]
    raw_master = master_medquad_floors()
    print("\nCLEAN-SPAN med_quad floors (test rows, same population as the master's ID cell) vs "
          "the CANONICAL raw floors:")
    print(f"{'floor':12s}{'clean-span':>12s}{'raw (master)':>14s}{'delta':>9s}")
    for agg, name in (("min", "msp_min"), ("perplexity", "perplexity"), ("sum", "msp_sum")):
        v = np.array([msp.msp_uncertainty(clean_records[i]["token_logprobs"], agg) for i in rows])
        p = results.prr(yte, v)
        print(f"{name:12s}{p:>+12.4f}{raw_master.get(name, float('nan')):>+14.4f}"
              f"{p - raw_master.get(name, float('nan')):>+9.4f}")
    print("\n(SENSITIVITY numbers — the canonical master is untouched. The ladder jobs must "
          "reproduce these three floor cells exactly: same slicing, second code path.)")


if __name__ == "__main__":
    main()
