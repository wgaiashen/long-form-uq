#!/usr/bin/env python
"""Build the med_quad WHITESPACE-TOLERANT clean-span shadow regime for a W-Models population.

WHY THIS IS A SEPARATE SCRIPT FROM `make_med_quad_clean_regime.py`
----------------------------------------------------------------------
That script builds the ORIGINAL clean-span shadow regime for the two development populations
(hardcoded to `meta-llama/Meta-Llama-3.1-8B`), using the FROZEN `luq.answer_span.answer_span`
cut rule. This script is its generalisation for the W-Models eight-dataset D6 extension on
`meta-llama/Llama-3.1-8B-Instruct` and `Qwen/Qwen2.5-32B`: it takes `--model`, `--regime` and
`--layer` on the command line (those two populations' med_quad generation lives under a named
chat-template regime, not the bare default namespace), and it cuts with
`luq.answer_span_v2.answer_span_v2` -- the whitespace-tolerant med_quad rule -- instead of the
frozen module. See `answer_span_v2.py`'s docstring for why that rule is a new module rather than
an edit to the frozen one.

APPLIED FROM THE OUTSET, NOT AS A LATER RECONCILIATION. Unlike the dev-population shadow regime
(built as a sensitivity AFTER raw scores had already shipped), this script runs BEFORE labelling
or ladder scoring for these two populations' med_quad. That is the whole point: it is the fix for
the label-vs-feature span mismatch found on the dev populations (STOCKTAKE_cleanspan.md), applied
here so the mismatch never happens in the first place. Labels are computed by the judge AFTER this
script runs, on the retained clean_v2 text this script writes into `gen_text` -- so, unlike the dev
shadow regime (where canonical `correctness` was ALREADY the clean-span label and only features
needed slicing), here BOTH the label and the features come from the same retained text by
construction.

WHAT IT WRITES, per record, under a NEW top-level cache namespace
`cache/<regime>_med_quad_clean_v2/{records,pertok,meta}` (never inside `cache/<regime>/`, so the
raw generation stays untouched and inspectable for audit):
  * records/<key>.jsonl -- a copy with `gen_token_ids[:cut_tok]`, `token_logprobs[:cut_tok]`,
    `gen_text` = the clean_v2 span, plus `span_cut_tok`, `span_reason`, `span_raw_len` provenance
    fields. No label fields exist yet at this stage (labelling runs afterward, on this file).
  * pertok/<key>__L<layer>.npz -- each per-token state array sliced to rows [0 : cut_tok+1]
    (row 0 = the last-prompt anchor, kept: the SAPLMA window convention, matches repool.py).
  * meta/<key>.prompthash -- copied verbatim (same prompts; only the generation tail is cut).

EDGE POLICY (predeclared, matching the dev-population script): cut_tok == 0 (empty clean span)
keeps 1 token -- an empty sequence breaks every scorer -- and the count is printed loudly.

GATES (fail loud BEFORE anything is written): G-window, G-logprob, G-pool (WARN above 1e-4, FAIL
above 0.05 -- med_quad is a documented GATE_TOLERANT set, see make_med_quad_clean_regime.py), G-slice.

REPORTED SEPARATELY, NEVER CONFLATED (per the D6 extension's MedQuAD protocol): the raw
template-restart rate (how much text this script actually cut, i.e. `cut rows / total rows`, the
diagnostic on the RAW generation) versus the retained clean_v2 response's own generation-quality
statistics (run `scripts/checks/generation_quality.py --regime <regime>_med_quad_clean_v2`
separately, AFTER this script, to get those -- this script does not compute them).

    python scripts/checks/make_med_quad_v2_clean_regime.py \
        --model meta-llama/Llama-3.1-8B-Instruct --regime llama31i_chat --layer 15
    python scripts/checks/make_med_quad_v2_clean_regime.py \
        --model Qwen/Qwen2.5-32B --regime <qwen32-chat-regime> --layer 31
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from luq import answer_span_v2 as A2      # noqa: E402
from luq import cache, repool             # noqa: E402
from luq.config import CACHE_DIR          # noqa: E402
import viz_common as V                    # noqa: E402  (load_tokenizer)

DATASET = "med_quad"
POOL_WARN, POOL_FAIL = 1e-4, 0.05         # med_quad is GATE_TOLERANT, see the dev-population script


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--regime", required=True,
                     help="the prompt-regime namespace the RAW med_quad generation lives under "
                          "(e.g. llama31i_chat) -- required, not defaulted: a silent default "
                          "namespace is exactly the failure mode this project avoids.")
    ap.add_argument("--layer", type=int, required=True,
                     help="the population's probe layer (15 for Llama-3.1-8B[-Instruct], 31 for "
                          "Qwen2.5-32B, 20 for gemma-2-9b) -- required, never guessed.")
    a = ap.parse_args()

    RAW_ROOT = CACHE_DIR / a.regime
    OUT_ROOT = CACHE_DIR / f"{a.regime}_med_quad_clean_v2"

    t0 = time.time()

    def tick(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    key = cache.run_key(a.model, DATASET, "ID")
    records = cache.load_records(RAW_ROOT, key)
    tick(f"records loaded ({len(records)}) from {RAW_ROOT}")
    z = np.load(RAW_ROOT / "pertok" / f"{key}__L{a.layer}.npz", allow_pickle=True)
    # Hoist z["states"] OUT of the comprehension -- indexing an NpzFile re-reads and
    # re-materialises the whole member every time (see make_med_quad_clean_regime.py's own note).
    st = z["states"]
    states = [np.asarray(st[k], dtype=np.float32) for k in range(len(records))]
    tick("pertok states loaded")
    if len(states) != len(records):
        raise SystemExit(f"G-window FAIL: {len(states)} states vs {len(records)} records")
    feats = np.ascontiguousarray(cache.load_features(RAW_ROOT, key, "saplma")[:, a.layer, :])
    tick("saplma feature plane loaded")
    if len(feats) != len(records):
        raise SystemExit(f"G-pool FAIL: feature cache {len(feats)} rows vs {len(records)} records")
    tok = V.load_tokenizer(a.model)
    tick("tokenizer loaded")

    # ---------------- gates on the RAW cache ----------------
    pool_diffs = []
    for i, (r, s) in enumerate(zip(records, states)):
        g = len(r["gen_token_ids"])
        if s.shape[0] != g + 1:
            raise SystemExit(f"G-window FAIL row {i}: {s.shape[0]} state rows vs G={g}")
        if len(r["token_logprobs"]) != g:
            raise SystemExit(f"G-logprob FAIL row {i}: {len(r['token_logprobs'])} logprobs vs G={g}")
        pool_diffs.append(float(np.max(np.abs(s.mean(axis=0) - feats[i]))))
    mx = max(pool_diffs)
    if mx > POOL_FAIL:
        raise SystemExit(f"G-pool FAIL: max|window-mean - cached feature| = {mx:.4g} > {POOL_FAIL}")
    print(f"G-window/G-logprob PASS on {len(records)} rows; G-pool max|delta| = {mx:.4g} "
          f"({'within the documented med_quad tolerance' if mx > POOL_WARN else 'tight'})")
    tick(f"raw gates done (G-pool max|delta| {mx:.4g})")

    # ---------------- the cut (answer_span_v2, whitespace-tolerant) ----------------
    n_cut = n_empty = 0
    clean_records, clean_states = [], []
    for i, (r, s) in enumerate(zip(records, states)):
        if i and i % 300 == 0:
            tick(f"cut loop {i}/{len(records)}")
        clean, cut_char, reason = A2.answer_span_v2(r["gen_text"], DATASET)
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
        s2 = np.ascontiguousarray(s[: ct + 1])
        if s2.shape[0] != ct + 1 or len(r2["gen_token_ids"]) != ct:
            raise SystemExit(f"G-slice FAIL idx={r['idx']}")
        clean_records.append(r2)
        clean_states.append(s2)
    print(f"RAW TEMPLATE-RESTART RATE (diagnostic, on the raw generation): "
          f"{n_cut}/{len(records)} rows cut ({100 * n_cut / len(records):.1f}%); "
          f"empty-span rows kept at 1 token: {n_empty}")
    print("This is a diagnostic on the RAW text. Run generation_quality.py on the retained "
          f"clean_v2 response separately (regime={a.regime}_med_quad_clean_v2) for the numbers "
          "that actually get scored -- the two must never be reported as one figure.")

    # ---------------- write the shadow root ----------------
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cache.save_records(clean_records, OUT_ROOT, key)
    ptdir = OUT_ROOT / "pertok"
    ptdir.mkdir(parents=True, exist_ok=True)
    tick("cut loop done; writing shadow root (uncompressed npz)")
    st_obj = np.empty(len(clean_states), dtype=object)
    for k, s in enumerate(clean_states):
        st_obj[k] = s
    np.savez(ptdir / f"{key}__L{a.layer}.npz", states=st_obj, layer=a.layer)
    tick("pertok written")
    (OUT_ROOT / "meta").mkdir(parents=True, exist_ok=True)
    src_hash = RAW_ROOT / "meta" / f"{key}.prompthash"
    if src_hash.exists():
        shutil.copy2(src_hash, OUT_ROOT / "meta" / src_hash.name)
    print(f"wrote {OUT_ROOT}/records/{key}.jsonl ({len(clean_records)} rows), "
          f"pertok ({sum(s.nbytes for s in clean_states) / 1e9:.2f} GB), meta")
    print("\nNEXT: label this file (scripts/02_label.py --prompt-regime "
          f"{a.regime}_med_quad_clean_v2), then run generation_quality.py and wmodels_gate.py on "
          "it before it enters the D6 ladder.")


if __name__ == "__main__":
    main()
