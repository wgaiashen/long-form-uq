#!/usr/bin/env python
"""Verify a transferred per-token cache set before it is used for anything.

Written for the Qwen2.5-14B DoC->RCS transfer (M6b), but model-agnostic.

WHY THIS EXISTS. A cross-cluster rsync can fail in ways that presence and mtime do not reveal, and
the project has been bitten before: a plain copy once truncated a 2.3 GB file silently. Worse, a
per-token cache can be intact as a FILE and still be wrong for the job: the wrong model's width, or a
window that is not [last_prompt_token] + gen_tokens. `answer_states` drops row 0 with no assertion, so
a bad window would MIS-ALIGN every token weight rather than crash.

FOUR CHECKS PER DATASET, all fail-loud:
  1. zip structure intact (a truncated npz raises BadZipFile)
  2. layer stamped in the file == the layer requested
  3. hidden dim == EXPECTED_HIDDEN_DIM[model] -- catches a mis-slugged cache from another model
  4. len(states) == len(records), and states[i].shape[0] == len(token_logprobs[i]) + 1 for EVERY i

Loads one dataset at a time and frees it, so peak memory is the largest single cache, not the set.

    qsub -l select=1:ncpus=4:mem=64gb -v LUQ_CMD="scripts/checks/verify_pertok_transfer.py \
        --model Qwen/Qwen2.5-14B --layer 23" pbs/audit_cpu.pbs
"""
import argparse
import gc
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache                                            # noqa: E402
from luq.config import Config                                    # noqa: E402
from attn_pool import EXPECTED_HIDDEN_DIM, PROMPT_REGIME         # noqa: E402

LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--datasets", default=",".join(LONG))
    args = ap.parse_args()

    slug = cache._slug(args.model)
    want_dim = EXPECTED_HIDDEN_DIM.get(args.model)
    if want_dim is None:
        raise SystemExit(f"no expected hidden dim registered for {args.model!r} -- refusing to verify "
                         "against nothing")
    print(f"VERIFY per-token transfer: {args.model} (slug {slug}), layer {args.layer}, "
          f"expected hidden dim {want_dim}")
    print("=" * 96)

    fails, checked = [], 0
    for d in args.datasets.split(","):
        cfg = Config(model_name=args.model, dataset=d, ood_setting="ID",
                     prompt_regime=PROMPT_REGIME.get(d, ""))
        p = cfg.cache_dir / "pertok" / f"{slug}__{d}__ID__L{args.layer}.npz"
        if not p.exists():
            fails.append(f"{d}: MISSING at {p}")
            print(f"  {d:15s} MISSING  {p}")
            continue
        try:
            zipfile.ZipFile(p).namelist()
        except Exception as e:
            fails.append(f"{d}: not a readable npz ({type(e).__name__})")
            print(f"  {d:15s} FAIL zip {type(e).__name__}")
            continue
        z = np.load(p, allow_pickle=True)
        lay = int(z["layer"])
        st = z["states"]
        recs = cache.load_records(cfg.cache_dir, cache.run_key(args.model, d, "ID"))
        dim = int(np.asarray(st[0]).shape[-1]) if len(st) else -1
        bad_win = sum(1 for i in range(min(len(st), len(recs)))
                      if np.asarray(st[i]).shape[0] != len(recs[i]["token_logprobs"]) + 1)
        probs = []
        if lay != args.layer:
            probs.append(f"layer stamp {lay} != {args.layer}")
        if dim != want_dim:
            probs.append(f"hidden dim {dim} != {want_dim}")
        if len(st) != len(recs):
            probs.append(f"{len(st)} states vs {len(recs)} records")
        if bad_win:
            probs.append(f"{bad_win} examples violate the G+1 window")
        status = "OK" if not probs else "FAIL"
        if probs:
            fails.append(f"{d}: " + "; ".join(probs))
        print(f"  {d:15s} {status:4s} n={len(st):5d} layer={lay} dim={dim} "
              f"G+1 ok={len(st)-bad_win}/{len(st)}")
        checked += 1
        del st, z, recs
        gc.collect()

    print("=" * 96)
    if fails:
        print(f"RESULT: FAIL -- {len(fails)} problem(s). Do not run the pass.")
        for f in fails:
            print(f"  x {f}")
        raise SystemExit(1)
    print(f"RESULT: PASS -- {checked}/{len(args.datasets.split(','))} datasets verified on all four checks.")


if __name__ == "__main__":
    main()
