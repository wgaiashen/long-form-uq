#!/usr/bin/env python
"""Derive the truncated population's per-token and SAPLMA caches by SLICING, not re-extracting.

⭐ WHY SLICING IS EXACT HERE, NOT AN APPROXIMATION. The model is causal: the hidden state at
position t is a function of positions <= t only. Truncating the END of a generation therefore
cannot change the hidden state of any RETAINED token. The states for a prefix of the generation are
bit-identical to the states the model would produce if it had stopped there. So

    truncated_states = full_states[:n_keep + 1]

is not "close enough" — it is the same tensor. Re-running `01h_pertoken` on the truncated records
would burn ~6 GPU-hours to recompute numbers we already hold, and would introduce nondeterminism
(fp32 kernel reduction order) that the slice does not.

⚠️ THE +1 IS THE WINDOW, AND IT IS ASSERTED, NOT ASSUMED. The per-token window is
`[last_prompt_token] + gen_tokens` = G+1 rows for G generated tokens (implementation_notes §6 —
this project has already paid once for getting it wrong). Every record is checked against its own
`token_logprobs` length before anything is written, and a single mismatch aborts the whole dataset.

WHAT IS WRITTEN, into the truncated namespace only:
  * `pertok/<slug>__<ds>__ID__L<layer>.npz`   — states sliced per row, same dtype, same layer field
  * `features/<slug>__<ds>__ID__saplma.npz`   — the mean-pool over the RETAINED window, recomputed
    from the sliced states (this is what `01e_repool` would produce, by the same definition)

The canonical caches are opened read-only. The guard `feature_pertok_consistency.py` still applies
afterwards and should pass by construction, since both sides are derived from the same slice.

    python scripts/checks/slice_caches_truncated.py --model Qwen/Qwen2.5-14B --suffix trunc_v1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache                                          # noqa: E402
from luq.config import Config                                  # noqa: E402
from attn_pool import PROMPT_REGIME                            # noqa: E402

QWEN = "Qwen/Qwen2.5-14B"
DATASETS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
# Middle layer per model, from the fixed rule ceil(n_layers/2) - 1. Never re-selected from results.
# ⚠️ gemma-2-9b-it has 42 layers -> 20. The discarded 2026-06 Gemma run used 21, which is the OTHER
# convention in this repo (03_probe.py's n_layers//2); it is not a precedent.
LAYER = {
    "Qwen/Qwen2.5-14B": 23,                  # 48 layers
    "meta-llama/Meta-Llama-3.1-8B": 15,      # 32
    "meta-llama/Llama-3.1-8B-Instruct": 15,  # 32
    "google/gemma-2-9b": 20,                 # 42 -- base, the population actually run
    "google/gemma-2-9b-it": 20,              # 42 -- instruct, WITHDRAWN 2026-08-15
    "Qwen/Qwen2.5-32B": 31,                  # 64
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN)
    ap.add_argument("--suffix", default="trunc_v1")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()
    layer = args.layer if args.layer is not None else LAYER[args.model]
    slug = cache._slug(args.model)

    print("=" * 104)
    print(f"SLICE CACHES -> truncated namespace   model={args.model}  layer={layer}")
    print("Causal model => a prefix's hidden states are UNCHANGED, so this is exact, not approximate.")
    print("The G+1 window is asserted per record; one mismatch aborts the dataset.")
    print("=" * 104)
    print(f"\n{'dataset':15s}{'rows':>7s}{'sliced':>8s}{'medKept':>9s}{'pertok MB':>11s}  status")

    for d in [x for x in args.datasets.split(",") if x]:
        base = PROMPT_REGIME.get(d, "")
        newns = f"{base}_{args.suffix}" if base else args.suffix
        src = Config(model_name=args.model, dataset=d, ood_setting="ID", prompt_regime=base)
        dst = Config(model_name=args.model, dataset=d, ood_setting="ID", prompt_regime=newns)
        key = cache.run_key(args.model, d, "ID")

        src_pt = Path(src.cache_dir) / "pertok" / f"{key}__L{layer}.npz"
        dst_rec = Path(dst.cache_dir) / "records" / f"{key}.jsonl"
        if not src_pt.exists():
            print(f"{d:15s}  no source pertok at {src_pt.name} — SKIPPED LOUDLY, cells stay ABSENT")
            continue
        if not dst_rec.exists():
            print(f"{d:15s}  no truncated records — run build_truncated_records.py first")
            continue

        trec = [json.loads(l) for l in open(dst_rec)]
        z = np.load(src_pt, allow_pickle=True)
        st = z["states"]
        if len(st) != len(trec):
            raise SystemExit(f"{d}: pertok has {len(st)} rows, truncated records {len(trec)} — "
                             f"refusing to align two different populations.")

        out, n_sliced, kept = [], 0, []
        for i, (s, r) in enumerate(zip(st, trec)):
            s = np.asarray(s, dtype=np.float32)
            g = len(r["token_logprobs"])
            # the window assertion, against the ORIGINAL generation length where the row was cut
            g_full = r.get("n_gen_raw", g)
            if len(s) != g_full + 1:
                raise SystemExit(f"{d}[{i}]: pertok rows {len(s)} != {g_full}+1 generated tokens. "
                                 f"The window assumption does not hold; refusing to slice.")
            if g < g_full:
                s = s[:g + 1]
                n_sliced += 1
                kept.append((g + 1) / (g_full + 1))
            out.append(s)

        dp = Path(dst.cache_dir) / "pertok"
        dp.mkdir(parents=True, exist_ok=True)
        target = dp / f"{key}__L{layer}.npz"
        if n_sliced == 0:
            # ⭐ NOTHING WAS CUT -> the truncated cache IS the canonical one. Symlink instead of
            # writing a multi-GB byte-for-byte duplicate (pubmed_qa alone is 4.5 GB). A symlink is
            # also self-documenting: it says "identical to v1", where a copy would merely look like
            # a separate artifact that happens to match.
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(src_pt.resolve())
            print(f"{d:15s}{len(out):>7d}{0:>8d}{100.0:>8.1f}%{'symlink':>10s}   identical to v1")
            fp = Path(dst.cache_dir) / "features"
            fp.mkdir(parents=True, exist_ok=True)
            src_f = Path(src.cache_dir) / "features" / f"{key}__saplma.npz"
            tgt_f = fp / f"{key}__saplma.npz"
            if src_f.exists():
                if tgt_f.exists() or tgt_f.is_symlink():
                    tgt_f.unlink()
                tgt_f.symlink_to(src_f.resolve())
            continue
        arr = np.empty(len(out), dtype=object)
        for i, s in enumerate(out):
            arr[i] = s
        # savez, NOT savez_compressed. These arrays are multi-GB and compression is the single
        # slowest step -- the first attempt was killed mid-write on pubmed_qa after 6 minutes and
        # 3.5 GB. The cache is regenerable scratch on a 15 TB volume, so disk is the cheap resource
        # here and wall-clock is the expensive one.
        np.savez(target, states=arr, layer=layer)

        # SAPLMA = mean-pool over the retained window, by the same definition 01e_repool uses.
        feats = np.stack([s.mean(axis=0) for s in out]).astype(np.float32)
        fp = Path(dst.cache_dir) / "features"
        fp.mkdir(parents=True, exist_ok=True)
        np.savez(fp / f"{key}__saplma.npz", feats=feats[:, None, :], layer=layer)

        mb = target.stat().st_size / 1e6
        med = 100 * float(np.median(kept)) if kept else 100.0
        print(f"{d:15s}{len(out):>7d}{n_sliced:>8d}{med:>8.1f}%{mb:>10.1f}M  ok")

    print("\n⚠️ The SAPLMA feature written here is a SINGLE-LAYER array (layer %d), not the "
          "all-layer\n   cache the canonical namespace holds. Anything wanting another layer must "
          "re-extract." % layer)


if __name__ == "__main__":
    main()
