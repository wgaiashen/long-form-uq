#!/usr/bin/env python
"""Cache per-token hidden states at SEVERAL layers from a single forward per record.

Registered use: prereg/M9_layer_distance_sensitivity.md, the layer sensitivity for the supervised
distance family.

WHY THIS EXISTS
---------------
A teacher-forced forward already computes every hidden layer; `recompute_states` then selects the
ones asked for. Extracting eleven layers therefore costs ONE forward per record, not eleven, and the
only extra cost is moving and compressing the additional arrays. Calling the single-layer script
eleven times would repeat the forward eleven times for no benefit.

WHAT IT WRITES
--------------
Exactly the format the single-layer script writes, one file per layer:

    cache/<regime>/pertok/<slug>__<dataset>__ID__L<layer>.npz   with states, idx, layer

Because the layer is part of the filename, a layer that already exists is REFUSED rather than
overwritten unless --overwrite is given. The canonical middle-layer caches took GPU hours to build
and are the input to every published result in this project; silently replacing one would be
unrecoverable.

    python scripts/01p_pertoken_multi.py --model meta-llama/Meta-Llama-3.1-8B \
        --dataset pubmed_qa --layers 0,3,6,9,12,18,21,24,27,30
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache, generate                                                 # noqa: E402
from luq.config import Config                                                   # noqa: E402

_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--layers", required=True,
                    help="comma-separated hidden layer indices, e.g. 0,3,6,9,12,18,21,24,27,30")
    # MUST match the dtype/attn the existing caches were built with, or the states differ and the
    # layer-15 reproduction gate in the registration cannot pass. fp32 + eager, as everywhere here.
    ap.add_argument("--dtype", default="fp32", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--attn", default="eager", choices=["auto", "eager", "sdpa"])
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag (must match the one used by 01_extract)")
    ap.add_argument("--overwrite", action="store_true",
                    help="permit replacing a layer file that already exists. Off by default: the "
                         "canonical middle-layer caches are the input to every published result here")
    ap.add_argument("--device-map", default="cuda",
                    help="passed to from_pretrained. 'cuda' (default) = one GPU, unchanged. "
                         "'auto' shards the weights across the visible GPUs, which is how a model "
                         "too large for one card is run on several smaller ones. Use with "
                         "--max-memory.")
    ap.add_argument("--max-memory", default="",
                    help="force a real split, e.g. '0=20GiB,1=20GiB'. accelerate fills GPU 0 first, "
                         "so --device-map auto on its own can silently place every layer on one "
                         "card; when this is set the split is asserted after loading rather than "
                         "assumed.")
    args = ap.parse_args()
    # Sharding the weights across several cards changes where the weights live and nothing else.
    # The dtype, the sequence handling and the values computed are identical to a single-card run.
    max_memory = None
    if args.max_memory:
        max_memory = {}
        for item in args.max_memory.split(","):
            k, v = item.split("=")
            max_memory[int(k.strip())] = v.strip()

    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    if len(set(layers)) != len(layers):
        sys.exit(f"--layers has duplicates: {layers}")

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    pdir = Path(cfg.cache_dir) / "pertok"
    pdir.mkdir(parents=True, exist_ok=True)

    # THE OVERWRITE GUARD RUNS FIRST, before the records are read and long before the model is
    # loaded, so a mistaken layer list costs seconds rather than minutes of record loading and a GPU
    # allocation. Nothing above this point touches the caches.
    paths = {L: pdir / f"{key}__L{L}.npz" for L in layers}
    existing = [L for L in layers if paths[L].exists()]
    if existing and not args.overwrite:
        sys.exit(f"REFUSING TO OVERWRITE: layer(s) {existing} already cached for {args.dataset} at\n"
                 f"  {[str(paths[L]) for L in existing]}\n"
                 f"Drop them from --layers to extract only what is missing, or pass --overwrite if "
                 f"replacing them is genuinely what you want.")

    records = cache.load_records(cfg.cache_dir, key)

    model, tok = generate.load_model(
        cfg.model_name,
        attn_implementation=None if args.attn == "auto" else args.attn,
        dtype=None if args.dtype == "auto" else _DTYPE[args.dtype],
        device_map=args.device_map, max_memory=max_memory)
    model.eval()
    # Prove the shard actually happened. A silent single-card placement would run out of memory part
    # way through the dataset, after hours of work, rather than here.
    if args.device_map == "auto":
        placed = getattr(model, "hf_device_map", {})
        n_dev = len({v for v in placed.values() if isinstance(v, int)})
        print(f"device map: {n_dev} GPU(s) hold weights", flush=True)
        if max_memory is not None and n_dev < 2:
            sys.exit("--max-memory asked for a split but every layer landed on one device; "
                     "the memory ceiling was too high or only one GPU is visible.")

    n_layers = model.config.num_hidden_layers + 1
    bad = [L for L in layers if not 0 <= L < n_layers]
    if bad:
        sys.exit(f"--layers {bad} out of range 0..{n_layers - 1} for {args.model}")

    print(f"{args.model} | {args.dataset} | {len(records)} records | layers {layers}", flush=True)

    per = {L: [] for L in layers}
    idxs = []
    for i, r in enumerate(records):
        p_ids = list(r["prompt_token_ids"])
        g_ids = list(r["gen_token_ids"])
        P, G = len(p_ids), len(g_ids)
        states = generate.recompute_states(model, tok, p_ids + g_ids, layers)
        lo, hi = P - 1, P + G          # last prompt position + every generated position, as 01h does
        for k, L in enumerate(layers):
            # .copy() is essential: tensor.numpy() shares memory with the torch tensor, which is
            # freed when `states` is reassigned next iteration, so a stored view would dangle into
            # reused memory and the cache would be silently corrupt.
            per[L].append(states[k][lo:hi].numpy().copy())
        idxs.append(r.get("idx", i))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(records)}", flush=True)

    idx_arr = np.array(idxs)
    for L in layers:
        ppath = paths[L]
        # Atomic write, as in the single-layer script: a kill or an out-of-quota error mid-save must
        # never leave a truncated file under the canonical name.
        tmp_path = ppath.with_name(ppath.name + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            with open(tmp_path, "wb") as fh:
                np.savez_compressed(fh, states=np.array(per[L], dtype=object), idx=idx_arr, layer=L)
            os.replace(tmp_path, ppath)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        size_gb = ppath.stat().st_size / 2 ** 30
        print(f"cached per-token (L{L}) -> {ppath}  ({size_gb:.2f} GB)", flush=True)

    # Reload every file written and confirm it opens and has the expected row count. Presence and
    # mtime lie: a truncated npz raises BadZipFile only when something actually reads it.
    print("\nRELOAD CHECK (a corrupt write is invisible until something reads it)", flush=True)
    for L in layers:
        z = np.load(paths[L], allow_pickle=True)
        st = z["states"]
        ok = len(st) == len(records)
        print(f"  L{L:<3d} rows {len(st):5d} expected {len(records):5d}  "
              f"dim {np.asarray(st[0]).shape[-1]}  {'OK' if ok else 'ROW COUNT MISMATCH'}")
        if not ok:
            sys.exit(f"L{L}: reloaded cache has {len(st)} rows, records have {len(records)}.")


if __name__ == "__main__":
    main()
