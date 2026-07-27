"""GPU step: cache per-token hidden states over the SAPLMA window, for aggregation experiments.

The aggregation methods (per-token / per-sentence / attention-pooling) need the INDIVIDUAL answer-
token hidden states, not the mean-pooled vector. This replays the cached [prompt + gen] tokens
(teacher-forced, like 01e_repool), takes the exact SAPLMA window -- the last-prompt position P-1
plus all answer tokens, slice [P-1:P+G] -- at one layer, and caches those per-token states.

It does NOT touch the cached SAPLMA features. As a sanity check it prints the PRR of mean-pooling
the per-token states it just cached: that must match the cached SAPLMA feature's PRR at the same
layer (otherwise the window is wrong, which is the bug that made the earlier per-token cache useless).

    python scripts/01h_pertoken.py --model meta-llama/Meta-Llama-3.1-8B --dataset sciq --layer 15

Writes cache/pertok/<key>__L<layer>.npz with states (object array of (n_tokens, hidden) fp16),
idx (record indices), layer. Records carry already-truncated gens (pubmed D1), so no re-truncation.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, generate, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402

_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--layer", type=int, default=15,
                    help="hidden layer to cache per-token (default 15, the verified middle)")
    # MUST match the dtype/attn the cached SAPLMA feature was extracted with, or the hidden states
    # differ and the per-token mean will not equal the feature. The Llama keystone used fp32 + eager
    # (slurm/llama_keystone.sbatch, slurm/repool_fix.sbatch), so those are the defaults here.
    ap.add_argument("--dtype", default="fp32", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--attn", default="eager", choices=["auto", "eager", "sdpa"])
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag (must match the one used by 01_extract).")
    ap.add_argument("--label-field", default="correctness",
                    help="record field the pertok sanity gate scores on (correctness | factuality | "
                         "consistency). Non-correctness sets (factscore/expertqa) MUST pass this.")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    dtype = None if args.dtype == "auto" else _DTYPE[args.dtype]
    attn = None if args.attn == "auto" else args.attn
    model, tok = generate.load_model(cfg.model_name, attn_implementation=attn, dtype=dtype)
    L = args.layer
    n_layers = model.config.num_hidden_layers + 1
    if not 0 <= L < n_layers:
        sys.exit(f"--layer {L} out of range 0..{n_layers - 1}")

    pertok, pertok_idx, means = [], [], []
    for i, r in enumerate(records):
        p_ids = list(r["prompt_token_ids"])
        g_ids = list(r["gen_token_ids"])     # already D1-truncated in the records where applicable
        P, G = len(p_ids), len(g_ids)
        states = generate.recompute_states(model, tok, p_ids + g_ids, [L])  # [(P+G, hidden)]
        lo, hi = P - 1, P + G                # last-prompt + all answer tokens, matches SAPLMA
        # .copy() is essential: tensor.numpy() shares memory with the torch tensor, which is
        # freed when `states` is reassigned next iteration, so a stored view would dangle into
        # reused memory and the cache would be silently corrupt (the in-memory mean, computed now,
        # would still look fine). Keep float32 (fp16 rounding of these large-magnitude states also
        # degrades the raw-feature SAPLMA signal).
        arr = states[0][lo:hi].numpy().copy()
        pertok.append(arr)
        pertok_idx.append(r.get("idx", i))
        means.append(arr.mean(axis=0))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(records)}", flush=True)

    pdir = Path(cfg.cache_dir) / "pertok"
    pdir.mkdir(parents=True, exist_ok=True)
    ppath = pdir / f"{key}__L{L}.npz"
    # Atomic write: save to a temp file then os.replace onto the final path, so a kill or an
    # out-of-quota error mid-save can never leave a corrupt file under the canonical name (that
    # exact failure once left a truncated npz that only the downstream reload gate caught). Pass an
    # open handle, not a path -- np.savez_compressed would otherwise append ".npz" to a ".tmp" name.
    tmp_path = ppath.with_name(ppath.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()  # clear a stale temp left by a previously killed run
    try:
        with open(tmp_path, "wb") as fh:
            np.savez_compressed(fh, states=np.array(pertok, dtype=object),
                                idx=np.array(pertok_idx), layer=L)
        os.replace(tmp_path, ppath)
    except BaseException:
        # Remove the partial temp on any failure (e.g. running out of disk quota mid-write) so it
        # does not leak gigabytes and push the project over its CephFS byte quota. BaseException so a
        # KeyboardInterrupt/SystemExit during the write also cleans up (a hard SIGKILL cannot be
        # caught, but the stale-temp unlink above clears that on the next run).
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    print(f"cached per-token (L{L}) -> {ppath}")

    # Sanity on the PERSISTED cache (reload it, do not trust the in-memory copy): mean-pool of the
    # reloaded per-token states must reproduce the cached SAPLMA feature's PRR. Reloading is what
    # catches a corrupt write (e.g. a dangling numpy view) that an in-memory check would miss.
    split = np.array([r["split"] for r in records])
    y = np.array([r[args.label_field] for r in records], dtype=float)
    tr, te = split == "train", split == "test"
    z = np.load(ppath, allow_pickle=True)
    # Align POSITIONALLY: the cache is written in record order and the record "idx" field is NOT
    # unique (train and test share idx 0..N), so a {idx: states} map would drop half the rows.
    st = z["states"]
    assert len(st) == len(records), "per-token cache length does not match records"
    Xmean = np.stack([np.asarray(st[k], np.float32).mean(axis=0) for k in range(len(records))])
    clf = probe.train_probe_mlp(Xmean[tr], y[tr])
    prr_pertok = results.prr(y[te], probe.uncertainty(clf, Xmean[te]))
    feat = cache.load_features(cfg.cache_dir, key, "saplma")
    clf2 = probe.train_probe_mlp(feat[tr, L, :], y[tr])
    prr_feat = results.prr(y[te], probe.uncertainty(clf2, feat[te, L, :]))
    flag = "OK" if abs(prr_pertok - prr_feat) < 0.03 else "MISMATCH -- cache is wrong"
    print(f"SANITY reloaded per-token mean-pool PRR {prr_pertok:.3f}  vs  cached SAPLMA L{L} PRR "
          f"{prr_feat:.3f}  {flag}")


if __name__ == "__main__":
    main()
