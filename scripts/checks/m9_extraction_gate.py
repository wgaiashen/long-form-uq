"""Acceptance gate 1 of the layer sensitivity: does the multi-layer extraction reproduce layer 15?

Registered in prereg/M9_layer_distance_sensitivity.md section 4:

    The layer-15 output of the multi-layer extractor must reproduce the existing layer-15 cache
    exactly, or within the tolerance already established for these caches (about 1e-6 on the window
    mean). A multi-layer extractor that does not reproduce the single-layer one cannot be trusted on
    the ten layers that have nothing to check against.

WHY IT CANNOT BE READ OFF THE CACHE. The extraction deliberately does not write layer 15: that cache
is the input to every published result in this project and the extractor refuses to overwrite it. So
the gate has to recompute layer 15 through the same call path the ten new layers went through, and
compare against the file already on disk. Nothing is written.

WHAT IT IS ACTUALLY TESTING, which is more than the prereg anticipated. Two things differ between the
cached layer 15 and the ten new layers, and this gate is the only place they are measured together:

1. The GPU architecture. The new layers were produced on cards of a different generation from the one
   that produced the cached layer 15, because those were the only cards this queue could reach. The
   precedent in this project for a cache recomputed on a different card is agreement to 1.9e-6.
2. The unread vocabulary projection, now computed at one position rather than all of them. That was
   separately gated at exact equality on hidden states, so it should contribute nothing here, and if
   this gate fails that claim is the first thing to re-examine.

Pass the same dtype, attention backend and device settings the extraction used, or the comparison is
between two different things.

  python scripts/checks/m9_extraction_gate.py --dataset med_quad --prompt-regime cleanv2 --n 40
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from luq import cache, generate                                                  # noqa: E402
from luq.config import Config                                                    # noqa: E402

_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prompt-regime", default="")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--n", type=int, default=40, help="records to recompute and compare")
    ap.add_argument("--tol-mean", type=float, default=1e-6,
                    help="the bar the pre-registration names, on the window mean")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--skip-lm-head", action="store_true",
                    help="match the extraction, which computed the unread projection at one position")
    ap.add_argument("--device-map", default="cuda")
    ap.add_argument("--max-memory", default="")
    args = ap.parse_args()

    max_memory = None
    if args.max_memory:
        max_memory = {}
        for item in args.max_memory.split(","):
            k, v = item.split("=")
            max_memory[int(k.strip())] = v.strip()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    path = Path(cfg.cache_dir) / "pertok" / f"{key}__L{args.layer}.npz"
    if not path.exists():
        sys.exit(f"no cached layer {args.layer} at {path}; nothing to gate against")
    z = np.load(path, allow_pickle=True)
    keys = list(z.keys())
    cached = z["states"] if "states" in keys else z[keys[0]]
    cached_idx = z["idx"] if "idx" in keys else None

    records = cache.load_records(cfg.cache_dir, key)
    if len(records) != len(cached):
        sys.exit(f"cache holds {len(cached)} rows but the record file holds {len(records)}; "
                 f"these are different populations and must not be compared")

    model, tok = generate.load_model(cfg.model_name, attn_implementation=args.attn,
                                     dtype=_DTYPE[args.dtype],
                                     device_map=args.device_map, max_memory=max_memory)
    model.eval()
    if args.device_map == "auto":
        placed = getattr(model, "hf_device_map", {})
        print(f"device map: {len({v for v in placed.values() if isinstance(v, int)})} GPU(s)")
    print(f"{args.model} | {args.dataset} (regime '{args.prompt_regime or 'canonical'}') "
          f"| layer {args.layer} | comparing {min(args.n, len(records))} of {len(records)} rows "
          f"| skip_lm_head={args.skip_lm_head}", flush=True)

    worst_elem, worst_mean, n_bad, n_cmp = 0.0, 0.0, 0, 0
    for i in range(min(args.n, len(records))):
        r = records[i]
        p_ids, g_ids = list(r["prompt_token_ids"]), list(r["gen_token_ids"])
        P, G = len(p_ids), len(g_ids)
        with torch.no_grad():
            st = generate.recompute_states(model, tok, p_ids + g_ids, [args.layer],
                                           logits_to_keep=1 if args.skip_lm_head else None)
        got = st[0][P - 1:P + G].numpy()          # the window the extractor stores
        ref = np.asarray(cached[i])
        if got.shape != ref.shape:
            print(f"  row {i}: SHAPE {got.shape} vs cached {ref.shape}")
            n_bad += 1
            continue
        de = float(np.max(np.abs(got - ref)))
        dm = float(np.max(np.abs(got.mean(axis=0) - ref.mean(axis=0))))
        worst_elem, worst_mean = max(worst_elem, de), max(worst_mean, dm)
        n_cmp += 1
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}: worst elementwise {worst_elem:.3e}, worst window mean "
                  f"{worst_mean:.3e}", flush=True)

    if cached_idx is not None:
        print(f"cached idx array present, length {len(cached_idx)}")
    print(f"\ncompared {n_cmp} rows | worst elementwise |d| = {worst_elem:.6e} "
          f"| worst window-mean |d| = {worst_mean:.6e} | shape mismatches: {n_bad}")
    print(f"pre-registered bar: window mean within {args.tol_mean:.1e}")
    if n_bad == 0 and worst_mean <= args.tol_mean:
        print("EXTRACTION GATE: PASS")
        return 0
    print("EXTRACTION GATE: FAIL. The pre-registration says the experiment stops and the failure "
          "is reported instead of a result.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
