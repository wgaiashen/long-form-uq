"""G1 diagnostic: resolve the PHYSICAL hidden layer each pipeline probes, so we match
the Hidden Failures "middle" by block, not by copying an index number.

HF `output_hidden_states=True` returns num_hidden_layers+1 tensors per step: index 0 is
the embedding output, index k is the k-th decoder layer's output. The Hidden Failures
reference implementation selects `ceil(num_hidden/2)-1` and indexes that
(embedding-included) tuple directly; our pipeline caches all layers
and `03_probe.py` defaults to `n_layers//2` over the same tuple. For Llama-3.1-8B (32
layers) the reference = index 15, ours = 16 — one slot apart. The paper's figures say "layer 16",
which is the embedding-as-layer-1 name for HF index 15. The empirical {15,16} sweep is the
final arbiter; this script confirms the structure (and that our cached features really do
include the embedding row at index 0).

Two ways to run (CPU-only):
  # from cached features — confirms our extraction includes the embedding row:
  python scripts/checks/check_layer_indexing.py --model google/gemma-2-9b-it --dataset sciq
  # or compute directly for a layer count (Llama-3.1-8B = 32, Gemma-2-9b = 42):
  python scripts/checks/check_layer_indexing.py --n-hidden 32
  # no args -> prints both reference models.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402


def report(num_hidden, source):
    n_states = num_hidden + 1  # HF per-step tuple length, incl. embedding at index 0
    ours = n_states // 2
    ref_layer = math.ceil(num_hidden / 2) - 1
    print(f"\n[{source}] num_hidden_layers={num_hidden}, "
          f"hidden_states length={n_states} (index 0 = embedding)")
    print(f"  ours  (n_layers//2) -> hidden_states[{ours}]")
    print(f"  ref   (ceil(N/2)-1) -> hidden_states[{ref_layer}]")
    if ours == ref_layer:
        print("  MATCH: same physical hidden state.")
    else:
        print(f"  MISMATCH by {ours - ref_layer}: pass `--layer {ref_layer}` to 03_probe.py to match the reference implementation "
              f"(and sweep {ref_layer}/{ours} to confirm against Table 14).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--dataset")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--n-hidden", type=int, default=None,
                    help="num_hidden_layers to compute directly (skip the cache)")
    args = ap.parse_args()

    did_something = False
    if args.n_hidden is not None:
        report(args.n_hidden, f"computed N={args.n_hidden}")
        did_something = True
    if args.model and args.dataset:
        cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
        key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
        feats = cache.load_features(cfg.cache_dir, key, method="saplma")  # (n, n_layers, hidden)
        n_states = feats.shape[1]
        print(f"\ncached features {key}__saplma: shape {tuple(feats.shape)} "
              f"-> hidden_states length {n_states}")
        report(n_states - 1, f"cached {args.model}")
        did_something = True
    if not did_something:
        report(32, "Llama-3.1-8B (reference)")
        report(42, "Gemma-2-9b (reference)")


if __name__ == "__main__":
    main()
