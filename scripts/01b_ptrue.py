"""GPU step: P(True) verdict-position features from the cached records.

P(True) needs no new generation. It replays the token IDs that 01_extract already
saved (Tier 1), appends "Is this true?", and reads the hidden state at the verdict
position. So run 01_extract FIRST, then this over the same run.

    python scripts/01b_ptrue.py --dataset sciq --ood ID

The output is a Tier-2 feature file keyed by method=`--name` (default "ptrue"), index-aligned
with the records (same order), so 03_probe.py and 04_eval.py treat it exactly like SAPLMA.

The verdict wording (`--wording`) and the cache feature-set name (`--name`) are flags so a new
wording can be extracted WITHOUT overwriting the old one — e.g. extract the unified wording under
`--name ptrue_accurate` while the old "true"-wording `ptrue` features stay on disk for comparison.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Make `src/` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, generate  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import ptrue  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--wording", default=ptrue.PTRUE_SUFFIX,
                    help="P(True) verification suffix (recorded hyperparameter)")
    ap.add_argument("--name", default="ptrue",
                    help="cache feature-set name (use a distinct name like 'ptrue_accurate' to "
                         "keep an existing 'ptrue' feature set instead of overwriting it)")
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag (must match the one used by 01_extract).")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)

    # Tier-1 records from 01_extract: prompt/gen token IDs are all we need.
    records = cache.load_records(cfg.cache_dir, key)
    model, tok = generate.load_model(cfg.model_name)
    # All layers, so layer choice stays a free sweep at probe time (same as SAPLMA).
    layers = list(range(model.config.num_hidden_layers + 1))

    # Process EVERY record, in order, so the feature rows stay aligned with the
    # records (03_probe asserts the two are the same length). A quick dev run is
    # controlled upstream by 01_extract --limit, which makes this records file small.
    print(f"wording: {args.wording!r} -> cache name {args.name!r}", flush=True)
    vectors = []
    for i, r in enumerate(records):
        vectors.append(ptrue.ptrue_vector(model, tok, r, layers, suffix=args.wording))
        if i % 10 == 0:
            print(f"{i}/{len(records)} done", flush=True)

    feats = np.stack(vectors)  # (n_examples, n_layers, hidden)
    path = cache.save_features(feats, cfg.cache_dir, key, method=args.name)
    print(f"saved features {feats.shape} -> {path}")


if __name__ == "__main__":
    main()
