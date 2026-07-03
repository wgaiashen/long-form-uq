"""ExpertQA Stage 0 (paper, no GPU, no spend): verify schema, confirm the token-length cap on the
real Llama tokenizer, freeze the prompt (show its hash), and write the deterministic pilot manifest.

The manifest (cache/expertqa/meta/pilot_manifest.jsonl) is the pilot's field/cluster metadata, in the
SAME order data.load('expertqa','pilot') emits prompts — so it joins to the Stage-1 records by idx and
Role C (domain-shift OOD) becomes a free re-slice later.

    python scripts/expertqa_stage0.py
"""
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, data, expertqa       # noqa: E402
from luq.config import Config               # noqa: E402


def main():
    recs = expertqa.load_records()
    print(f"usable ExpertQA records (factual filter): {len(recs)}")
    print("by cluster:", dict(Counter(r["cluster"] for r in recs)))
    print("top fields:", Counter(r["field"] for r in recs).most_common(6))

    # token-length cap check on the real Llama tokenizer (the one Stage-0 number needing confirmation)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    gl = np.array([len(tok(r["gold"], add_special_tokens=False).input_ids) for r in recs])
    print(f"\ngold-length tokens: p90 {np.percentile(gl,90):.0f}  p95 {np.percentile(gl,95):.0f}  "
          f"max {gl.max()}  | cap MAX_NEW_TOKENS={expertqa.MAX_NEW_TOKENS} covers {np.mean(gl<=expertqa.MAX_NEW_TOKENS):.1%}")

    # frozen prompt + its hash (the cache guard). Build via data.load so this IS what 01_extract sees.
    _, eval_ds = data.load("expertqa", "pilot")
    digest = cache.prompt_hash(list(eval_ds.x), list(eval_ds.y))
    print(f"\nfrozen prompt (repr): {expertqa.PROMPT!r}")
    print(f"pilot n={len(eval_ds.x)} (seed {expertqa.PILOT_SEED})  prompt_hash={digest}")
    print("example prompt:\n" + eval_ds.x[0][:220])

    # pilot stratification (from source_ids = 'field||cluster')
    fields = [s.split("||")[0] for s in eval_ds.source_ids]
    clusters = [s.split("||")[1] for s in eval_ds.source_ids]
    print("\npilot by cluster:", dict(Counter(clusters)))
    print("pilot distinct fields:", len(set(fields)))

    # write the manifest (idx-aligned to records the Stage-1 generation will produce)
    cfg = Config(model_name="meta-llama/Meta-Llama-3.1-8B", dataset="expertqa",
                 ood_setting="pilot", prompt_regime="expertqa")
    mdir = Path(cfg.cache_dir) / "meta"
    mdir.mkdir(parents=True, exist_ok=True)
    mpath = mdir / "pilot_manifest.jsonl"
    picked = expertqa.pilot_sample(expertqa.load_records(), n=expertqa.PILOT_N, seed=expertqa.PILOT_SEED)
    with open(mpath, "w") as f:
        for idx, r in enumerate(picked):
            f.write(json.dumps({"idx": idx, "field": r["field"], "cluster": r["cluster"],
                                "question_types": r["question_types"], "question": r["question"]}) + "\n")
    print(f"\nwrote pilot manifest ({len(picked)} rows) -> {mpath}")
    print("\nStage 0 done. Stage 1 (GPU, needs go-ahead):")
    print("  python scripts/01_extract.py --dataset expertqa --ood pilot "
          "--prompt-regime expertqa --max-new-tokens-cap 384")
    print("  # NO --truncate-long: ExpertQA is long-form, newlines are legitimate; truncating at the")
    print("  # first newline would corrupt multi-paragraph expert-style answers.")


if __name__ == "__main__":
    main()
