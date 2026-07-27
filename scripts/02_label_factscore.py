"""FActScore-Bio factuality labelling — three-state judge vs the entity's Wikipedia article.

Same three-state judge as ExpertQA (commensurable): factuality = SUPPORTED/(SUPPORTED+CONTRADICTED) =
factual PRECISION, reference = the person's enwiki page (fetched by entity title). Writes an explicit
`factuality` field (stamped `factuality_model`), NEVER the shared `correctness`. Resumable (skips rows
already stamped). The SEVERE-degeneracy distrust rule matches ExpertQA (factuality=0.0, not judged).

Run on DoC (has the master enwiki db + API/internet; no GPU needed for labelling):
    source /vol/gpudata/gs925-msc_project/.openai_key            # OPENAI_API_KEY
    export FACTSCORE_DIR=/vol/gpudata/gs925-msc_project/factscore_data
    python scripts/02_label_factscore.py --prompt-regime factscore_rp12 --judge gpt-5-mini
Validate first: run with --limit 10 and eyeball, then spot-check gpt-5-mini vs gpt-5 (as for ExpertQA).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, degeneracy, factscore          # noqa: E402
from luq.config import Config                           # noqa: E402
from luq.labels import factscore_judge as judge         # noqa: E402
from luq.labels import llm_judge                         # noqa: E402


def label_records(records, entities, judge_model, save_cb, save_every=25):
    """Label every unlabelled record in place (three-state judge vs the enwiki reference)."""
    n_new = 0
    for i, r in enumerate(records):
        if r.get("factuality_model"):
            continue                                      # already labelled (resume)
        if degeneracy.is_severe(r["gen_text"]):
            # Distrust label, NOT judged (same as ExpertQA). uncovered=None (not 0.0): the judge was never
            # called, so a 0.0 coverage number would be a fabricated measurement about an unlooked-at row.
            r.update(factuality=0.0, uncovered=None, coherent=False, factuality_quarantined=True)
        else:
            entity = entities[r["idx"]]["entity"]         # idx aligns to file order (eval-only design)
            reference = factscore.wiki_reference(entity)
            if reference is None:                          # title not in the enwiki db -> no reference, no score
                r.update(factuality=None, uncovered=None, coherent=None, factuality_no_reference=True)
                r["factuality_model"] = judge_model
                n_new += 1
                continue
            parsed = judge.parse(llm_judge._gpt_response(
                judge.fill(entity, reference, r["gen_text"]), judge_model))
            if parsed is None:
                r.update(factuality=None, uncovered=None, coherent=None, factuality_parse_fail=True)
            else:
                if not parsed["coherent"]:                 # derailed marginal survivor -> distrust
                    parsed["factuality"], parsed["uncovered"] = 0.0, None
                r.update(factuality=parsed["factuality"], uncovered=parsed["uncovered"],
                         coherent=parsed["coherent"], factuality_quarantined=False)
        r["factuality_model"] = judge_model              # provenance stamp + resume marker
        n_new += 1
        if n_new % save_every == 0:
            save_cb()
            print(f"labelled {n_new} new (at {i + 1}/{len(records)})", flush=True)
    return n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--prompt-regime", default="factscore_rp12")
    ap.add_argument("--judge", default="gpt-5-mini")
    ap.add_argument("--limit", type=int, default=None, help="label only the first N (mechanics/validation)")
    args = ap.parse_args()

    cfg = Config(dataset="factscore", ood_setting=args.ood, model_name=args.model,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(args.model, "factscore", args.ood)
    records = cache.load_records(cfg.cache_dir, key)
    entities = factscore.load_records()                    # 500, file order -> idx-aligned to records
    assert len(entities) >= max(r["idx"] for r in records) + 1, "entities / record idx misaligned"
    todo = records if args.limit is None else records[:args.limit]
    print(f"labelling {len(todo)}/{len(records)} factscore records with {args.judge} "
          f"(already done: {sum(1 for r in todo if r.get('factuality_model'))})", flush=True)

    def save():
        cache.save_records(records, cfg.cache_dir, key)

    n_new = label_records(todo, entities, args.judge, save)
    save()
    lab = [r for r in todo if r.get("factuality_model")]
    fdef = [r["factuality"] for r in lab if r.get("factuality") is not None]
    n_noref = sum(1 for r in lab if r.get("factuality_no_reference"))
    n_quar = sum(1 for r in lab if r.get("factuality_quarantined"))
    import numpy as np
    print(f"\nDONE: {n_new} newly labelled -> {cache.records_path(cfg.cache_dir, key)}")
    print(f"  factuality defined: {len(fdef)}" + (f" (mean {np.mean(fdef):.3f})" if fdef else ""))
    print(f"  no-reference (title absent from enwiki): {n_noref} | distrust-quarantined: {n_quar}")


if __name__ == "__main__":
    main()
