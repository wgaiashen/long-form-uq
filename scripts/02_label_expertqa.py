"""ExpertQA three-state factuality labelling — the "ship now, partial label" decision (2026-07-06).

Produces, per record, the PARTIAL faithfulness-to-gold label from the three-state judge:
  - faithfulness : SUPPORTED / (SUPPORTED + CONTRADICTED) over COVERED claims; None if all-uncovered
                   (no covered claims to score) -> that instance has no faithfulness signal.
  - uncovered    : fraction of the answer's substantive claims the gold+citations do not address.
  - coherent     : False if the generation collapsed (word-salad/code/etc).
Downgraded claim (stamped): the label is blind to ~58% of the output (the uncovered fraction), so
ExpertQA leans on its Role-C domain-shift role; retrieval-based coverage extension (option C) is
deferred. Judge = gpt-5-mini (validated vs gpt-5: r=0.86 / MAD 0.084 on well-defined cases).

DISTRUST rule (uniform, per the stamped decision): a detector-SEVERE generation OR a judge
coherent=false verdict => faithfulness=0.0 (a distrust label, KEPT not dropped). SEVERE is not sent
to the judge (saves the call); coherent=false is forced to 0.0 after the call.

Writes an EXPLICIT label field (`faithfulness`, stamped `faithfulness_model`) — never the shared
`correctness` field (which is last-labeller-wins; the probe selects --label-field faithfulness).
Resumable: skips records already stamped with faithfulness_model.

Run (needs OPENAI_API_KEY + internet; login node / VM, no GPU):
    source /vol/gpudata/gs925-msc_project/.openai_key
    python scripts/02_label_expertqa.py --prompt-regime expertqa_rp12 --judge gpt-5-mini
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/ (for the gate import)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, degeneracy                       # noqa: E402
from luq.config import Config                            # noqa: E402
from luq.labels import expertqa_judge_draft as judge     # noqa: E402
from luq.labels import llm_judge                          # noqa: E402
from checks.expertqa_label_gate import load_eval_with_evidence  # noqa: E402  (shared evidence loader)


def label_records(records, raw, judge_model, save_cb, save_every=25):
    """Label every unlabelled record in place; call save_cb() periodically for crash-safety."""
    n_new = 0
    for i, r in enumerate(records):
        if r.get("faithfulness_model"):
            continue                                    # already labelled (resume)
        if degeneracy.is_severe(r["gen_text"]):
            # distrust label, not judged
            r.update(faithfulness=0.0, uncovered=0.0, coherent=False, faithfulness_quarantined=True)
        else:
            src = raw[r["idx"]]
            q = r["prompt"].split("Question:")[-1].split("\nAnswer:")[0].strip()
            reference = judge.build_reference(src["gold"], src["evidence"], max_ev_chars=4000)
            parsed = judge.parse(llm_judge._gpt_response(judge.fill(q, reference, r["gen_text"]), judge_model))
            if parsed is None:
                r.update(faithfulness=None, uncovered=None, coherent=None, faithfulness_parse_fail=True)
            else:
                if not parsed["coherent"]:              # distrust: derailed marginal survivor
                    parsed["faithfulness"], parsed["uncovered"] = 0.0, 0.0
                r.update(faithfulness=parsed["faithfulness"], uncovered=parsed["uncovered"],
                         coherent=parsed["coherent"], faithfulness_quarantined=False)
        r["faithfulness_model"] = judge_model            # provenance stamp (also the resume marker)
        n_new += 1
        if n_new % save_every == 0:
            save_cb()
            print(f"labelled {n_new} new (at {i + 1}/{len(records)})", flush=True)
    return n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--prompt-regime", default="expertqa_rp12")
    ap.add_argument("--judge", default="gpt-5-mini")
    ap.add_argument("--limit", type=int, default=None, help="label only the first N (for a mechanics test)")
    args = ap.parse_args()

    cfg = Config(dataset="expertqa", ood_setting=args.ood, model_name=args.model,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(args.model, "expertqa", args.ood)
    records = cache.load_records(cfg.cache_dir, key)
    raw = load_eval_with_evidence()
    assert len(raw) >= max(r["idx"] for r in records) + 1, "raw pool / record idx misaligned"
    todo = records if args.limit is None else records[:args.limit]
    print(f"labelling {len(todo)}/{len(records)} records with {args.judge} "
          f"(already done: {sum(1 for r in todo if r.get('faithfulness_model'))})", flush=True)

    def save():
        cache.save_records(records, cfg.cache_dir, key)

    n_new = label_records(todo, raw, args.judge, save)
    save()
    # summary
    lab = [r for r in todo if r.get("faithfulness_model")]
    fdef = [r["faithfulness"] for r in lab if r.get("faithfulness") is not None]
    n_alluncov = sum(1 for r in lab if r.get("faithfulness") is None and not r.get("faithfulness_parse_fail"))
    n_quar = sum(1 for r in lab if r.get("faithfulness_quarantined"))
    import numpy as np
    print(f"\nDONE: {n_new} newly labelled -> {cache.records_path(cfg.cache_dir, key)}")
    print(f"  faithfulness defined: {len(fdef)} (mean {np.mean(fdef):.2f})" if fdef else "  no defined faithfulness")
    print(f"  all-uncovered (no faithfulness signal): {n_alluncov} | distrust-quarantined: {n_quar}")


if __name__ == "__main__":
    main()
