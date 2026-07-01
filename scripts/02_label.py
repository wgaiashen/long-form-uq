"""Label the cached records for correctness.

Short-form (sciq, trivia_qa, qa): string match, no model, runs anywhere.
Long-form (pubmed_qa, xsum, cnn_dailymail): Joe's LLM judge, LOGIN NODE only.

    python scripts/02_label.py --dataset sciq --ood ID

Optionally ALSO judge a short-form dataset (into a separate `correctness_judge`
field, leaving string-match as the canonical `correctness`) to compare the two
labels by hand — see scripts/checks/inspect_label_disagreement.py:

    python scripts/02_label.py --dataset sciq --ood ID --judge-short-form --judge gpt-5-mini
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, data  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.labels import llm_judge, string_match  # noqa: E402


def judge_into(records, field, cfg, judge_model, save_every=25, strip_newlines=False):
    """Score every record with the LLM judge, writing the result into record[field].

    LOGIN NODE only; needs OPENAI_API_KEY; each call costs money. Resumable: skip
    records already scored on THIS field, and checkpoint to disk every `save_every`
    new scores, so a crash or rate-limit partway through never re-spends on work
    already done (just rerun the command to pick up where it stopped). Each new
    label is stamped with `field`_model so we always know which judge produced it
    (the "never mix judges within one comparison" discipline). `strip_newlines`
    reproduces Joe's newline-collapsed judge input (faithful Hidden Failures repro).
    """
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    new_count = n_failed = 0
    for i, r in enumerate(records):
        if r.get(field) is not None:
            continue  # already judged on a previous (possibly partial) run
        r[field] = llm_judge.judge(r, cfg.dataset, model=judge_model,
                                   strip_newlines=strip_newlines)
        new_count += 1
        if r[field] is None:
            n_failed += 1
        else:
            r[f"{field}_model"] = judge_model  # provenance: which judge gave this label
        if new_count % save_every == 0:
            cache.save_records(records, cfg.cache_dir, key)  # checkpoint
            print(f"judged {new_count} new (at {i + 1}/{len(records)})", flush=True)
    if n_failed:
        print(f"WARNING: {n_failed} records returned no valid score (None)")
    return new_count


def _report(records, field, label):
    """Print how many records carry a numeric `field` label and their mean."""
    scores = [r[field] for r in records if isinstance(r.get(field), (int, float))]
    mean = sum(scores) / len(scores) if scores else float("nan")
    print(f"{label}: {len(scores)}/{len(records)} records, mean {mean:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--judge", default=llm_judge.MODEL,
                    help="judge model (default: pinned GPT-5; pass e.g. gpt-5-mini "
                         "for the cheaper judge). Used for long-form labels, and for "
                         "short-form only with --judge-short-form.")
    ap.add_argument("--judge-short-form", action="store_true",
                    help="ALSO run the LLM judge on a short-form dataset, into a "
                         "SEPARATE `correctness_judge` field (string-match stays the "
                         "canonical `correctness`, so downstream stages are unchanged). "
                         "For comparing the two labels by hand. LOGIN NODE only; "
                         "costs money. Resumable — Ctrl-C and rerun to continue.")
    ap.add_argument("--strip-newlines", action="store_true",
                    help="Collapse all newlines out of the model answer before judging, "
                         "matching Joe's Hidden Failures judge input "
                         "(collect_llm_judge_inputs.py:92). Use ONLY for faithful "
                         "reproduction of his labels (e.g. the Llama keystone).")
    ap.add_argument("--promote-judge", action="store_true",
                    help="STANDARDISE this short-form dataset onto the judge label: copy the "
                         "already-computed `correctness_judge` into the canonical `correctness` "
                         "(the field 03/04 read) and keep string-match as `correctness_strmatch`. "
                         "Use after --judge-short-form has finished EVERY record. No API calls — "
                         "this only rewrites label fields. Retrain the probe (03) afterwards, "
                         "since the cached probe was fit on the old label.")
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag (must match the one used by 01_extract).")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    if args.promote_judge:
        # GATE for the cross-task work: make the judge the canonical correctness for this
        # short-form dataset, so the whole OOD matrix runs on ONE label semantics (the
        # judge), not a string-match/judge mix. Non-destructive — the deterministic
        # string-match label is preserved in `correctness_strmatch` for the record.
        if cfg.dataset not in data.SHORT_FORM:
            sys.exit(f"--promote-judge is only for short-form datasets; long-form already "
                     f"uses the judge as `correctness`. {cfg.dataset} is not short-form.")
        missing = sum(1 for r in records if r.get("correctness_judge") is None)
        if missing:
            sys.exit(f"{missing}/{len(records)} records have no `correctness_judge` yet. Run "
                     f"`02_label.py --dataset {cfg.dataset} --ood {cfg.ood_setting} "
                     f"--judge-short-form` to completion first (it is resumable). Refusing to "
                     "promote a partial label — it would leave the dataset half string-match.")
        for r in records:
            # Keep the deterministic string-match label on record for the comparison, then
            # make the judge score the canonical label that 03_probe / 04_eval read.
            r["correctness_strmatch"] = string_match.match(r["gen_text"], r["target"])
            r["correctness"] = r["correctness_judge"]
        cache.save_records(records, cfg.cache_dir, key)
        _report(records, "correctness", "promoted: canonical = judge")
        _report(records, "correctness_strmatch", "string-match (kept for comparison)")
        return

    if cfg.dataset in data.SHORT_FORM:
        # String match: does the gold answer (or any alias) appear in the output?
        # This stays the canonical `correctness` for short-form: deterministic, free,
        # no judge bias — the label stages 03/04 read.
        for r in records:
            r["correctness"] = string_match.match(r["gen_text"], r["target"])
        if args.judge_short_form:
            # Also score with the judge into a separate field so the two labels
            # coexist for the disagreement study. Does NOT touch `correctness`.
            judge_into(records, "correctness_judge", cfg, args.judge,
                       strip_newlines=args.strip_newlines)
    else:
        # Long-form: Joe's LLM judge is the only label.
        judge_into(records, "correctness", cfg, args.judge,
                   strip_newlines=args.strip_newlines)

    # Re-save the records in place: the labels become part of the Tier-1 record, so
    # stages 03/04 read one file and labels can never drift out of line with their
    # examples. Relabelling is cheap for string match; for the judge this is the
    # final checkpoint that captures the tail since the last periodic save.
    cache.save_records(records, cfg.cache_dir, key)

    _report(records, "correctness", "labelled")
    if args.judge_short_form and cfg.dataset in data.SHORT_FORM:
        _report(records, "correctness_judge", "judge-labelled")


if __name__ == "__main__":
    main()
