"""Label the cached records for correctness.

Short-form (sciq, trivia_qa, qa): string match, no model, runs anywhere.
Long-form (pubmed_qa, xsum, cnn_dailymail): the LLM judge, LOGIN NODE only.

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
    reproduces the newline-collapsed judge input (faithful Hidden Failures repro).
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


def _stamped_judges(cache_dir, key, field):
    """The set of judge models already stamped on these records for `field`.

    Records carry `<field>_model` provenance (written by judge_into). Returns a set, because more than
    one value means the file is ALREADY mixed and that is itself a finding.

    An ABSENT stamp is not evidence that nothing was judged — sciq/trivia were judged and then had the
    judge label promoted into `correctness` without carrying the stamp across (see the audit note further
    down this file). So absence is reported as unknown, never as "safe to use any judge".
    """
    try:
        recs = cache.load_records(cache_dir, key)
    except Exception:
        return set(), 0
    stamps = {r.get(f"{field}_model") for r in recs if r.get(f"{field}_model")}
    n_labelled = sum(1 for r in recs if r.get(field) is not None)
    return stamps, n_labelled


def _resolve_judge(args, cfg, key, field):
    """Decide which judge model to use, and REFUSE rather than silently pick a different one.

    The failure this prevents (caught on DoC, 2026-08-03, before any money was spent): `--judge` used to
    default to the pinned GPT-5, while v1 med_quad is stamped `gpt-5-mini` on all 1800 rows. Labelling a
    regenerated arm without remembering `--judge gpt-5-mini` would have produced a £-paid label set from a
    DIFFERENT judge and confounded the pre-registered v1-vs-v2 comparison — with nothing in the output
    saying so. "Never mix judges within one comparison" was a rule that lived only in a person's memory,
    and a flag you have to remember is a defect, not a safeguard.

    Resolution order:
      1. The judge already used on THESE records (resuming a partial run).
      2. When --prompt-regime is set, the judge used on the v1 CANONICAL records for the same dataset --
         because a regenerated arm exists to be compared against v1, so it must share v1's yardstick.
      3. Only if nothing has ever been judged: the pinned default.
    An explicit --judge that contradicts 1 or 2 is refused unless --allow-judge-mismatch.
    """
    own, own_n = _stamped_judges(cfg.cache_dir, key, field)

    # The v1 counterpart matters only for a namespaced regime; for v1 itself, `own` already IS it.
    v1, v1_n = (set(), 0)
    if cfg.prompt_regime:
        v1_cfg = Config(model_name=cfg.model_name, dataset=cfg.dataset,
                        ood_setting=cfg.ood_setting, prompt_regime="")
        v1, v1_n = _stamped_judges(v1_cfg.cache_dir, key, field)

    prior = own | v1
    if len(prior) > 1:
        sys.exit(f"REFUSING TO LABEL: the records for {cfg.dataset} already carry MORE THAN ONE judge "
                 f"stamp on `{field}`: {sorted(prior)}. That comparison is already mixed and adding a "
                 "third judge cannot fix it. Inspect the cache before labelling anything.")

    if prior:
        inherited = next(iter(prior))
        src = "these records" if own else f"the v1 records for {cfg.dataset}"
        if args.judge and args.judge != inherited:
            if not args.allow_judge_mismatch:
                sys.exit(
                    f"REFUSING TO LABEL: you asked for judge `{args.judge}`, but {src} are labelled by "
                    f"`{inherited}` ({own_n or v1_n} rows). Two £-paid label sets from different judges "
                    "would confound the comparison, and a probe can learn a judge's biases. Either drop "
                    f"--judge (it will inherit `{inherited}`), or pass --allow-judge-mismatch if you "
                    "genuinely intend a judge-vs-judge study into a separate field.")
            print(f"JUDGE MISMATCH ACCEPTED: labelling with `{args.judge}` while {src} carry "
                  f"`{inherited}`. You passed --allow-judge-mismatch.", flush=True)
            return args.judge
        print(f"judge = `{inherited}` (inherited from {src}; not the pinned default). "
              f"{own_n or v1_n} rows already carry this stamp.", flush=True)
        return inherited

    # Nothing stamped anywhere. For a namespaced regime that is suspicious enough to stop on: the arm
    # exists to be compared against a v1 set, so silently inventing a yardstick is the whole bug.
    if cfg.prompt_regime and args.judge is None:
        # Two different situations reach here and they deserve different explanations.
        if v1_n:
            # A v1 counterpart EXISTS and is labelled, but carries no stamp. This is the dangerous one:
            # there IS a yardstick to match and we cannot read what it was.
            sys.exit(
                f"REFUSING TO LABEL: --prompt-regime {cfg.prompt_regime} was passed, and the v1 records "
                f"for {cfg.dataset} have {v1_n} labelled rows on `{field}` but NO `{field}_model` stamp. "
                "Absence of a stamp is NOT evidence of no judge (sciq/trivia were judged and promoted "
                "without carrying the stamp across). A regenerated arm must share v1's judge or the "
                "comparison is confounded, so check what actually labelled v1 and pass --judge explicitly.")
        sys.exit(
            f"REFUSING TO LABEL: --prompt-regime {cfg.prompt_regime} was passed and nothing has been "
            f"judged on `{field}` here, nor is there a v1 counterpart for {cfg.dataset} to inherit from. "
            "If this is a REGENERATED ARM, the missing v1 set is the real problem -- find it first. If it "
            "is simply a namespaced primary dataset (asqa/expertqa/factscore), pass --judge explicitly; "
            "naming the judge in the command is a one-word cost and it puts the yardstick on the record.")

    chosen = args.judge or llm_judge.MODEL
    print(f"judge = `{chosen}` ({'explicit' if args.judge else 'pinned default'}; nothing judged yet "
          f"for `{field}`).", flush=True)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--judge", default=None,
                    help="judge model. DEFAULT IS NOT A FIXED MODEL: it is whichever judge already "
                         "labelled this dataset (read from the `<field>_model` stamps on the existing "
                         "records), falling back to the pinned "
                         f"{llm_judge.MODEL} only when nothing has been judged yet. Pass explicitly to "
                         "override; a value that contradicts the existing stamps is refused unless "
                         "--allow-judge-mismatch. See _resolve_judge().")
    ap.add_argument("--allow-judge-mismatch", action="store_true",
                    help="Deliberately label with a DIFFERENT judge from the one already on these "
                         "records. Only legitimate when building a judge-vs-judge comparison into a "
                         "separate field. Never use it to 'get the run going'.")
    ap.add_argument("--judge-short-form", action="store_true",
                    help="ALSO run the LLM judge on a short-form dataset, into a "
                         "SEPARATE `correctness_judge` field (string-match stays the "
                         "canonical `correctness`, so downstream stages are unchanged). "
                         "For comparing the two labels by hand. LOGIN NODE only; "
                         "costs money. Resumable — Ctrl-C and rerun to continue.")
    ap.add_argument("--strip-newlines", action="store_true",
                    help="Collapse all newlines out of the model answer before judging, "
                         "matching the Hidden Failures judge input "
                         "(collect_llm_judge_inputs.py:92). Use ONLY for faithful "
                         "reproduction of those labels (e.g. the Llama keystone).")
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

    # READ THIS BEFORE CONCLUDING WHICH LABEL A DATASET ACTUALLY USES (audit, 2026-08-02).
    # This routing has now twice been read as "sciq/trivia are string-matched, so their labels are
    # overlap labels". THAT IS NOT WHAT IS IN THE CACHE. On the shipped records, sciq's and trivia's
    # `correctness` is BYTE-IDENTICAL to `correctness_judge` (max|diff| 0.0) and carries graded values
    # (0.1, 0.2, 0.3 ...) that string-match, which only ever emits 0.0/1.0, cannot produce. The judge
    # labels were promoted into `correctness` afterwards and the `correctness_model` stamp was not
    # carried across -- so the ABSENT STAMP IS NOT EVIDENCE OF STRING MATCHING. Check the values.
    # `correctness_strmatch` is the unused secondary field; nothing trains on it.
    # ALL TEN datasets are judge-labelled: sciq/trivia by gpt-5, the rest by gpt-5-mini.
    # For the record, on the two sets where overlap would have been most defensible it still disagreed
    # with the judge on 9.5% (sciq) / 6.0% (trivia) of rows, always penalising correct paraphrase.
    if cfg.dataset in data.SHORT_FORM:
        # String match: does the gold answer (or any alias) appear in the output?
        # Written to `correctness` HERE, but see the note above: on the shipped caches it was later
        # superseded by the judge. Deterministic, free, no judge bias.
        for r in records:
            r["correctness"] = string_match.match(r["gen_text"], r["target"])
        if args.judge_short_form:
            # Also score with the judge into a separate field so the two labels
            # coexist for the disagreement study. Does NOT touch `correctness`.
            judge = _resolve_judge(args, cfg, key, "correctness_judge")
            judge_into(records, "correctness_judge", cfg, judge,
                       strip_newlines=args.strip_newlines)
    else:
        # Long-form: the LLM judge is the only label. The judge is RESOLVED, not defaulted --
        # see _resolve_judge() for why a forgettable flag was the wrong safeguard.
        judge = _resolve_judge(args, cfg, key, "correctness")
        judge_into(records, "correctness", cfg, judge,
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
