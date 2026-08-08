"""GPU step: generate, then cache the Tier-1 records and Tier-2 pooled features.

Run under Slurm (see slurm/extract.sbatch). Download the model on the login node
first, because compute nodes have no internet.

    python scripts/01_extract.py --dataset sciq --ood ID
"""
import argparse
import random
import sys
from pathlib import Path

import torch

# Make `src/` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import answer_span, cache, data, generate  # noqa: E402
from luq.config import Config  # noqa: E402

# Map the --dtype flag to a torch dtype. "auto" -> None lets load_model pick its
# per-model default (bf16 for Gemma, fp16 otherwise).
_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"],
                    help="model load dtype. ⚠️ auto = load_model's per-model default, which is bf16 for "
                         "Gemma and **fp16 for everything else** — including Qwen. Every canonical "
                         "Llama long-eval cache was extracted fp32, and token logprobs ARE the msp_min "
                         "signal, so omitting this silently produces a cache in a different precision "
                         "from the one it will be compared against. PASS IT EXPLICITLY: fp32 matches "
                         "the keystone runs. See --require-explicit-dtype.")
    ap.add_argument("--require-explicit-dtype", action="store_true", default=None,
                    help="refuse to run if --dtype was left at auto. On by default for any model that "
                         "is NOT the Llama keystone, because 'auto' is only harmless where its result "
                         "happens to match what the existing caches used. Pass --no-require-explicit-dtype "
                         "to override deliberately.")
    ap.add_argument("--no-require-explicit-dtype", dest="require_explicit_dtype",
                    action="store_false", help=argparse.SUPPRESS)
    ap.add_argument("--attn", default="auto", choices=["auto", "eager", "sdpa"],
                    help="attention backend. auto = load_model's default (eager for "
                         "Gemma, HF default otherwise). Use eager to match Joe's runs.")
    ap.add_argument("--truncate-long", action="store_true",
                    help="ALSO truncate long-form generations at the first newline (D1). "
                         "Matches Joe's generate_until=['\\n'] for every dataset; off by "
                         "default since our convention leaves long-form untruncated. Use for "
                         "the pubmed_qa keystone reproduction.")
    ap.add_argument("--sample-n", type=int, default=None,
                    help="generate a RANDOM subsample of N examples per split, instead of --limit's "
                         "first N. ⚠️ --limit is a HEAD SLICE, not a sample: on expertqa the first 200 "
                         "rows have gold p90 450 against 349 for the full set, outside the [316,376] "
                         "range of random 200-row draws. Any statistic read off a --limit run (length, "
                         "degeneracy, capping) is therefore biased. Use this for anything measured.")
    ap.add_argument("--sample-seed", type=int, default=1,
                    help="seed for --sample-n. The chosen indices are stored in each record's `idx`, "
                         "so the sample is auditable and reproducible after the fact.")
    ap.add_argument("--limit", type=int, default=None,
                    help="optional cap on #examples for a quick run")
    ap.add_argument("--max-new-tokens-cap", type=int, default=None,
                    help="override the safety ceiling on generation length (default 128 from "
                         "Config). REQUIRED for datasets whose MAX_NEW_TOKENS exceeds 128, e.g. "
                         "expertqa (384) — else the budget is silently clipped to 128.")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="REPLACE this dataset's data.MAX_NEW_TOKENS budget for this run. Distinct from "
                         "--max-new-tokens-cap, which is a safety CEILING and can therefore only ever "
                         "LOWER the budget (the effective value is min(table, cap)) -- so the cap alone "
                         "cannot raise samsum 56 -> 96 for the v2 pilot. Use this instead of editing the "
                         "table: data.MAX_NEW_TOKENS is also what truncation_confound.py uses to decide "
                         "which v1 rows were capped, so mutating it would silently rewrite the v1 "
                         "truncation analysis. ALWAYS pair with a fresh --prompt-regime.")
    ap.add_argument("--truncate-answer-span", action="store_true",
                    help="LONG-FORM sibling of --truncate-long. Cut the generation at the point the "
                         "model stops answering and starts inventing a fresh Question:/Answer: pair "
                         "(luq.answer_span's per-dataset rules). Applied BEFORE the logprobs and the "
                         "hidden-state pooling, so record, MSP floors, features and label all describe "
                         "the same text. On med_quad's 768-token generations this takes the fabricated-"
                         "continuation rate from 92.6%% to 2.3%% with ZERO rows emptied. Off by default.")
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag for one ProbeDrift prompt set. Empty = the "
                         "frozen original cache; use e.g. 'pdnew' for the updated ProbeDrift "
                         "(different prompts) so the two never share records/features.")
    ap.add_argument("--repetition-penalty", type=float, default=None,
                    help="OPT-IN decoding penalty on repeated tokens (HF default 1.0 = off). Set "
                         "e.g. 1.3 for open-ended prompts where the base model loops (ExpertQA). "
                         "Leave unset for the frozen runs so generation is byte-identical.")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=None,
                    help="OPT-IN: forbid repeating any n-gram of this size (HF default 0 = off). "
                         "Use with --repetition-penalty to kill loops; leave unset for frozen runs.")
    args = ap.parse_args()

    # Validate the argument combination BEFORE anything expensive: loading fp32 Qwen-14B is ~59GB and
    # several minutes, and a config error should not cost that.
    if args.sample_n is not None and args.limit is not None:
        raise SystemExit("--sample-n and --limit are both subsetting rules; pass one, not both. "
                         "--limit takes the FIRST n (a head slice); --sample-n takes a random n.")

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    if args.max_new_tokens_cap is not None:
        cfg.max_new_tokens_cap = args.max_new_tokens_cap
    # Validate the budget override BEFORE loading an 8B model -- a config error should cost a second,
    # not a GPU allocation and several minutes of weight loading.
    # ⚠️ --truncate-answer-span only does anything for datasets answer_span has a RULE for. For any
    # other name it returns "no-cut" silently, so the run would generate uncut while appearing to have
    # been cut -- and the whole point of the flag is that the record, logprobs, features and label
    # describe the same text. Fail here, before the model loads, rather than produce a plausible cache.
    if args.truncate_answer_span and args.dataset not in answer_span.DATASETS_WITH_RULES:
        raise SystemExit(f"--truncate-answer-span: no cut rule for {args.dataset!r}. answer_span has "
                         f"rules for {sorted(answer_span.DATASETS_WITH_RULES)}; for anything else it "
                         "returns 'no-cut' SILENTLY, which would look like a successful cut run.")
    if args.max_new_tokens is not None:
        if not args.prompt_regime:
            raise SystemExit("--max-new-tokens changes the generations, so it MUST be paired with a fresh "
                             "--prompt-regime; refusing to write non-default-budget records into the "
                             "default cache namespace alongside the frozen v1 records.")
        if args.max_new_tokens > cfg.max_new_tokens_cap:
            raise SystemExit(f"--max-new-tokens {args.max_new_tokens} exceeds the safety ceiling "
                             f"--max-new-tokens-cap {cfg.max_new_tokens_cap} and would be silently "
                             "clipped. Raise the ceiling explicitly rather than generating at a budget "
                             "you did not ask for.")
    train_ds, eval_ds = data.load(cfg.dataset, cfg.ood_setting)

    # ⚠️ THE fp16 TRAP (guard added 2026-08-08). `--dtype auto` resolves in generate.load_model to
    # bf16 for Gemma and **fp16 for every other model**. Every canonical Llama long-eval cache was
    # extracted with an explicit `--dtype fp32`, so `auto` was never exercised there -- but for a NEW
    # model it silently produces a cache in a different precision from the one it will be compared
    # against, and token logprobs are the msp_min signal. Crash instead of guessing.
    _KEYSTONE = "meta-llama/Meta-Llama-3.1-8B"
    require_explicit = (args.require_explicit_dtype if args.require_explicit_dtype is not None
                        else cfg.model_name != _KEYSTONE)
    if require_explicit and args.dtype == "auto":
        raise SystemExit(
            f"--dtype was left at 'auto' for {cfg.model_name}, which resolves to fp16.\n"
            f"The Llama keystone caches are fp32 + eager, so an auto-dtype cache is NOT comparable "
            f"to them. Pass --dtype fp32 --attn eager (what every canonical run used), or pass\n"
            f"--no-require-explicit-dtype if you genuinely intend a different precision.")

    # auto -> None so load_model keeps its per-model defaults; otherwise override.
    dtype = None if args.dtype == "auto" else _DTYPE[args.dtype]
    attn = None if args.attn == "auto" else args.attn
    model, tok = generate.load_model(cfg.model_name, attn_implementation=attn, dtype=dtype)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)

    # Guard against silently extending a cache built from different prompts (e.g. a
    # ProbeDrift change). Stamp the hash of the exact prompts+targets. If a stored hash
    # exists and differs, stop loudly instead of mixing two prompt sets in one cache.
    digest = cache.prompt_hash(list(train_ds.x) + list(eval_ds.x),
                               list(train_ds.y) + list(eval_ds.y))
    stored = cache.load_prompt_hash(cfg.cache_dir, key)
    if stored is not None and stored != digest:
        sys.exit(f"PROMPT MISMATCH for {key} in {cfg.cache_dir}:\n"
                 f"  cached prompts hash {stored}\n  current prompts hash {digest}\n"
                 f"The prompts changed under this cache. Use a fresh --prompt-regime "
                 f"or clear this namespace; do not mix prompt sets.")
    cache.save_prompt_hash(digest, cfg.cache_dir, key)

    # Record WHICH probe_drift produced those prompts. The hash guard above is necessary but not
    # sufficient: it only compares a cache against its OWN stored hash, so a fresh --prompt-regime
    # starts empty and nothing compares it against the v1 cache it will be read beside. That is
    # exactly how the xsum probe (2026-08-01) ended up varying prompt, examples and budget together
    # while looking like a budget-only change -- 0 of 400 prompts shared with its own v1 cache.
    # `probe_drift` is an editable install, so which checkout answers is invisible at the call site.
    # Stamping it makes the question answerable later and ACROSS namespaces, where the hash cannot
    # help. scripts/checks/source_provenance.py reads these, and reconstructs the answer for caches
    # written before this existed.
    prov = cache.source_provenance()
    why = cache.provenance_mismatch(cache.load_source_provenance(cfg.cache_dir, key), prov)
    if why:
        sys.exit(f"SOURCE MISMATCH for {key} in {cfg.cache_dir}:\n  {why}\n"
                 f"A different probe_drift built this cache. Use a fresh --prompt-regime or clear "
                 f"this namespace; do not mix libraries in one cache.")
    cache.save_source_provenance(prov, cfg.cache_dir, key)
    print(f"probe_drift: {prov['path']} "
          f"(dataset_configs sha256 {str(prov['dataset_configs_sha256'])[:16]})", flush=True)

    # Few-shot short-form QA: the answer ends at the first newline; after that the
    # model just imitates the prompt format. Long-form output keeps its newlines by
    # default, UNLESS --truncate-long is set (Joe's generate_until=['\n'] for the
    # pubmed_qa keystone reproduction; see D1 in the analysis).
    truncate = (cfg.dataset in data.SHORT_FORM) or args.truncate_long

    # Resume: at 7-9B a job can hit the Slurm time limit before finishing, so we
    # reload whatever a previous run already cached and skip those examples instead
    # of regenerating them. `pooled_list` holds one (n_layers, hidden) array per
    # example, in the SAME order as `records`, so Tier 1 and Tier 2 stay aligned;
    # 03_probe.py separates train/test by each record's split tag.
    records, pooled_list = cache.load_checkpoint(cfg.cache_dir, key)
    done = {(r["split"], r["idx"]) for r in records}
    if done:
        print(f"resuming: {len(done)} examples already cached", flush=True)

    # Checkpoint every CKPT_EVERY new examples. A kill loses at most this many.
    CKPT_EVERY = 200
    n_new = 0
    # --max-new-tokens REPLACES the table value (it may raise it); --max-new-tokens-cap remains a pure
    # ceiling applied afterwards. Print the resolved budget so the log records what was actually used --
    # a silently-clipped budget is the exact failure this pair of flags exists to make visible.
    _table = data.MAX_NEW_TOKENS[cfg.dataset]
    _want = args.max_new_tokens if args.max_new_tokens is not None else _table
    budget = min(_want, cfg.max_new_tokens_cap)
    if args.max_new_tokens is not None:
        print(f"BUDGET OVERRIDE: {cfg.dataset} table={_table} -> effective={budget} "
              f"(ceiling={cfg.max_new_tokens_cap}, regime={cfg.prompt_regime!r})", flush=True)
    for split, ds in [("train", train_ds), ("test", eval_ds)]:
        # Random subsample. Selected up front over the whole split so the draw does not depend on how
        # far a resumed run got, which keeps a killed-and-restarted job on the SAME sample.
        keep = None
        if args.sample_n is not None:
            n_total = len(ds)
            k = min(args.sample_n, n_total)
            keep = set(random.Random(args.sample_seed).sample(range(n_total), k))
            print(f"SAMPLE: {split} drawing {k}/{n_total} at random (seed={args.sample_seed}); "
                  f"selected idx are recorded per row", flush=True)
        for idx, batch in enumerate(ds):
            if keep is not None and idx not in keep:
                continue
            # The updated ProbeDrift yields (x, y); the old one yielded (x, y, mnt). Take
            # the first two either way, and use our own per-dataset budget (data.MAX_NEW_TOKENS).
            xb, yb = batch[0], batch[1]
            if args.limit is not None and idx >= args.limit:
                break
            if (split, idx) in done:
                continue  # already cached by a previous run
            prompt, target = xb[0], yb[0]  # batch_size=1: unwrap the lists

            record, pooled = generate.generate(model, tok, prompt, budget,
                                               truncate_at_newline=truncate,
                                               truncate_answer_span=(cfg.dataset if args.truncate_answer_span else None),
                                               repetition_penalty=args.repetition_penalty,
                                               no_repeat_ngram_size=args.no_repeat_ngram_size)
            record |= {"idx": idx, "split": split, "target": target}
            records.append(record)
            pooled_list.append(pooled.numpy())  # float32 array, (n_layers, hidden)
            n_new += 1

            if n_new % CKPT_EVERY == 0:
                cache.save_checkpoint(records, pooled_list, cfg.cache_dir, key)
                print(f"[{split}] {idx} done (checkpointed {len(records)})", flush=True)
            elif idx % 10 == 0:
                # flush=True: Slurm buffers stdout, so unflushed prints make a
                # healthy job look hung.
                print(f"[{split}] {idx} done", flush=True)

    records_path, features_path = cache.save_checkpoint(
        records, pooled_list, cfg.cache_dir, key)
    print(f"saved {len(records)} records  ({n_new} new this run) -> {records_path}")
    print(f"saved features ({len(pooled_list)}, ...) -> {features_path}")


if __name__ == "__main__":
    main()
