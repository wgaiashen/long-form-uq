"""GPU step: unsupervised P(True) score from the cached records.

Unlike the P(True) probe (01b, which reads the verdict-position hidden state and trains a probe
on it), this reads the model's OWN yes/no answer: it appends the verification question, runs one
forward pass, and renormalises the next-token probability over the yes and no tokens. The score
needs no training and no labels, so it is an unsupervised baseline like MSP, but it does need a
GPU forward (the cached token logprobs do not cover the appended question).

It writes two fields back into each Tier-1 record, in place:
    ptrue_unsup        uncertainty = 1 - P(yes)/(P(yes)+P(no))   (higher = more uncertain)
    ptrue_unsup_mass   P(yes)+P(no)   (share of next-token mass on a yes/no verdict; a sanity
                       signal, not used by PRR)
plus ptrue_unsup_model for provenance. 04_eval picks ptrue_unsup up automatically, like MSP.

    python scripts/01g_ptrue_unsup.py --model meta-llama/Meta-Llama-3.1-8B --dataset sciq --ood ID

Resumable: rerun to continue (records already scored are skipped unless --overwrite).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, generate  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import ptrue  # noqa: E402

CHECKPOINT_EVERY = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--wording", default=ptrue.PTRUE_SUFFIX,
                    help="P(True) verification suffix (recorded hyperparameter)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rescore every record even if it already has ptrue_unsup")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    todo = [r for r in records
            if args.overwrite or not isinstance(r.get("ptrue_unsup"), (int, float))]
    if not todo:
        print(f"{args.dataset}: all {len(records)} records already have ptrue_unsup; nothing to do")
        return

    model, tok = generate.load_model(cfg.model_name)
    yes_ids, no_ids = ptrue.yes_no_token_ids(tok)
    print(f"wording: {args.wording!r}", flush=True)
    print(f"yes ids {yes_ids} -> {[tok.decode([i]) for i in yes_ids]}")
    print(f"no  ids {no_ids} -> {[tok.decode([i]) for i in no_ids]}")
    if not yes_ids or not no_ids:
        sys.exit("ERROR: no single-token yes or no surface form for this tokenizer; "
                 "the verdict cannot be read. Check the tokenizer before scoring.")

    done = 0
    masses = []
    for i, r in enumerate(records):
        if not (args.overwrite or not isinstance(r.get("ptrue_unsup"), (int, float))):
            continue
        conf, p_yes, p_no = ptrue.ptrue_unsup_confidence(
            model, tok, r, suffix=args.wording, yes_ids=yes_ids, no_ids=no_ids)
        r["ptrue_unsup"] = 1.0 - conf            # uncertainty: higher = model less sure it is accurate
        r["ptrue_unsup_mass"] = p_yes + p_no
        r["ptrue_unsup_model"] = cfg.model_name
        masses.append(p_yes + p_no)
        done += 1
        if done % CHECKPOINT_EVERY == 0:
            cache.save_records(records, cfg.cache_dir, key)
            print(f"{done}/{len(todo)} scored (checkpoint)", flush=True)

    cache.save_records(records, cfg.cache_dir, key)
    mean_mass = sum(masses) / len(masses) if masses else float("nan")
    print(f"done: scored {done} records, mean yes/no mass {mean_mass:.3f} -> saved records")


if __name__ == "__main__":
    main()
