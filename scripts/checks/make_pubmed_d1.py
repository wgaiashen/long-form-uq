"""Prep pubmed for the D1 (newline-stop) reproduction. CPU-only, no model.

D1 = match Joe's generate_until=["\\n"] on long-form: cut the cached pubmed generations at
the first newline. That changes the answer text, so this script ALSO clears the gpt-5 labels
(02_label must re-judge the truncated text). The untruncated originals are backed up to
*.preD1 so our own "long-form never truncated" convention is recoverable.

Run order for the full pubmed D1: this -> 01e_repool (re-pool truncated gens; NO --truncate-long,
records are already cut) -> 02_label --judge gpt-5 (re-label, the spend) -> probe -> eval.
The pubmed_d1.sh orchestrator chains these.

    python scripts/checks/make_pubmed_d1.py --model meta-llama/Meta-Llama-3.1-8B --dataset pubmed_qa --ood ID
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402

_LABEL_FIELDS = ("correctness", "correctness_model", "correctness_judge",
                 "correctness_judge_model", "correctness_strmatch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--dataset", default="pubmed_qa")
    ap.add_argument("--ood", default="ID")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    rp = cache.records_path(cfg.cache_dir, key)
    fp = cache.features_path(cfg.cache_dir, key, "saplma")

    # Back up the untruncated originals (idempotent: never clobber an existing backup).
    for p in (rp, fp):
        bak = p.with_suffix(p.suffix + ".preD1")
        if p.exists() and not bak.exists():
            shutil.copy2(p, bak)
            print(f"backed up {p.name} -> {bak.name}")

    tok = AutoTokenizer.from_pretrained(args.model)
    recs = cache.load_records(cfg.cache_dir, key)
    n_cut = 0
    for r in recs:
        g = list(r["gen_token_ids"])
        cut = len(g)
        for i, tid in enumerate(g):
            if "\n" in tok.decode([tid]):
                cut = max(i, 1)  # keep at least one token (mirrors generate.py)
                break
        if cut < len(g):
            n_cut += 1
        r["gen_token_ids"] = g[:cut]
        r["gen_text"] = tok.decode(g[:cut], skip_special_tokens=True)
        if isinstance(r.get("token_logprobs"), list):
            r["token_logprobs"] = r["token_logprobs"][:cut]
        for f in _LABEL_FIELDS:  # clear so 02_label re-judges the truncated text
            r.pop(f, None)

    cache.save_records(recs, cfg.cache_dir, key)
    print(f"truncated {n_cut}/{len(recs)} pubmed gens at first newline; labels cleared; saved {rp.name}")
    print("restore the untruncated version any time with: "
          f"cp {rp.name}.preD1 {rp.name} && cp {fp.name}.preD1 {fp.name}")


if __name__ == "__main__":
    main()
