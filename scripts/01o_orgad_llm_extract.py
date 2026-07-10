"""Cache the MODEL'S OWN extracted short answer per record (Joe-style Orgad, gpt-5-mini).

For each record we ask gpt-5-mini to extract the model's own short answer from its generation (Joe's
prompt), giving a span for correct AND incorrect rows (unlike gold-string-match, which only fires when
the answer is right). Cached to cache/orgad_llm/<key>.json (idx -> extracted string). Resumable: skips
records already extracted, checkpoints every 25. LOGIN NODE (API); costs gpt-5-mini £. SHORT-FORM only.

    export OPENAI_API_KEY=...   # source .openai_key
    python scripts/01o_orgad_llm_extract.py --dataset sciq
    python scripts/01o_orgad_llm_extract.py --dataset trivia_qa
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import orgad_llm  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"


def question_of(prompt):
    """The last question (short-form few-shot format), for the extraction prompt context."""
    for marker in ("\nQuestion:", "\n\nQuestion:", "\nQ:", "\n\nQ:"):
        idx = prompt.rfind(marker)
        if idx != -1:
            seg = prompt[idx + len(marker):]
            for stop in ("\nAnswer:", "\nA:", "\n"):
                j = seg.find(stop)
                if j != -1:
                    seg = seg[:j]
            return seg.strip()
    return prompt[-300:].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--extract-model", default="gpt-5-mini")
    ap.add_argument("--workers", type=int, default=12, help="concurrent API calls (I/O-bound)")
    args = ap.parse_args()

    cfg = Config(model_name=MODEL, dataset=args.dataset, ood_setting="ID")
    recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, args.dataset, "ID"))
    out_dir = ROOT / "cache" / "orgad_llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{cache._slug(MODEL)}__{args.dataset}__ID.json"
    done = json.loads(out.read_text()) if out.exists() else {}   # idx(str) -> extracted

    tok = AutoTokenizer.from_pretrained(MODEL)
    todo = [r for r in recs if f"{r['split']}:{r['idx']}" not in done]
    print(f"[{args.dataset}] {len(done)} cached, {len(todo)} to extract, {args.workers} workers", flush=True)

    def work(r):
        # split:idx is unique; idx alone collides across train/test
        q = None if orgad_llm.is_summarisation(args.dataset) else question_of(r["prompt"])
        val = orgad_llm.extract_important(args.dataset, q, r["gen_text"], model=args.extract_model)
        return f"{r['split']}:{r['idx']}", val

    n_new = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:      # API calls are I/O-bound
        futs = [pool.submit(work, r) for r in todo]
        for fut in as_completed(futs):                             # collect single-threaded -> no cache race
            key, val = fut.result()
            done[key] = val
            n_new += 1
            if n_new % 50 == 0:
                out.write_text(json.dumps(done))
                print(f"extracted {n_new}/{len(todo)}", flush=True)
    out.write_text(json.dumps(done))

    # report located rate (span found in the generation) -- this is the number that should NO LONGER
    # track correctness (the leak fix)
    n_loc = 0
    for r in recs:
        ex = done.get(f"{r['split']}:{r['idx']}", "NO ANSWER")
        _, found = orgad_llm.locate_important_rows(tok, r["gen_token_ids"], ex)
        n_loc += found
    print(f"\n[{args.dataset}] extracted {len(done)}/{len(recs)} | located span {n_loc} "
          f"({100*n_loc/len(recs):.0f}%) | wrote {out}", flush=True)


if __name__ == "__main__":
    main()
