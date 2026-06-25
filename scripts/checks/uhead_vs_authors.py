"""Verify OUR uhead glue against the AUTHORS' own inference path (needs a GPU).

We use the authors' PRETRAINED head through their OWN `luh` package, so there is no head
to reproduce — but the thin glue that turns the head's per-token output into one score
(build the combined ids, set context_lengths, slice the generated span, mean-of-sigmoid)
is OUR code (src/luq/features/uhead.py). This check confirms that glue produces the same
number as running the authors' real pipeline:

    authors:  CalculatorInferLuh.infer_cached(...)  ->  per-token uncertainty_logits
              LuhClaimEstimator(reduce_type="mean")  ->  expit(logits).mean()
    ours:     uhead.uhead_instance_score(...)        ->  sigmoid(logits[span]).mean()

on the same cached records. If they agree to tolerance, the uhead row can be trusted (not
just "looks right by inspection"). Per-record, batch size 1 — same as how we score.

    python scripts/checks/uhead_vs_authors.py --dataset sciq --ood ID \
        --model google/gemma-2-9b-it --n 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache, generate  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import uhead  # noqa: E402

from luh.calculator_infer_luh import CalculatorInferLuh  # noqa: E402
from luh.disk_cache import DiskCache  # noqa: E402
from lm_polygraph.utils.model import WhiteboxModel  # noqa: E402

SCRATCH = "/vol/gpudata/gs925-msc_project/tmp/uhead_verify_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--n", type=int, default=10, help="how many records to cross-check")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max allowed |ours - authors| per record")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)
    # First N records with a non-empty generation (our score path needs gen tokens).
    records = [r for r in records if r.get("gen_token_ids")][: args.n]
    if not records:
        sys.exit("no records with generated tokens to check")

    # Same load as 01d_uhead: eager backend (the head reads attentions), bf16 Gemma.
    model, tok = generate.load_model(cfg.model_name, attn_implementation="eager")
    head = uhead.load_uhead(model)

    # Wrap our HF model/tokenizer in the lm-polygraph Model the calculator expects, and
    # build the authors' calculator around the SAME head object we use.
    wb = WhiteboxModel(model, tok, model_path=cfg.model_name)
    calc = CalculatorInferLuh(head, predict_token_uncertainties=True,
                              generations_cache_dir=None, device=str(model.device))

    # The authors' infer_cached reads the generation tokens from a DiskCache keyed by the
    # prompt text; pre-populate it with our cached gen_token_ids so both paths score the
    # SAME generation.
    Path(SCRATCH).mkdir(parents=True, exist_ok=True)
    dc = DiskCache(str(Path(SCRATCH) / f"{key}.sqlite"))

    reduce_mean = lambda x: float(expit(np.asarray(x, dtype=np.float64)).mean())

    print(f"{'idx':>6} {'ours':>10} {'authors':>10} {'|diff|':>10}  tok_match")
    max_diff = 0.0
    n_tok_mismatch = 0
    for r in records:
        text = r["prompt"]
        dc.get(text, (lambda g: (lambda: list(g)))(r["gen_token_ids"]))  # store gen tokens

        ours = uhead.uhead_instance_score(model, tok, head, r)

        out = calc.infer_cached(dc, [text], wb, max_new_tokens=len(r["gen_token_ids"]))
        ulogits = out["uncertainty_logits"][0]            # per-token logits for the gen span
        authors = reduce_mean(ulogits)

        # Did the authors' re-tokenisation of the prompt match our cached prompt tokens? If
        # not, a score gap could be a tokenisation artifact rather than a glue bug — flag it.
        retok = wb.tokenize([text])["input_ids"][0].tolist()
        tok_match = (retok == r["prompt_token_ids"])
        n_tok_mismatch += (not tok_match)

        diff = abs(ours - authors)
        max_diff = max(max_diff, diff)
        print(f"{r['idx']:>6} {ours:>10.6f} {authors:>10.6f} {diff:>10.2e}  {tok_match}")

    print(f"\nmax |ours - authors| over {len(records)} records: {max_diff:.2e} (tol {args.tol:.0e})")
    if n_tok_mismatch:
        print(f"NOTE: {n_tok_mismatch} record(s) had a prompt re-tokenisation mismatch — a gap "
              "on those reflects tokenisation, not the scoring glue.")
    if max_diff <= args.tol:
        print("PASS: our uhead glue reproduces the authors' inference path. uhead row can be trusted.")
    else:
        sys.exit(f"FAIL: max diff {max_diff:.2e} exceeds tol {args.tol:.0e} — investigate the glue "
                 "before trusting the uhead numbers.")


if __name__ == "__main__":
    main()
