"""Which generation-time logits processors produced the cached token log-probabilities?

WHY THIS EXISTS
---------------
The probability-augmented distance hybrids need a mean token entropy, which is the entropy of the
model's full next-token distribution at each generated position. The Tier-1 record stores only the
chosen token's log-probability, so the distribution has to be recomputed, and
`scripts/01n_token_entropy.py` proves it recomputed the right positions by checking that the chosen
token's log-probability comes back the same.

That check passes on three datasets and fails on the rest, and the failures do not look like an
off-by-one: on one dataset the first positions agree to about 1e-5 and later ones differ by several
nats, on another every position differs from the first. Both are the signature of a DIFFERENT
DISTRIBUTION, not a misaligned window.

The likely reason is that generation applied logits processors. What a model's `generate` returns in
`scores`, and therefore what was cached, is the distribution AFTER those processors run, while a
plain teacher-forced forward reproduces the distribution before them. A repetition penalty reweights
every token already in the context, which includes the prompt, so it shifts the very first generated
position when the prompt is long and only later positions when it is short. A repeated-n-gram ban
cannot bite until enough tokens have been generated to form one.

This script does not assume which was used. It replays each candidate configuration through the
authors' own processor classes and reports which one, if any, reproduces the cached values. If none
does, that is the answer and it says so.

    python scripts/checks/m11_entropy_regime.py --dataset asqa --prompt-regime asqa_rp12 --n 6
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache, generate  # noqa: E402
from luq.config import Config  # noqa: E402

_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# Configurations seen in this project's generation jobs, plus the empty one. Named so the report
# reads as a statement about generation rather than as a list of numbers.
CANDIDATES = [
    ("none", dict()),
    ("rep1.2", dict(repetition_penalty=1.2)),
    ("rep1.2+ngram3", dict(repetition_penalty=1.2, no_repeat_ngram_size=3)),
    ("ngram3", dict(no_repeat_ngram_size=3)),
    ("rep1.3+ngram3", dict(repetition_penalty=1.3, no_repeat_ngram_size=3)),
]


def build_processors(cfg):
    """The processors HF's generate would install for this configuration, in its order."""
    from transformers import (LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
                              NoRepeatNGramLogitsProcessor)
    procs = LogitsProcessorList()
    if cfg.get("repetition_penalty"):
        procs.append(RepetitionPenaltyLogitsProcessor(penalty=float(cfg["repetition_penalty"])))
    if cfg.get("no_repeat_ngram_size"):
        procs.append(NoRepeatNGramLogitsProcessor(int(cfg["no_repeat_ngram_size"])))
    return procs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prompt-regime", default="")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tol", type=float, default=2e-2,
                    help="the tolerance the entropy extractor uses, unchanged")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--device-map", default="cuda")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)
    print(f"{args.model} | {args.dataset} (regime '{args.prompt_regime or 'canonical'}') | "
          f"{len(records)} records, checking {min(args.n, len(records))}", flush=True)

    model, tok = generate.load_model(args.model, attn_implementation=args.attn,
                                     dtype=_DTYPE[args.dtype], device_map=args.device_map)
    model.eval()

    worst = {name: 0.0 for name, _ in CANDIDATES}
    over = {name: 0 for name, _ in CANDIDATES}
    n_pos = 0

    for i in range(min(args.n, len(records))):
        r = records[i]
        p_ids, g_ids = list(r["prompt_token_ids"]), list(r["gen_token_ids"])
        cached = np.asarray(r["token_logprobs"], dtype=float)
        P, G = len(p_ids), len(g_ids)
        if G == 0 or len(cached) != G:
            print(f"  row {i}: G={G} but {len(cached)} cached logprobs -> skipped")
            continue
        ids = torch.tensor(p_ids + g_ids)[None].to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[0].float().cpu()

        for name, conf in CANDIDATES:
            procs = build_processors(conf)
            for t in range(G):
                # The distribution that produced generated token t sits at position P-1+t, and the
                # processors see the context as it stood then: the prompt plus the tokens already
                # generated. That growing context is the whole point; a processor applied to the
                # final context would be a different function.
                ctx = torch.tensor(p_ids + g_ids[:t])[None]
                scores = logits[P - 1 + t][None].clone()
                if len(procs):
                    scores = procs(ctx, scores)
                lp = torch.log_softmax(scores.float(), dim=-1)[0, g_ids[t]].item()
                d = abs(lp - cached[t])
                worst[name] = max(worst[name], d)
                if d > args.tol:
                    over[name] += 1
            if name == CANDIDATES[0][0]:
                n_pos += G
        print(f"  row {i}: G={G} | " + " | ".join(
            f"{name} {worst[name]:.3g}" for name, _ in CANDIDATES), flush=True)

    print(f"\n=== {args.dataset} | {n_pos} positions ===")
    print(f"{'configuration':<16} {'worst |d|':>12} {'positions over tol':>20}")
    for name, _ in CANDIDATES:
        print(f"{name:<16} {worst[name]:>12.4e} {over[name]:>20}")
    best = min(CANDIDATES, key=lambda c: worst[c[0]])[0]
    print(f"\ntolerance {args.tol:.1e}")
    if worst[best] <= args.tol:
        print(f"READING: the cached log-probabilities are reproduced by '{best}'. Entropy for this "
              f"dataset must be computed under that configuration, not under a plain forward.")
        return 0
    print("READING: no candidate configuration reproduces the cached log-probabilities. The cause is "
          "something other than a logits processor and must be found before any entropy is cached.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
