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

# TWO AXES ARE SWEPT, because two different explanations are on the table and only a measurement
# separates them.
#
#   The NUMERICAL axis: the dtype and attention backend of the forward pass. An earlier diagnostic
#   concluded that the cached values came from a forward configuration the current fp32-plus-eager
#   recomputation does not reproduce, and named dtype or the attention backend as the likely
#   difference. It did not test them.
#
#   The PROCESSOR axis: logits processors active at generation time. What `generate` returns in
#   `scores`, and therefore what was cached, is the distribution AFTER any processor runs, while a
#   plain forward reproduces it before. The same earlier diagnostic rejected a repetition penalty on
#   the grounds that the diverging positions were not enriched for repeated tokens. That argument is
#   weaker than it looks: a penalty renormalises the whole distribution, so it shifts every position
#   rather than only the repeats, and which positions then exceed a tolerance is a question of
#   magnitude, not of whether that particular token was a repeat. Replaying the processor settles it
#   directly, which is what this does.
#
# The numerical axis is the outer loop because changing it means reloading the model; the processor
# axis is inner and costs one vector operation per position.
NUMERICAL = [
    ("fp32/eager", "fp32", "eager"),
    ("fp16/eager", "fp16", "eager"),
    ("bf16/eager", "bf16", "eager"),
    ("fp32/sdpa", "fp32", "sdpa"),
    ("fp16/sdpa", "fp16", "sdpa"),
]

PROCESSORS = [
    ("none", dict()),
    ("rep1.2", dict(repetition_penalty=1.2)),
    ("rep1.2+ngram3", dict(repetition_penalty=1.2, no_repeat_ngram_size=3)),
    ("ngram3", dict(no_repeat_ngram_size=3)),
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
    ap.add_argument("--max-memory", default="",
                    help="per-device ceilings when sharding, e.g. '0=17GiB,1=17GiB'. Needed with "
                         "--device-map auto: on its own that fills the first card and a float32 8B "
                         "model then runs out of memory on a 24 GB one rather than sharding.")
    args = ap.parse_args()
    max_memory = None
    if args.max_memory:
        max_memory = {}
        for item in args.max_memory.split(","):
            k, v = item.split("=")
            max_memory[int(k.strip())] = v.strip()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)
    n = min(args.n, len(records))
    print(f"{args.model} | {args.dataset} (regime '{args.prompt_regime or 'canonical'}') | "
          f"{len(records)} records, checking {n}", flush=True)

    rows = []
    for i in range(n):
        r = records[i]
        p_ids, g_ids = list(r["prompt_token_ids"]), list(r["gen_token_ids"])
        cached = np.asarray(r["token_logprobs"], dtype=float)
        if len(g_ids) == 0 or len(cached) != len(g_ids):
            print(f"  row {i}: {len(g_ids)} generated tokens but {len(cached)} cached "
                  f"log-probabilities -> skipped")
            continue
        rows.append((p_ids, g_ids, cached))
    if not rows:
        sys.exit("no comparable rows")
    n_pos = sum(len(g) for _, g, _ in rows)

    results = {}
    for num_name, dtype, attn in NUMERICAL:
        try:
            model, tok = generate.load_model(args.model, attn_implementation=attn,
                                             dtype=_DTYPE[dtype], device_map=args.device_map,
                                             max_memory=max_memory)
            model.eval()
        except Exception as e:
            print(f"  {num_name}: could not load ({type(e).__name__}) -> NOT TESTED", flush=True)
            for proc_name, _ in PROCESSORS:
                results[(num_name, proc_name)] = None
            continue

        worst = {pn: 0.0 for pn, _ in PROCESSORS}
        over = {pn: 0 for pn, _ in PROCESSORS}
        for p_ids, g_ids, cached in rows:
            P, G = len(p_ids), len(g_ids)
            ids = torch.tensor(p_ids + g_ids)[None].to(model.device)
            with torch.no_grad():
                logits = model(ids).logits[0].float().cpu()
            for proc_name, conf in PROCESSORS:
                procs = build_processors(conf)
                for t in range(G):
                    # The distribution that produced generated token t sits at position P-1+t, and a
                    # processor sees the context as it stood then: the prompt plus the tokens already
                    # generated. That growing context is the point; a processor applied to the final
                    # context would be a different function.
                    scores = logits[P - 1 + t][None].clone()
                    if len(procs):
                        ctx = torch.tensor(p_ids + g_ids[:t])[None]
                        scores = procs(ctx, scores)
                    lp = torch.log_softmax(scores.float(), dim=-1)[0, g_ids[t]].item()
                    d = abs(lp - cached[t])
                    worst[proc_name] = max(worst[proc_name], d)
                    if d > args.tol:
                        over[proc_name] += 1
        for proc_name, _ in PROCESSORS:
            results[(num_name, proc_name)] = (worst[proc_name], over[proc_name])
        best_here = min((worst[pn] for pn, _ in PROCESSORS))
        print(f"  {num_name}: best worst-case {best_here:.4e}", flush=True)
        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"\n=== {args.dataset} | {len(rows)} rows | {n_pos} positions ===")
    print(f"{'forward':<12} {'processors':<15} {'worst |d|':>12} {'over tol':>10}")
    best, best_key = None, None
    for num_name, _, _ in NUMERICAL:
        for proc_name, _ in PROCESSORS:
            v = results.get((num_name, proc_name))
            if v is None:
                print(f"{num_name:<12} {proc_name:<15} {'not tested':>12} {'-':>10}")
                continue
            w, o = v
            print(f"{num_name:<12} {proc_name:<15} {w:>12.4e} {o:>10}")
            if best is None or w < best:
                best, best_key = w, (num_name, proc_name)

    print(f"\ntolerance {args.tol:.1e}")
    if best is None:
        print("READING: nothing was tested. This is a failure of the job, not a finding.")
        return 2
    if best <= args.tol:
        print(f"READING: the cached log-probabilities ARE reproduced, by forward '{best_key[0]}' "
              f"with processors '{best_key[1]}'. Entropy for this dataset must be computed that way.")
        return 0
    print(f"READING: no combination reproduces the cached log-probabilities. The closest is "
          f"'{best_key[0]}' with '{best_key[1]}' at {best:.4e}, still above {args.tol:.1e}. "
          f"Neither the forward configuration nor a logits processor explains it on its own.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
