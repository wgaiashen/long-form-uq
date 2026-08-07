"""G1 — does fp32 multi-GPU SHARDING change the numbers?

WHY THIS EXISTS
---------------
The second model (`Qwen/Qwen2.5-14B`) must run fp32 + eager to match the Llama keystone, and fp32
14B is ~59GB — bigger than any single RCS card (L40S/A40 = 48GB). So it has to be SHARDED across
two GPUs with `device_map="auto"`. Before trusting that on a model whose numbers we do not yet
know, prove it on a model whose numbers we DO know: Llama-3.1-8B, against its frozen cache.

WHAT IS AND IS NOT BEING TESTED
-------------------------------
The question is narrow: *does splitting the layers across two devices perturb fp32 arithmetic?*
It is answered by running the SAME code path twice on the SAME node over the SAME inputs, once
single-GPU and once sharded, and diffing. Anything else is a confound:

  ⚠️ Do NOT compare a sharded TEACHER-FORCED pass against the cached AUTOREGRESSIVE logprobs and
     call the difference "sharding". The cache was written by generate() decoding one token at a
     time with a KV cache; a teacher-forced forward over the same ids is a DIFFERENT computation
     and is documented in this project to differ by ~1e-4 relative (the same AR-vs-TF gap that
     forced the `01e_repool` rule). That gap would swamp the ~1e-6 sharding signal and fail the
     gate for the wrong reason. Check C below MEASURES that gap separately so it cannot be
     mistaken for the thing under test.

  ⚠️ `device_map="auto"` ALONE DOES NOT SHARD. accelerate fills GPU 0 first, and a 32GB fp32
     Llama-8B fits entirely on one 48GB card — so the "sharded" arm would be the single-GPU arm
     and the test would pass vacuously while proving nothing. We force a real split with
     max_memory and then ASSERT the model actually spans >1 device.

THE THREE CHECKS
----------------
  A (THE GATE, decisive)  teacher-forced per-token logprobs, single-GPU vs SHARDED, over the full
                          driver eval population. Gate: max |Δ| <= --tol (default 1e-6).
                          Also recomputes msp_min/perplexity/sum PRR under each arm; a floor PRR
                          that moves between arms is a hard failure regardless of the logprob diff.
  B (practical)           free-running generate() on a slice, single vs sharded. Greedy decoding
                          amplifies any numeric drift into a DIFFERENT TOKEN, so this asks the
                          deployment question: would a sharded extraction write the same cache?
  C (context, NOT a gate) single-GPU teacher-forced vs the CACHED autoregressive logprobs. This
                          quantifies the known AR-vs-TF gap and, incidentally, proves the harness
                          is reading the right cache for the right model.

POPULATION (matters — got this wrong twice before it was traced)
----------------------------------------------------------------
The ladder drivers drop UNLABELLED rows FIRST and only then carve the train/test split, so:
    pubmed_qa : 3800 rows, real baked-in split      -> 2000 test   -> msp_min PRR +0.3710
    factscore :  500 rows, 45 unlabelled dropped    -> 455 kept    -> 30% carve at seed 0
                                                    -> 136 test    -> msp_min PRR +0.4283
Reproduced exactly by this script's `driver_population()`. Using the raw `split` field instead
gives +0.4597, and splitting before filtering gives +0.5953 — neither is the ladder's number.

USAGE
    python scripts/checks/shard_equivalence.py                    # full gate, both datasets
    python scripts/checks/shard_equivalence.py --limit 50         # quick smoke
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "checks"))

from luq import generate as gen                        # noqa: E402
from luq import msp                                    # noqa: E402
from luq.results import prr                            # noqa: E402
from xl_rungs import eval_split, label_of              # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"

# The frozen reference numbers this must reproduce, from the master ladder
# (results/pdl_master__meta-llama_Meta-Llama-3.1-8B.csv). Floors are rung-invariant, so one
# value per dataset covers all five rungs.
RECORDS = {
    "pubmed_qa": REPO / "cache/records/meta-llama_Meta-Llama-3.1-8B__pubmed_qa__ID.jsonl",
    "factscore": REPO / "cache/factscore_rp12/records/meta-llama_Meta-Llama-3.1-8B__factscore__ID.jsonl",
}
REFERENCE_PRR = {
    "pubmed_qa": {"min": 0.3710},
    "factscore": {"min": 0.4283, "perplexity": 0.3260, "sum": 0.2821},
}


# ---------------------------------------------------------------------------------------------
def driver_population(dataset):
    """The EXACT rows the ladder scores: drop unlabelled, THEN carve the split. Order matters."""
    path = RECORDS[dataset]
    if not path.exists():
        raise SystemExit(f"{dataset}: missing records cache {path}")
    recs = [json.loads(line) for line in open(path)]
    # Fail loud on a wrong-model cache rather than silently scoring it (CLAUDE.md: no model-agnostic globs).
    if "Meta-Llama-3.1-8B" not in path.name:
        raise SystemExit(f"{dataset}: {path.name} is not the Llama cache — refusing to run")
    lab = label_of(dataset)
    y_all = np.array([r.get(lab, np.nan) for r in recs], dtype=float)
    keep = np.where(np.isfinite(y_all))[0]
    recs = [recs[i] for i in keep]
    _, te = eval_split(np.array([r["split"] for r in recs]))
    rows = [recs[i] for i in te]
    y = np.array([r[lab] for r in rows], dtype=float)
    return rows, y, lab, len(y_all), len(keep)


@torch.no_grad()
def tf_logprobs(model, record):
    """Teacher-forced logprob of each GENERATED token, the like-for-like counterpart of generate()'s.

    generate() records log_softmax(logits at step i)[gen_ids[i]] from AR decoding. Feeding
    [prompt + gen] through in one pass, the logits that predict gen_ids[i] sit at position
    P-1+i. Same quantity, one forward instead of G.
    """
    p_ids = list(record["prompt_token_ids"])
    g_ids = list(record["gen_token_ids"])
    P, G = len(p_ids), len(g_ids)
    dev = model.get_input_embeddings().weight.device      # correct under accelerate sharding
    ids = torch.tensor(p_ids + g_ids, device=dev)[None]
    logits = model(ids).logits[0, P - 1: P + G - 1].float()   # (G, vocab): predicts each gen token
    logp = torch.log_softmax(logits, dim=-1)
    tgt = torch.tensor(g_ids, device=logp.device)
    return logp.gather(1, tgt[:, None])[:, 0].cpu().numpy().astype(np.float64)


def load(device_map, max_memory, want_shard):
    """Load Llama fp32 + eager, and PROVE the device layout is what we asked for."""
    model, tok = gen.load_model(MODEL, attn_implementation="eager", dtype=torch.float32,
                                device_map=device_map, max_memory=max_memory)
    dmap = getattr(model, "hf_device_map", None) or {}
    devs = {v for v in dmap.values() if isinstance(v, int) or (isinstance(v, str) and v not in ("cpu", "disk"))}
    n_dev = len(devs)
    if want_shard:
        if n_dev < 2:
            raise SystemExit(f"❌ SHARDING DID NOT HAPPEN — model sits on {n_dev} device(s): {devs}. "
                             "The test would be vacuous. Lower max_memory and retry.")
        if any(v in ("cpu", "disk") for v in dmap.values()):
            raise SystemExit("❌ part of the model was offloaded to CPU/disk — that is a different "
                             "computation, not GPU sharding. Raise max_memory.")
        print(f"    ✅ genuinely sharded across {n_dev} devices: {sorted(map(str, devs))}", flush=True)
    else:
        print(f"    single-device load ({n_dev} device: {sorted(map(str, devs))})", flush=True)
    return model, tok


def run_tf(model, rows, tag):
    t0 = time.time()
    out = []
    for i, r in enumerate(rows):
        out.append(tf_logprobs(model, r))
        if (i + 1) % 250 == 0:
            print(f"      {tag}: {i+1}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"      {tag}: {len(rows)}/{len(rows)} done in {time.time()-t0:.0f}s", flush=True)
    return out


def floors(lp_list):
    """msp_min / perplexity / sum from a list of per-record logprob vectors."""
    return {a: np.array([msp.msp_uncertainty(lp, a) for lp in lp_list]) for a in ("min", "perplexity", "sum")}


def maxdiff(a_list, b_list):
    m, where = 0.0, None
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        if len(a) != len(b):
            raise SystemExit(f"length mismatch at row {i}: {len(a)} vs {len(b)}")
        d = float(np.max(np.abs(a - b))) if len(a) else 0.0
        if d > m:
            m, where = d, i
    return m, where


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="pubmed_qa,factscore")
    ap.add_argument("--limit", type=int, default=0, help="0 = full driver eval population")
    ap.add_argument("--tol", type=float, default=1e-6, help="gate on max |Δ logprob|, single vs sharded")
    ap.add_argument("--gen-rows", type=int, default=20, help="rows for check B (free-running generate)")
    ap.add_argument("--shard-mem", default="20GiB", help="per-GPU cap that FORCES a real split")
    ap.add_argument("--out", default=str(REPO / "results/g1_shard_equivalence.json"))
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    ngpu = torch.cuda.device_count()
    print(f"G1 shard-equivalence | {MODEL} | fp32 + eager | {ngpu} GPU(s) visible", flush=True)
    if ngpu < 2:
        raise SystemExit(f"❌ need >=2 GPUs to test sharding, found {ngpu}")
    for i in range(ngpu):
        p = torch.cuda.get_device_properties(i)
        print(f"    GPU{i}: {p.name} {p.total_memory/2**30:.0f}GiB", flush=True)

    pops = {}
    for d in datasets:
        rows, y, lab, n_all, n_kept = driver_population(d)
        if args.limit:
            rows, y = rows[:args.limit], y[:args.limit]
        pops[d] = (rows, y, lab)
        print(f"    {d}: {n_all} rows -> {n_kept} labelled ({lab}) -> {len(rows)} eval rows"
              + ("  [LIMITED — SMOKE TEST, not the gate]" if args.limit else ""), flush=True)

    results = {"model": MODEL, "dtype": "fp32", "attn": "eager", "tol": args.tol,
               "limit": args.limit, "smoke_test": bool(args.limit), "datasets": {}}

    # ---- arm 1: single GPU -------------------------------------------------------------------
    print("\n=== ARM 1: single GPU (device_map='cuda:0') ===", flush=True)
    model, tok = load({"": 0}, None, want_shard=False)
    single, gen_single = {}, {}
    for d in datasets:
        print(f"  {d}: teacher-forced pass", flush=True)
        single[d] = run_tf(model, pops[d][0], d)
        rows = pops[d][0][:args.gen_rows]
        print(f"  {d}: free-running generate on {len(rows)} rows (check B)", flush=True)
        gen_single[d] = [gen.generate(model, tok, r["prompt"], len(r["gen_token_ids"]))[0] for r in rows]
    del model
    torch.cuda.empty_cache()

    # ---- arm 2: sharded ----------------------------------------------------------------------
    print(f"\n=== ARM 2: SHARDED (device_map='auto', max_memory={args.shard_mem}/GPU) ===", flush=True)
    model, tok = load("auto", {i: args.shard_mem for i in range(ngpu)}, want_shard=True)
    shard, gen_shard = {}, {}
    for d in datasets:
        print(f"  {d}: teacher-forced pass", flush=True)
        shard[d] = run_tf(model, pops[d][0], d)
        rows = pops[d][0][:args.gen_rows]
        print(f"  {d}: free-running generate on {len(rows)} rows (check B)", flush=True)
        gen_shard[d] = [gen.generate(model, tok, r["prompt"], len(r["gen_token_ids"]))[0] for r in rows]
    del model
    torch.cuda.empty_cache()

    # ---- compare -----------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("CHECK A — THE GATE: teacher-forced logprobs, single vs SHARDED (same code path, same node)")
    print("=" * 92)
    all_pass = True
    for d in datasets:
        rows, y, lab = pops[d]
        md, where = maxdiff(single[d], shard[d])
        ok = md <= args.tol
        all_pass &= ok
        print(f"\n  {d}: max |Δ logprob| = {md:.3e}   (tol {args.tol:.0e})   "
              f"{'✅ PASS' if ok else '❌ FAIL'}" + (f"   worst row {where}" if where is not None else ""))
        fs, fh = floors(single[d]), floors(shard[d])
        ent = {"max_abs_logprob_diff": md, "pass": bool(ok), "n_rows": len(rows), "label": lab, "prr": {}}
        for agg in ("min", "perplexity", "sum"):
            p_s, p_h = prr(y, fs[agg]), prr(y, fh[agg])
            ref = REFERENCE_PRR.get(d, {}).get(agg)
            drift = abs(p_s - p_h)
            moved = drift > 1e-6
            all_pass &= not moved
            line = (f"    msp_{agg:10} PRR  single {p_s:+.4f}   sharded {p_h:+.4f}   "
                    f"Δ {drift:.2e} {'❌ MOVED' if moved else '✓'}")
            if ref is not None and not args.limit:
                line += f"   | cached-AR ref {ref:+.4f} (Δ {abs(p_s-ref):+.4f}, AR-vs-TF)"
            print(line)
            ent["prr"][agg] = {"single": p_s, "sharded": p_h, "reference_cached_ar": ref}
        results["datasets"][d] = ent

    print("\n" + "=" * 92)
    print("CHECK B — free-running generate(): would a sharded extraction write the same cache?")
    print("=" * 92)
    for d in datasets:
        same_tok = same_txt = 0
        lp_md = 0.0
        for a, b in zip(gen_single[d], gen_shard[d]):
            same_tok += int(list(a["gen_token_ids"]) == list(b["gen_token_ids"]))
            same_txt += int(a["gen_text"] == b["gen_text"])
            n = min(len(a["token_logprobs"]), len(b["token_logprobs"]))
            if n:
                lp_md = max(lp_md, float(np.max(np.abs(
                    np.array(a["token_logprobs"][:n]) - np.array(b["token_logprobs"][:n])))))
        n = len(gen_single[d])
        ok = (same_tok == n)
        all_pass &= ok
        print(f"  {d}: identical gen_token_ids {same_tok}/{n}   identical gen_text {same_txt}/{n}   "
              f"max |Δ logprob| {lp_md:.3e}   {'✅' if ok else '❌ GREEDY DECODE DIVERGED'}")
        results["datasets"][d]["generate"] = {"n": n, "identical_token_ids": same_tok,
                                              "identical_text": same_txt, "max_abs_logprob_diff": lp_md}

    print("\n" + "=" * 92)
    print("CHECK C — context, NOT a gate: single-GPU TEACHER-FORCED vs the CACHED AUTOREGRESSIVE logprobs")
    print("         (the known AR-vs-TF gap, ~1e-4. Isolated here so it cannot be read as sharding.)")
    print("=" * 92)
    for d in datasets:
        cached = [np.asarray(r["token_logprobs"], dtype=np.float64) for r in pops[d][0]]
        md, _ = maxdiff(cached, single[d])
        print(f"  {d}: max |Δ| = {md:.3e}   {'(consistent with the documented AR-vs-TF gap)' if md > args.tol else '(AR and TF agree to the gate tolerance)'}")
        results["datasets"][d]["ar_vs_tf_max_abs_diff"] = md

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results["verdict"] = "PASS" if all_pass else "FAIL"
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)

    print("\n" + "=" * 92)
    if args.limit:
        print("⚠️  SMOKE TEST (--limit set) — NOT the gate. Re-run without --limit before trusting this.")
    print(f"G1 VERDICT: {'✅ PASS — sharding is numerically safe; proceed to Qwen' if all_pass else '❌ FAIL — DO NOT proceed to Qwen on this path'}")
    print("=" * 92, flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
