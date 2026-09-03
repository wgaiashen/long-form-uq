"""Fitted background statistics for the relative distance, at any set of layers.

WHY THIS EXISTS
---------------
The relative token Mahalanobis distance subtracts a background distance, estimated on a general
corpus, from each token's distance to the training centroid. `scripts/01m_background_c4.py` builds
that corpus and writes the per-token hidden states it produced, at one layer, for one generation
budget ceiling. At 4096 dimensions that file is about 2.3 GB.

Reproducing the published estimators at their full layer set needs the background at thirty-two
layers and four budgets. Storing the states for that is about 73 GB, all of which is only ever
reduced to a centroid and an inverse covariance of about 67 MB. So this script separates the two
things `01m` does at once:

  STAGE ONE, `--emit-records`. Generate the background continuations and write ONLY the token
  identities, about 40 MB. This is the expensive stage and it runs once per model. The selection,
  the prompt cap and the generation call are imported from `01m_background_c4.py` rather than
  restated, so the corpus cannot drift between the two scripts.

  STAGE TWO, `--from-records` or `--from-states`. Run one teacher-forced forward per background row,
  take the requested layers out of it, and fit the statistics for each budget. Output is one small
  file per layer per budget. This stage is cheap enough to run inside the same job that extracts a
  layer of the grid, which is what keeps the background off the critical path.

`--from-states` reads an existing states file from `01m` instead of recomputing. It exists so that
the code path added here can be checked against the one already used, on the same numbers: fitting
from the same states through both paths must agree exactly, which isolates this change from the
separate question of whether a recomputed hidden state matches a cached one.

WINDOW
------
The cached window is the last prompt position through the final generated position. The reference
implementation's window is the last prompt position through the second to last generated position,
one row shorter. `--window ref` reproduces the reference; `--window project` keeps the window every
existing result in this project was measured on. The background must always use the SAME window as
the data it is subtracted from, or the difference is between two different quantities.

  python scripts/01q_background_stats.py --model meta-llama/Meta-Llama-3.1-8B --emit-records
  python scripts/01q_background_stats.py --model meta-llama/Meta-Llama-3.1-8B \
      --from-records --layers 0,1,2 --budgets 56,128,256,384 --window ref
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache, generate  # noqa: E402
from luq import mahalanobis as MD  # noqa: E402

_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# The four generation budgets the long-form grid uses, so the background can be sliced to whichever
# one an evaluation dataset was generated at.
DEFAULT_BUDGETS = [56, 128, 256, 384]


def _load_01m():
    """Import the corpus selection from 01m_background_c4.py by path.

    The module name begins with a digit, so it cannot be imported by name. Importing it rather than
    copying `load_background_texts` is deliberate: the C4 shard, the 100,000 row window, the seed and
    the subsample are the reference implementation's selection, and there must be exactly one copy of
    them in this repository.
    """
    path = ROOT / "scripts" / "01m_background_c4.py"
    spec = importlib.util.spec_from_file_location("_bg01m", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_window(arr, window):
    """Return the per-token window this run is measured on.

    The cached array runs from the last prompt position through the final generated position. The
    reference implementation concatenates the last prompt position with one state per generation
    step, which is one row shorter. Dropping the final row therefore turns one into the other exactly,
    with no re-extraction.
    """
    if window == "project":
        return arr
    return arr[:-1] if len(arr) else arr


def records_path(slug, budget):
    return ROOT / "cache" / "background_c4" / f"{slug}__records__b{budget}.npz"


def stats_path(slug, layer, budget, window):
    tag = "" if window == "project" else "__refwin"
    return ROOT / "cache" / "background_c4" / f"{slug}__bgstats__L{layer}__b{budget}{tag}.npz"


def emit_records(args, slug):
    """Stage one: generate the background continuations, store token identities only."""
    m01 = _load_01m()
    texts, c4_idx = m01.load_background_texts(args.n, args.seed)
    print(f"{args.model} | {len(texts)} C4 rows | budget {args.budget}", flush=True)

    model, tok = generate.load_model(
        args.model, attn_implementation=None if args.attn == "auto" else args.attn,
        dtype=None if args.dtype == "auto" else _DTYPE[args.dtype],
        device_map=args.device_map, max_memory=_max_memory(args))
    model.eval()

    # The same prompt cap 01m applies, for the same reason: a handful of C4 documents run to tens of
    # thousands of tokens and would otherwise dominate the background.
    capped, n_trunc = [], 0
    for t in texts:
        ids = tok(t).input_ids
        if len(ids) > args.max_prompt_tokens:
            t = tok.decode(ids[:args.max_prompt_tokens], skip_special_tokens=True)
            n_trunc += 1
        capped.append(t)
    print(f"prompt cap {args.max_prompt_tokens}: truncated {n_trunc}/{len(capped)} documents",
          flush=True)

    todo = capped[:args.limit] if args.limit else capped
    prompt_ids, gen_ids, nl_out, keep = [], [], [], []
    newline_id_cache = {}
    t0 = time.time()
    for i, prompt in enumerate(todo):
        rec, _ = generate.generate(model, tok, prompt, max_new_tokens=args.budget,
                                   truncate_at_newline=False)
        p_ids = np.asarray(rec["prompt_token_ids"], dtype=np.int64)
        g_ids = np.asarray(rec["gen_token_ids"], dtype=np.int64)
        nl = -1
        for t_i, tid in enumerate(g_ids.tolist()):
            if tid not in newline_id_cache:
                newline_id_cache[tid] = "\n" in tok.decode([tid])
            if newline_id_cache[tid]:
                nl = t_i
                break
        prompt_ids.append(p_ids)
        gen_ids.append(g_ids)
        nl_out.append(nl)
        keep.append(int(c4_idx[i]))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(todo)} rows ({time.time() - t0:.0f}s)", flush=True)

    out = records_path(slug, args.budget)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             prompt_token_ids=np.array(prompt_ids, dtype=object),
             gen_token_ids=np.array(gen_ids, dtype=object),
             newline_pos=np.array(nl_out), c4_index=np.array(keep),
             budget=np.int64(args.budget), seed=np.int64(args.seed),
             n_prompt_truncated=np.int64(n_trunc),
             max_prompt_tokens=np.int64(args.max_prompt_tokens),
             model=str(args.model))
    tot = int(sum(len(g) for g in gen_ids))
    print(f"\nwrote {out}\n  {len(gen_ids)} rows | {tot} generated tokens | "
          f"{sum(1 for n in nl_out if n >= 0)} rows contain a newline")


def records_from_existing_states(args, slug):
    """Stage one without regenerating anything.

    The background states file already on disk carries the generated token identities for every row,
    and the prompts are a deterministic function of the C4 selection, the seed and the prompt cap. So
    the token records can be rebuilt from it with a tokenizer alone, no model and no GPU, and the
    corpus is then the SAME corpus the existing single-layer background was fitted on rather than a
    fresh sample that merely follows the same recipe.

    The rebuild is gated, not assumed: every row's re-derived prompt must have exactly the length the
    states file recorded for it, and the C4 row indices must match one for one. A tokenizer or cap
    that had drifted would change those lengths, and the job stops instead of writing a corpus that
    only looks right.
    """
    from transformers import AutoTokenizer

    z = np.load(ROOT / args.from_existing_states, allow_pickle=True)
    gen_ids = z["gen_token_ids"]
    plen = np.asarray(z["prompt_len"], dtype=np.int64)
    c4_idx = np.asarray(z["c4_index"], dtype=np.int64)
    budget = int(z["budget"])
    seed = int(z["seed"])
    cap = int(z["max_prompt_tokens"])
    print(f"rebuilding from {args.from_existing_states}\n  {len(gen_ids)} rows | budget {budget} "
          f"| seed {seed} | prompt cap {cap}", flush=True)

    m01 = _load_01m()
    texts, sel_idx = m01.load_background_texts(len(gen_ids), seed)
    if not np.array_equal(np.asarray(sel_idx, dtype=np.int64), c4_idx):
        sys.exit("FATAL: the C4 selection re-derived here is not the one the states file recorded. "
                 "The corpus would differ, so nothing is written.")

    tok = AutoTokenizer.from_pretrained(args.model)
    prompt_ids, bad = [], 0
    for i, t in enumerate(texts):
        ids = tok(t).input_ids
        if len(ids) > cap:
            t = tok.decode(ids[:cap], skip_special_tokens=True)
            ids = tok(t).input_ids
        prompt_ids.append(np.asarray(ids, dtype=np.int64))
        if len(ids) != int(plen[i]):
            bad += 1
            if bad <= 5:
                print(f"  row {i}: re-derived prompt is {len(ids)} tokens, states file says "
                      f"{int(plen[i])}")
    print(f"PROMPT LENGTH GATE: {len(texts) - bad}/{len(texts)} rows exact", flush=True)
    if bad:
        sys.exit("FATAL: the re-derived prompts are not the prompts that were generated from. "
                 "Nothing written.")

    out = records_path(slug, budget)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             prompt_token_ids=np.array(prompt_ids, dtype=object),
             gen_token_ids=np.array(gen_ids, dtype=object),
             newline_pos=np.asarray(z["newline_pos"]), c4_index=c4_idx,
             budget=np.int64(budget), seed=np.int64(seed),
             n_prompt_truncated=np.int64(z["n_prompt_truncated"]),
             max_prompt_tokens=np.int64(cap), model=str(args.model),
             rebuilt_from=str(args.from_existing_states))
    tot = int(sum(len(g) for g in gen_ids))
    print(f"\nwrote {out}\n  {len(gen_ids)} rows | {tot} generated tokens")


def rows_from_states(path, budget, window):
    """Background rows taken from an existing states file written by 01m."""
    z = np.load(path, allow_pickle=True)
    stored_budget = int(z["budget"])
    if budget > stored_budget:
        sys.exit(f"FATAL: asked for budget {budget} but the states were generated at "
                 f"{stored_budget}. A longer budget cannot be recovered by slicing.")
    rows = []
    for s in z["states"]:
        s = np.asarray(s, dtype=np.float32)
        if len(s) == 0:
            continue
        s = apply_window(s[:budget + 1], window)
        if len(s):
            rows.append(s)
    return rows, int(z["layer"])


def fit_and_write(rows, layer, budget, window, slug, args, provenance):
    """Fit one background statistic and persist it.

    The fit is the same call the single-layer driver makes, with no correctness filter, because the
    reference does not filter its background corpus.
    """
    stats = MD.fit_md(rows, layer=layer,
                      row_ids=[("__c4__", i) for i in range(len(rows))],
                      kind=f"bg{budget}")
    out = stats_path(slug, layer, budget, window)
    out.parent.mkdir(parents=True, exist_ok=True)
    MD.save_stats(out, stats, layer=layer, budget=budget, window=window,
                  n_bg_rows=len(rows), **provenance)
    print(f"  L{layer:<3d} b{budget:<4d} {len(rows)} rows | {stats.n_tokens} tokens | "
          f"jitter {stats.jitter:g} -> {out.name}", flush=True)
    return stats


def _max_memory(args):
    if not args.max_memory:
        return None
    mm = {}
    for item in args.max_memory.split(","):
        k, v = item.split("=")
        mm[int(k.strip())] = v.strip()
    return mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--emit-records", action="store_true", help="stage one, by generating")
    ap.add_argument("--from-existing-states", default="",
                    help="stage one, by rebuilding the token records from a states file 01m already "
                         "wrote. No GPU and no generation, and the corpus is provably the same one.")
    ap.add_argument("--from-records", action="store_true", help="stage two, recomputing states")
    ap.add_argument("--from-states", default="",
                    help="stage two, reading states from an existing 01m file. Equivalence check.")
    ap.add_argument("--layers", default="", help="comma list, stage two only")
    ap.add_argument("--budgets", default=",".join(str(b) for b in DEFAULT_BUDGETS))
    ap.add_argument("--window", default="ref", choices=["project", "ref"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=int, default=384, help="generation ceiling, stage one")
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--dtype", default="fp32", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--attn", default="eager", choices=["auto", "eager", "sdpa"])
    ap.add_argument("--device-map", default="cuda")
    ap.add_argument("--max-memory", default="")
    ap.add_argument("--limit", type=int, default=0, help="debug only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    slug = cache._slug(args.model)

    if args.from_existing_states:
        if records_path(slug, args.budget).exists() and not args.overwrite:
            print(f"SKIP: {records_path(slug, args.budget)} exists. Pass --overwrite to replace.")
            return 0
        records_from_existing_states(args, slug)
        return 0

    if args.emit_records:
        if records_path(slug, args.budget).exists() and not args.overwrite:
            print(f"SKIP: {records_path(slug, args.budget)} exists. Pass --overwrite to replace.")
            return 0
        emit_records(args, slug)
        return 0

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]

    if args.from_states:
        # The equivalence path. One layer, whichever the states file holds.
        for b in budgets:
            rows, layer = rows_from_states(ROOT / args.from_states, b, args.window)
            fit_and_write(rows, layer, b, args.window, slug, args,
                          dict(source="states", states_file=str(args.from_states)))
        return 0

    if not args.from_records:
        sys.exit("choose one of --emit-records, --from-existing-states, --from-records, --from-states")

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    if not layers:
        sys.exit("--from-records needs --layers")

    rec_path = records_path(slug, args.budget)
    if not rec_path.exists():
        sys.exit(f"FATAL: no background records at {rec_path}. Run --emit-records first.")
    z = np.load(rec_path, allow_pickle=True)
    p_all, g_all = z["prompt_token_ids"], z["gen_token_ids"]
    n = len(g_all) if not args.limit else min(args.limit, len(g_all))
    print(f"{args.model} | {n} background rows | layers {layers} | budgets {budgets} "
          f"| window {args.window}", flush=True)

    # Skip layers whose every budget is already on disk, so a job that ran out of walltime resumes.
    todo = [L for L in layers
            if args.overwrite or not all(stats_path(slug, L, b, args.window).exists()
                                         for b in budgets)]
    if not todo:
        print("every requested layer and budget already exists. Nothing recomputed.")
        return 0
    print(f"layers still to do: {todo}", flush=True)

    model, tok = generate.load_model(
        args.model, attn_implementation=None if args.attn == "auto" else args.attn,
        dtype=None if args.dtype == "auto" else _DTYPE[args.dtype],
        device_map=args.device_map, max_memory=_max_memory(args))
    model.eval()
    card = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"accelerator: {card}", flush=True)

    # One forward per row yields every requested layer, so the cost is one pass over the corpus
    # regardless of how many layers this job covers.
    per_layer = {L: [] for L in todo}
    t0 = time.time()
    for i in range(n):
        p_ids, g_ids = list(p_all[i]), list(g_all[i])
        P, G = len(p_ids), len(g_ids)
        if G == 0:
            continue
        with torch.no_grad():
            st = generate.recompute_states(model, tok, p_ids + g_ids, todo, logits_to_keep=1)
        for k, L in enumerate(todo):
            arr = st[k][P - 1:P + G].numpy().astype(np.float32)
            if arr.shape[0] != G + 1:
                sys.exit(f"FATAL row {i} layer {L}: window gave {arr.shape[0]} states for G={G}. "
                         "Refusing to fit a misaligned background.")
            per_layer[L].append(arr)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n} rows ({time.time() - t0:.0f}s)", flush=True)

    print(f"\nfitting statistics ({time.time() - t0:.0f}s of forwards)", flush=True)
    for L in todo:
        for b in budgets:
            rows = [apply_window(a[:b + 1], args.window) for a in per_layer[L]]
            rows = [r for r in rows if len(r)]
            fit_and_write(rows, L, b, args.window, slug, args,
                          dict(source="records", card=card, seed=args.seed))
        # Free the layer as soon as its statistics exist. Holding several layers of 2000 rows at 384
        # tokens is what this two-stage split exists to avoid.
        per_layer[L] = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
