"""E1 (Round-3 method track): dump the attention pooler's per-token weights when TRAINED ON AN OOD SOURCE
POOL (a rung), for the B2 diagnostic; save the trained poolers so C1 needs no retrain.

WHY a new script (not dump_viz_attention): that one is ID-only and derives train/test from a single dataset's
`split` field (record positions). E1 must train on a POOLED OOD source set (via the ladder's build_rows) and
apply to the eval's test set -- a different assembly. We reuse the LADDER's exact pool construction
(cells_long -> build_rows -> sampled_train_idx -> train_attn) so the sidecar pooler == the §C.3 ladder pooler.

CPU ONLY: trains AttnPool on cached per-token states (like the ladders); the GPU self-attention render is not
produced here (the B2 entropy diagnostic needs only pool_w).

Outputs, per (eval X, rung, seed):
  * sidecar  cache/viz/<run_key(model,X,"ID")>__attn[__<rung>].npz  (ID rung -> no suffix), with:
      record_pos_all = X's ORIGINAL test record positions (X_te)   <- aligns pool_w to X's ID records
      pool_w         = object[]  (G+1,)  per test example
      meta keys: rung, seed, layer, best_T, pool_config, n_train
  * checkpoint (--save-pooler) cache/probes/<slug>__<X>__ID__attnpool_<rung>_s<seed>__L<layer>.pkl (pickled AttnPool)

REPRODUCTION GATE (ID cells only): §C.2 is still narrow-pool so OOD cells are not a valid target yet, but ID
trains on X's own eval_split carve (no source pool) so its PRR is pool-independent. We assert the retrained ID
pooler reproduces the realised ID attention PRR within GATE_TOL; a fail means the diagnostic would describe a
different model than the table -> STOP for that eval.

    python scripts/tools/dump_ood_attention.py --evals cnn_dailymail,expertqa,pubmed_qa,xsum \
        --rungs ID,LOO,DiffTask --seed 1 --save-pooler
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]         # repo root (scripts/tools/<file> -> parents[2])
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, results                                              # noqa: E402
import attn_pool as ap                                                      # noqa: E402
from aggregation_table import attn_unc                                      # noqa: E402
from xl_rungs import build_rows, eval_split, label_of                       # noqa: E402
import probedriftlong as pdl                                                # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15
# realised ID attention PRR (from the ladder CSVs; ID_ANCHOR lacks xsum/cnn/expertqa). The reproduction target.
ID_TARGET = {"xsum": 0.577, "cnn_dailymail": 0.594, "pubmed_qa": 0.731, "expertqa": 0.645,
             "sciq": 0.932, "trivia_qa": 0.847}   # med_quad/samsum/asqa: no prior ID sidecar -> recorded, not gated
GATE_TOL = 0.03


def main():
    apr = argparse.ArgumentParser()
    apr.add_argument("--evals", default="cnn_dailymail,expertqa,pubmed_qa,xsum")
    apr.add_argument("--rungs", default="ID,LOO,DiffTask")
    apr.add_argument("--seed", type=int, default=1)
    apr.add_argument("--layer", type=int, default=LAYER)
    apr.add_argument("--save-pooler", action="store_true")
    args = apr.parse_args()
    evals = args.evals.split(","); want_rungs = set(args.rungs.split(","))
    sd = args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | seed {sd} | evals {evals} | rungs {sorted(want_rungs)}", flush=True)

    # Load PT (eval targets + every training source), drop unlabelled rows. FAST PATH: ID cells train on the
    # eval's OWN split (no source pool), so when only ID is requested load just the evals -- avoids pulling the
    # full ~30GB source set for the P2b ID-only regeneration.
    id_only = (want_rungs == {"ID"})
    to_load = set(evals) if id_only else (set(pdl.LONG_SRC) | set(evals))
    if id_only:
        print("  ID-only fast path: loading eval datasets only (no source pool)", flush=True)
    PT = {}
    for d in sorted(to_load):
        loaded = ap.load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip", flush=True); continue
        states, split, y, _lyr, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: fully unlabelled ({label_of(d)}) -> skip", flush=True); continue
        orig = np.arange(len(records))                 # map filtered-row -> ORIGINAL record position
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]; orig = keep
        # PT[d][4] = orig: needed because filtered indices must be translated back to ORIGINAL record positions
        # for the sidecar's record_pos_all, else the diagnostic (which reloads the FULL records) misaligns.
        PT[d] = (states, split, y, records, orig)
        print(f"  {d}: {len(states)} rows (label={label_of(d)})", flush=True)
    sources = set(PT)

    viz = ROOT / "cache" / "viz"; viz.mkdir(parents=True, exist_ok=True)
    probes = ROOT / "cache" / "probes"; probes.mkdir(parents=True, exist_ok=True)
    gate_rows = []

    # cells_long only emits an ID cell for LONG-universe datasets (probedriftlong.py:93) -> it SKIPS the SHORT
    # sets' ID (sciq/trivia). For ID-only mode build the ID cells directly (the ID cell is trivially [(X,None)]
    # for ANY eval) so every requested dataset is covered; otherwise use the full cells_long rung set.
    cell_iter = ([("ID", X, [(X, None)]) for X in evals if X in sources] if id_only
                 else pdl.cells_long(sources, [e for e in evals if e in sources]))
    for rung, X, spec in cell_iter:
        base_rung = rung.replace("-long", "")                    # "SameTask-long" -> "SameTask"; "ID" stays
        if base_rung not in want_rungs or X not in PT:
            continue
        train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
        if not train_rows or not test_rows:
            print(f"  [{rung:14s}] {X}: empty -> skip", flush=True); continue
        n_tr = len(train_rows); tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
        allrows = train_rows + test_rows
        y = np.array([PT[d][2][i] for d, i in allrows], float)
        states = [PT[d][0][i] for d, i in allrows]
        best_T, _ = ap.select_temperature(states, y, tr_idx, device, sd, False, False)
        pooler = ap.train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T)
        pooler.eval()

        # per-example attention weights on the eval test set (CPU, no grad)
        pool_w = []
        with torch.no_grad():
            for k in te_idx:
                Xk, mask, pos = ap.pad_batch([states[k]], device)
                _, a = pooler(Xk, mask, pos)
                pool_w.append(a[0, : states[k].shape[0]].float().cpu().numpy())
        # X's ORIGINAL test record positions (translate filtered idx -> original via PT[X][4]); this is what the
        # diagnostic (which reloads the FULL records) needs to align pool_w to gen_token_ids.
        record_pos_all = np.array([int(PT[X][4][i]) for _d, i in test_rows])

        key = cache.run_key(MODEL, X, "ID")
        # S1/P0 FIX (2026-07-28): DO NOT strip "-long". The pool is built from the LONG ladder
        # (cells_long/LONG_SRC), so a stripped "LOO" suffix asserted a STANDARD rung that was never trained --
        # any join then silently mixed a long-pool quantity with a standard-pool one. Keep the TRUE rung name
        # in BOTH the filename and the metadata, and stamp `ladder_family` so a consumer can tell the family
        # without re-parsing the name.
        ladder_family = "ID" if base_rung == "ID" else ("LONG" if rung.endswith("-long") else "STANDARD")
        if any(c in rung for c in ">/\\"):     # e.g. "Long->Short" -- would make a malformed filename
            raise SystemExit(f"unsafe rung name for a filename: {rung!r}; add a sanitised mapping first "
                             "(short-set OOD is out of scope for this dump)")
        suffix = "" if base_rung == "ID" else f"__{rung}"          # full rung, e.g. "__LOO-long" (was "__LOO")
        out = viz / f"{key}__attn{suffix}.npz"
        np.savez_compressed(out, record_pos_all=record_pos_all,
                            pool_w=np.array(pool_w, dtype=object),
                            rung=rung, base_rung=base_rung, ladder_family=ladder_family,
                            seed=sd, layer=args.layer, best_T=float(best_T),
                            pool_config="post-TaskA-widened", n_train=n_tr)
        # reproduction gate on ID cells
        yte = np.array([y[i] for i in te_idx], float)
        prr = results.prr(yte, np.asarray(attn_unc(pooler, states, te_idx, device), float))
        if base_rung == "ID" and X in ID_TARGET:
            d = abs(prr - ID_TARGET[X]); ok = d < GATE_TOL
            gate_rows.append((X, prr, ID_TARGET[X], d, ok))
        if args.save_pooler:
            pk = probes / f"{cache._slug(MODEL)}__{X}__ID__attnpool_{rung}_s{sd}__L{args.layer}.pkl"
            with open(pk, "wb") as f:
                pickle.dump({"model": pooler, "best_T": float(best_T), "rung": rung, "base_rung": base_rung,
                             "ladder_family": ladder_family, "seed": sd, "eval": X, "layer": args.layer,
                             "pool_config": "post-TaskA-widened"}, f)
        print(f"  [{rung:14s}] {X}: n_tr={n_tr} T*={best_T} PRR={prr:+.3f} -> {out.name}"
              f"{'  [+pooler]' if args.save_pooler else ''}", flush=True)

    print("\n=== ID REPRODUCTION GATE (retrained ID pooler vs realised §C.1 target, tol "
          f"{GATE_TOL}) ===", flush=True)
    all_ok = True
    for X, prr, tgt, d, ok in gate_rows:
        print(f"  {X:14s} retrained {prr:+.3f}  target {tgt:+.3f}  |Δ|={d:.3f}  {'PASS' if ok else 'FAIL <== STOP'}",
              flush=True)
        all_ok = all_ok and ok
    print(f"GATE: {'ALL PASS -> read attention / run B2' if all_ok else 'FAIL -> do NOT run B2 on failing evals'}",
          flush=True)


if __name__ == "__main__":
    main()
