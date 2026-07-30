"""head_aggregation_2x2.py -- the missing 2x2 cell: learned attention x MLP head.

The completed ProbeDriftLong table varied AGGREGATION ~ten ways but never the HEAD. Three of four
head x aggregation cells exist; this driver fills the fourth and, crucially, RE-RUNS all four in ONE
paired loop so the comparison needs no cross-run join:

                        linear head            MLP head (256/128/64)
    mean-pool           meanpool_linear(=armB)  meanpool_mlp
    learned attention   attention_linear(=armA) attention_mlp   <- NEW

PRE-REGISTRATION (mirrors results/head_aggregation_2x2_PREREGISTRATION.md; written BEFORE any run):
  * PRIMARY contrast   : attention_mlp(60ep) vs meanpool_mlp(60ep) -- the aggregation effect AT the MLP
                         head, fully recipe-controlled.
  * SECONDARY contrast : meanpool_mlp(60ep) vs armB(60ep) -- the CLEAN, recipe-controlled head effect.
                         The motivating +0.038 "head effect" compared armB(linear,60ep,wd=1e-2) vs
                         SAPLMA(MLP,5ep,wd=0) -- head arch AND recipe confounded, so it is NOT
                         established. If this delta is well below +0.038, part of that margin was
                         epochs/weight-decay and the headline gets rewritten.
  * The ~+0.260 expectation was derived from the confounded margin -> a rough expectation, NOT a target.
  * INFORMATIVE FAILURE: attention_mlp <= meanpool_mlp means the axes interact (a stronger head cannot
                         rescue an attention distribution that dissolves under shift) -- a real finding.

THE UNIFIED RECIPE (all four cells, one recipe -> a paired 2x2): train_attn's recipe -- 60 epochs,
Adam lr=1e-3, wd=1e-2 on head params, wd_query=0 (un-decayed query), bs=32, BCEWithLogitsLoss.
Temperature for the two learned-attention cells is RE-SELECTED with the head in place (select_temperature
with head_hidden), not reused from armA -- armA's protocol is "select T for THIS architecture". A stale T
would handicap only attention_mlp (biased against the hypothesis), hence re-select.

MANDATORY q-MOVED ASSERTION (before any PRR is trusted): a 5-epoch MLP head cannot pull the query off
its zero init (the collapse the code comment records). If it also fails to train at 60ep, attention_mlp
silently degenerates to mean-pool and any number is a FALSE null. So after training we record ||q|| and
RETRACT the attention_mlp cell (blank PRR, retracted=True) if ||q|| ~ 0. This is the S6 q_rest check.

BRIDGE + DIAGNOSTIC LADDER (each step isolates ONE variable, so the bridge is interpretable):
  saplma_ref (5ep, wd=0, conf_meanpool)
    -> meanpool_mlp_5ep (5ep, wd=0)      [GATE: must reproduce saplma_ref]
    -> meanpool_mlp     (60ep, wd=1e-2)  (isolates the RECIPE effect)
    -> vs armB          (60ep, wd=1e-2)  (isolates the HEAD effect -- the clean secondary contrast)
  attention_mlp_5ep (5ep, T=1) is EXPECTED to collapse to meanpool_mlp_5ep (query can't train) -- a
  positive diagnostic confirming why 60ep (or --early-stop) is required.

RECIPE RISK: the MLP head is ~1M params (256x4096) against ~1800 rows (p/n ~ 580); SAPLMA's 5 epochs
IS the regularisation. 60ep at wd=1e-2 may overfit. Pubmed is a DECISION GATE (run it alone first,
report, WAIT). --early-stop switches the MLP-head cells to validation-selected epochs (the linear cells
stay fixed-60ep armA/armB so their gates still pass).

CPU or GPU; runs on the per-token cache (layer 15). Reuses probedriftlong's EXACT pool construction so
the linear/saplma cells reproduce the reference runs. Does NOT recompute floors/wMSP.

    python scripts/checks/head_aggregation_2x2.py --evals pubmed_qa --out results/head_aggregation_2x2_pubmed_qa__<slug>.csv
"""
import argparse
import csv as _csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402
from luq import cache  # noqa: E402
from aggregation_table import load_per_token, attn_unc, paired_bootstrap, conf_meanpool  # noqa: E402
from attn_pool import train_attn, select_temperature, pad_batch  # noqa: E402
from xl_rungs import build_rows, eval_split, label_of, different_label_projection  # noqa: E402
from probedriftlong import cells_long, sampled_train_idx, _provenance, LONG_SRC, MODEL  # noqa: E402

MLP_HIDDEN = (256, 128, 64)      # the SAPLMA architecture (probe.train_probe_mlp)
EPOCHS_FULL = 60                 # the linear-pooler recipe (armA/armB)
EPOCHS_SHORT = 5                 # SAPLMA's early-stopping recipe
EPOCH_GRID = [5, 10, 20, 40, 60]  # --early-stop: val-selected epochs for the MLP-head cells
# q-moved retract thresholds: attention_mlp's ||q|| must be well off zero AND a non-trivial fraction of
# the linear-attention query's ||q|| (the in-cell reference scale). Either failing -> retract the cell.
RETRACT_ABS = 0.1
RETRACT_REL = 0.05


def _qnorm(model):
    return float(model.q.detach().norm().cpu())


def _mean_attn_entropy(model, states, te_idx, device):
    """Mean attention entropy over the test set (single-head; a is (B,T)). The mechanism column: learned
    attention is claimed to DISSOLVE toward uniform under shift, so ID->OOD entropy is the diagnostic."""
    model.eval()
    ents = []
    with torch.no_grad():
        for k in te_idx:
            X, mask, pos = pad_batch([states[k]], device)
            _, a = model(X, mask, pos)
            a = a[0, : states[k].shape[0]].cpu().numpy()
            ents.append(float(-(a * np.log(a + 1e-12)).sum()))
    return float(np.mean(ents)) if ents else float("nan")


def _select_epochs(states, y, tr_idx, device, seed, temperature, head_hidden, freeze_query):
    """--early-stop: carve a val split from train (never test) and pick the epoch count with the best val
    PRR from EPOCH_GRID. Mirrors select_temperature's val logic so the MLP head stops where it should."""
    from attn_pool import VAL_FRAC
    g = np.random.RandomState(seed)
    order = list(tr_idx); g.shuffle(order)
    n_val = int(len(order) * VAL_FRAC)
    val_idx, sub_tr = order[:n_val], order[n_val:]
    best_e, best_v = EPOCH_GRID[0], -1e9
    for e in EPOCH_GRID:
        m = train_attn(states, y, sub_tr, device, seed=seed, temperature=temperature,
                       freeze_query=freeze_query, head_hidden=head_hidden, epochs=e)
        v = attn_unc(m, states, val_idx, device)
        from luq import results as _r
        pr = _r.prr([y[i] for i in val_idx], v)
        if pr > best_v:
            best_v, best_e = pr, e
    return best_e


def _save_pooler(pooler, best_T, method, rung, X, sd, layer, states, te_idx, test_rows, PT, device):
    """Persist a trained learned-attention pooler (attention_linear / attention_mlp), method-tagged so it
    never clobbers probedriftlong's armA sidecar. Same byte layout as _save_pooler in probedriftlong /
    dump_ood_attention (pool_w + record_pos_all) so RCS's post-hoc pass can read it."""
    if any(c in rung for c in ">/\\"):
        print(f"  [save-pooler] skip {X} {rung!r}: unsafe rung name", flush=True); return
    base_rung = rung.replace("-long", "")
    probes = ROOT / "cache" / "probes"; probes.mkdir(parents=True, exist_ok=True)
    viz = ROOT / "cache" / "viz"; viz.mkdir(parents=True, exist_ok=True)
    pooler.eval()
    pool_w = []
    with torch.no_grad():
        for k in te_idx:
            Xk, mask, pos = pad_batch([states[k]], device)
            _, a = pooler(Xk, mask, pos)
            pool_w.append(a[0, : states[k].shape[0]].float().cpu().numpy())
    record_pos_all = np.array([int(PT[X][4][i]) for _d, i in test_rows])
    key = cache.run_key(MODEL, X, "ID")
    suffix = "" if base_rung == "ID" else f"__{rung}"
    np.savez_compressed(viz / f"{key}__attn_{method}{suffix}.npz", record_pos_all=record_pos_all,
                        pool_w=np.array(pool_w, dtype=object), rung=rung, base_rung=base_rung,
                        method=method, seed=sd, layer=layer, best_T=float(best_T),
                        pool_config="head2x2")
    pk = probes / f"{cache._slug(MODEL)}__{X}__ID__{method}_{rung}_s{sd}__L{layer}.pkl"
    with open(pk, "wb") as f:
        pickle.dump({"model": pooler, "best_T": float(best_T), "rung": rung, "method": method,
                     "seed": sd, "eval": X, "layer": layer, "pool_config": "head2x2"}, f)
    print(f"  [save-pooler] {X} {rung} {method} s{sd} -> {pk.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="pubmed_qa",
                    help="comma-separated evals. Run pubmed_qa ALONE first (the decision gate).")
    ap.add_argument("--rungs", default=None,
                    help="comma-separated BASE rungs to KEEP: ID,SameTask,DiffTask,LOO,1ds-Diff. Default all.")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out (guard the reference lacks)")
    ap.add_argument("--save-pooler", action="store_true",
                    help="persist seed-1 attention_linear + attention_mlp poolers (method-tagged; no clobber)")
    ap.add_argument("--early-stop", action="store_true",
                    help="MLP-head cells use validation-selected epochs (EPOCH_GRID) instead of fixed 60; "
                         "linear cells stay fixed-60ep armA/armB so their gates still pass. The pubmed-gate "
                         "fallback if fixed 60ep overfits.")
    args = ap.parse_args()

    # Overwrite guard (probedriftlong has none) + dirty-tree provenance abort (refuse to stamp a lie).
    out = Path(args.out) if args.out else (ROOT / "results" /
                                           f"head_aggregation_2x2__{cache._slug(MODEL)}.csv")
    if out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {out} (pass --force). Per-eval jobs must use distinct --out.")
    prov = _provenance()
    print(f"provenance: {prov}", flush=True)

    want_rungs = set(args.rungs.split(",")) if args.rungs else None
    evals = args.evals.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recipe = "early-stop(val-selected epochs on MLP cells)" if args.early_stop else f"fixed {EPOCHS_FULL}ep"
    print(f"device {device} | seeds {seeds} | evals {evals} | MLP recipe: {recipe}", flush=True)
    print(f"UNIFIED RECIPE: {EPOCHS_FULL}ep, Adam lr=1e-3, wd=1e-2 head / wd_query=0, bs=32; "
          f"T re-selected with head in place. MLP head = {MLP_HIDDEN}.", flush=True)

    # Build PT dict exactly as probedriftlong (load per-token, drop unlabelled rows). No segments/tokenizer
    # needed (no wMSP). Sources = LONG training pool U the evals requested.
    PT = {}
    for d in sorted(set(LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip", flush=True); continue
        states, split, y, _, records = loaded
        orig = np.arange(len(records))
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: fully unlabelled ({label_of(d)}) -> skip", flush=True); continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]; orig = keep
        PT[d] = (states, split, y, records, orig)
        print(f"  {d}: {len(states)} rows (label={label_of(d)})", flush=True)
    sources = set(PT)

    METHODS = ["saplma_ref", "meanpool_mlp_5ep", "meanpool_linear", "attention_linear",
               "meanpool_mlp", "attention_mlp", "attention_mlp_5ep"]
    out_rows = []
    for rung, X, spec in cells_long(sources, evals):
        if want_rungs is not None and rung.replace("-long", "") not in want_rungs:
            continue
        if X not in PT:
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        xlbl = different_label_projection(X)
        per = {m: [] for m in METHODS}           # per-seed PRR
        unc_acc = {m: [] for m in METHODS}       # per-seed uncertainty vectors (for paired bootstrap)
        qn = {"attention_linear": [], "attention_mlp": [], "attention_mlp_5ep": []}
        ent = {"attention_linear": [], "attention_mlp": []}
        bestT = {"attention_linear": None, "attention_mlp": None}
        yte_ref = None; srcs = ""; train_rows = []
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows); tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float); yte_ref = yte
            states = [PT[d][0][i] for d, i in allrows]
            Xmean = np.stack([s.mean(axis=0) for s in states])
            mlp_epochs = (_select_epochs(states, y, tr_idx, device, sd, 1.0, MLP_HIDDEN, True)
                          if args.early_stop else EPOCHS_FULL)

            v = {}
            # --- bridge + 5ep diagnostics -------------------------------------------------------
            v["saplma_ref"] = 1.0 - conf_meanpool(Xmean, tr_idx, te_idx, y, sd)          # SAPLMA (5ep, wd=0)
            m_mlp5 = train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True,   # GATE vs saplma_ref
                                head_hidden=MLP_HIDDEN, epochs=EPOCHS_SHORT, weight_decay=0.0)
            v["meanpool_mlp_5ep"] = np.asarray(attn_unc(m_mlp5, states, te_idx, device), float)
            m_amlp5 = train_attn(states, y, tr_idx, device, seed=sd, temperature=1.0,    # collapse diagnostic
                                 head_hidden=MLP_HIDDEN, epochs=EPOCHS_SHORT, weight_decay=0.0)
            v["attention_mlp_5ep"] = np.asarray(attn_unc(m_amlp5, states, te_idx, device), float)
            qn["attention_mlp_5ep"].append(_qnorm(m_amlp5))

            # --- linear cells (== armB / armA; fixed 60ep, unchanged recipe) --------------------
            m_ml = train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True)     # armB
            v["meanpool_linear"] = np.asarray(attn_unc(m_ml, states, te_idx, device), float)
            bT_lin, _ = select_temperature(states, y, tr_idx, device, sd, False, False)  # T for linear head
            m_al = train_attn(states, y, tr_idx, device, seed=sd, temperature=bT_lin)    # armA
            v["attention_linear"] = np.asarray(attn_unc(m_al, states, te_idx, device), float)
            qn["attention_linear"].append(_qnorm(m_al))
            ent["attention_linear"].append(_mean_attn_entropy(m_al, states, te_idx, device))
            bestT["attention_linear"] = bT_lin

            # --- MLP cells (the head axis; unified recipe, T re-selected WITH the MLP head) ------
            m_mm = train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True,     # meanpool_mlp
                              head_hidden=MLP_HIDDEN, epochs=mlp_epochs)
            v["meanpool_mlp"] = np.asarray(attn_unc(m_mm, states, te_idx, device), float)
            bT_mlp = select_temperature(states, y, tr_idx, device, sd, False, False,
                                        head_hidden=MLP_HIDDEN)[0]                        # T for MLP head
            m_am = train_attn(states, y, tr_idx, device, seed=sd, temperature=bT_mlp,    # attention_mlp (NEW)
                              head_hidden=MLP_HIDDEN, epochs=mlp_epochs)
            v["attention_mlp"] = np.asarray(attn_unc(m_am, states, te_idx, device), float)
            qn["attention_mlp"].append(_qnorm(m_am))
            ent["attention_mlp"].append(_mean_attn_entropy(m_am, states, te_idx, device))
            bestT["attention_mlp"] = bT_mlp

            if args.save_pooler and sd == seeds[0]:
                _save_pooler(m_al, bT_lin, "attnlin", rung, X, sd, args.layer, states, te_idx, test_rows, PT, device)
                _save_pooler(m_am, bT_mlp, "attnmlp", rung, X, sd, args.layer, states, te_idx, test_rows, PT, device)

            from luq import results as _r
            for m in METHODS:
                per[m].append(_r.prr(yte, v[m])); unc_acc[m].append(v[m])

            # realised per-source train label (honest; last seed, seed-stable)
            _real = {}
            for _d, _i in train_rows:
                _real[_d] = _real.get(_d, 0) + 1
            srcs = "+".join(f"{d}:{_real.get(d, 0)}" for d in dict.fromkeys(d for d, _c in spec))

        if yte_ref is None:
            continue
        stats = {m: (float(np.mean(per[m])), float(np.std(per[m]))) for m in per if per[m]}
        avg = {m: np.mean(np.stack(unc_acc[m]), 0) for m in unc_acc if unc_acc[m]}
        qnorm_mean = {k: (float(np.mean(vv)) if vv else float("nan")) for k, vv in qn.items()}
        ent_mean = {k: (float(np.mean(vv)) if vv else float("nan")) for k, vv in ent.items()}

        # q-MOVED assertion: retract attention_mlp if its query never trained (mean-pool in a costume).
        q_mlp = qnorm_mean["attention_mlp"]; q_lin = qnorm_mean["attention_linear"]
        retracted = (q_mlp < RETRACT_ABS) or (q_lin > 0 and q_mlp < RETRACT_REL * q_lin)

        xf = "  [CROSS-LABEL]" if (xlbl and rung != "ID") else ""
        print(f"\n[{rung:14s}] eval={X} ({label_of(X)}) train={srcs}{xf}", flush=True)
        print(f"    q_norm attention_mlp={q_mlp:.3f} (linear ref={q_lin:.3f}) "
              f"{'RETRACTED (q~0: mean-pool in a costume)' if retracted else 'moved -> real'}", flush=True)
        for m in METHODS:
            if m not in stats:
                continue
            row = {"rung": rung, "eval": X, "train": srcs, "method": m,
                   "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                   "n_seeds": len(per[m]), "different_label_projection": bool(xlbl and rung != "ID"),
                   "q_final_norm": round(qnorm_mean[m], 4) if m in qnorm_mean else "",
                   "attn_entropy": round(ent_mean[m], 4) if m in ent_mean else "",
                   "best_T": bestT.get(m, ""), "retracted": ""}
            if m == "attention_mlp" and retracted:               # never report a PRR for an untrained q
                row["prr_mean"] = ""; row["prr_std"] = ""; row["retracted"] = True
            print(f"    {m:18s} {stats[m][0]:+.3f} +/- {stats[m][1]:.3f}"
                  + (f"  q={qnorm_mean[m]:.2f}" if m in qnorm_mean else "")
                  + (f"  H={ent_mean[m]:.2f}" if m in ent_mean else ""), flush=True)
            out_rows.append(row)

        # paired-bootstrap verdicts. Primary: attention_mlp vs meanpool_mlp (aggregation at the MLP head).
        # Secondary: meanpool_mlp vs armB (clean head effect). Plus attention_mlp vs armA. Skip the primary
        # / attn-vs-armA verdicts if attention_mlp is retracted (no valid vector to compare).
        verdicts = [("meanpool_mlp_vs_armB", "meanpool_mlp", "meanpool_linear")]
        if not retracted:
            verdicts = [("attention_mlp_vs_meanpool_mlp", "attention_mlp", "meanpool_mlp"),
                        ("attention_mlp_vs_armA", "attention_mlp", "attention_linear")] + verdicts
        for vk, a, b in verdicts:
            if a in avg and b in avg:
                mg, lo, hi, p, sig = paired_bootstrap(yte_ref, avg[a], avg[b])
                print(f"    [verdict] {vk:32s} margin {mg:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} "
                      f"{'SIG' if sig else 'ns'}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"VERDICT:{vk}",
                                 "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                 "boot_p": round(p, 4), "significant": bool(sig), "n_seeds": len(seeds)})

    for _r in out_rows:
        _r.update(prov)
    fieldnames = ["rung", "eval", "train", "method", "prr_mean", "prr_std", "n_seeds",
                  "different_label_projection", "q_final_norm", "attn_entropy", "best_T", "retracted",
                  "ci_lo", "ci_hi", "boot_p", "significant", "git_sha", "cluster", "env_hash"]
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
