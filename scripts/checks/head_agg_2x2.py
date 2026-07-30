"""The missing 2x2 cell: aggregation (mean-pool | learned attention) x head (linear | SAPLMA MLP).

All FOUR cells are recomputed together per (rung, eval, seed) so the comparison is PAIRED and needs no
cross-run join. The three reference cells ARE the identical functions used by the base run
(probedriftlong), so they reproduce it by construction; the fourth (attention_mlp) is the only new
computation:

  meanpool_linear  (armB)    = attn_unc(train_attn(freeze_query=True))     -- the base `uniform`
  meanpool_mlp     (SAPLMA)  = 1 - conf_meanpool(Xmean)                     -- the base `saplma`
  attention_linear (armA)    = attn_unc(train_attn(learned query))         -- the base `attention`
  attention_mlp    (NEW)     = 1 - conf_meanpool(Xattn)                     -- armA's learned-attention
      pooled vectors (sum_t a_t x_t, the SAME attention as attention_linear) re-headed with the
      BYTE-IDENTICAL SAPLMA MLP (probe.train_probe_mlp). A head swap of armA, not a joint retrain --
      so the MLP is identical to SAPLMA and the only thing that varies vs meanpool_mlp is Xmean->Xattn.

Order of the three reference computations matches probedriftlong exactly (saplma, then best_T, uniform,
attention), and every trainer re-seeds torch from `sd` internally, so the reference cells are byte-exact
reproductions -- verified by the reproduction gate vs the widened base CSV.

Lean: floors/wMSP are NOT recomputed. Provenance (git_sha/cluster/env_hash + dirty-tree refusal) and the
widened pool / cells_long rungs are inherited from probedriftlong.
"""
import argparse, csv as _csv, glob, os, sys, pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "checks"))
import torch  # noqa: E402
from luq import cache, results  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from aggregation_table import load_per_token, attn_unc, paired_bootstrap, conf_meanpool  # noqa: E402
from attn_pool import train_attn, select_temperature, pad_batch  # noqa: E402
from xl_rungs import build_rows, eval_split, label_of, different_label_projection  # noqa: E402
import probedriftlong as pdl  # noqa: E402  (MODEL, LONG_SRC, cells_long, sampled_train_idx, _provenance)

MODEL = pdl.MODEL
CELLS = ["meanpool_linear", "meanpool_mlp", "attention_linear", "attention_mlp"]
# reference cell -> base-run column it must reproduce (attention_mlp is NEW, no external reference)
GATE_MAP = {"meanpool_linear": "uniform", "meanpool_mlp": "saplma", "attention_linear": "attention"}
GATE_TOL = 0.02   # reproduction should be near-exact; flag anything above seed noise


def attn_pooled(model, states, idx, device, bs=64):
    """Reconstruct the learned-attention pooled vector sum_t a_t x_t for each example in idx, exactly as
    AttnPool.forward computes it internally (forward returns the attention `a`, not the pooled vector)."""
    model.eval()
    d = states[0].shape[1]
    out = np.zeros((len(idx), d), dtype=np.float64)
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            sub = idx[b: b + bs]
            X, mask, pos = pad_batch([states[i] for i in sub], device)
            _, a = model(X, mask, pos)                       # a: (B, T) single-head
            pooled = (a.unsqueeze(-1) * X).sum(dim=1)         # (B, d) == AttnPool's internal pooled
            out[b: b + len(sub)] = pooled.cpu().numpy()
    return out


def load_base_ref(eval_name):
    """Per-rung {base_method: prr_mean} from the WIDENED base CSV, for the reproduction gate. Returns
    (None, None) if no widened reference is present on this cluster (then the gate is deferred)."""
    cands = sorted(glob.glob(str(ROOT / "results" / f"probedriftlong_{eval_name}_widened_wmsp__*.csv")))
    if not cands:
        return None, None
    ref = {}
    for r in _csv.DictReader(open(cands[0])):
        if r["method"] in ("uniform", "saplma", "attention") and r["prr_mean"]:
            ref.setdefault(r["rung"], {})[r["method"]] = float(r["prr_mean"])
    return ref, os.path.basename(cands[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--evals", required=True, help="comma-separated (one eval per job; writes at eval-end)")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-pooler", action="store_true",
                    help="persist the seed-1 attention_mlp bundle {armA pooler, SAPLMA MLP} under a DISTINCT "
                         "namespace (headagg2x2) so it can never collide with the base/arm poolers at rsync.")
    args = ap.parse_args()
    prov = pdl._provenance()   # aborts if the tracked tree is dirty
    print(f"PROVENANCE: git_sha={prov['git_sha'][:12]} cluster={prov['cluster']} env_hash={prov['env_hash']}"
          + ("  [+save-pooler seed-1]" if args.save_pooler else ""), flush=True)
    evals = args.evals.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | seeds {seeds} | evals {evals} | 2x2 cells {CELLS}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    _ = tok  # (kept for parity with the base loader; sentence ids not needed here)
    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip", flush=True); continue
        states, split, y, _lyr, records = loaded
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

    probes = ROOT / "cache" / "probes"

    for X in evals:
        ref, ref_name = load_base_ref(X)
        print(f"\n=== eval {X} | gate ref = {ref_name or 'NONE on this cluster (gate deferred to RCS)'} ===",
              flush=True)
        out_rows = []
        for rung, Xe, spec in pdl.cells_long(sources, [X]):
            if Xe != X or X not in PT:
                continue
            _, X_te = eval_split(PT[X][1])
            if len(X_te) == 0:
                continue
            xlbl = different_label_projection(X)
            per = {m: [] for m in CELLS}; unc_acc = {m: [] for m in CELLS}; yte_ref = None
            srcs = ""
            for sd in seeds:
                train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
                if not train_rows or not test_rows:
                    continue
                n_tr = len(train_rows); tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
                allrows = train_rows + test_rows
                y = np.array([PT[d][2][i] for d, i in allrows], float)
                yte = np.array([y[i] for i in te_idx], float); yte_ref = yte
                states = [PT[d][0][i] for d, i in allrows]
                # ---- ORDER MATCHES probedriftlong (saplma, best_T, uniform, attention) for byte-exact repro ----
                Xmean = np.stack([s.mean(axis=0) for s in states])
                v = {}
                v["meanpool_mlp"] = 1.0 - conf_meanpool(Xmean, tr_idx, te_idx, y, sd)
                best_T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
                armB = train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True)
                v["meanpool_linear"] = np.asarray(attn_unc(armB, states, te_idx, device), float)
                armA = train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T)
                v["attention_linear"] = np.asarray(attn_unc(armA, states, te_idx, device), float)
                # ---- NEW cell: armA's learned-attention pooled vectors -> IDENTICAL SAPLMA MLP ----
                Xattn = attn_pooled(armA, states, list(range(len(states))), device)
                v["attention_mlp"] = 1.0 - conf_meanpool(Xattn, tr_idx, te_idx, y, sd)
                for m in CELLS:
                    per[m].append(results.prr(yte, v[m])); unc_acc[m].append(v[m])
                # realised source composition (last seed, seed-stable)
                _real = {}
                for _d, _i in train_rows:
                    _real[_d] = _real.get(_d, 0) + 1
                srcs = "+".join(f"{d}:{_real.get(d, 0)}" for d in dict.fromkeys(d for d, _c in spec))
                if args.save_pooler and sd == seeds[0]:
                    try:
                        mlp = __import__("luq.probe", fromlist=["train_probe_mlp"]).train_probe_mlp(
                            Xattn[tr_idx], y[tr_idx], seed=sd)
                        pk = probes / f"{cache._slug(MODEL)}__{X}__ID__headagg2x2_attnmlp_{rung}_s{sd}__L{args.layer}.pkl"
                        with open(pk, "wb") as f:
                            pickle.dump({"attn_pooler": armA, "mlp": mlp, "best_T": float(best_T), "rung": rung,
                                         "eval": X, "seed": sd, "layer": args.layer,
                                         "pool_config": "post-TaskA-widened", "cell": "attention_mlp"}, f)
                        print(f"    [save-pooler] {rung} -> {pk.name}", flush=True)
                    except Exception as e:                    # secondary artifact -- loud, but do not lose the CSV
                        print(f"    [save-pooler] WARNING {rung}: {type(e).__name__}: {e} (CSV unaffected)", flush=True)
            if yte_ref is None:
                continue
            stats = {m: (float(np.mean(per[m])), float(np.std(per[m]))) for m in per if per[m]}
            avg = {m: np.mean(np.stack(unc_acc[m]), 0) for m in unc_acc if unc_acc[m]}
            print(f"\n[{rung:14s}] eval={X} ({label_of(X)}) train={srcs}"
                  + ("  [CROSS-LABEL]" if (xlbl and rung != 'ID') else ''), flush=True)
            for m in CELLS:
                gmarg = ""
                if m in GATE_MAP and ref and rung in ref and GATE_MAP[m] in ref[rung]:
                    gmarg = round(stats[m][0] - ref[rung][GATE_MAP[m]], 4)
                    flag = "" if abs(gmarg) <= GATE_TOL else "  <== GATE FAIL"
                    print(f"    {m:16s} {stats[m][0]:+.4f} +/- {stats[m][1]:.3f}   gate({GATE_MAP[m]})={gmarg:+.4f}{flag}", flush=True)
                else:
                    print(f"    {m:16s} {stats[m][0]:+.4f} +/- {stats[m][1]:.3f}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": m,
                                 "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                                 "n_seeds": len(per[m]), "different_label_projection": bool(xlbl and rung != "ID"),
                                 "gate_ref_method": GATE_MAP.get(m, ""), "gate_margin": gmarg})
            for vk, a, b in [("attnmlp_vs_saplma", "attention_mlp", "meanpool_mlp"),
                             ("attnmlp_vs_armA", "attention_mlp", "attention_linear")]:
                if a in avg and b in avg:
                    mg, lo, hi, p, sig = paired_bootstrap(yte_ref, avg[a], avg[b])
                    print(f"    [verdict] {vk:20s} margin {mg:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p:.3f} {'SIG' if sig else 'ns'}", flush=True)
                    out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"VERDICT:{vk}",
                                     "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                     "boot_p": round(p, 4), "significant": bool(sig), "n_seeds": len(seeds)})
        for _r in out_rows:
            _r.update(prov)
        out = Path(args.out) if args.out else (ROOT / "results" / f"head_agg_2x2_{X}__{cache._slug(MODEL)}.csv")
        with open(out, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["rung", "eval", "train", "method", "prr_mean", "prr_std",
                                               "n_seeds", "different_label_projection", "gate_ref_method",
                                               "gate_margin", "ci_lo", "ci_hi", "boot_p", "significant",
                                               "git_sha", "cluster", "env_hash"])
            w.writeheader(); w.writerows(out_rows)
        print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
