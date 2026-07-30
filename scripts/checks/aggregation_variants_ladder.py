#!/usr/bin/env python
"""3A -- POST-HOC aggregation variants off the saved seed-1 attention poolers (NO training).

Our AttnPool head is LINEAR, so logit = W.pooled + b = sum_j a_j (W.x_j) + b: the attention-probe logit is
EXACTLY an attention-weighted average of per-token scalar scores s_j = W.x_j. So MultiMax (eq 9, Kramar et al.
2026), Max-of-Rolling-Means (eq 10), and the score-standardised variants are exact aggregation swaps read off
a single trained pooler, with zero retraining. We reference every variant against `armA_s1` -- the SAME seed-1
pooler's OWN softmax-weighted-average aggregation -- never the CSV 3-seed armA (seed-consistent paired design).

Variants (all on the SAME token set = the G+1 window, same as armA):
  armA_s1   sum_j a_j s_j + b                         (incumbent, paired reference)
  multimax  max_j s_j + b                              (eq 9)
  rolling_w<w>  max over FIXED-WIDTH-w contiguous windows of the window-renormalised weighted mean + b (eq 10)
  rolling_wT    same with w = seq length -> ONE whole-sequence window == armA_s1  (LIMIT GATE, <1e-6)
  zstd_V1   z-score raw scores (x.q), softmax(z/Tz) with ONE global Tz fit on the ID cells' entropy target
  zstd_V2   z-score raw scores, per-example Tz solved so H(a')/log(n) hits the incumbent ID entropy target
            (= Joe idea 1 / C1-revisited: sharpen OOD attention back toward ID sharpness, the untested direction)
  layernorm_ctrl  standardise the POOLED vector (magnitude, not the attention distribution) -> W.pooled_ln + b
            (ablation: if V1/V2 help and this does not, the gain is the attention distribution, not generic norm)

GATES (fail loud): (G1) recomputed a_j == the pooler's own attention (sidecar pool_w) <1e-6; (G2) rolling_wT PRR
== armA_s1 PRR <1e-6; (G3) armA_s1 PRR == the pooler's attn_unc PRR <1e-9. Also reports attention normalised
entropy ID vs OOD per variant (mechanistic check: the ID->OOD entropy gap must SHRINK for V1/V2, be unchanged
for layernorm). CPU; loads one eval's pertok at a time.
"""
import argparse
import csv
import glob
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import attn_pool as ap                                                    # noqa: E402
from aggregation_table import attn_unc                                    # noqa: E402
from xl_rungs import eval_split, label_of                                 # noqa: E402
from luq import cache, results                                           # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15
PROBES = ROOT / "cache" / "probes"
VIZ = ROOT / "cache" / "viz"
OUT_DEFAULT = ROOT / "results" / f"aggregation_variants_3A__{cache._slug(MODEL)}.csv"

EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "expertqa", "samsum", "factscore"]
RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]  # all 5 (poolers now exist at all rungs);
#                                                 (samsum OOD + factscore arrive from DoC JOB 2 -> re-run then)
SEED = 1


def load_pooler(pk):
    """Load a saved-on-CUDA pooler onto CPU; backfill attrs missing from older pickles (single-head)."""
    if not torch.cuda.is_available():
        torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu",
                                                              weights_only=False)
    d = pickle.load(open(pk, "rb"))
    m = d["model"]
    m.eval()
    for a, v in [("n_query", 1), ("n_head", 1), ("frozen_prior", False), ("beta", 1.0),
                 ("use_position", False), ("q_rest", None), ("heads_rest", None), ("pos", None)]:
        if not hasattr(m, a):
            setattr(m, a, v)
    return m, float(d["best_T"])


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def norm_entropy(a):
    n = len(a)
    if n <= 1:
        return 0.0
    h = -(a * np.log(a + 1e-12)).sum()
    return float(h / np.log(n))


def rolling_max(a, s, w):
    """max over FIXED-WIDTH-w contiguous windows of the window-renormalised attention-weighted mean of s.
    Only full-width windows [i, i+w-1] (i=0..n-w) -- so w>=n gives exactly ONE whole-sequence window == armA."""
    n = len(s)
    w = min(w, n)
    num = a * s
    best = -np.inf
    for i in range(0, n - w + 1):
        den = a[i:i + w].sum()
        if den <= 0:
            continue
        v = num[i:i + w].sum() / den
        if v > best:
            best = v
    return float(best)


def solve_Tz(z, target, lo=0.02, hi=80.0, it=60):
    """bisection: find Tz so norm_entropy(softmax(z/Tz)) == target (entropy increases monotonically with Tz)."""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if norm_entropy(softmax(z / mid)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cell_examples(states_full, y_full, split, rung, X):
    """(record_pos list, a_list) for the cell's test examples, preferring the sidecar (gives pool_w + the exact
    positions the pooler was scored on). Returns (record_pos, pool_w_or_None)."""
    key = cache.run_key(MODEL, X, "ID")
    suffix = "" if rung == "ID" else f"__{rung}"
    sc = VIZ / f"{key}__attn{suffix}.npz"
    if sc.exists():
        z = np.load(sc, allow_pickle=True)
        return [int(p) for p in z["record_pos_all"]], list(z["pool_w"])
    # fallback: reconstruct the eval_split test set (finite-y filtered, as build_rows does) -- no a-vectors
    finite = np.isfinite(y_full)
    keep = np.where(finite)[0]
    sub_split = split[keep]
    _, te = eval_split(sub_split)
    return [int(keep[i]) for i in te], None


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--evals", default=",".join(EVALS))
    ap_.add_argument("--rolling-w", type=int, default=10)
    ap_.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap_.parse_args()
    device = "cpu"
    evals = [e for e in args.evals.split(",") if e]

    # discover which (eval,rung) cells have a saved pooler
    cells = []
    for X in evals:
        for rung in RUNGS:
            pk = PROBES / f"{cache._slug(MODEL)}__{X}__ID__attnpool_{rung}_s{SEED}__L{LAYER}.pkl"
            if pk.exists():
                cells.append((X, rung, pk))
    print(f"[3A] {len(cells)} cells with a saved seed-1 pooler:", flush=True)
    for X, rung, _ in cells:
        print(f"     {X:14s} {rung}", flush=True)

    # ---- PASS 1: incumbent ID normalised-entropy target (mean H/logn over ID-cell test examples) ----
    id_ents = []
    per_eval_states = {}   # cache the loaded eval states across passes (one eval at a time is cheaper, but the
    #                        ID pass + main pass both need them; load lazily and keep only the current eval)

    def load_eval(X):
        loaded = ap.load_per_token(MODEL, X, LAYER, label_of(X))
        if loaded is None:
            return None
        states, split, y, _l, records = loaded
        return states, np.asarray(split), np.asarray(y, float)

    # PASS 1 (ID cells only)
    for X, rung, pk in cells:
        if rung != "ID":
            continue
        m, bestT = load_pooler(pk)
        q = m.q.detach().numpy(); scale = float(m.scale); T = float(m.temperature)
        ev = load_eval(X)
        if ev is None:
            continue
        states_full, split, y_full = ev
        rec_pos, pool_w = cell_examples(states_full, y_full, split, rung, X)
        for k, rp in enumerate(rec_pos):
            x = states_full[rp]
            a = np.asarray(pool_w[k]) if pool_w is not None else softmax((x @ q) / (scale * T))
            id_ents.append(norm_entropy(a))
    target_H = float(np.mean(id_ents)) if id_ents else 0.5
    print(f"\n[3A] incumbent ID normalised-entropy target H/logn = {target_H:.4f}  (n={len(id_ents)} ID test ex)",
          flush=True)

    # ---- PASS 1b: fit ONE global Tz for V1 so the z-scored ID attention hits target_H on average ----
    id_z = []   # collect raw z-scored scores on ID cells
    for X, rung, pk in cells:
        if rung != "ID":
            continue
        m, _ = load_pooler(pk)
        q = m.q.detach().numpy()
        ev = load_eval(X)
        if ev is None:
            continue
        states_full, split, y_full = ev
        rec_pos, _ = cell_examples(states_full, y_full, split, rung, X)
        for rp in rec_pos:
            x = states_full[rp]
            raw = x @ q
            id_z.append((raw - raw.mean()) / (raw.std() + 1e-9))

    def mean_id_entropy_at(Tz):
        return float(np.mean([norm_entropy(softmax(z / Tz)) for z in id_z])) if id_z else 0.0

    lo, hi = 0.02, 80.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mean_id_entropy_at(mid) < target_H:
            lo = mid
        else:
            hi = mid
    global_Tz = 0.5 * (lo + hi)
    print(f"[3A] global V1 Tz = {global_Tz:.4f}  (reproduces ID target_H to "
          f"{abs(mean_id_entropy_at(global_Tz)-target_H):.4f})", flush=True)

    # ---- MAIN PASS: all cells, all variants ----
    VARIANTS = ["armA_s1", "multimax", f"rolling_w{args.rolling_w}", "rolling_wT",
                "zstd_V1", "zstd_V2", "layernorm_ctrl"]
    rows = []
    gate_fail = []
    stale_sidecars = []
    # group cells by eval so each eval's pertok loads once
    by_eval = {}
    for X, rung, pk in cells:
        by_eval.setdefault(X, []).append((rung, pk))

    for X, rung_list in by_eval.items():
        ev = load_eval(X)
        if ev is None:
            print(f"  {X}: no pertok -> skip", flush=True); continue
        states_full, split, y_full = ev
        for rung, pk in rung_list:
            m, bestT = load_pooler(pk)
            q = m.q.detach().numpy(); W = m.head.weight.detach().numpy().ravel()
            b = float(m.head.bias.detach().numpy()[0]); scale = float(m.scale); T = float(m.temperature)
            rec_pos, pool_w = cell_examples(states_full, y_full, split, rung, X)
            yte = np.array([y_full[rp] for rp in rec_pos], float)

            unc = {v: np.zeros(len(rec_pos)) for v in VARIANTS}
            ent = {v: [] for v in ["armA_s1", "zstd_V1", "zstd_V2"]}   # entropy tracked for the attention-changing ones
            a_recompute_maxdiff = 0.0
            for k, rp in enumerate(rec_pos):
                x = states_full[rp]                    # (n, d)
                a_re = softmax((x @ q) / (scale * T))  # recomputed incumbent attention
                if pool_w is not None:
                    a_recompute_maxdiff = max(a_recompute_maxdiff,
                                              float(np.max(np.abs(a_re - np.asarray(pool_w[k])))))
                # The .pkl query is AUTHORITATIVE; the cache/viz sidecar's pool_w can go stale (a later dump/rsync
                # rewrote some sidecars while the .pkl stayed put -> G1 caught it). Always recompute `a` from q;
                # the sidecar is used only for record_pos_all (which examples, deterministic) and the G1 warning.
                a = a_re
                s = x @ W                              # per-token scalar scores
                n = len(s)
                # armA
                unc["armA_s1"][k] = 1.0 - 1.0 / (1.0 + np.exp(-((a * s).sum() + b)))
                ent["armA_s1"].append(norm_entropy(a))
                # multimax
                unc["multimax"][k] = 1.0 - 1.0 / (1.0 + np.exp(-(s.max() + b)))
                # rolling
                unc[f"rolling_w{args.rolling_w}"][k] = 1.0 - 1.0 / (1.0 + np.exp(-(rolling_max(a, s, args.rolling_w) + b)))
                unc["rolling_wT"][k] = 1.0 - 1.0 / (1.0 + np.exp(-(rolling_max(a, s, n) + b)))
                # z-score variants
                raw = x @ q
                z = (raw - raw.mean()) / (raw.std() + 1e-9)
                a1 = softmax(z / global_Tz)
                unc["zstd_V1"][k] = 1.0 - 1.0 / (1.0 + np.exp(-((a1 * s).sum() + b)))
                ent["zstd_V1"].append(norm_entropy(a1))
                Tz2 = solve_Tz(z, target_H)
                a2 = softmax(z / Tz2)
                unc["zstd_V2"][k] = 1.0 - 1.0 / (1.0 + np.exp(-((a2 * s).sum() + b)))
                ent["zstd_V2"].append(norm_entropy(a2))
                # layernorm control
                pooled = (a[:, None] * x).sum(0)
                pln = (pooled - pooled.mean()) / (pooled.std() + 1e-9)
                unc["layernorm_ctrl"][k] = 1.0 - 1.0 / (1.0 + np.exp(-(float(W @ pln) + b)))

            prr = {v: results.prr(yte, unc[v]) for v in VARIANTS}
            # G3: armA_s1 == the module's own attn_unc
            m.n_query = 1; m.n_head = 1
            pooler_prr = results.prr(yte, np.asarray(attn_unc(m, states_full, rec_pos, device), float))
            g3 = abs(prr["armA_s1"] - pooler_prr)
            g2 = abs(prr["rolling_wT"] - prr["armA_s1"])
            note = ""
            # GATES. G2 (rolling_wT == armA_s1) and G3 (armA_s1 == the module's own attn_unc) are the CORRECTNESS
            # gates -- both use the authoritative .pkl and are exact to float32 eps (<1e-4), so a failure means the
            # decomposition/reconstruction is wrong -> the cell must not enter the table. G1 (recomputed a vs the
            # sidecar's cached pool_w) is now only a STALE-SIDECAR WARNING: the aggregation uses a_re (from the
            # .pkl), so a stale sidecar does NOT corrupt the numbers -- it just means the viz cache is out of date.
            if pool_w is not None and a_recompute_maxdiff > 1e-5:
                stale_sidecars.append(f"{X}/{rung} (G1 maxdiff {a_recompute_maxdiff:.2e})")
            if g2 > 1e-4:
                gate_fail.append(f"{X}/{rung} G2 rolling_wT vs armA {g2:.2e}")
            if g3 > 1e-4:
                gate_fail.append(f"{X}/{rung} G3 armA vs attn_unc {g3:.2e}")
            print(f"  [{rung:14s}] {X:13s} n={len(rec_pos):4d}  "
                  + "  ".join(f"{v.split('_')[0][:6]}:{prr[v]:+.3f}" for v in VARIANTS)
                  + f"  | G1 {a_recompute_maxdiff:.1e} G2 {g2:.1e} G3 {g3:.1e}", flush=True)
            for v in VARIANTS:
                rows.append({"eval": X, "rung": rung, "method": v, "prr": round(prr[v], 4),
                             "n": len(rec_pos),
                             "mean_norm_entropy": round(float(np.mean(ent[v])), 4) if v in ent else "",
                             "target_H": round(target_H, 4), "global_Tz": round(global_Tz, 4),
                             "g1_amaxdiff": f"{a_recompute_maxdiff:.2e}",
                             "g2_rollingwT_vs_armA": f"{g2:.2e}", "g3_armA_vs_attnunc": f"{g3:.2e}"})

    out = Path(args.out)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["eval", "rung", "method", "prr", "n", "mean_norm_entropy",
                                           "target_H", "global_Tz", "g1_amaxdiff", "g2_rollingwT_vs_armA",
                                           "g3_armA_vs_attnunc"])
        w.writeheader()
        w.writerows(rows)

    # ---- summary: mean PRR per variant, ID vs OOD; entropy-gap check ----
    print("\n" + "=" * 78)
    print("[3A] mean PRR per variant  (ID = ID cells; OOD = LOO-long + DiffTask-long)")
    print("=" * 78)
    print(f"{'method':16s}{'ID':>9s}{'OOD':>9s}{'Δ(OOD-ID)':>11s}   vs armA_s1 OOD")
    def mean_prr(v, rungs):
        xs = [r["prr"] for r in rows if r["method"] == v and r["rung"] in rungs]
        return float(np.mean(xs)) if xs else float("nan")
    armA_ood = mean_prr("armA_s1", {"LOO-long", "DiffTask-long"})
    for v in VARIANTS:
        idm = mean_prr(v, {"ID"}); oodm = mean_prr(v, {"LOO-long", "DiffTask-long"})
        print(f"{v:16s}{idm:>9.3f}{oodm:>9.3f}{oodm-idm:>11.3f}   {oodm-armA_ood:+.3f}")
    print("\n[3A] attention ID->OOD normalised-entropy GAP (mechanism: should SHRINK for V1/V2):")
    for v in ["armA_s1", "zstd_V1", "zstd_V2"]:
        idg = np.mean([r["mean_norm_entropy"] for r in rows if r["method"] == v and r["rung"] == "ID" and r["mean_norm_entropy"] != ""])
        oog = np.mean([r["mean_norm_entropy"] for r in rows if r["method"] == v and r["rung"] in ("LOO-long", "DiffTask-long") and r["mean_norm_entropy"] != ""])
        print(f"   {v:14s} ID {idg:.3f}  OOD {oog:.3f}  gap {oog-idg:+.3f}")

    print("\n[3A] CORRECTNESS GATES (G2 rolling_wT==armA, G3 armA==attn_unc):",
          "ALL PASS" if not gate_fail else "FAIL")
    for g in gate_fail:
        print("   FAIL:", g)
    if stale_sidecars:
        print(f"\n[3A] STALE-SIDECAR WARNINGS ({len(stale_sidecars)} cells): the cache/viz pool_w disagrees with the "
              f".pkl query -> the viz cache is out of date, but the aggregation uses a_re (from the .pkl), so the "
              f"NUMBERS ARE UNAFFECTED. Re-dump these sidecars to refresh the visualiser:")
        for s in stale_sidecars:
            print("   stale:", s)
    print(f"\nwrote {out}  ({len(rows)} rows)")
    if gate_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
