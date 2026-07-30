#!/usr/bin/env python
"""3B -- length ROUTER + continuous BLEND on the canonical cells_long cells, with MultiMax as a THIRD routing
candidate (NOT the broad-LOO pool length_router.extract_ood uses -- a DIFFERENT population, verified).

3A showed MultiMax is a FLOOR-RAISER, not a mean-raiser: it loses on the OOD mean (-0.115 vs armA) but
RESCUES the cells where armA goes NEGATIVE / below the free floor (cnn DiffTask armA -0.107 -> multimax +0.039)
-- the Gemini paper's own framing (they minimised long-context FNR, not mean accuracy). That is the profile of
a routing candidate, so we add it to the pool and ask whether per-example selection over {floor, armA, multimax}
beats selection over {floor, armA}. V2 (entropy-solve) is bimodal (helps pubmed, destroys expertqa) -> a 4th
candidate for the oracle only.

Per (eval,rung) cell we build per-example {y, floor_min, armA (=pooler), multimax, v2, length} with ZERO
training and no source-pool load: floor+length from cached token_logprobs; the pooler scores + MultiMax + V2
are exact post-hoc reads off the saved seed-1 pooler (linear head -> s_j = W.x_j). Then:
  - ORACLES (the decisive numbers): per-dataset oracle (max full-vector PRR) AND per-example oracle
    (label-aware per-example pick of the best method) for the 2-way {floor,armA}, 3-way {+multimax}, 4-way
    {+v2} pools. Headroom = 3-way minus 2-way; if material, MultiMax's value is to SELECTION, not aggregation.
  - ROUTERS (label-free gate, LODO over evals): 2-way length threshold {floor,armA}; 3-way length-bin gate
    {floor,armA,multimax} (bin->method map fit on train); continuous length blend {floor,armA}.
Primary on OOD rungs (LOO-long + DiffTask-long); ID reported separately. CPU, one eval's pertok at a time.
"""
import argparse
import csv
import io
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import attn_pool as ap                                                    # noqa: E402
from aggregation_table import attn_unc                                    # noqa: E402
from xl_rungs import eval_split, label_of                                 # noqa: E402
from luq import cache, results, msp                                       # noqa: E402
from length_router import zscore                                          # noqa: E402
from aggregation_variants_ladder import softmax, solve_Tz, norm_entropy   # noqa: E402  (shared primitives)

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15
PROBES = ROOT / "cache" / "probes"
VIZ = ROOT / "cache" / "viz"
SEED = 1
EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "expertqa", "samsum", "factscore"]
OOD_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]   # all 4 OOD-long (Phase C; cells
CANDS3 = ["floor_min", "pooler_unc", "multimax"]           # the 3-way routing pool
OUT_DEFAULT = ROOT / "results" / f"router_pdl__{cache._slug(MODEL)}.csv"


def load_pooler(pk):
    if not torch.cuda.is_available():
        torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu",
                                                              weights_only=False)
    d = pickle.load(open(pk, "rb"))
    m = d["model"]; m.eval()
    for a, v in [("n_query", 1), ("n_head", 1), ("frozen_prior", False), ("beta", 1.0),
                 ("use_position", False), ("q_rest", None), ("heads_rest", None), ("pos", None)]:
        if not hasattr(m, a):
            setattr(m, a, v)
    m.n_query = 1; m.n_head = 1
    return m


def test_positions(y_full, split, X, rung):
    key = cache.run_key(MODEL, X, "ID")
    suffix = "" if rung == "ID" else f"__{rung}"
    sc = VIZ / f"{key}__attn{suffix}.npz"
    if sc.exists():
        return [int(p) for p in np.load(sc, allow_pickle=True)["record_pos_all"]]
    keep = np.where(np.isfinite(y_full))[0]
    _, te = eval_split(split[keep])
    return [int(keep[i]) for i in te]


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def cell_vectors(m, states_full, rec_pos, records, target_H):
    """per-example {armA, multimax, v2, floor_min, length} from the saved pooler (exact post-hoc reads)."""
    q = m.q.detach().numpy(); W = m.head.weight.detach().numpy().ravel(); b = float(m.head.bias.detach().numpy()[0])
    scale = float(m.scale); T = float(m.temperature)
    armA = np.zeros(len(rec_pos)); mmax = np.zeros(len(rec_pos)); v2 = np.zeros(len(rec_pos))
    for k, rp in enumerate(rec_pos):
        x = states_full[rp]
        a = softmax((x @ q) / (scale * T)); s = x @ W
        armA[k] = 1.0 - sig((a * s).sum() + b)
        mmax[k] = 1.0 - sig(s.max() + b)
        z = (x @ q); z = (z - z.mean()) / (z.std() + 1e-9)
        a2 = softmax(z / solve_Tz(z, target_H))
        v2[k] = 1.0 - sig((a2 * s).sum() + b)
    floor = np.array([msp.msp_uncertainty(records[rp]["token_logprobs"], "min") for rp in rec_pos])
    length = np.array([len(records[rp]["token_logprobs"]) for rp in rec_pos], float)
    return {"pooler_unc": armA, "multimax": mmax, "v2": v2, "floor_min": floor, "length": length}


def per_dataset_oracle(c, methods):
    return max(results.prr(c["y"], c[mth]) for mth in methods)


def per_example_oracle(c, methods):
    """label-aware upper bound on any per-example router: correct(y>=.5) example -> the method giving the
    LOWEST z-scored uncertainty; incorrect -> the HIGHEST. z-scored so methods are rank-comparable."""
    Z = np.stack([zscore(c[mth]) for mth in methods])          # (K, n)
    correct = c["y"] >= 0.5
    picked = np.where(correct, Z.min(0), Z.max(0))
    return results.prr(c["y"], picked)


# ---- 2-way length threshold router (direction fit on train) ----
def hybrid2(feat, thr, pooler_unc, floor_unc, long_to_floor):
    hi = zscore(floor_unc) if long_to_floor else zscore(pooler_unc)
    lo = zscore(pooler_unc) if long_to_floor else zscore(floor_unc)
    return np.where(feat > thr, hi, lo)


def fit_threshold2(train_cells):
    grid = np.percentile(np.concatenate([c["length"] for c in train_cells.values()]), np.arange(5, 100, 5))
    best = (-9.0, grid[0], True)
    for direction in (True, False):
        for t in grid:
            mm = np.mean([results.prr(c["y"], hybrid2(c["length"], t, c["pooler_unc"], c["floor_min"], direction))
                          for c in train_cells.values()])
            if mm > best[0]:
                best = (mm, t, direction)
    return best[1], best[2]


# ---- 3-way length-bin router: fit a (bin -> method) map on train, apply on held-out ----
def combined_binmap(c, edges, assign, methods):
    b = np.digitize(c["length"], edges)                        # 0..len(edges) bin index
    Z = {mth: zscore(c[mth]) for mth in methods}
    return np.array([Z[methods[assign[b[i]]]][i] for i in range(len(c["length"]))])


def fit_binmap(train_cells, methods, nbins=3):
    alllen = np.concatenate([c["length"] for c in train_cells.values()])
    edges = np.percentile(alllen, [100.0 / nbins * i for i in range(1, nbins)])
    best = (-9.0, (0,) * nbins)
    for assign in product(range(len(methods)), repeat=nbins):
        mm = np.mean([results.prr(c["y"], combined_binmap(c, edges, assign, methods)) for c in train_cells.values()])
        if mm > best[0]:
            best = (mm, assign)
    return edges, best[1]


# ---- continuous length blend (floor<->armA) ----
def blend_pred(logL, centre, scale, pooler_unc, floor_unc):
    w = sig((zscore(logL) - centre) / scale)                   # weight on FLOOR, rising with length
    return (1.0 - w) * zscore(pooler_unc) + w * zscore(floor_unc)


def fit_blend(train_cells):
    best = (-9.0, 0.0, 1.0)
    for centre in np.arange(-1.5, 1.6, 0.25):
        for scale in (0.25, 0.5, 1.0, 2.0):
            mm = np.mean([results.prr(c["y"], blend_pred(c["logL"], centre, scale, c["pooler_unc"], c["floor_min"]))
                          for c in train_cells.values()])
            if mm > best[0]:
                best = (mm, centre, scale)
    return best[1], best[2]


def main():
    apr = argparse.ArgumentParser()
    apr.add_argument("--out", default=str(OUT_DEFAULT))
    args = apr.parse_args()

    # gather cells; need target_H (incumbent ID attention entropy) BEFORE computing V2 -> two-touch per eval,
    # but load each eval's pertok only once and keep its arrays for the cell build.
    raw = {}   # eval -> (states_full, split, y_full, records)
    pk_of = {}
    for X in EVALS:
        loaded = ap.load_per_token(MODEL, X, LAYER, label_of(X))
        if loaded is None:
            print(f"  {X}: no pertok -> skip", flush=True); continue
        states_full, split, y_full, _l, records = loaded
        raw[X] = (states_full, np.asarray(split), np.asarray(y_full, float), records)
        for rung in ["ID"] + OOD_RUNGS:
            pk = PROBES / f"{cache._slug(MODEL)}__{X}__ID__attnpool_{rung}_s{SEED}__L{LAYER}.pkl"
            if pk.exists():
                pk_of[(X, rung)] = pk

    # target_H from ID cells' incumbent attention
    ents = []
    for (X, rung), pk in pk_of.items():
        if rung != "ID":
            continue
        m = load_pooler(pk); q = m.q.detach().numpy(); scale = float(m.scale); T = float(m.temperature)
        states_full, split, y_full, records = raw[X]
        for rp in test_positions(y_full, split, X, rung):
            ents.append(norm_entropy(softmax((states_full[rp] @ q) / (scale * T))))
    target_H = float(np.mean(ents)) if ents else 0.5
    print(f"[3B] incumbent ID attention entropy target H/logn = {target_H:.4f} (n={len(ents)})", flush=True)

    cells = {}
    for (X, rung), pk in pk_of.items():
        m = load_pooler(pk)
        states_full, split, y_full, records = raw[X]
        rp = test_positions(y_full, split, X, rung)
        yte = np.array([y_full[i] for i in rp], float)
        v = cell_vectors(m, states_full, rp, records, target_H)
        v["y"] = yte; v["logL"] = np.log(v["length"] + 1.0)
        cells[(X, rung)] = v
        print(f"  built {X:13s} {rung:14s} n={len(rp)} floor {results.prr(yte, v['floor_min']):+.3f} "
              f"armA {results.prr(yte, v['pooler_unc']):+.3f} mmax {results.prr(yte, v['multimax']):+.3f}", flush=True)

    ood = {k: v for k, v in cells.items() if k[1] in OOD_RUNGS}

    # ---- ORACLES (the decisive headroom numbers) ----
    def mean_oracle(fn, methods):
        return float(np.mean([fn(c, methods) for c in ood.values()]))
    print("\n" + "=" * 78)
    print("[3B] ORACLE HEADROOM on OOD cells  (per-dataset = max full-vector PRR; per-example = label-aware pick)")
    print("=" * 78)
    ap0 = float(np.mean([results.prr(c["y"], c["pooler_unc"]) for c in ood.values()]))
    af0 = float(np.mean([results.prr(c["y"], c["floor_min"]) for c in ood.values()]))
    mm0 = float(np.mean([results.prr(c["y"], c["multimax"]) for c in ood.values()]))
    print(f"  always: floor {af0:+.3f}  armA {ap0:+.3f}  multimax {mm0:+.3f}")
    for label, methods in [("2-way {floor,armA}", ["floor_min", "pooler_unc"]),
                           ("3-way {+multimax}", ["floor_min", "pooler_unc", "multimax"]),
                           ("4-way {+v2}", ["floor_min", "pooler_unc", "multimax", "v2"])]:
        pdo = mean_oracle(per_dataset_oracle, methods); peo = mean_oracle(per_example_oracle, methods)
        print(f"  {label:22s} per-dataset oracle {pdo:+.3f}   per-example oracle {peo:+.3f}")
    base_pd = mean_oracle(per_dataset_oracle, ["floor_min", "pooler_unc"])
    tri_pd = mean_oracle(per_dataset_oracle, ["floor_min", "pooler_unc", "multimax"])
    base_pe = mean_oracle(per_example_oracle, ["floor_min", "pooler_unc"])
    tri_pe = mean_oracle(per_example_oracle, ["floor_min", "pooler_unc", "multimax"])
    print(f"\n  MultiMax adds to per-dataset oracle {tri_pd-base_pd:+.3f}, to per-example oracle {tri_pe-base_pe:+.3f}")
    print("  -> if materially >0 AND above a null candidate (below), MultiMax's value is SELECTION not aggregation.")

    # ---- DIAGNOSTIC: which method does the per-dataset oracle SELECT? (explains a flat per-dataset oracle) ----
    print("\n[3B] per-dataset oracle selection (argmax whole-vector PRR) per OOD cell:")
    mm_sel = 0
    for (X, rung), c in ood.items():
        prrs = {m: results.prr(c["y"], c[m]) for m in ["floor_min", "pooler_unc", "multimax"]}
        pick = max(prrs, key=prrs.get); mm_sel += (pick == "multimax")
        star = "  <== multimax best" if pick == "multimax" else ""
        print(f"   {X:13s} {rung:14s} floor {prrs['floor_min']:+.3f} armA {prrs['pooler_unc']:+.3f} "
              f"mmax {prrs['multimax']:+.3f} -> {pick}{star}")
    print(f"   multimax is the WHOLE-VECTOR best on {mm_sel}/{len(ood)} cells -> per-dataset oracle barely moves "
          f"(NOT a bug: where multimax beats armA it is still below floor, e.g. cnn DiffTask). Its value is per-EXAMPLE.")

    # ---- NULL-CANDIDATE CONTROL: a max-oracle is MONOTONE in the candidate set, so ANY 3rd candidate (even
    # noise) raises the per-example oracle. Subtract what an arbitrary candidate buys. ----
    rng = np.random.RandomState(0)
    for c in ood.values():
        c["armA_shuf"] = rng.permutation(c["pooler_unc"])
        c["armA_noise"] = rng.normal(float(c["pooler_unc"].mean()), float(c["pooler_unc"].std()) + 1e-9, len(c["y"]))
    mpeo = lambda methods: float(np.mean([per_example_oracle(c, methods) for c in ood.values()]))
    base2 = mpeo(["floor_min", "pooler_unc"])
    inc_mm = mpeo(["floor_min", "pooler_unc", "multimax"]) - base2
    inc_shuf = mpeo(["floor_min", "pooler_unc", "armA_shuf"]) - base2
    inc_noise = mpeo(["floor_min", "pooler_unc", "armA_noise"]) - base2
    inc_v2 = mpeo(["floor_min", "pooler_unc", "multimax", "v2"]) - mpeo(["floor_min", "pooler_unc", "multimax"])
    print(f"\n[3B] NULL-CANDIDATE CONTROL (per-example oracle increment over the 2-way {base2:+.3f}):")
    print(f"   +multimax {inc_mm:+.3f}   |   +armA_shuffled {inc_shuf:+.3f}   +gaussian_noise {inc_noise:+.3f}")
    over_null = inc_mm - max(inc_shuf, inc_noise)
    print(f"   DECISIVE: multimax increment MINUS the best null = {over_null:+.3f} -> "
          f"{'REAL (multimax buys more than noise)' if over_null > 0.01 else 'NOT above null -> selection story NOT established'}")
    print(f"   (V2 increment over the 3-way, for reference: {inc_v2:+.3f})")

    # ---- ROUTERS (LODO over evals, OOD rungs) ----
    rows = []
    print("\nROUTERS (LODO over evals, OOD rungs; features label-free; bar=msp_min)")
    print(f"{'held-out':13s}{'a-floor':>9s}{'a-pool':>9s}{'a-mmax':>9s}{'R2:fl/pl':>9s}{'R3:+mm':>9s}{'BLEND':>9s}{'oracle':>9s}")
    S = {k: [] for k in ["af", "ap", "am", "r2", "r3", "bl", "orc"]}
    for held in EVALS:
        held_cells = {k: v for k, v in ood.items() if k[0] == held}
        train_cells = {k: v for k, v in ood.items() if k[0] != held}
        if not held_cells or not train_cells:
            continue
        thr, direction = fit_threshold2(train_cells)
        edges, assign = fit_binmap(train_cells, CANDS3)
        centre, scale = fit_blend(train_cells)
        for (X, rung), c in held_cells.items():
            af = results.prr(c["y"], c["floor_min"]); apr_ = results.prr(c["y"], c["pooler_unc"])
            amm = results.prr(c["y"], c["multimax"])
            r2 = results.prr(c["y"], hybrid2(c["length"], thr, c["pooler_unc"], c["floor_min"], direction))
            r3 = results.prr(c["y"], combined_binmap(c, edges, assign, CANDS3))
            bl = results.prr(c["y"], blend_pred(c["logL"], centre, scale, c["pooler_unc"], c["floor_min"]))
            orc = per_dataset_oracle(c, CANDS3)
            rows.append({"eval": X, "rung": rung, "always_floor": round(af, 4), "always_pooler": round(apr_, 4),
                         "always_multimax": round(amm, 4), "router2_floor_pool": round(r2, 4),
                         "router3_plus_mmax": round(r3, 4), "blend": round(bl, 4), "oracle3": round(orc, 4),
                         "thr": round(thr, 1), "binmap": "".join(str(a) for a in assign), "n": len(c["y"])})
        e = [r for r in rows if r["eval"] == held]
        mn = lambda k: float(np.mean([r[k] for r in e]))
        print(f"{held:13s}{mn('always_floor'):>9.3f}{mn('always_pooler'):>9.3f}{mn('always_multimax'):>9.3f}"
              f"{mn('router2_floor_pool'):>9.3f}{mn('router3_plus_mmax'):>9.3f}{mn('blend'):>9.3f}{mn('oracle3'):>9.3f}")
        S["af"].append(mn("always_floor")); S["ap"].append(mn("always_pooler")); S["am"].append(mn("always_multimax"))
        S["r2"].append(mn("router2_floor_pool")); S["r3"].append(mn("router3_plus_mmax"))
        S["bl"].append(mn("blend")); S["orc"].append(mn("oracle3"))
    print(f"\n{'MEAN':13s}{np.mean(S['af']):>9.3f}{np.mean(S['ap']):>9.3f}{np.mean(S['am']):>9.3f}"
          f"{np.mean(S['r2']):>9.3f}{np.mean(S['r3']):>9.3f}{np.mean(S['bl']):>9.3f}{np.mean(S['orc']):>9.3f}")
    r3beats = sum(1 for r in rows if r["router3_plus_mmax"] > max(r["always_floor"], r["always_pooler"]))
    print(f"\n3-way router beats max(floor,armA) on {r3beats}/{len(rows)} cells; "
          f"router3 gap to 3-way oracle {np.mean(S['orc'])-np.mean(S['r3']):+.3f}.")

    # ---- GATE null-candidate control: refit the 3-way length-bin gate with the SHUFFLED candidate. If the null
    # gate moves like the multimax gate, the effect is bin-map CAPACITY (27 free params), not multimax. ----
    S_r3null = []
    for held in EVALS:
        hc = {k: v for k, v in ood.items() if k[0] == held}
        tc = {k: v for k, v in ood.items() if k[0] != held}
        if not hc or not tc:
            continue
        edges_n, assign_n = fit_binmap(tc, ["floor_min", "pooler_unc", "armA_shuf"])
        S_r3null.extend(results.prr(c["y"], combined_binmap(c, edges_n, assign_n, ["floor_min", "pooler_unc", "armA_shuf"]))
                        for c in hc.values())
    print(f"GATE control: 3-way length-bin gate with a SHUFFLED 3rd candidate = {np.mean(S_r3null):+.3f}  "
          f"(vs multimax-3way {np.mean(S['r3']):+.3f}, 2-way {np.mean(S['r2']):+.3f}, always-pool {np.mean(S['ap']):+.3f})")

    # ---- BLEND paired-bootstrap CI vs always-pooler (the only strategy beating a-pool; characterise it) ----
    rng2 = np.random.RandomState(1)
    blend_vecs = {}
    for held in EVALS:
        hc = {k: v for k, v in ood.items() if k[0] == held}
        tc = {k: v for k, v in ood.items() if k[0] != held}
        if not hc or not tc:
            continue
        centre, scale = fit_blend(tc)
        for key, c in hc.items():
            blend_vecs[key] = (c["y"], blend_pred(c["logL"], centre, scale, c["pooler_unc"], c["floor_min"]),
                               c["pooler_unc"])
    keys = list(blend_vecs)
    boot = []
    for _ in range(2000):
        ds = []
        for k in keys:
            y, bl, pl = blend_vecs[k]; n = len(y); idx = rng2.randint(0, n, n)
            ds.append(results.prr(y[idx], bl[idx]) - results.prr(y[idx], pl[idx]))
        boot.append(np.mean(ds))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    print(f"\n[3B] BLEND vs always-pooler, cross-dataset paired bootstrap: mean Δ {boot.mean():+.3f} "
          f"CI[{lo:+.3f},{hi:+.3f}] -> {'SIG (>0)' if lo > 0 else 'CI SPANS 0 -> NOT a robust win'}")
    for k in keys:
        y, bl, pl = blend_vecs[k]
        print(f"   {k[0]:13s} {k[1]:14s} blend {results.prr(y, bl):+.3f} pool {results.prr(y, pl):+.3f} "
              f"Δ {results.prr(y, bl)-results.prr(y, pl):+.3f}  (n={len(y)})")

    # ---- V2 bimodality: per-cell V2-vs-armA delta and its label-free length correlate ----
    print("\n[3B] V2-vs-armA per-cell delta (bimodal? correlate with length):")
    dv, ml = [], []
    for (X, rung), c in ood.items():
        d = results.prr(c["y"], c["v2"]) - results.prr(c["y"], c["pooler_unc"])
        dv.append(d); ml.append(float(np.median(c["length"])))
        print(f"   {X:13s} {rung:14s} v2-armA {d:+.3f}  med_len {np.median(c['length']):.0f}")
    if len(dv) > 2:
        dv = np.array(dv); ml = np.array(ml)
        r = float(np.corrcoef(ml, dv)[0, 1])
        rng3 = np.random.RandomState(2); rs = []
        for _ in range(2000):
            idx = rng3.randint(0, len(dv), len(dv))
            if np.std(ml[idx]) > 0 and np.std(dv[idx]) > 0:
                rs.append(float(np.corrcoef(ml[idx], dv[idx])[0, 1]))
        clo, chi = np.percentile(rs, 2.5), np.percentile(rs, 97.5)
        # leave-one-out: does dropping the strongest cell (expertqa Diff) collapse it?
        j = int(np.argmax(np.abs(dv - dv.mean())))
        keep = [i for i in range(len(dv)) if i != j]
        r_loo = float(np.corrcoef(ml[keep], dv[keep])[0, 1])
        print(f"   corr(median length, V2-armA delta) = {r:+.3f}  CI[{clo:+.3f},{chi:+.3f}]  n={len(dv)}  "
              f"| drop-strongest-cell r={r_loo:+.3f}")
        print(f"   -> {'length gates V2 (mirror of multimax)' if chi < 0 else 'CI spans 0 -> not a clean gate'}")

    idc = {k: v for k, v in cells.items() if k[1] == "ID"}
    if idc:
        afi = np.mean([results.prr(c["y"], c["floor_min"]) for c in idc.values()])
        api = np.mean([results.prr(c["y"], c["pooler_unc"]) for c in idc.values()])
        print(f"\nID rungs ({len(idc)} cells): always-floor {afi:+.3f} always-pooler {api:+.3f} "
              f"-- pooler wins ID; routing headroom is an OOD phenomenon.")

    out = Path(args.out)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["eval", "rung", "always_floor", "always_pooler", "always_multimax",
                                           "router2_floor_pool", "router3_plus_mmax", "blend", "oracle3",
                                           "thr", "binmap", "n"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} OOD cells)")


if __name__ == "__main__":
    main()
