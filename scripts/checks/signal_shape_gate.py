#!/usr/bin/env python
"""ITEM 1 (per-example SIGNAL-SHAPE gating, 3-way {msp_min, perplexity, probe}) + ITEM 2 (length-gated V2),
both label-free, post-hoc on the saved seed-1 poolers, on the ProbeDriftLong OOD cells (router_pdl's 32).

Pre-registration: results/gate_experiments_PREREG.md (written BEFORE this ran). Controls are the point:
  - shuffled-FEATURE control (shuffle each feature within each cell, refit) -> report real − shuffled FIRST.
  - nested length-only gate (the incumbent) -> the gate must beat it, paired.
  - label-free assertion: features are a pure function of the logprobs; asserted in code (no y, no train set).
  - baselines: always-floor / always-perplexity / always-probe / dataset-oracle / per-example oracle (2+3-way).
Paired CIs = per-cell differences bootstrapped over cells. Oracles are DIAGNOSTICS, never method rows.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch                                                                  # noqa: E402
from luq import cache, results, msp                                          # noqa: E402
from length_router import zscore                                             # noqa: E402
import attn_pool as ap                                                        # noqa: E402
from aggregation_table import attn_unc                                        # noqa: E402
from xl_rungs import eval_split, label_of                                     # noqa: E402
from aggregation_variants_ladder import softmax, solve_Tz, norm_entropy       # noqa: E402
import router_pdl as rp                                                       # noqa: E402
from sklearn.linear_model import LogisticRegression                          # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"; LAYER = 15
EVALS = rp.EVALS; OOD = rp.OOD_RUNGS
QA_FAMILY = {"pubmed_qa", "med_quad", "asqa", "expertqa", "factscore"}
SUMM_FAMILY = {"xsum", "cnn_dailymail", "samsum"}
FEATNAMES = ["min_mean_gap", "eff_k", "top_decile_frac", "skew", "kurt", "loglen"]


def signal_features(lp):
    """label-free per-example features from the token logprobs ONLY (no y, no training set)."""
    lp = np.asarray(lp, float); n = len(lp); s = -lp                          # surprise
    mean = float(lp.mean()); mn = float(lp.min())
    ssum = float(s.sum())
    eff_k = (ssum * ssum) / (float(np.sum(s * s)) + 1e-12) if ssum > 0 else float(n)   # participation ratio
    k = max(1, int(np.ceil(0.1 * n))); topdec = float(np.sort(s)[::-1][:k].sum() / (ssum + 1e-12))
    sd = float(lp.std()) + 1e-12
    skew = float(((lp - mean) ** 3).mean() / sd ** 3); kurt = float(((lp - mean) ** 4).mean() / sd ** 4)
    return [mean - mn, eff_k, topdec, skew, kurt, float(np.log(n + 1.0))]


def paired_ci(diffs, n=5000, seed=0):
    d = np.asarray(diffs, float); rng = np.random.RandomState(seed)
    b = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(n)]
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def per_example_oracle_prr(y, cand_z):
    """diagnostic upper bound: correct->lowest-unc candidate, incorrect->highest (z-scored)."""
    Z = np.stack(cand_z); correct = y >= 0.5
    return results.prr(y, np.where(correct, Z.min(0), Z.max(0)))


def oracle_choice(y, cand_z):
    Z = np.stack(cand_z); correct = y >= 0.5
    return np.where(correct, np.argmin(Z, 0), np.argmax(Z, 0))


def main():
    device = "cpu"
    # ---- build per-cell per-example vectors (reusing router_pdl machinery + adding perplexity + features) ----
    raw = {}
    for X in EVALS:
        loaded = ap.load_per_token(MODEL, X, LAYER, label_of(X))
        if loaded is None:
            print(f"  {X}: no pertok -> skip", flush=True); continue
        states, split, y, _l, records = loaded
        raw[X] = (states, np.asarray(split), np.asarray(y, float), records)
    # target_H from ID cells (for V2) -- reuse router_pdl's approach
    ents = []
    for X in EVALS:
        if X not in raw:
            continue
        pk = ROOT / "cache" / "probes" / f"{cache._slug(MODEL)}__{X}__ID__attnpool_ID_s1__L15.pkl"
        if not pk.exists():
            continue
        m = rp.load_pooler(pk); q = m.q.detach().numpy(); scale = float(m.scale); T = float(m.temperature)
        states, split, yv, records = raw[X]
        for rp_i in rp.test_positions(yv, split, X, "ID"):
            ents.append(norm_entropy(softmax((states[rp_i] @ q) / (scale * T))))
    target_H = float(np.mean(ents)) if ents else 0.5
    print(f"[gate] target_H={target_H:.4f}", flush=True)

    cells = {}
    for X in EVALS:
        if X not in raw:
            continue
        states, split, yv, records = raw[X]
        for rung in OOD:
            pk = ROOT / "cache" / "probes" / f"{cache._slug(MODEL)}__{X}__ID__attnpool_{rung}_s1__L15.pkl"
            if not pk.exists():
                continue
            m = rp.load_pooler(pk)
            rpos = rp.test_positions(yv, split, X, rung)
            v = rp.cell_vectors(m, states, rpos, records, target_H)          # pooler_unc, v2, floor_min, length
            ppl = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity") for i in rpos])
            feats = np.array([signal_features(records[i]["token_logprobs"]) for i in rpos])
            yte = np.array([yv[i] for i in rpos], float)
            cells[(X, rung)] = {"y": yte, "floor": v["floor_min"], "ppl": ppl, "probe": v["pooler_unc"],
                                "v2": v["v2"], "length": v["length"], "feat": feats}
            print(f"  built {X:13s} {rung:14s} n={len(rpos)}", flush=True)
    # LABEL-FREE ASSERT (in code, not a comment): (1) signal_features' signature takes ONLY logprobs -- no label,
    # no train set; (2) it is deterministic given the logprobs (recompute == stored). Together these prove every
    # gate feature is computable at test time from the generation alone.
    import inspect
    params = list(inspect.signature(signal_features).parameters)
    assert params == ["lp"], f"signal_features must take only logprobs, got {params}"
    (X0, r0), c0 = next(iter(cells.items()))
    rec0 = raw[X0][3]
    f_re = np.array([signal_features(rec0[i]["token_logprobs"]) for i in rp.test_positions(raw[X0][2], raw[X0][1], X0, r0)])
    assert f_re.shape == c0["feat"].shape and np.allclose(f_re, c0["feat"]), "features not a deterministic fn of logprobs"
    print("[gate] LABEL-FREE ASSERT PASSED: signal_features(lp) — pure function of token_logprobs, no y / no train set.", flush=True)

    datasets = sorted(set(k[0] for k in cells))
    rows = []

    # ================= ITEM 1: 3-way signal-shape gate, LODO =================
    print("\n" + "=" * 84 + "\n[ITEM 1] per-example SIGNAL-SHAPE gate  {msp_min, perplexity, probe}, LODO\n" + "=" * 84)
    CANDS = ["floor", "ppl", "probe"]

    # candidate scores are named so the gate can use a 2-way {floor,probe} OR 3-way {floor,ppl,probe} set. The
    # 3-way is Item-1's registered gate; the 2-way is the RECONCILIATION with §D.3/3B (whose length router routes
    # {floor,probe} only). Same feature, same LODO construction — only the candidate SET differs.
    CAND3 = ["floor", "ppl", "probe"]
    CAND2 = ["floor", "probe"]

    def cand_z(c, keys=CAND3):
        return [zscore(c[k]) for k in keys]

    def gate_prr_lodo(feat_cols, keys=CAND3):
        """train a multinomial logistic on `feat_cols` predicting the per-example oracle choice over `keys`, LODO.
        Returns per-cell {(X,rung): (gate_prr, [always-per-key], dataset_oracle, peo2, peo3)}, nparam."""
        out = {}; nparam = None
        for held in datasets:
            Xtr = []; ytr = []
            for (X, rung), c in cells.items():
                if X == held:
                    continue
                cz = cand_z(c, keys); ch = oracle_choice(c["y"], cz)
                Xtr.append(c["feat"][:, feat_cols]); ytr.append(ch)
            Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            clf = LogisticRegression(max_iter=2000, C=1.0)   # multinomial is the default for >2 classes
            clf.fit((Xtr - mu) / sd, ytr)
            nparam = clf.coef_.size + clf.intercept_.size
            for (X, rung), c in cells.items():
                if X != held:
                    continue
                cz = cand_z(c, keys); Z = np.stack(cz)
                pred = clf.predict((c["feat"][:, feat_cols] - mu) / sd)
                # guard: a class the LODO-train never saw can't be predicted; clip into range
                pred = np.clip(pred, 0, len(keys) - 1)
                comb = Z[pred, np.arange(len(pred))]
                base = [results.prr(c["y"], c[k]) for k in keys]
                cz3 = cand_z(c, CAND3)
                out[(X, rung)] = (results.prr(c["y"], comb), base, max(base),
                                  per_example_oracle_prr(c["y"], [cz3[0], cz3[2]]),
                                  per_example_oracle_prr(c["y"], cz3))
        return out, nparam

    allcols = list(range(len(FEATNAMES)))
    lencol = [FEATNAMES.index("loglen")]
    gate_real, np_real = gate_prr_lodo(allcols)
    gate_len, np_len = gate_prr_lodo(lencol)
    # shuffled-feature control: shuffle each feature WITHIN each cell, refit
    rngc = np.random.RandomState(1)
    for c in cells.values():
        c["_feat_orig"] = c["feat"].copy()
        c["feat"] = np.column_stack([rngc.permutation(c["feat"][:, j]) for j in range(c["feat"].shape[1])])
    gate_shuf, _ = gate_prr_lodo(allcols)
    for c in cells.values():
        c["feat"] = c["_feat_orig"]

    def mean_over(d, idx=0):
        return float(np.mean([v[idx] if idx is not None else v for v in d.values()]))
    g_real = mean_over(gate_real, 0); g_len = mean_over(gate_len, 0); g_shuf = mean_over(gate_shuf, 0)
    aprobe = float(np.mean([v[1][2] for v in gate_real.values()]))
    afloor = float(np.mean([v[1][0] for v in gate_real.values()]))
    appl = float(np.mean([v[1][1] for v in gate_real.values()]))
    dorc = float(np.mean([v[2] for v in gate_real.values()]))
    peo2 = float(np.mean([v[3] for v in gate_real.values()])); peo3 = float(np.mean([v[4] for v in gate_real.values()]))
    # CONTROL MARGINS FIRST
    dshuf = paired_ci([gate_real[k][0] - gate_shuf[k][0] for k in gate_real])
    dlen = paired_ci([gate_real[k][0] - gate_len[k][0] for k in gate_real])
    dprobe = paired_ci([gate_real[k][0] - gate_real[k][1][2] for k in gate_real])
    print(f"\n[ITEM 1] CONTROL MARGINS (reported FIRST; params: real gate {np_real}, length gate {np_len}):")
    print(f"   real − SHUFFLED-feature : {dshuf[0]:+.3f} CI[{dshuf[1]:+.3f},{dshuf[2]:+.3f}]  "
          f"{'REAL (info, not capacity)' if dshuf[1] > 0 else 'NULL — capacity not information'}")
    print(f"   real − LENGTH-only gate : {dlen[0]:+.3f} CI[{dlen[1]:+.3f},{dlen[2]:+.3f}]  "
          f"{'beats incumbent' if dlen[1] > 0 else 'does NOT beat the length gate'}")
    print(f"   real − always-probe     : {dprobe[0]:+.3f} CI[{dprobe[1]:+.3f},{dprobe[2]:+.3f}]")
    print(f"\n[ITEM 1] scores: signal-gate {g_real:+.3f} | length-gate {g_len:+.3f} | shuffled {g_shuf:+.3f} | "
          f"always: floor {afloor:+.3f} ppl {appl:+.3f} probe {aprobe:+.3f}")
    print(f"   DIAGNOSTIC oracles (never a method row): dataset-oracle {dorc:+.3f} | per-ex 2-way {peo2:+.3f} | 3-way {peo3:+.3f}")
    print(f"\n[ITEM 1] per-dataset: signal-gate − length-gate (PREDICTION: biggest on pubmed/cnn/xsum, least on expertqa/med_quad):")
    for X in datasets:
        ks = [k for k in gate_real if k[0] == X]
        d = np.mean([gate_real[k][0] - gate_len[k][0] for k in ks])
        print(f"   {X:14s} Δ(signal−length) {d:+.3f}   (signal {np.mean([gate_real[k][0] for k in ks]):+.3f})")

    # ---- RECONCILE with §D.3/3B (the length gate is +0.239 there, ~+0.177 here) ----
    # Hypothesis: 3-way vs 2-way candidate SET. §D.3/3B routes {floor,probe} only (2-way); this gate adds
    # perplexity as a 3rd option, which gives the LODO gate a way to be WRONG. Re-fit the SAME length gate with a
    # 2-way {floor,probe} candidate set and see if it recovers toward the 3B number.
    gate_len2, _ = gate_prr_lodo(lencol, CAND2)
    gate_real2, _ = gate_prr_lodo(allcols, CAND2)
    g_len2 = mean_over(gate_len2, 0); g_real2 = mean_over(gate_real2, 0)
    print(f"\n[ITEM 1 · RECONCILE with §D.3/3B] length gate: 3-way {{floor,ppl,probe}} {g_len:+.3f}  vs  "
          f"2-way {{floor,probe}} {g_len2:+.3f}   (Δ from dropping perplexity {g_len2 - g_len:+.3f})")
    print(f"                                  signal gate: 3-way {g_real:+.3f}  vs  2-way {g_real2:+.3f}")
    print(f"   §D.3/3B references (2-way, DIFFERENT construction = threshold/sigmoid hybrid, not oracle-choice "
          f"logistic): binary length router +0.239, continuous blend +0.247.")
    print("   READ: if the 2-way length gate > the 3-way, the perplexity candidate is what makes THIS construction "
          "worse; the residual gap to +0.239 is the construction (logistic-on-oracle-choice vs threshold-hybrid).")

    for (X, rung), v in gate_real.items():
        rows.append({"item": 1, "eval": X, "rung": rung, "signal_gate": round(v[0], 4),
                     "length_gate": round(gate_len[(X, rung)][0], 4), "shuffled_gate": round(gate_shuf[(X, rung)][0], 4),
                     "always_floor": round(v[1][0], 4), "always_ppl": round(v[1][1], 4), "always_probe": round(v[1][2], 4),
                     "dataset_oracle": round(v[2], 4), "peo_2way": round(v[3], 4), "peo_3way": round(v[4], 4)})

    # ================= ITEM 2: length-gated V2, LODO, BY FAMILY =================
    print("\n" + "=" * 84 + "\n[ITEM 2] length-gated V2 (C1-revisited, conditional), LODO, BY FAMILY\n" + "=" * 84)

    def fit_len_threshold(train_cells):
        # grid over pooled log-lengths; choose threshold maximising mean train PRR of (short->V2, long->armA)
        alll = np.concatenate([c["length"] for c in train_cells.values()])
        grid = np.percentile(alll, np.arange(10, 95, 5)); best = (-9, grid[0])
        for t in grid:
            m = np.mean([results.prr(c["y"], np.where(c["length"] <= t, zscore(c["v2"]), zscore(c["probe"])))
                         for c in train_cells.values()])
            if m > best[0]:
                best = (m, t)
        return best[1]

    def item2_lodo(shuffle_len=False):
        out = {}
        for held in datasets:
            tr = {k: c for k, c in cells.items() if k[0] != held}
            t = fit_len_threshold(tr)
            for (X, rung), c in cells.items():
                if X != held:
                    continue
                L = c["length"].copy()
                if shuffle_len:
                    np.random.RandomState(2).shuffle(L)
                gated = results.prr(c["y"], np.where(L <= t, zscore(c["v2"]), zscore(c["probe"])))
                out[(X, rung)] = (gated, results.prr(c["y"], c["probe"]), results.prr(c["y"], c["v2"]),
                                  results.prr(c["y"], c["floor"]))
        return out
    g2 = item2_lodo(False); g2s = item2_lodo(True)
    dctrl = paired_ci([g2[k][0] - g2s[k][0] for k in g2])
    print(f"\n[ITEM 2] CONTROL: length-gated V2 − SHUFFLED-length gate = {dctrl[0]:+.3f} CI[{dctrl[1]:+.3f},{dctrl[2]:+.3f}] "
          f"{'(gate uses length info)' if dctrl[1] > 0 else '(NULL — length not used)'}")
    for fam, name in [(QA_FAMILY, "QA/factuality"), (SUMM_FAMILY, "summarisation")]:
        ks = [k for k in g2 if k[0] in fam]
        gated = np.mean([g2[k][0] for k in ks]); armA = np.mean([g2[k][1] for k in ks])
        v2 = np.mean([g2[k][2] for k in ks]); fl = np.mean([g2[k][3] for k in ks])
        dpair = paired_ci([g2[k][0] - g2[k][1] for k in ks])
        print(f"   {name:14s} ({len(ks)} cells): gatedV2 {gated:+.3f} | armA {armA:+.3f} | V2 {v2:+.3f} | floor {fl:+.3f} "
              f"| gated−armA {dpair[0]:+.3f} CI[{dpair[1]:+.3f},{dpair[2]:+.3f}]")
    print("   PREDICTION: gated>armA on QA, gated≈armA on summ. If summ improves too, the gate is not doing what it claims.")
    for (X, rung), v in g2.items():
        rows.append({"item": 2, "eval": X, "rung": rung, "gatedV2": round(v[0], 4), "armA": round(v[1], 4),
                     "V2": round(v[2], 4), "floor": round(v[3], 4),
                     "family": "QA" if X in QA_FAMILY else "summ"})

    # ================= ITEM 3: FAMILY RULE (task identity = label-free INPUT), LODO =================
    print("\n" + "=" * 84 + "\n[ITEM 3] FAMILY RULE — task identity is an INPUT (label-free), LODO\n" + "=" * 84)

    def family_rule_lodo(shuffle_family=False):
        """Learn per-family best of {floor, probe} on TRAIN datasets, apply to held-out by its family, LODO."""
        base_fam = {X: ("QA" if X in QA_FAMILY else "summ") for X in datasets}
        fam = dict(base_fam)
        if shuffle_family:
            vals = list(base_fam.values()); np.random.RandomState(3).shuffle(vals)
            fam = dict(zip(datasets, vals))
        out = {}
        for held in datasets:
            best = {}
            for f in ("QA", "summ"):
                tr = [c for (X, r), c in cells.items() if X != held and fam[X] == f]
                if not tr:
                    best[f] = "probe"; continue
                pf = np.mean([results.prr(c["y"], c["floor"]) for c in tr])
                pp = np.mean([results.prr(c["y"], c["probe"]) for c in tr])
                best[f] = "floor" if pf >= pp else "probe"
            for (X, rung), c in cells.items():
                if X != held:
                    continue
                m = best[fam[X]]
                out[(X, rung)] = (results.prr(c["y"], c[m]), results.prr(c["y"], c["probe"]),
                                  results.prr(c["y"], c["floor"]), m)
        return out
    fr = family_rule_lodo(False); frs = family_rule_lodo(True)
    dctrl3 = paired_ci([fr[k][0] - frs[k][0] for k in fr])
    dprobe3 = paired_ci([fr[k][0] - fr[k][1] for k in fr])
    fr_mean = float(np.mean([v[0] for v in fr.values()])); frs_mean = float(np.mean([v[0] for v in frs.values()]))
    print(f"\n[ITEM 3] CONTROL (reported FIRST): family-rule − SHUFFLED-family = {dctrl3[0]:+.3f} "
          f"CI[{dctrl3[1]:+.3f},{dctrl3[2]:+.3f}]  "
          f"{'(family identity is used)' if dctrl3[1] > 0 else '(NULL — family identity not used)'}")
    print(f"[ITEM 3] family-rule − always-probe = {dprobe3[0]:+.3f} CI[{dprobe3[1]:+.3f},{dprobe3[2]:+.3f}]  "
          f"(ceiling = dataset-oracle +0.252 vs always-probe +0.213)")
    print(f"[ITEM 3] scores: family-rule {fr_mean:+.3f} | shuffled-family {frs_mean:+.3f}")
    for fam_name, famset in [("QA/factuality", QA_FAMILY), ("summarisation", SUMM_FAMILY)]:
        ks = [k for k in fr if k[0] in famset]
        gr = np.mean([fr[k][0] for k in ks]); pr = np.mean([fr[k][1] for k in ks]); flr = np.mean([fr[k][2] for k in ks])
        dp = paired_ci([fr[k][0] - fr[k][1] for k in ks])
        picks = sorted(set(fr[k][3] for k in ks))
        print(f"   {fam_name:14s} ({len(ks)} cells, rule picks {picks}): family-rule {gr:+.3f} | probe {pr:+.3f} "
              f"| floor {flr:+.3f} | rule−probe {dp[0]:+.3f} CI[{dp[1]:+.3f},{dp[2]:+.3f}]")
    print("   PREDICTION: helps QA (floor wins there), no-op on summ (routes to probe). If summ improves, something else.")
    for (X, rung), v in fr.items():
        rows.append({"item": 3, "eval": X, "rung": rung, "family_rule": round(v[0], 4),
                     "shuffled_family": round(frs[(X, rung)][0], 4), "always_probe": round(v[1], 4),
                     "floor": round(v[2], 4), "rule_pick": v[3], "family": "QA" if X in QA_FAMILY else "summ"})

    out = ROOT / "results" / f"gate_experiments__{cache._slug(MODEL)}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["item", "eval", "rung", "col", "value"])
        for r in rows:
            for kk, vv in r.items():
                if kk not in ("item", "eval", "rung"):
                    w.writerow([r["item"], r["eval"], r["rung"], kk, vv])
    print(f"\nwrote {out}  ({len(rows)} cell-rows)")


if __name__ == "__main__":
    main()
