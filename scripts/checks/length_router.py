"""ROUND-3 TASK D — length-gated router: pick the MSP floor or the attention pooler per example, label-free.

§B.3's per-dataset pattern IS a router spec: the pooler wins almost exactly when the floor is weak, and the
floor is weak almost exactly on long output (PART A gives the crossover). This turns that diagnosis into a
method. Unlike Workstream C (attention internals, speculative), this uses results already in hand and needs
no GPU.

DESIGN (guards against the two traps):
  * GATE FEATURES ARE LABEL-FREE AT TEST TIME: output length = len(token_logprobs) (free); floor-distribution
    shape = spread of the per-token probabilities (entropy / max-min of exp(logprob)) (free). No test labels.
  * FIT ON TRAINING DATASETS ONLY, EVALUATE LEAVE-ONE-DATASET-OUT (LODO): a router tuned on the datasets it is
    scored on is an oracle and proves nothing.
  * SCALE: floor uncertainty (NLL-based) and pooler uncertainty ([0,1]) live on different scales, and PRR is a
    RANKING metric, so mixing raw values corrupts the ranking. We z-score each method's vector on the eval set
    (label-free) before the per-example switch -> a comparable hybrid ranking.

BASELINES it must beat: (i) always-floor, (ii) always-pooler, (iii) ORACLE (per-dataset pick the better method
-- the upper bound; the gap to it says how much the label-free gate captures).

TWO PHASES (cache-then-iterate, the project's discipline):
  --extract : per ID dataset, train the pooler + collect {pooler_unc, floor_min, floor_ppl, y, length,
              floor_entropy} on the eval test set -> cache/router/<dataset>.npz  (the GPU/CPU-heavy step)
  (default) : load the cached vectors, run the LODO router + baselines, report per-dataset + cross-dataset
              vs the pre-registered msp_min bar (dual-report perplexity).  (pure numpy, instant)

    python scripts/checks/length_router.py --extract      # once (qsub; CPU on cached pertok states)
    python scripts/checks/length_router.py                # the router analysis
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import msp, results                                   # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
MODEL_SLUG = "meta-llama_Meta-Llama-3.1-8B"
LAYER = 15
# the §B.3 nine (each has an ID eval cell; med_quad/samsum via eval_split carve)
EVALS = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa", "asqa"]
LABEL = {"expertqa": "factuality", "factscore": "factuality"}
CACHE = ROOT / "cache" / "router"


def floor_entropy(token_logprobs):
    """label-free floor-distribution-shape feature: normalised entropy of the per-token probabilities."""
    p = np.exp(np.asarray(token_logprobs, float))
    p = np.clip(p, 1e-12, 1.0)
    q = p / p.sum()
    return float(-(q * np.log(q)).sum() / np.log(len(q))) if len(q) > 1 else 0.0


def extract():
    """Heavy step: per dataset, train the ID pooler and collect the per-example vectors + gate features."""
    import torch                                                # local: heavy import only when extracting
    from attn_pool import load_per_token, train_attn, select_temperature, pad_batch
    from aggregation_table import attn_unc
    from xl_rungs import eval_split
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def pool_attn_entropy(pooler, states, te_idx, bs=64):
        """P6: per-example NORMALISED entropy of the POOLER's attention weights. High = the learned attention
        has dissolved toward uniform (B2 finding) = don't trust the probe -> a label-free gate feature."""
        pooler.eval(); out = np.zeros(len(te_idx))
        for b in range(0, len(te_idx), bs):
            idx = te_idx[b:b + bs]
            X, mask, pos = pad_batch([states[i] for i in idx], device)
            _, a = pooler(X, mask, pos)
            a = a.cpu().numpy()
            for r, i in enumerate(idx):
                T = states[i].shape[0]
                w = a[r, :T]; w = w / max(w.sum(), 1e-12)
                out[b + r] = float(-(w * np.log(w + 1e-12)).sum() / np.log(max(T, 2)))
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    for d in EVALS:
        loaded = load_per_token(MODEL, d, LAYER, LABEL.get(d, "correctness"))
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip", flush=True); continue
        states, split, y, _layer, records = loaded
        tr_idx, te_idx = eval_split(split)                      # baked for core, deterministic carve for XL
        tr_idx, te_idx = list(tr_idx), list(te_idx)
        ok = np.isfinite(y)
        tr_idx = [i for i in tr_idx if ok[i]]; te_idx = [i for i in te_idx if ok[i]]
        if len(tr_idx) < 50 or len(te_idx) < 30:
            print(f"  {d}: too few labelled (tr={len(tr_idx)} te={len(te_idx)}) -> skip", flush=True); continue
        best_T, _ = select_temperature(states, y, tr_idx, device, 1, False, False)
        pooler = train_attn(states, y, tr_idx, device, seed=1, temperature=best_T)   # train once, read twice
        pooler_unc = np.asarray(attn_unc(pooler, states, te_idx, device), float)
        pool_ent = pool_attn_entropy(pooler, states, te_idx)                          # P6 gate feature
        floor_min = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "min") for i in te_idx])
        floor_ppl = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity") for i in te_idx])
        length = np.array([len(records[i]["token_logprobs"]) for i in te_idx], float)
        f_ent = np.array([floor_entropy(records[i]["token_logprobs"]) for i in te_idx])
        yte = np.array([y[i] for i in te_idx], float)
        np.savez_compressed(CACHE / f"{MODEL_SLUG}__{d}.npz",
                            pooler_unc=pooler_unc, floor_min=floor_min, floor_ppl=floor_ppl,
                            length=length, floor_entropy=f_ent, pool_entropy=pool_ent, y=yte)
        print(f"  {d}: cached n_te={len(te_idx)}  pooler_PRR={results.prr(yte, pooler_unc):+.3f}  "
              f"floor_min_PRR={results.prr(yte, floor_min):+.3f}  med_len={np.median(length):.0f}", flush=True)


def zscore(v):
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else v - v.mean()


def hybrid(feat, thr, pooler_unc, floor_unc):
    """per-example switch on a label-free feature, z-scored so the two scales are comparable for ranking."""
    use_pooler = feat > thr
    return np.where(use_pooler, zscore(pooler_unc), zscore(floor_unc))


def load_cells():
    cells = {}
    for d in EVALS:
        p = CACHE / f"{MODEL_SLUG}__{d}.npz"
        if p.exists():
            z = np.load(p)
            cells[d] = {k: z[k] for k in z.files}
    return cells


def fit_threshold(train_cells, feat_key):
    """choose the crossover threshold that maximises MEAN per-dataset PRR of the hybrid across TRAIN datasets."""
    grid = np.percentile(np.concatenate([c[feat_key] for c in train_cells.values()]), np.arange(5, 100, 5))
    best_t, best = grid[0], -9
    for t in grid:
        m = np.mean([results.prr(c["y"], hybrid(c[feat_key], t, c["pooler_unc"], c["floor_min"]))
                     for c in train_cells.values()])
        if m > best:
            best, best_t = m, t
    return float(best_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="heavy step: build per-dataset vector caches")
    ap.add_argument("--feature", default="length", choices=["length", "floor_entropy", "pool_entropy"])
    args = ap.parse_args()
    if args.extract:
        extract(); return
    cells = load_cells()
    if len(cells) < 3:
        raise SystemExit(f"only {len(cells)} cached cells -- run --extract first (qsub pbs/length_router_extract.pbs)")

    print(f"\nLENGTH-GATED ROUTER (LODO) — gate feature = {args.feature}; bar = msp_min (perplexity dual)\n")
    print(f"{'held-out':14s}{'always-floor':>13s}{'always-pool':>12s}{'ROUTER':>9s}{'oracle':>8s}{'thr':>7s}")
    rows = []
    for held in cells:
        train = {d: c for d, c in cells.items() if d != held}
        thr = fit_threshold(train, args.feature)                # fit on the OTHER datasets only
        c = cells[held]
        af = results.prr(c["y"], c["floor_min"])
        ap_ = results.prr(c["y"], c["pooler_unc"])
        rt = results.prr(c["y"], hybrid(c[args.feature], thr, c["pooler_unc"], c["floor_min"]))
        oracle = max(af, ap_)                                    # per-dataset best of the two = upper bound
        rows.append((held, af, ap_, rt, oracle))
        print(f"{held:14s}{af:>13.3f}{ap_:>12.3f}{rt:>9.3f}{oracle:>8.3f}{thr:>7.1f}")
    A = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    print(f"\n{'MEAN':14s}{A[:,0].mean():>13.3f}{A[:,1].mean():>12.3f}{A[:,2].mean():>9.3f}{A[:,3].mean():>8.3f}")
    print(f"\nrouter beats always-floor on {sum(1 for r in rows if r[3] > r[1])}/{len(rows)} datasets, "
          f"always-pooler on {sum(1 for r in rows if r[3] > r[2])}/{len(rows)}; "
          f"gap to oracle = {A[:,3].mean() - A[:,2].mean():+.3f} (mean).")


if __name__ == "__main__":
    main()
