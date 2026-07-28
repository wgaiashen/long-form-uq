"""S4.2 — is per-example ATTENTION ENTROPY a usable uncertainty score in its own right?

Reframed from an "is-this-OOD AUROC" (which needs OOD labels): instead ask whether the pooler's per-example
attention entropy PREDICTS PROBE ERROR. High entropy = the attention dissolved = the pooler is near mean-pool
= (hypothesis) less trustworthy. Score it exactly like any UQ method: PRR(correctness, entropy), per dataset,
ID and OOD. Label-free at test time. If PRR > 0 it is a free contribution; if ≈0 it is not.

Reuses the SAME sidecars as the dissolution join (pool_w per example) + the judge correctness label. No
training, no GPU — a light read over cache/viz sidecars + records.

    python scripts/checks/entropy_prr.py
"""
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, results                                              # noqa: E402
from luq.config import Config                                              # noqa: E402
from xl_rungs import label_of                                             # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}


def records_for(ds):
    cfg = Config(model_name=MODEL, dataset=ds, ood_setting="ID", prompt_regime=REGIME.get(ds, ""))
    return cache.load_records(cfg.cache_dir, cache.run_key(MODEL, ds, "ID"))


def per_example_entropy(pool_w, record):
    """Normalised entropy H/log(T) for one example, matching pool_attention_ood_diag (full G+1 window)."""
    w = np.asarray(pool_w, float)
    if w.sum() <= 0:
        return None
    w = w / w.sum()
    G = len(record["gen_token_ids"])
    if len(w) != G + 1:
        return None
    T = len(w)
    H = float(-(w * np.log(w + 1e-12)).sum())
    return H / np.log(T)


def cell_prr(sidecar, records, label_field):
    z = np.load(sidecar, allow_pickle=True)
    pos = z["record_pos_all"].tolist(); pw = z["pool_w"]
    ent, y = [], []
    for p, w in zip(pos, pw):
        e = per_example_entropy(w, records[int(p)])
        c = records[int(p)].get(label_field, np.nan)
        if e is None or not np.isfinite(c):
            continue
        ent.append(e); y.append(float(c))
    if len(y) < 20:
        return None, len(y)
    # high entropy -> uncertain; PRR expects higher = more uncertain
    return results.prr(np.array(y), np.array(ent)), len(y)


def main():
    viz = ROOT / "cache" / "viz"
    datasets = sorted({Path(p).name.split("__")[1] for p in glob.glob(str(viz / f"{SLUG}__*__attn*.npz"))})
    print(f"{'dataset':13s} {'rung':16s} {'n':>5s} {'entropy-PRR':>12s}")
    for ds in datasets:
        recs = records_for(ds); lf = label_of(ds)
        key = cache.run_key(MODEL, ds, "ID")
        for sc in sorted(glob.glob(str(viz / f"{key}__attn*.npz"))):
            stem = Path(sc).name[len(f"{key}__attn"):-len(".npz")]
            rung = "ID" if stem == "" else stem.lstrip("_")
            prr, n = cell_prr(sc, recs, lf)
            if prr is None:
                continue
            print(f"{ds:13s} {rung:16s} {n:5d} {prr:+12.3f}")


if __name__ == "__main__":
    main()
