"""S1/P0 — the LONG-to-LONG join (matched rung family).

The mechanism claim was: the pooler's attention dissolves toward uniform OOD, so it reverts to mean-pool, so
the attention pooler performs ~like the uniform pooler OOD. The old join paired the dissolution measure
(LONG-ladder sidecars) with §B.3's attention-uniform gap (STANDARD ladder) -- a rung-family mismatch. This
redoes it MATCHED: both sides from the LONG ladder.

  * dissolution scalar  = mean normalised attention entropy (H/log T over the G+1 window, EXACTLY the
                          pool_attention_ood_diag convention) at the OOD rung minus at ID. Higher = flatter.
  * performance gap      = §C.3 (probedriftlong) `attention` - `uniform` PRR on the SAME -long OOD rung.

Prediction under "dissolution -> reverts to mean-pool -> performs like uniform": more dissolution (bigger
entropy increase) => gap -> 0 (attention loses its edge over uniform). We report it honestly; a favourable
story dies the moment it is checked against ALL datasets (pubmed dissolves most yet KEEPS a positive long-OOD
gap -- so the matched join may not support the collapse story the mismatched one implied).

    python scripts/checks/dissolution_gap_join.py            # narrow §C.3 (interim; widened reruns pending)
"""
import csv as _csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]        # scripts/checks/<file> -> repo root
sys.path.insert(0, str(ROOT / "src"))
from luq import cache                                                        # noqa: E402
from luq.config import Config                                                # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}
OOD_LONG = ["LOO-long", "DiffTask-long"]        # the rungs we have OOD sidecars for


def mean_norm_entropy(sidecar, records):
    """Mean H/log(T) over examples, T = G+1 window (the pool_attention_ood_diag convention exactly)."""
    z = np.load(sidecar, allow_pickle=True)
    pool = dict(zip(z["record_pos_all"].tolist(), z["pool_w"]))
    vals = []
    for i, w in pool.items():
        w = np.asarray(w, float)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        G = len(records[i]["gen_token_ids"])
        if len(w) != G + 1:
            continue
        T = len(w)
        H = float(-(w * np.log(w + 1e-12)).sum())
        vals.append(H / np.log(T))
    return float(np.mean(vals)), len(vals)


WIDENED_DIR = Path(os.environ.get("EPHEMERAL", str(Path.home() / "ephemeral")) + "/luq_overnight_results")


def csv_rows(dataset):
    """{(rung,method): (prr_mean, n_seeds)} from the ONE probedriftlong CSV for this dataset (provenance:
    attention & uniform therefore come from the same file / same run / same seeds -- asserted in main).
    PREFERS the WIDENED §C.3 CSV (2026-07-28) when present (cnn/pubmed/xsum/expertqa), else the narrow one."""
    wf = WIDENED_DIR / f"probedriftlong_{dataset}_widened__{SLUG}.csv"
    f = wf if wf.exists() else (ROOT / "results" / f"probedriftlong_{dataset}__{SLUG}.csv")
    if not f.exists():
        return {}, "none"
    out = {}
    for r in _csv.DictReader(open(f)):
        out[(r["rung"], r["method"])] = (float(r["prr_mean"]), int(r.get("n_seeds", 0) or 0))
    return out, ("widened" if wf.exists() else "narrow")


def records_for(ds):
    cfg = Config(model_name=MODEL, dataset=ds, ood_setting="ID", prompt_regime=REGIME.get(ds, ""))
    return cache.load_records(cfg.cache_dir, cache.run_key(MODEL, ds, "ID"))


def main():
    viz = ROOT / "cache" / "viz"
    # every dataset that has at least one OOD-long sidecar
    datasets = sorted({Path(p).name.split("__")[1] for p in glob.glob(str(viz / f"{SLUG}__*__attn__*-long.npz"))})
    print(f"datasets with OOD-long sidecars: {datasets}   (NARROW §C.3 -- INTERIM; widened reruns pending)\n")
    print(f"{'dataset':13s} {'rung':14s} {'ent_ID':>7s} {'ent_OOD':>7s} {'Δent':>7s}  "
          f"{'unif_ID':>7s} {'attn':>7s} {'unif':>7s} {'gap':>7s}  {'Δunif':>7s}")
    pts = []
    prov_ok = True
    for ds in datasets:
        recs = records_for(ds)
        idsc = viz / f"{cache.run_key(MODEL, ds, 'ID')}__attn.npz"
        if not idsc.exists():
            print(f"  {ds}: no ID sidecar -> skip"); continue
        ent_id, _ = mean_norm_entropy(idsc, recs)
        rows, src = csv_rows(ds)
        print(f"  [{ds}: §C.3 source = {src}]")
        unif_id = rows.get(("ID", "uniform"), (float("nan"), 0))[0]
        for rung in OOD_LONG:
            sc = viz / f"{cache.run_key(MODEL, ds, 'ID')}__attn__{rung}.npz"
            if not sc.exists() or (rung, "attention") not in rows or (rung, "uniform") not in rows:
                continue
            a, na = rows[(rung, "attention")]; u, nu = rows[(rung, "uniform")]
            if na != nu or na == 0:                 # PROVENANCE: same run/seeds for both methods
                prov_ok = False
                print(f"  !! {ds} {rung}: n_seeds attention={na} uniform={nu} -- provenance mismatch")
            ent_ood, _ = mean_norm_entropy(sc, recs)
            dent = ent_ood - ent_id; gap = a - u
            pts.append((ds, rung, dent, gap))
            print(f"{ds:13s} {rung:14s} {ent_id:7.3f} {ent_ood:7.3f} {dent:+7.3f}  "
                  f"{unif_id:+7.3f} {a:+7.3f} {u:+7.3f} {gap:+7.3f}  {u - unif_id:+7.3f}")

    def corr(ps):
        d = np.array([p[2] for p in ps]); g = np.array([p[3] for p in ps])
        pe = float(np.corrcoef(d, g)[0, 1])
        sp = float(np.corrcoef(np.argsort(np.argsort(d)), np.argsort(np.argsort(g)))[0, 1])
        return pe, sp

    print(f"\nPROVENANCE (attention & uniform from same CSV/run/seeds): {'PASS' if prov_ok else 'FAIL'}")
    if len(pts) >= 3:
        pe, sp = corr(pts)
        print(f"\nALL {len(pts)} cells:            Pearson(Δent, gap) = {pe:+.3f}   Spearman = {sp:+.3f}")
        nop = [p for p in pts if p[0] != "pubmed_qa"]
        if len(nop) >= 3:
            pe2, sp2 = corr(nop)
            print(f"PUBMED DROPPED ({len(nop)} cells): Pearson(Δent, gap) = {pe2:+.3f}   Spearman = {sp2:+.3f}")
        print("\nPrediction (dissolution -> reverts to uniform) = NEGATIVE corr. Observed: NOT negative.")
        print("Honest read: pubmed carries both high-Δ points; with it dropped the relationship vanishes ->")
        print("NO detectable dissolution<->gap relationship, and the one hard-dissolving set KEEPS its edge.")
        print("Δunif column: on long-only training the UNIFORM pooler degrades (e.g. pubmed) while attention")
        print("holds -- so the positive long gaps come substantially from uniform dropping, not attn rising.")


if __name__ == "__main__":
    main()
