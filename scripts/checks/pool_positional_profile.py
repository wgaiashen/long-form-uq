"""FIGURE 4 (P3): positional profile of the pooler's attention — mean weight vs RELATIVE position, per
(dataset, rung). Tests whether the pooler keys on POSITION rather than content (pubmed's ID peak sits at
relative position ~0.11 — a structural cue). Relative position (0=first gen token, 1=last) makes datasets of
different length comparable.

Reads the committed viz sidecars (`cache/viz/<key>__attn[__<rung>].npz`); reuses the ID-vs-OOD sidecar glob
and the regime record dirs. Output: a per-bin CSV (robust) + a matplotlib PNG if matplotlib is importable.

    python scripts/checks/pool_positional_profile.py --datasets pubmed_qa,cnn_dailymail,xsum,expertqa
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luq import cache  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}
NBIN = 10


def sidecars_for(dataset, cache_dir):
    key = cache.run_key(MODEL, dataset, "ID")
    out = {}
    for p in sorted(glob.glob(str(cache_dir / "viz" / f"{key}__attn*.npz"))):
        stem = Path(p).name[len(f"{key}__attn"):-len(".npz")]
        out["ID" if stem == "" else stem.lstrip("_")] = p
    return out


def profile(sidecar_path, records):
    """mean pooler weight per relative-position bin (NBIN), averaged over examples (each example's gen-token
    weights re-normalised to sum 1, then accumulated into the bin its relative position falls in)."""
    z = np.load(sidecar_path, allow_pickle=True)
    pool = dict(zip(z["record_pos_all"].tolist(), z["pool_w"]))
    acc = np.zeros(NBIN); cnt = np.zeros(NBIN)
    n = 0
    for i, w in pool.items():
        w = np.asarray(w, float)
        if w.sum() <= 0:
            continue
        gids = records[i]["gen_token_ids"]; G = len(gids)
        if len(w) != G + 1 or G < 2:
            continue
        gw = w[1:] / w[1:].sum() if w[1:].sum() > 0 else w[1:]
        rel = np.arange(G) / (G - 1)                       # 0..1 relative position
        b = np.minimum((rel * NBIN).astype(int), NBIN - 1)
        for k in range(G):
            acc[b[k]] += gw[k]; cnt[b[k]] += 1
        n += 1
    mean = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    # scale to "× uniform": a flat pooler puts 1/G per token; per bin that is (#tokens in bin)/G of the mass,
    # so mean-weight-per-token = 1/G on average -> we report mean-weight-per-token / (1/G_typical). Simpler and
    # length-robust: report the per-bin mean weight normalised so a flat profile = 1.0.
    flat = np.nanmean(mean) if np.isfinite(mean).any() else 1.0
    return (mean / flat if flat else mean), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="pubmed_qa,cnn_dailymail,xsum,expertqa")
    ap.add_argument("--cache-dir", default=str(ROOT / "cache"))
    ap.add_argument("--out-csv", default=str(ROOT / "results" / "pool_positional_profile.csv"))
    ap.add_argument("--out-png", default=str(ROOT / "results" / "viz" / "FIG4_positional_profile.png"))
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)
    centres = (np.arange(NBIN) + 0.5) / NBIN
    curves = {}
    import csv as _csv
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        wr = _csv.writer(f); wr.writerow(["dataset", "rung", "n"] + [f"bin{b}" for b in range(NBIN)])
        for d in args.datasets.split(","):
            scs = sidecars_for(d, cache_dir)
            if not scs:
                print(f"  {d}: no sidecar -> skip"); continue
            rec_dir = cache_dir / REGIME[d] if d in REGIME else cache_dir
            recs = cache.load_records(rec_dir, cache.run_key(MODEL, d, "ID"))
            # rung-family-agnostic (S1/P0 fix): iterate the sidecars present, ID first, so the honest
            # "-long" OOD names are used instead of the old stripped "LOO"/"DiffTask".
            for rung in (["ID"] if "ID" in scs else []) + sorted(k for k in scs if k != "ID"):
                curve, n = profile(scs[rung], recs)
                curves[f"{d}/{rung}"] = curve
                wr.writerow([d, rung, n] + [f"{v:.4f}" for v in curve])
                peak_bin = int(np.nanargmax(curve))
                print(f"  {d:14s} {rung:9s} n={n:5d}  peak bin {peak_bin} (rel-pos ~{centres[peak_bin]:.2f}), "
                      f"×uniform {np.nanmax(curve):.2f}", flush=True)
    print(f"wrote {args.out_csv}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        for name, c in curves.items():
            ls = "-" if name.endswith("/ID") else "--"
            plt.plot(centres, c, ls, marker=".", label=name)
        plt.axhline(1.0, color="grey", lw=0.8, ls=":")
        plt.xlabel("relative position in generation (0=first, 1=last)")
        plt.ylabel("mean pooler weight (× uniform; 1.0 = flat)")
        plt.title("Figure 4 — pooler attention positional profile (ID solid, OOD dashed)")
        plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(args.out_png, dpi=130)
        print(f"wrote {args.out_png}")
    except Exception as e:
        print(f"(matplotlib unavailable: {e} — CSV written, plot it yourself)")


if __name__ == "__main__":
    main()
