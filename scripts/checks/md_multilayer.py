#!/usr/bin/env python
"""Pass two of the layer sensitivity: the published regression over several layers' distances.

Registered in prereg/M9_layer_distance_sensitivity.md. Every row this produces is a LAYER
SENSITIVITY, never a reproduction of the published methods: the reference aggregates every layer and
this aggregates a frozen subset.

The regression itself IS the published one, taken from run_polygraph.py:360-380 and
average_token_mahalanobis_distance.py: a ten-component reduction of the layer-wise distance matrix,
fitted on the development half and only transformed at evaluation, then an unconstrained ridge on
target `1 - correctness`.

    python scripts/checks/md_multilayer.py --model meta-llama/Meta-Llama-3.1-8B \
        --layers 0,3,6,9,12,15,18,21,24,27,30
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from sklearn.decomposition import PCA                                           # noqa: E402
from sklearn.linear_model import Ridge                                          # noqa: E402

from luq import cache, results                                                  # noqa: E402
from luq import mahalanobis as MD                                               # noqa: E402
from md_hybrids import N_COMPONENTS                                             # noqa: E402


def prr_or_blank(y, vec, label, degen):
    """A constant score is ranked by row position and would score whatever that permutation gives.
    Blank plus an explicit count, never a number -- the same rule the hybrid driver applies."""
    vec = np.asarray(vec, float)
    if np.isfinite(vec).all() and float(np.ptp(vec)) == 0.0:
        degen[label] = degen.get(label, 0) + 1
        return float("nan")
    return results.prr(y, vec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--scan-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="11layer", help="suffix for the method names in the output")
    args = ap.parse_args()

    slug = cache._slug(args.model)
    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    scan = ROOT / (args.scan_dir or f"results/hybrids/mdscan__{slug}")
    if not scan.is_dir():
        sys.exit(f"FATAL: no pass-one output at {scan}. Run md_layer_scan.py for every layer first.")

    # Which cells does EVERY layer have? A cell present at some layers and not others would give the
    # regression a different feature set per cell, which is not the registered experiment.
    per_layer = {}
    for L in layers:
        per_layer[L] = {p.name.split("__", 1)[1] for p in scan.glob(f"L{L}__*__{slug}.npz")}
    common = set.intersection(*per_layer.values()) if per_layer else set()
    for L in layers:
        extra = len(per_layer[L] - common)
        print(f"  L{L:<3d} {len(per_layer[L]):3d} cells" + (f"  ({extra} not shared)" if extra else ""))
    if not common:
        sys.exit("FATAL: no cell is present at every requested layer -- nothing can be combined.")
    incomplete = sorted(set().union(*per_layer.values()) - common)
    if incomplete:
        print(f"\n{len(incomplete)} cell(s) absent from at least one layer -> LEFT OUT, never "
              f"partially combined:\n  " + "\n  ".join(incomplete[:10]), flush=True)
    print(f"\nCOMBINING {len(common)} cells over {len(layers)} layers: {layers}\n", flush=True)

    rows_out, diag_out = [], []
    for name in sorted(common):
        stem = name.rsplit(f"__{slug}.npz", 1)[0]
        X, rung = stem.split("__", 1)
        z = {L: np.load(scan / f"L{L}__{name}", allow_pickle=True) for L in layers}
        seeds = [int(s) for s in np.asarray(z[layers[0]]["seeds"]).ravel()]
        per, degen = {}, {}
        for sd in seeds:
            key = f"dev_md__{sd}"
            if any(key not in z[L].files for L in layers):
                continue
            # (n_dev, n_layers) and (n_test, n_layers), columns ordered by the frozen layer list.
            Xd = np.column_stack([np.nan_to_num(z[L][f"dev_md__{sd}"]) for L in layers])
            Xt = np.column_stack([np.nan_to_num(z[L][f"test_md__{sd}"]) for L in layers])
            y_dev = np.asarray(z[layers[0]][f"y_dev__{sd}"], float)
            y_te = np.asarray(z[layers[0]][f"y_test__{sd}"], float)
            msp_dev = np.asarray(z[layers[0]][f"msp_dev__{sd}"], float)
            msp_te = np.asarray(z[layers[0]][f"msp_test__{sd}"], float)
            target = np.nan_to_num(1.0 - y_dev, nan=1.0)

            # The published reduction: fitted on the development matrix, transform-only at evaluation.
            pca = None
            if Xd.shape[1] >= N_COMPONENTS:
                pca = PCA(n_components=N_COMPONENTS).fit(Xd)
                Zd, Zt = pca.transform(Xd), pca.transform(Xt)
            else:
                Zd, Zt = Xd, Xt
            r = Ridge(positive=False).fit(Zd, target)
            satmd, satmd_dev = r.predict(Zt), r.predict(Zd)

            _, tmin, tmax, alpha = MD.grid_search_hp(
                satmd_dev, msp_dev, y_dev, prr_fn=lambda yy, uu: results.prr(yy, uu))
            huq = MD.total_uncertainty_linear_step(
                np.concatenate([satmd, satmd_dev]), np.concatenate([msp_te, msp_dev]),
                tmin, tmax, alpha)[:len(msp_te)]

            per.setdefault(f"satmd_{args.tag}", []).append(
                prr_or_blank(y_te, satmd, f"satmd_{args.tag}", degen))
            per.setdefault(f"huq_satmd_{args.tag}", []).append(
                prr_or_blank(y_te, huq, f"huq_satmd_{args.tag}", degen))
            # Each layer's RAW distance, so a gain from combining can be told apart from a gain that
            # was available at one layer all along. Registered prediction P4 turns on this.
            for k, L in enumerate(layers):
                per.setdefault(f"md_mean_L{L}", []).append(
                    prr_or_blank(y_te, Xt[:, k], f"md_mean_L{L}", degen))

            coefs = np.ravel(r.coef_)
            diag_out.append(dict(model=args.model, eval=X, rung=rung, seed=sd,
                                 n_layers=len(layers), n_features_in=Xd.shape[1],
                                 pca_applied=int(pca is not None),
                                 n_zero_coef=int(np.sum(np.abs(coefs) < 1e-12)),
                                 n_coef=len(coefs),
                                 huq_t_min=float(tmin), huq_t_max=float(tmax),
                                 huq_alpha=float(alpha)))
        if not per:
            continue
        for m, vals in sorted(per.items()):
            a = np.array(vals, float)
            ok = a[np.isfinite(a)]
            rows_out.append(dict(model=args.model, eval=X, rung=rung, method=m,
                                 prr_mean=(round(float(ok.mean()), 6) if len(ok) else ""),
                                 prr_std=(round(float(ok.std()), 6) if len(ok) else ""),
                                 n_seeds=len(ok), degenerate_seeds=degen.get(m, 0),
                                 layers=";".join(str(L) for L in layers)))
        head = [m for m in (f"satmd_{args.tag}", f"huq_satmd_{args.tag}") if m in per]
        line = "  ".join(f"{m} {np.nanmean(per[m]):+.3f}" for m in head)
        print(f"  [{rung:14s}/{X:14s}] {line}", flush=True)

    if not rows_out:
        sys.exit("FATAL: nothing combined.")
    out = ROOT / (args.out or f"results/hybrids/pdl_mdlayers_{args.tag}__{slug}.csv")
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    dpath = out.with_name(out.stem + "__diagnostics.csv")
    with open(dpath, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(diag_out[0])); w.writeheader(); w.writerows(diag_out)
    print(f"\nwrote {out.relative_to(ROOT)} ({len(rows_out)} rows)")
    print(f"wrote {dpath.relative_to(ROOT)} ({len(diag_out)} rows)")
    print("\nEvery row above is a LAYER SENSITIVITY. The published methods aggregate every layer; "
          "this aggregates a frozen subset, so no row here reproduces a published number.")


if __name__ == "__main__":
    main()
