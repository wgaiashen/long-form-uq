#!/usr/bin/env python
"""Pass two of the layer sensitivity: the published regression over several layers' distances.

What a row from this script IS depends entirely on which layers were passed to it.

  * The full published layer set, registered in prereg/M11_alllayer_published_distance_baselines.md,
    makes it a REPRODUCTION of the published estimators.
  * Any subset makes it a SENSITIVITY, and it must be labelled as one wherever it is quoted. The
    earlier subset registration, prereg/M9_layer_distance_sensitivity.md, ended at its first
    acceptance gate and produces nothing here.

The script cannot tell which case it is in, so it prints the layer count and the reader decides. The
`--tag` suffix is the mechanism for keeping the two apart in a merged table.

The regression itself is the published one, taken from run_polygraph.py:360-380 and
average_token_mahalanobis_distance.py: a ten-component reduction of the layer-wise distance matrix,
fitted on the development half and only transformed at evaluation, then an unconstrained ridge on
target `1 - correctness`.

FEATURE ORDER IN THE PROBABILITY-AUGMENTED VARIANTS
---------------------------------------------------
The reference reduces the DISTANCE COLUMNS ALONE and appends the sequence probability and the mean
token entropy afterwards:

    X = pca.fit_transform(train_dists)
    X = np.hstack([X, msp.reshape(-1, 1), ent.reshape(-1, 1)])

That order is reproduced here. At one cached layer it was unobservable, because the reduction needs
ten columns and there were at most three, so the two possible orders coincided. Across the full layer
set they do not.

    python scripts/checks/md_multilayer.py --model meta-llama/Meta-Llama-3.1-8B \
        --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32 \
        --tag alllayer
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


# The layer list the released driver builds: list(range(num_hidden_layers - 1)) + [-1], resolved
# against the 33-entry hidden state tuple of a 32-block model, so -1 is entry 32. Entry 31 is absent
# because the released code does not include it, and that asymmetry is reproduced rather than fixed.
PUBLISHED_LAYERS = list(range(31)) + [32]


def prr_or_blank(y, vec, label, degen):
    """A constant score is ranked by row position and would score whatever that permutation gives.
    Blank plus an explicit count, never a number -- the same rule the hybrid driver applies."""
    vec = np.asarray(vec, float)
    if np.isfinite(vec).all() and float(np.ptp(vec)) == 0.0:
        degen[label] = degen.get(label, 0) + 1
        return float("nan")
    return results.prr(y, vec)


def fit_family(Xd, Xt, target, msp_dev, msp_te, ent_dev, ent_te):
    """The published estimator over one distance matrix, plus its two derived variants.

    `Xd` and `Xt` are the development and evaluation layer-distance matrices, one column per layer.
    Returns the evaluation scores, the development scores the hybrid search needs, and diagnostics.

    Called once on the ordinary distances and once on the relative ones, so the two families are the
    same code applied to different inputs rather than two implementations that have to be kept in
    step by hand.
    """
    pca = None
    if Xd.shape[1] >= N_COMPONENTS:
        pca = PCA(n_components=N_COMPONENTS).fit(Xd)
        Zd, Zt = pca.transform(Xd), pca.transform(Xt)
    else:
        Zd, Zt = Xd, Xt
    r = Ridge(positive=False).fit(Zd, target)
    sat, sat_dev = r.predict(Zt), r.predict(Zd)

    # The probability-augmented variant. The two extra columns are appended AFTER the reduction,
    # which is the reference's order. Absent entropy leaves the variant unfitted rather than fitted
    # on a substituted value.
    msp_sat = None
    if ent_dev is not None and np.isfinite(ent_dev).all() and np.isfinite(ent_te).all():
        Ad = np.hstack([Zd, np.asarray(msp_dev, float).reshape(-1, 1),
                        np.asarray(ent_dev, float).reshape(-1, 1)])
        At = np.hstack([Zt, np.asarray(msp_te, float).reshape(-1, 1),
                        np.asarray(ent_te, float).reshape(-1, 1)])
        msp_sat = Ridge(positive=False).fit(Ad, target).predict(At)

    diag = dict(n_features_in=int(Xd.shape[1]), pca_applied=int(pca is not None),
                n_coef=len(np.ravel(r.coef_)),
                n_zero_coef=int(np.sum(np.abs(np.ravel(r.coef_)) < 1e-12)))
    return sat, sat_dev, msp_sat, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--scan-dir", default="")
    ap.add_argument("--window", default="ref", choices=["project", "ref"],
                    help="which scan directory to read, when --scan-dir is not given. It must match "
                         "the window the scan was run under; the cell files record theirs and a "
                         "mismatch across layers is refused.")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="11layer", help="suffix for the method names in the output")
    args = ap.parse_args()

    slug = cache._slug(args.model)
    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    _wtag = "" if args.window == "project" else "_refwin"
    scan = ROOT / (args.scan_dir or f"results/hybrids/mdscan{_wtag}__{slug}")
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
    print(f"\nCOMBINING {len(common)} cells over {len(layers)} layers: {layers}")
    if set(layers) == set(PUBLISHED_LAYERS):
        print("this IS the published layer set")
    else:
        miss = sorted(set(PUBLISHED_LAYERS) - set(layers))
        print(f"this is a SUBSET of the published layer set, missing {len(miss)} layers: {miss}")
    print("", flush=True)

    rows_out, diag_out = [], []
    for name in sorted(common):
        stem = name.rsplit(f"__{slug}.npz", 1)[0]
        X, rung = stem.split("__", 1)
        z = {L: np.load(scan / f"L{L}__{name}", allow_pickle=True) for L in layers}
        # Two layers scanned under different token windows are two different quantities, and
        # combining them would produce a number with no method behind it. The scan records which
        # window it used, so this is checked rather than assumed.
        wins = {str(z[L]["window"]) for L in layers if "window" in z[L].files}
        if len(wins) > 1:
            sys.exit(f"FATAL [{name}]: layers were scanned under different token windows {sorted(wins)}. "
                     "Refusing to combine them.")
        cell_window = wins.pop() if wins else "project"
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

            # THE KEY BEING PRESENT IS NOT THE SAME AS THE ENTROPY BEING USABLE. The scan always
            # writes these arrays and fills them with nan for a dataset that has no entropy cache, so
            # testing for the key reports availability everywhere and is simply wrong. What decides
            # whether the probability-augmented variants can be fitted is whether every value is
            # finite, which is what the fit itself checks, so the flag must agree with the fit.
            ent_key = f"ent_dev__{sd}"
            ent_dev = ent_te = None
            if all(ent_key in z[L].files for L in layers[:1]):
                _d = np.asarray(z[layers[0]][ent_key], float)
                _t = np.asarray(z[layers[0]][f"ent_test__{sd}"], float)
                if np.isfinite(_d).all() and np.isfinite(_t).all():
                    ent_dev, ent_te = _d, _t
            has_ent = ent_dev is not None

            # The relative family, present only where every layer carried a background statistic. A
            # partial layer set would be a different feature matrix per cell, which is not the
            # registered method, so it is all or nothing.
            rmd_key = f"dev_rmd__{sd}"
            has_rmd = all(rmd_key in z[L].files for L in layers)
            Rd = Rt = None
            if has_rmd:
                Rd = np.column_stack([np.nan_to_num(z[L][f"dev_rmd__{sd}"]) for L in layers])
                Rt = np.column_stack([np.nan_to_num(z[L][f"test_rmd__{sd}"]) for L in layers])

            fam = [("satmd", Xd, Xt)] + ([("satrmd", Rd, Rt)] if has_rmd else [])
            seed_diag = dict(model=args.model, eval=X, rung=rung, seed=sd, n_layers=len(layers),
                             has_rmd=int(has_rmd), has_entropy=int(has_ent))
            for base, Md, Mt in fam:
                sat, sat_dev, msp_sat, d = fit_family(Md, Mt, target, msp_dev, msp_te,
                                                      ent_dev, ent_te)
                _, tmin, tmax, alpha = MD.grid_search_hp(
                    sat_dev, msp_dev, y_dev, prr_fn=lambda yy, uu: results.prr(yy, uu))
                huq = MD.total_uncertainty_linear_step(
                    np.concatenate([sat, sat_dev]), np.concatenate([msp_te, msp_dev]),
                    tmin, tmax, alpha)[:len(msp_te)]

                name = f"{base}_{args.tag}"
                per.setdefault(name, []).append(prr_or_blank(y_te, sat, name, degen))
                hname = f"huq_{base}_{args.tag}"
                per.setdefault(hname, []).append(prr_or_blank(y_te, huq, hname, degen))
                if msp_sat is not None:
                    mname = f"msp_{base}_{args.tag}"
                    per.setdefault(mname, []).append(prr_or_blank(y_te, msp_sat, mname, degen))
                seed_diag.update({f"{base}_{k}": v for k, v in d.items()})
                seed_diag.update({f"huq_{base}_t_min": float(tmin),
                                  f"huq_{base}_t_max": float(tmax),
                                  f"huq_{base}_alpha": float(alpha)})

            # The raw distance at each layer, so a gain from combining can be told apart from a gain
            # that was available at one layer all along.
            for k, L in enumerate(layers):
                per.setdefault(f"md_mean_L{L}", []).append(
                    prr_or_blank(y_te, Xt[:, k], f"md_mean_L{L}", degen))
                if has_rmd:
                    per.setdefault(f"rmd_mean_L{L}", []).append(
                        prr_or_blank(y_te, Rt[:, k], f"rmd_mean_L{L}", degen))

            diag_out.append(seed_diag)
        if not per:
            continue
        for m, vals in sorted(per.items()):
            a = np.array(vals, float)
            ok = a[np.isfinite(a)]
            rows_out.append(dict(model=args.model, eval=X, rung=rung, method=m, window=cell_window,
                                 prr_mean=(round(float(ok.mean()), 6) if len(ok) else ""),
                                 prr_std=(round(float(ok.std()), 6) if len(ok) else ""),
                                 n_seeds=len(ok), degenerate_seeds=degen.get(m, 0),
                                 layers=";".join(str(L) for L in layers)))
        head = [m for m in (f"satmd_{args.tag}", f"huq_satmd_{args.tag}",
                            f"satrmd_{args.tag}", f"huq_satrmd_{args.tag}") if m in per]
        line = "  ".join(f"{m} {np.nanmean(per[m]):+.3f}" for m in head)
        print(f"  [{rung:14s}/{X:14s}] {line}", flush=True)

    if not rows_out:
        sys.exit("FATAL: nothing combined.")
    out = ROOT / (args.out or f"results/hybrids/pdl_mdlayers_{args.tag}__{slug}.csv")
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    dpath = out.with_name(out.stem + "__diagnostics.csv")
    # A cell without a background carries fewer keys than one with, so the header is the union rather
    # than whatever the first row happened to have. Taking the first row would silently drop columns.
    fields = []
    for d in diag_out:
        for k in d:
            if k not in fields:
                fields.append(k)
    with open(dpath, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader(); w.writerows(diag_out)
    print(f"\nwrote {out.relative_to(ROOT)} ({len(rows_out)} rows)")
    print(f"wrote {dpath.relative_to(ROOT)} ({len(diag_out)} rows)")
    if set(layers) == set(PUBLISHED_LAYERS):
        print("\nThe layer set is the published one, so these rows are reproductions of the "
              "published estimators. Their fidelity still rests on the acceptance gates in "
              "prereg/M11_alllayer_published_distance_baselines.md, which are checked separately.")
    else:
        print(f"\nEvery row above is a LAYER SENSITIVITY over {len(layers)} of "
              f"{len(PUBLISHED_LAYERS)} published layers. The published methods aggregate the full "
              "set, so no row here reproduces a published number.")


if __name__ == "__main__":
    main()
