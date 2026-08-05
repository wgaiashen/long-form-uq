"""0.2 — does the ID attention entropy ALONE predict the OOD drop?

Pre-registered in `prereg/0.2_id_entropy_as_predictor.md`, committed before this ran.

⚠️ NOT BLIND, and the prereg says so up front: the `ne_ID` and `d_PRR` columns were already printed by the
A1 run before the prereg was written. This fixes the PROCEDURE and the DECISION RULE, not a blind
prediction, and no result from it may be presented as if it were pre-registered in the 0.1 sense.

WHY IT IS WORTH ONE BOUNDED TEST. Delta-entropy (0.1) is dead as a switching signal, but it also needed an
OOD pass to compute -- you had to run the probe on the target data to know how far its attention moved.
`ne_ID` is a property of the TRAINED PROBE ALONE, measurable once before deployment with no target data.
So if it carried signal it would be a CHEAPER gate than the one we just lost, not merely a replacement.

⚠️ THE HONEST n IS 8, NOT 32. `ne_ID` is CONSTANT within a dataset, so a per-cell correlation over the 32
cells repeats each predictor value four times, adds no independent information, and would narrow the CI by
roughly a factor of two for free. The dataset-level n=8 test IS the test; the per-cell number is printed
only so nobody recomputes it and thinks it was missed, and it is labelled pseudo-replicated in the output.

Registered TWO-SIDED: "sharp ID attention is brittle" predicts a positive correlation with the drop,
"flat ID attention means nothing was learned" predicts a negative one. Both are plausible, so a result in
either direction counts and its sign is reported with the mechanism it supports.

Reads the A1 output CSV (`entropy_delta_vs_drop__<slug>.csv`) -- everything needed is already in it, so
this touches no cache and takes seconds.

    python scripts/checks/id_entropy_vs_drop.py
"""
import argparse
import csv
import itertools
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SLUG = "meta-llama_Meta-Llama-3.1-8B"


def spearman(x, y):
    from scipy import stats
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(stats.spearmanr(x, y)[0])


def pearson(x, y):
    from scipy import stats
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(stats.pearsonr(x, y)[0])


def boot_ci(x, y, n_boot=20000, seed=0):
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_boot):
        k = rng.randint(0, len(x), len(x))
        if np.std(x[k]) == 0 or np.std(y[k]) == 0:
            continue
        out.append(stats.spearmanr(x[k], y[k])[0])
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan, np.nan)


def exact_perm_p(x, y):
    """EXACT two-sided permutation p for Spearman at n=8: all 8! = 40,320 orderings are enumerable, so
    there is no sampling error in the null and no reason to approximate it."""
    from scipy import stats
    obs = abs(stats.spearmanr(x, y)[0])
    y = np.asarray(y, float)
    n_ge = tot = 0
    for perm in itertools.permutations(range(len(y))):
        r = stats.spearmanr(x, y[list(perm)])[0]
        tot += 1
        if abs(r) >= obs - 1e-12:
            n_ge += 1
    return n_ge / tot, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1", default=str(ROOT / "results" / f"entropy_delta_vs_drop__{SLUG}.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / f"id_entropy_vs_drop__{SLUG}.csv"))
    ap.add_argument("--plot", default=str(ROOT / "results" / f"id_entropy_vs_drop__{SLUG}.png"))
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.a1)) if r.get("status") == "ok"]
    if not rows:
        raise SystemExit(f"no usable rows in {args.a1} — run entropy_delta_vs_drop.py first")

    agg = defaultdict(list)
    for r in rows:
        agg[r["dataset"]].append(r)

    ds, ne_id, drop, level, mlen, ncell = [], [], [], [], [], []
    for d in sorted(agg):
        rs = agg[d]
        ids = {float(r["ne_ID"]) for r in rs}
        if len(ids) > 1:                          # ne_ID must be constant within a dataset, by construction
            raise SystemExit(f"{d}: ne_ID is not constant across rungs ({sorted(ids)}) — the A1 CSV is "
                             "not what this analysis assumes; refusing to average a predictor that moves.")
        ds.append(d)
        ne_id.append(ids.pop())
        drop.append(float(np.mean([float(r["d_PRR"]) for r in rs])))
        level.append(float(np.mean([float(r["prr_OOD"]) for r in rs])))
        mlen.append(float(np.mean([float(r["mean_len"]) for r in rs])))
        ncell.append(len(rs))

    print("0.2 — ID attention entropy as a predictor of the OOD drop")
    print(f"  population: ProbeDriftLong cells_long, method=attention (armA); source {Path(args.a1).name}")
    print("  ⚠️ POST-HOC: the inputs were visible before the prereg was written. Procedure and decision")
    print("     rule were fixed in advance; the prediction was not blind.")
    print("  Registered TWO-SIDED. Honest n = 8 (ne_ID is constant within a dataset).\n")

    print(f"  {'dataset':16s}{'ne_ID':>9s}{'drop':>9s}{'OOD level':>11s}{'mean len':>10s}{'cells':>7s}")
    for i, d in enumerate(ds):
        print(f"  {d:16s}{ne_id[i]:9.3f}{drop[i]:+9.3f}{level[i]:+11.3f}{mlen[i]:10.1f}{ncell[i]:7d}")

    n = len(ds)
    print(f"\n  ================ PRIMARY (Target A: the drop), n={n} ================")
    s = spearman(ne_id, drop)
    p = pearson(ne_id, drop)
    lo, hi = boot_ci(ne_id, drop)
    pp, tot = exact_perm_p(ne_id, drop)
    print(f"  Spearman {s:+.3f}  bootstrap CI [{lo:+.3f}, {hi:+.3f}]   Pearson {p:+.3f}")
    print(f"  EXACT permutation p (two-sided, all {tot:,} orderings) = {pp:.4f}")

    print(f"\n  ================ SECONDARY (Target B: the OOD PRR level), n={n} ================")
    s2 = spearman(ne_id, level)
    p2 = pearson(ne_id, level)
    lo2, hi2 = boot_ci(ne_id, level)
    pp2, _ = exact_perm_p(ne_id, level)
    print(f"  Spearman {s2:+.3f}  bootstrap CI [{lo2:+.3f}, {hi2:+.3f}]   Pearson {p2:+.3f}")
    print(f"  EXACT permutation p = {pp2:.4f}")
    print("  (a router picks the best method AT THE TARGET, so the level is arguably more decision-relevant")
    print("   than the change — reported alongside, never instead of, the primary)")

    print("\n  ---- LENGTH CONFOUND (registered: reported regardless of the result) ----")
    print(f"  mean length vs ne_ID   Spearman {spearman(mlen, ne_id):+.3f}   "
          f"Pearson {pearson(mlen, ne_id):+.3f}")
    print(f"  mean length vs drop    Spearman {spearman(mlen, drop):+.3f}   "
          f"Pearson {pearson(mlen, drop):+.3f}")

    print("\n  ---- DECLARED SECONDARY: the U-shape test (Bonferroni alpha = 0.025) ----")
    dev = list(np.abs(np.array(ne_id) - float(np.median(ne_id))))
    su = spearman(dev, drop)
    ppu, _ = exact_perm_p(dev, drop)
    print(f"  |ne_ID - median| vs drop   Spearman {su:+.3f}   exact permutation p = {ppu:.4f}")
    print("  Declared in the prereg, not invented afterwards. Exploratory: it cannot by itself reopen")
    print("  entropy, and it carries a multiple-comparisons cost against the primary.")

    # pseudo-replicated per-cell, printed ONLY so nobody recomputes it and thinks it was overlooked
    pc_x = [float(r["ne_ID"]) for r in rows]
    pc_y = [float(r["d_PRR"]) for r in rows]
    print(f"\n  [pseudo-replicated, NOT the test] per-cell n={len(rows)} Spearman "
          f"{spearman(pc_x, pc_y):+.3f} — each ne_ID repeated 4x, effective n is still 8. "
          "Never quote this CI.")

    print("\n  ================ DECISION (rule fixed in advance) ================")
    adopt = (s is not None and abs(s) >= 0.70 and pp < 0.05 and not (lo <= 0 <= hi))
    if adopt:
        mech = ("sharp ID attention is brittle" if s > 0 else "flat ID attention means nothing was learned")
        print(f"  CLEARS THE BAR (|rho| >= 0.70, CI excludes 0, perm p < 0.05). Sign supports: {mech}.")
        print("  ⚠️ SUSPECT until the required follow-up clears it: leave-one-dataset-out — refit on 7,")
        print("     predict the 8th, report whether the ordering holds out of sample. n=8 with a post-hoc")
        print("     rule is far too easy to fit. Then head-to-head against the LENGTH gate before adoption.")
    else:
        why = []
        if s is None or abs(s) < 0.70:
            why.append(f"|rho|={abs(s):.3f} < 0.70")
        if pp >= 0.05:
            why.append(f"perm p={pp:.3f} >= 0.05")
        if lo <= 0 <= hi:
            why.append("bootstrap CI includes 0")
        print(f"  DOES NOT CLEAR THE BAR ({'; '.join(why)}).")
        print("  -> ENTROPY IS CLOSED, in all its forms (delta-entropy 0.1 and ID entropy here).")
        print("     The stopping rule is binding: no further entropy variant is run. Move to the length")
        print("     gate and the top-k surprisal prior.")

    cols = ["dataset", "ne_ID", "drop_mean", "ood_level_mean", "mean_len", "n_cells"]
    tmp = args.out + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, d in enumerate(ds):
            w.writerow([d, f"{ne_id[i]:.4f}", f"{drop[i]:.4f}", f"{level[i]:.4f}",
                        f"{mlen[i]:.1f}", ncell[i]])
    os.replace(tmp, args.out)
    print(f"\n  wrote {args.out}")
    plot(args.plot, ds, ne_id, drop, level, s, s2)


def plot(path, ds, ne_id, drop, level, s, s2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, y, ylab, r, ttl in ((axes[0], drop, "Δ PRR (OOD − ID)   ↓ worse", s, "Target A: the drop"),
                                (axes[1], level, "OOD PRR (mean over rungs)", s2, "Target B: the OOD level")):
        ax.scatter(ne_id, y, s=90, color="#3060b0")
        for i, d in enumerate(ds):
            ax.annotate(d, (ne_id[i], y[i]), fontsize=7, xytext=(5, 4), textcoords="offset points")
        ax.set_xlabel("ID normalised attention entropy (H / log T)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{ttl}   n={len(ds)}   Spearman {r:+.3f}" if r is not None else ttl)
        ax.axhline(0, lw=0.7, color="0.6")
    fig.suptitle("0.2 — ID attention entropy as a predictor (POST-HOC; n=8, ne_ID constant within dataset)   "
                 "population: cells_long, method=attention", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
