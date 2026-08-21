#!/usr/bin/env python
"""Full-population statistics 1, 3, 4 (§D.7) -- cached data, no GPU, no sampling.

STAT 1  pooler-argmax vs floor-argmin OVERLAP, per dataset, vs a length-matched random baseline,
        then correlate per-dataset overlap against the OOD (floor - pooler) PRR gap.
STAT 3  positional signal profile: per relative-position bin, AUROC of token logprob for
        predicting example-level error, per dataset.
STAT 4  global / fractional FIXED-k (from the k-sweep CSV): one k for all datasets chosen LODO.

Alignment (verified): len(token_logprobs)+1 == len(pool_w) (G+1 window, row-0 = prompt anchor), so the
pooler's gen-token pick is argmax(pool_w[1:]) and the floor's pick is argmin(token_logprobs).
"""
import sys, csv as _csv
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "checks"))
from luq import cache                       # noqa: E402
from luq.config import Config               # noqa: E402
from xl_rungs import eval_split, label_of   # noqa: E402
from attn_pool import PROMPT_REGIME         # noqa: E402
from sklearn.metrics import roc_auc_score   # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
DATASETS = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail",
            "med_quad", "samsum", "expertqa", "asqa"]
VIZ = ROOT / "cache" / "viz"
EPHEM = Path(os.environ.get("EPHEMERAL", str(Path.home() / "ephemeral"))) / "luq_overnight_results"
SWEEP_CSV = EPHEM / "topk_floor_sweep__meta-llama_Meta-Llama-3.1-8B.csv"

# OOD (floor, pooler) PRR per dataset -- from the length_router --ood extract (3494057; verified in §D.3).
# always-floor = msp_min OOD, always-pool = attention pooler OOD.
OOD_FLOOR_POOL = {
    "sciq": (0.755, 0.770), "trivia_qa": (0.747, 0.616), "pubmed_qa": (0.371, 0.313),
    "xsum": (-0.015, 0.230), "cnn_dailymail": (0.120, 0.248), "med_quad": (0.149, 0.322),
    "samsum": (-0.024, 0.343), "expertqa": (0.220, 0.183), "asqa": (0.250, 0.388),
}


def load_records(dataset):
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    return cache.load_records(cfg.cache_dir, cache.run_key(MODEL, dataset, "ID"))


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_corr_ci(x, y, fn, n=2000):
    x = np.asarray(x, float); y = np.asarray(y, float); m = len(x)
    rng = np.random.RandomState(0)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, m, m)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        vals.append(fn(x[idx], y[idx]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def stat1():
    print("=" * 82); print("STAT 1 -- pooler-argmax vs floor-argmin OVERLAP (vs length-matched random)"); print("=" * 82)
    print(f"{'dataset':14s}{'n':>6s}{'overlap%':>10s}{'random%':>9s}{'lift':>7s}{'anchor%':>9s}{'OOD floor-pool':>16s}")
    ov, gap = [], []
    rows = []
    for d in DATASETS:
        recs = load_records(d)
        z = np.load(VIZ / f"{SLUG}__{d}__ID__attn.npz", allow_pickle=True)
        pw = z["pool_w"]; rp = z["record_pos_all"]
        overlaps, rands, anchors = [], [], []
        for i in range(len(pw)):
            lp = np.asarray(recs[int(rp[i])]["token_logprobs"], float)
            G = len(lp)
            if G == 0 or len(pw[i]) != G + 1:      # alignment guard -- skip/flag, never silently mis-align
                continue
            w = np.asarray(pw[i], float)
            gen_pick = int(np.argmax(w[1:]))       # pooler's top GEN token (0-based, row-0 anchor dropped)
            floor_pick = int(np.argmin(lp))        # msp_min's token
            overlaps.append(gen_pick == floor_pick)
            rands.append(1.0 / G)                  # chance the top gen token == the argmin (1/T)
            anchors.append(int(np.argmax(w)) == 0)  # pooler's GLOBAL peak is the prompt anchor
        o = float(np.mean(overlaps)); r = float(np.mean(rands)); a = float(np.mean(anchors))
        fl, po = OOD_FLOOR_POOL[d]; g = fl - po
        ov.append(o); gap.append(g)
        rows.append((d, len(overlaps), o, r, a, g))
        print(f"{d:14s}{len(overlaps):>6d}{100 * o:>10.1f}{100 * r:>9.2f}{o / r:>7.1f}{100 * a:>9.1f}{g:>+16.3f}")
    pear = float(np.corrcoef(ov, gap)[0, 1]); spear = spearman(ov, gap)
    pc = boot_corr_ci(ov, gap, lambda x, y: float(np.corrcoef(x, y)[0, 1]))
    print(f"\ncorr(overlap, OOD floor-pool gap): Pearson {pear:+.3f} CI[{pc[0]:+.3f},{pc[1]:+.3f}] | "
          f"Spearman {spear:+.3f}   (n=9 -- read with caution)")
    print("PREDICTION: HIGH overlap -> floor wins (positive corr).",
          "CONFIRMED" if pear > 0 else "DISCONFIRMED / null", flush=True)
    return rows, (pear, spear, pc)


def stat3():
    print("\n" + "=" * 82); print("STAT 3 -- positional signal profile: AUROC(token logprob -> example error) per rel-pos bin"); print("=" * 82)
    NB = 10
    hdr = "".join(f"{f'b{b}':>7s}" for b in range(NB))
    print(f"{'dataset':14s}{'posrate':>9s}  {hdr}")
    out = {}
    for d in DATASETS:
        recs = load_records(d)
        split = np.array([r["split"] for r in recs])
        lf = label_of(d)
        y = np.array([r.get(lf, np.nan) for r in recs], float)
        finite = np.isfinite(y)
        keep = np.where(finite)[0]
        recs_f = [recs[k] for k in keep]; split_f = split[keep]; y_f = y[keep]
        _, te = eval_split(split_f)
        # per-bin pools of (token_logprob, example_error)
        bin_lp = [[] for _ in range(NB)]; bin_err = [[] for _ in range(NB)]
        err_all = []
        for i in te:
            lp = np.asarray(recs_f[i]["token_logprobs"], float)
            G = len(lp)
            if G == 0:
                continue
            err = 1 if y_f[i] < 0.5 else 0      # example-level error (graded correctness binarised at 0.5)
            err_all.append(err)
            relpos = (np.arange(G) + 0.5) / G   # token relative position in the response
            b = np.minimum((relpos * NB).astype(int), NB - 1)
            for j in range(G):
                bin_lp[b[j]].append(lp[j]); bin_err[b[j]].append(err)
        aucs = []
        for b in range(NB):
            e = np.array(bin_err[b]); s = -np.asarray(bin_lp[b])   # low logprob -> high error score
            if len(e) == 0 or len(np.unique(e)) < 2:
                aucs.append(float("nan"))
            else:
                aucs.append(roc_auc_score(e, s))
        posrate = float(np.mean(err_all)) if err_all else float("nan")
        out[d] = (posrate, aucs)
        print(f"{d:14s}{posrate:>9.2f}  " + "".join(f"{a:>7.2f}" if a == a else f"{'--':>7s}" for a in aucs))
    return out


def stat4():
    print("\n" + "=" * 82); print("STAT 4 -- global / fractional FIXED-k, LODO (from the k-sweep CSV)"); print("=" * 82)
    rows = list(_csv.DictReader(open(SWEEP_CSV)))
    def grid(sweep):
        ks, tab = [], {}
        for r in rows:
            if r["sweep"] != sweep:
                continue
            k = r["k"]; tab.setdefault(k, {})[r["dataset"]] = float(r["prr"])
            if k not in ks:
                ks.append(k)
        return ks, tab
    for sweep, name in [("abs", "ABSOLUTE-k (one integer k for all)"), ("frac", "FRACTIONAL-k (one fraction of T for all)")]:
        ks, tab = grid(sweep)
        # in-sample best global k (context)
        mean_by_k = {k: float(np.mean([tab[k][d] for d in DATASETS])) for k in ks}
        best_glob = max(ks, key=lambda k: mean_by_k[k])
        # LODO: pick k on the other 8, apply to held-out
        lodo, kpicks = [], []
        for held in DATASETS:
            tr = [d for d in DATASETS if d != held]
            kbest = max(ks, key=lambda k: np.mean([tab[k][d] for d in tr]))
            lodo.append(tab[kbest][held]); kpicks.append(kbest)
        msp_min = float(np.mean([tab[ks[0]][d] for d in DATASETS]))     # k=1 / smallest frac
        ppl = float(np.mean([tab[ks[-1]][d] for d in DATASETS]))         # k=all / frac=1.0
        print(f"\n{name}")
        print(f"  in-sample best global k={best_glob}  mean PRR {mean_by_k[best_glob]:+.4f}")
        print(f"  LODO global-k          mean PRR {np.mean(lodo):+.4f}   (picks: {dict(zip(DATASETS, kpicks))})")
        print(f"  reference: always-msp_min(k={ks[0]}) {msp_min:+.4f} | always-ppl(k={ks[-1]}) {ppl:+.4f}")
        verdict = ("BEATS msp_min -> a better label-free floor" if np.mean(lodo) > msp_min + 1e-9
                   else "does NOT beat msp_min -> stays a diagnostic")
        print(f"  VERDICT: LODO global {sweep}-k {verdict}")


def main():
    rows1, corr1 = stat1()
    out3 = stat3()
    stat4()
    # persist STAT1 + STAT3 to a small CSV
    outp = EPHEM / ("four_stats_1_3__" + SLUG + ".csv")
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["stat", "dataset", "field", "value"])
        for d, n, o, r, a, g in rows1:
            w.writerow(["stat1", d, "overlap", o]); w.writerow(["stat1", d, "random", r])
            w.writerow(["stat1", d, "anchor", a]); w.writerow(["stat1", d, "ood_floor_minus_pool", g])
        for d, (pr, aucs) in out3.items():
            w.writerow(["stat3", d, "posrate", pr])
            for b, au in enumerate(aucs):
                w.writerow(["stat3", d, f"auc_bin{b}", au])
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
