"""Why is the shrinkage gain WEAKER on a replication population than on the development one?

Written 2026-08-19 after gemma-2-9b replicated only weakly (macro Delta_shrink +0.056 / +0.061 vs
Llama base's +0.088 / +0.113 on the identical reduced six-dataset grid). It tests, and mostly
REFUTES, the obvious explanations. Every number it prints is quoted in STOCKTAKE_gemma2_9b.md, which
is why it is committed rather than left as scratch.

THE FOUR CANDIDATE EXPLANATIONS AND WHAT THIS SCRIPT DOES TO EACH
  1. absolute-PRR decomposition -- is wmsp_shrink2 weak, or is wmsp_norm strong (no room)?
  2. capped/run-on generations   -- does the gain live in generations that hit the token cap?
  3. per-example length          -- within ONE dataset, does the gain concentrate in long outputs?
  4. cross-model rank agreement  -- does WHICH dataset benefits transfer between models?

⚠️ PER-EXAMPLE TOKEN COUNTS COME FROM THE SIDECAR, NOT FROM A JOIN TO THE RECORDS.
`unc__floor_sum` is msp_sum = sum(NLL) and `unc__floor_ppl` is perplexity = mean(NLL), so their
ratio is n_tokens exactly (verified to 7e-15 against the records). An earlier version joined on the
NLL sum and failed on four of six datasets, because a model that emits the same short summary many
times produces many records with an identical NLL sum. The ratio has no such failure mode.

⚠️ PRR IS AVERAGED OVER SEEDS, never computed on a seed-averaged vector (that inflates it ~0.05).
⚠️ Subsets smaller than 40 rows are reported as `n/a`, never as a number.

    python scripts/checks/wmodels_shrink_diagnosis.py
"""
import argparse, csv, glob, os, statistics as st, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luq import results  # noqa: E402

PANEL = ["pubmed_qa", "xsum", "cnn_dailymail", "samsum", "asqa", "factscore"]
RUNGS = ["DiffTask-long", "1ds-Diff-long"]
BUDGET = {"pubmed_qa": 128, "xsum": 56, "cnn_dailymail": 128, "samsum": 56, "asqa": 256, "factscore": 256}
MIN_SUBSET = 40


def prr(y, u):
    return results.prr(list(map(float, y)), list(map(float, u)))


def load_grid(slug):
    d = {}
    for f in glob.glob(str(ROOT / f"results/wmodels_stageA__{slug}__*.csv")):
        for r in csv.DictReader(open(f)):
            try:
                d[(r["eval"], r["rung"], r["method"])] = float(r["prr_mean"])
            except (KeyError, ValueError):
                pass
    return d


def sidecar(slug, ds, rung):
    p = ROOT / f"results/perex_wmodels/{slug}/{ds}__{rung}__{slug}.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def ntokens(z):
    """floor_sum / floor_ppl = sum(NLL) / mean(NLL) = n_tokens."""
    n = z["unc__floor_sum"][0] / z["unc__floor_ppl"][0]
    if np.abs(n - np.round(n)).max() > 1e-6:
        raise SystemExit("token-count recovery is not integral -- the floor definitions have changed")
    return np.round(n).astype(int)


def delta(z, mask):
    if mask.sum() < MIN_SUBSET:
        return None
    return float(np.mean([prr(z["y"][mask], z["unc__wmsp_shrink2"][s][mask])
                          - prr(z["y"][mask], z["unc__wmsp_norm"][s][mask])
                          for s in range(z["unc__wmsp_shrink2"].shape[0])]))


def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for p, i in enumerate(s): r[i] = p + 1
        return r
    rx, ry, n = rank(x), rank(y), len(x)
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (n * (n * n - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default="meta-llama_Meta-Llama-3.1-8B")
    ap.add_argument("--rep", default="google_gemma-2-9b")
    a = ap.parse_args()
    MODELS = [(a.dev, "dev"), (a.rep, "rep")]
    grids = {s: load_grid(s) for s, _ in MODELS}

    print("=" * 92)
    print("1. ABSOLUTE PRR -- is shrink weak, or is norm strong?")
    print("=" * 92)
    for rung in RUNGS:
        print(f"\n[{rung}]  macro PRR over the six datasets")
        print(f"  {'method':14s} {a.dev[:22]:>22s} {a.rep[:22]:>22s} {'diff':>9s}")
        for m in ["floor_min", "saplma", "attention", "wmsp_norm", "wmsp_shrink2"]:
            v = {}
            for s, _ in MODELS:
                got = [grids[s][(e, rung, m)] for e in PANEL if (e, rung, m) in grids[s]]
                v[s] = st.mean(got) if len(got) == 6 else None
            if None in v.values():
                continue
            print(f"  {m:14s} {v[a.dev]:+22.4f} {v[a.rep]:+22.4f} {v[a.rep]-v[a.dev]:+9.4f}")

    print("\n" + "=" * 92)
    print("2. CAPPED vs UNCAPPED -- does the gain live in run-on generations?")
    print("=" * 92)
    for slug, role in MODELS:
        print(f"\n{slug} ({role})")
        print(f"  {'dataset':14s} {'rung':15s} {'%cap':>5s} {'D all':>8s} {'D uncap':>8s} {'D cap':>8s}")
        for ds in PANEL:
            for rung in RUNGS:
                z = sidecar(slug, ds, rung)
                if z is None or "unc__wmsp_shrink2" not in z:
                    continue
                cap = ntokens(z) >= BUDGET[ds]
                vals = [delta(z, np.ones(len(z["y"]), bool)), delta(z, ~cap), delta(z, cap)]
                g = lambda v: "     n/a" if v is None else f"{v:+8.3f}"
                print(f"  {ds:14s} {rung:15s} {100*cap.mean():5.1f} " + " ".join(g(v) for v in vals))

    print("\n" + "=" * 92)
    print("3. WITHIN-DATASET LENGTH SPLIT -- holds dataset, task and labels fixed; only length varies")
    print("=" * 92)
    rows = []
    for slug, role in MODELS:
        print(f"\n{slug} ({role})")
        print(f"  {'dataset':14s} {'rung':15s} {'med':>5s} {'D short':>8s} {'D long':>8s} {'long-short':>11s}")
        for ds in PANEL:
            for rung in RUNGS:
                z = sidecar(slug, ds, rung)
                if z is None or "unc__wmsp_shrink2" not in z:
                    continue
                n = ntokens(z); med = float(np.median(n)); short = n <= med
                sh, lo = delta(z, short), delta(z, ~short)
                g = lambda v: "     n/a" if v is None else f"{v:+8.3f}"
                gap = "        n/a" if (sh is None or lo is None) else f"{lo-sh:+11.3f}"
                print(f"  {ds:14s} {rung:15s} {med:5.0f} {g(sh)} {g(lo)} {gap}")
                if sh is not None and lo is not None:
                    rows.append((sh, lo))
    if rows:
        sh = [r[0] for r in rows]; lo = [r[1] for r in rows]
        print(f"\n  SHORT half mean {st.mean(sh):+.4f} | LONG half mean {st.mean(lo):+.4f} | "
              f"long>short in {sum(1 for s,l in rows if l>s)}/{len(rows)} splits")
        print("  ⚠️ A coin-flip split count means per-example length is NOT the mechanism, whatever the")
        print("     cross-DATASET correlation says -- that one is confounded with task identity.")

    print("\n" + "=" * 92)
    print("4. CROSS-MODEL RANK AGREEMENT -- does WHICH dataset benefits transfer?")
    print("=" * 92)
    px, py = [], []
    for rung in RUNGS:
        dv = [grids[a.dev][(e, rung, "wmsp_shrink2")] - grids[a.dev][(e, rung, "wmsp_norm")] for e in PANEL]
        rv = [grids[a.rep][(e, rung, "wmsp_shrink2")] - grids[a.rep][(e, rung, "wmsp_norm")] for e in PANEL]
        print(f"  {rung:15s} spearman = {spearman(dv, rv):+.3f}   "
              f"best: {PANEL[dv.index(max(dv))]} (dev) vs {PANEL[rv.index(max(rv))]} (rep)")
        px += dv; py += rv
    print(f"  pooled (n={len(px)})   spearman = {spearman(px, py):+.3f}")


if __name__ == "__main__":
    sys.exit(main())
