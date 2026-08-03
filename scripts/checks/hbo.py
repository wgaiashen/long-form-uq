"""R4b — Joe's HBO (Hybrid Back-Off), implemented from the paper and VALIDATED before use as a baseline.

WHY THIS IS A STANDALONE STEP
-----------------------------
Our proposed system is structurally HBO. If HBO were implemented *as our comparison's baseline*, there
would be a pull toward specifying it in whatever form makes us look good. So it is built and validated
on its own first, against criteria fixed in `prereg/R4b_hbo_reproduction.md` BEFORE the numbers are seen.

THE METHOD (paper §5.2, equations 2-4) -- quoted, not remembered
----------------------------------------------------------------
Per test example, an OOD score from the Mahalanobis distance (following Vazhentsev et al. 2025b):
  1. Split the training data in half. Use the FIRST half for a mean and covariance.
  2. Compute the MD of each of the REMAINING training examples under that mean/covariance.
  3. Compute the MD of each TEST example under a mean/covariance from the ENTIRE training set.
  4. r = how many training MDs are smaller than this test MD ("a combination of the training MDs and the
     added test instance MD").
  5. R = r / (N + 1).
Then:
     W_usv = R + 0.5  if R <= 0.5 else 1
     W_sv  = 1 - W_usv
     UQ_hyb = W_sv * UQ_sv + W_usv * UQ_usv
with UQ_sv = SAPLMA (middle layer), UQ_usv = MSP, both RANK-NORMALISED before combining.

⚠️ TWO PROPERTIES OF THE FORMULA, so they are not later mistaken for bugs:
  * W_sv never exceeds 0.5 (at R=0 the weights are 0.5/0.5 -- the paper's "even weighting" ID case).
    HBO never leans supervised.
  * W_sv is exactly 0 whenever R > 0.5, so HBO collapses to PURE MSP for the more-OOD half of the test
    set. That is why the paper's HBO row equals its MSP row on the far rungs.

⚠️ AN AMBIGUITY IN THE PAPER, RESOLVED EXPLICITLY RATHER THAN SILENTLY
----------------------------------------------------------------------
"R = r/(N+1) where N is the size of the training data" -- but the ranking pool is described as "a
combination of the training MDs and the added test instance MD", and the training MDs come only from the
SECOND half (step 2). So N is read here as **the number of training MDs used for ranking**, i.e. the size
of the second half. Reading N as the FULL training size instead would cap R at ~0.5, the `otherwise`
branch of W_usv would never fire, and HBO could never collapse to MSP.

That is not a matter of taste: registered criterion **V3** (HBO within 0.02 of MSP on the far rungs) is
exactly the check that fails if this reading is wrong. The ambiguity is therefore *testable*, and the
test is pre-registered. We cannot check against the authors' code (not released with the PDF we hold), so
this is marked UNCONFIRMED against source and rests on V3.

    python scripts/checks/hbo.py
"""
import csv as _csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache  # noqa: E402
from xl_rungs import build_rows, cells as xl_cells  # noqa: E402
from contribution_ladder import sampled_train_idx as cl_sampled  # noqa: E402
from aggregation_table import prr_from_conf  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = cache._slug(MODEL)
LAYER = 15
SEED = 1
SHORT = ["sciq", "trivia_qa"]          # the EVAL targets: short-form, which is where the paper validates
# ⚠️ THE TRAINING SOURCE POOL IS NOT THE EVAL LIST. A short-form eval's DiffTask/OneDatasetDiffTask rungs
# train on LONG-form data (sciq DiffTask <- samsum+xsum+cnn). The first version of this script passed only
# the short-form sets as `sources`, and `cells()` silently drops any spec entry not in `sources` -- so
# every rung except ID and LOO vanished with no warning and V3 came back "n/a". That is the standing
# evaluation rule's failure mode exactly: a partial grid reported as if it were the whole one. The pool is
# therefore every dataset with BOTH records and pooled features.
SOURCES = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum"]
RIDGE = 1e-3          # covariance regularisation; 4096-dim features with ~1800 rows are singular
HALF_SEED = 0         # the train/half split for the MD reference is fixed, so R is not seed noise


def load_dataset(ds):
    """(features_L15, split, y, msp_uncertainty) for one dataset, or None if not fully cached."""
    base = ROOT / "cache" / "records"
    hits = [h for h in base.glob(f"*__{ds}__ID.jsonl") if "Meta-Llama-3.1-8B" in h.name]
    if len(hits) != 1:                       # NEVER a model-agnostic glob, and never glob[0]
        return None
    recs = [json.loads(l) for l in open(hits[0])]
    key = cache.run_key(MODEL, ds, "ID")
    try:
        feats = cache.load_features(ROOT / "cache", key, "saplma")
    except Exception:
        return None
    if feats.ndim == 3:                      # (n, layers, dim) -> pick the middle layer
        feats = feats[:, LAYER, :]
    y = np.array([r.get("correctness", np.nan) for r in recs], dtype=float)
    split = np.array([r["split"] for r in recs])
    # MSP as the paper uses it: uncertainty = 1 - min token probability over the generation.
    msp = np.array([1.0 - float(np.exp(np.min(r["token_logprobs"]))) if r.get("token_logprobs") else np.nan
                    for r in recs], dtype=float)
    if len(feats) != len(recs):
        sys.exit(f"{ds}: {len(feats)} feature rows vs {len(recs)} records -- refusing to align by guess.")
    return feats, split, y, msp, recs


def rank_normalise(v):
    """Map to [0,1] by rank. The paper combines RANK-normalised estimates, which is what makes a probe
    score and a probability comparable at all -- they are on different scales otherwise."""
    v = np.asarray(v, float)
    order = np.argsort(np.argsort(v))
    return order / max(len(v) - 1, 1)


def _md(X, mu, prec):
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, prec, d)


def hbo_R(train_X, test_X):
    """The normalised OOD rank R per test example, per paper §5.2 steps 1-5."""
    rng = np.random.RandomState(HALF_SEED)
    perm = rng.permutation(len(train_X))
    h1, h2 = perm[:len(perm) // 2], perm[len(perm) // 2:]

    # steps 1-2: reference MDs for the SECOND half, under the FIRST half's mean/covariance
    mu1 = train_X[h1].mean(0)
    cov1 = np.cov(train_X[h1], rowvar=False) + RIDGE * np.eye(train_X.shape[1])
    ref = _md(train_X[h2], mu1, np.linalg.pinv(cov1))

    # step 3: test MDs under the ENTIRE training set's mean/covariance
    mu_all = train_X.mean(0)
    cov_all = np.cov(train_X, rowvar=False) + RIDGE * np.eye(train_X.shape[1])
    te = _md(test_X, mu_all, np.linalg.pinv(cov_all))

    # steps 4-5: rank each test MD among the reference MDs. N = number of reference MDs (see the module
    # docstring for why this reading, and V3 for the test that catches it if it is wrong).
    ref_sorted = np.sort(ref)
    r = np.searchsorted(ref_sorted, te, side="left").astype(float)
    return r / (len(ref) + 1.0)


def saplma_uncertainty(Xtr, ytr, Xte, seed):
    """SAPLMA = Azaria & Mitchell MLP on the pooled hidden state. Uncertainty = 1 - P(correct)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=5, batch_size=32,
                        random_state=seed)
    yb = (np.asarray(ytr) >= 0.5).astype(int)
    if len(np.unique(yb)) < 2:
        return None
    clf.fit(sc.transform(Xtr), yb)
    return 1.0 - clf.predict_proba(sc.transform(Xte))[:, 1]


def main():
    PT, DATA = {}, {}
    for ds in SOURCES:
        got = load_dataset(ds)
        if got is None:
            print(f"  {ds}: not fully cached -> SKIPPED LOUDLY (left absent, never zero)", flush=True)
            continue
        feats, split, y, msp, recs = got
        DATA[ds] = (feats, split, y, msp)
        PT[ds] = (None, split, None, recs)
    sources = set(PT)
    evals = [e for e in SHORT if e in sources]
    if not evals:
        sys.exit("no short-form EVAL dataset is cached. Refusing to report an empty validation.")

    # State the grid and its coverage BEFORE the numbers (standing evaluation rule).
    planned = xl_cells(sources, evals)
    print(f"R4b HBO validation | evals {evals} | training-source pool {sorted(sources)} | "
          f"layer {LAYER} seed {SEED}")
    by_rung = {}
    for rung, X, _ in planned:
        by_rung.setdefault(rung, []).append(X)
    print(f"GRID: {len(planned)} cells across {len(by_rung)} rungs -> "
          + ", ".join(f"{k}:{len(v)}" for k, v in sorted(by_rung.items())))
    missing = [r for r in ("ID", "LOO", "SameTask", "DiffTask", "OneDatasetDiffTask") if r not in by_rung]
    if missing:
        print(f"⚠️ RUNGS ABSENT FROM THIS GRID: {missing} — the validation does NOT cover them and no "
              f"claim is made about them.")
    print(flush=True)

    rows = []
    for rung, X, spec in planned:
        tr_rows, te_rows = build_rows(X, spec, PT, SEED, cl_sampled)
        Xtr = np.stack([DATA[d][0][i] for d, i in tr_rows])
        ytr = np.array([DATA[d][2][i] for d, i in tr_rows])
        Xte = np.stack([DATA[d][0][i] for d, i in te_rows])
        yte = np.array([DATA[d][2][i] for d, i in te_rows])
        msp_te = np.array([DATA[d][3][i] for d, i in te_rows])

        ok = np.isfinite(ytr)
        if ok.sum() < 50 or not np.isfinite(yte).all():
            print(f"  [{rung:18s}] {X:10s} insufficient labels -> SKIPPED", flush=True)
            continue
        Xtr, ytr = Xtr[ok], ytr[ok]

        u_sv = saplma_uncertainty(Xtr, ytr, Xte, SEED)
        if u_sv is None:
            print(f"  [{rung:18s}] {X:10s} single-class train -> SKIPPED", flush=True)
            continue
        R = hbo_R(Xtr, Xte)
        w_usv = np.where(R <= 0.5, R + 0.5, 1.0)
        w_sv = 1.0 - w_usv
        uq = w_sv * rank_normalise(u_sv) + w_usv * rank_normalise(msp_te)

        prr_msp = prr_from_conf(yte, -msp_te)
        prr_sap = prr_from_conf(yte, -u_sv)
        prr_hbo = prr_from_conf(yte, -uq)
        rows.append({"rung": rung, "eval": X, "n_train": int(len(ytr)), "n_test": int(len(yte)),
                     "prr_msp": round(prr_msp, 4), "prr_saplma": round(prr_sap, 4),
                     "prr_hbo": round(prr_hbo, 4), "mean_R": round(float(R.mean()), 4),
                     "frac_R_gt_0.5": round(float((R > 0.5).mean()), 4),
                     "mean_w_sv": round(float(w_sv.mean()), 4)})
        print(f"  [{rung:18s}] {X:10s} MSP {prr_msp:+.4f}  SAPLMA {prr_sap:+.4f}  HBO {prr_hbo:+.4f}   "
              f"| mean R {R.mean():.3f}, frac(R>0.5) {(R > 0.5).mean():.3f}", flush=True)

    if not rows:
        sys.exit("no cells produced. Refusing to write an empty validation.")

    # ---- REGISTERED VALIDATION (prereg §3). Mechanical, so the verdict is not eyeballed.
    print("\n" + "=" * 78 + "\nREGISTERED VALIDATION CRITERIA (fixed before the numbers)\n" + "=" * 78)
    ood = [r for r in rows if r["rung"] != "ID"]
    idc = [r for r in rows if r["rung"] == "ID"]
    far = [r for r in rows if r["rung"] in ("DiffTask", "OneDatasetDiffTask")]

    v1 = all(r["prr_hbo"] >= r["prr_msp"] - 0.01 for r in rows)
    v2 = all(r["prr_hbo"] >= r["prr_saplma"] - 0.01 for r in ood) if ood else None
    v3 = all(abs(r["prr_hbo"] - r["prr_msp"]) <= 0.02 for r in far) if far else None
    v4 = all(r["prr_hbo"] >= max(r["prr_msp"], r["prr_saplma"]) - 0.01 for r in idc) if idc else None
    for tag, val, desc in [("V1", v1, "HBO >= MSP at every rung"),
                           ("V2", v2, "HBO >= SAPLMA on every OOD rung"),
                           ("V3", v3, "HBO within 0.02 of MSP on the far rungs (near-unit-test)"),
                           ("V4", v4, "HBO >= both at ID")]:
        mark = "n/a (no such cell)" if val is None else ("PASS" if val else "FAIL")
        print(f"  {tag}  {mark:18s} {desc}")

    vals = [v for v in (v1, v2, v3, v4) if v is not None]
    if all(vals):
        print("\n✅ HBO VALIDATED — it reproduces the paper's reported short-form behaviour and may be "
              "used as a named baseline.")
    else:
        print("\n❌ HBO NOT VALIDATED — we have implemented something other than HBO. Per the "
              "pre-registration this is reported as a FAILURE TO REPRODUCE and is NOT re-tuned until it "
              "passes. Comparisons against it are meaningless until resolved.")

    # The diagnostic that decides whether the validation was vacuous.
    mean_far_frac = np.mean([r["frac_R_gt_0.5"] for r in idc]) if idc else float("nan")
    print(f"\n⚠️ VACUITY CHECK: at the ID rung, frac(R>0.5) = {mean_far_frac:.3f}. If this is near 1.0 "
          "then HBO is pure MSP everywhere, V1-V3 pass trivially, and the validation says nothing.")

    out = ROOT / "results" / f"regime_R4b_hbo_validation__{SLUG}.csv"
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
