"""Train/test carving, source sampling, and the row-order guard.

WHY THE CARVE RUNS BEFORE THE LABEL FILTER
-----------------------------------------
On the two partially-labelled datasets the quality label is absent because of what that particular
model generated: `factuality` is None when `uncovered == 1.0`, meaning the judge ran and found no
claim the reference could adjudicate. Dropping unlabelled rows before carving the test set would
therefore give a different model a different test set, and two models' numbers would not be on the
same population.

So the carve runs over all rows, and whichever of those carry labels are then scored. Only the two
datasets whose label coverage varies by model are affected; where every row is labelled, the two
orders give the same test set.

This does not fully equalise the populations. Each model is still scored on the subset its own
quality labels cover, so the compared row sets still differ. What the change buys is that the
difference is fixed, visible and measurable through `coverage`, rather than silent. It should be
reported rather than described as equalised.

WHY POSITIONAL, NOT HASHED
--------------------------
A content hash would be model-independent by construction, but it would also reassign rows on
med_quad/samsum/asqa, which have no drift problem -- five datasets re-scored instead of two, for
no gain. Positional is reproducible here because row order is a deterministic function of the
data: records are loaded in file order with no sort, and the carve-relevant datasets are
single-split, contiguous and index-ordered. `assert_canonical_order` below enforces that rather
than assuming it.
"""

import numpy as np

XL_TEST_FRAC = 0.30      # held-out test fraction carved from a split-less eval target
CARVE_SEED = 0           # fixed, so the test set is stable across the per-run seeds


# ---- the row-order guard -------------------------------------------------------------------
def assert_canonical_order(records, dataset="<unknown>"):
    """Fail loud unless records are in canonical (split, idx) order with no gaps or duplicates.

    The positional carve is only reproducible if row position IS `idx`. That holds for every
    cached dataset today, but nothing enforced it -- so a records file rebuilt or merged in a
    different order would produce a different test set with no error and no visible symptom. A
    silent change of test population is worse than a failure, so this raises rather than warns.
    """
    keys = [(r["split"], int(r["idx"])) for r in records]
    if len(set(keys)) != len(keys):
        dupes = len(keys) - len(set(keys))
        raise SystemExit(f"{dataset}: records contain {dupes} duplicate (split, idx) keys — "
                         "the positional carve is not reproducible on this file.")
    for s in sorted({k[0] for k in keys}):
        ids = [i for ss, i in keys if ss == s]
        if ids != sorted(ids):
            raise SystemExit(f"{dataset}: split '{s}' rows are not in ascending idx order — "
                             "row position is not idx, so the positional carve would drift.")
        if ids != list(range(ids[0], ids[0] + len(ids))):
            raise SystemExit(f"{dataset}: split '{s}' idx values are not contiguous "
                             f"({ids[0]}..{ids[-1]}, n={len(ids)}) — refusing to carve.")
    return True


# ---- the carve -----------------------------------------------------------------------------
def eval_split(split, labelled=None, *, seed=CARVE_SEED, test_frac=XL_TEST_FRAC, carve="all-rows"):
    """(train_idx, test_idx) for an EVAL TARGET, as positions in the array you passed in.

    A dataset with a real baked-in train/test split (pubmed_qa, xsum, cnn_dailymail) uses it
    untouched. A split-less dataset gets a fixed deterministic carve at `seed`, so there is ONE
    stable test set and the per-run seed varies only the training subsample.

    carve="legacy"    filter first, then carve. Pass an already-filtered split array and leave
                      `labelled` as None. This is the rule the published numbers were produced
                      under, so it is kept for reproducing them.
    carve="all-rows"  the fix. Pass the FULL split array plus a boolean `labelled` mask of the
                      same length; the carve runs over ALL rows and unlabelled rows are then
                      dropped from BOTH sides. Returned indices are positions in the FULL array
                      (use `remap_to_filtered` if the caller works on a filtered array).
    """
    split = np.asarray(split)
    if carve not in ("legacy", "all-rows"):
        raise ValueError(f"carve must be 'legacy' or 'all-rows', got {carve!r}")
    if carve == "legacy" and labelled is not None:
        raise ValueError("carve='legacy' takes a pre-filtered split array and no `labelled` mask")
    if carve == "all-rows" and labelled is None:
        raise ValueError("carve='all-rows' needs the `labelled` mask — that is the whole point. "
                         "Pass carve='legacy' if you deliberately want filter-then-carve.")

    if len(np.unique(split)) >= 2:                      # real baked-in split
        tr = np.where(split == "train")[0]
        te = np.where(split == "test")[0]
    else:                                               # split-less: deterministic carve
        n = len(split)
        perm = np.random.RandomState(seed).permutation(n)
        n_te = int(round(n * test_frac))
        tr, te = perm[n_te:], perm[:n_te]

    if carve == "legacy":
        return tr, te
    lab = np.asarray(labelled, dtype=bool)
    if len(lab) != len(split):
        raise ValueError(f"`labelled` has length {len(lab)}, split has {len(split)}")
    return tr[lab[tr]], te[lab[te]]


def remap_to_filtered(idx, labelled):
    """Positions in the FULL array -> positions in the labelled-only (filtered) array.

    Raises on an unlabelled index rather than dropping it silently: a caller that hands us a row
    the label cannot score is confused about which array it holds, and quietly returning a
    shorter list is how an index mismatch becomes a wrong-rows-scored bug.
    """
    lab = np.asarray(labelled, dtype=bool)
    pos = np.full(len(lab), -1, dtype=np.int64)
    pos[lab] = np.arange(int(lab.sum()))
    out = pos[np.asarray(idx, dtype=np.int64)]
    if (out < 0).any():
        raise ValueError(f"{int((out < 0).sum())} index/indices refer to unlabelled rows")
    return out


# ---- source sampling -----------------------------------------------------------------------
def sampled_train_idx(split, seed, cap):
    """Draw up to `cap` TRAINING rows from a source dataset.

    A source with no dedicated train split (the eval-only sets) draws from ALL its rows. That is
    safe because `cells_long` always excludes a dataset from its own eval's sources, so no row
    can be both trained on and tested on. Filtering to split=="train" unconditionally would return
    nothing for an eval-only source, so a listed source such as `expertqa:360` would contribute
    zero rows while the pool label still claimed it.
    """
    split = np.asarray(split)
    tr = np.where(split == "train")[0]
    if len(tr) == 0:
        tr = np.arange(len(split))
    if cap is None or cap >= len(tr):
        return tr
    return tr[np.random.RandomState(seed).permutation(len(tr))[:cap]]


def build_rows(X, spec, splits, seed, sampled_fn=sampled_train_idx, labelled=None,
               carve="all-rows"):
    """(train_rows, test_rows) for one cell, as lists of (dataset, idx).

    `splits`   : {dataset: split array}. Only the split array is ever read; this function needs
                 neither hidden states nor records.
    `labelled` : {dataset: bool mask}, required when carve="all-rows".

    The eval target X uses its fixed `eval_split`; OOD sources are drawn by `sampled_fn`.
    """
    if carve not in ("legacy", "all-rows"):
        raise ValueError(f"carve must be 'legacy' or 'all-rows', got {carve!r}")
    if carve == "all-rows" and labelled is None:
        raise ValueError(
            "build_rows(carve='all-rows') needs `labelled` — a {dataset: bool mask} dict over the "
            "UNFILTERED rows. Without it the carve would silently run on whatever array it was "
            "handed, which is the legacy population wearing the new rule's name. Pass the masks, "
            "or pass carve='legacy' if that is what you actually want.")
    lab_X = None if carve == "legacy" else np.asarray(labelled[X], dtype=bool)
    X_tr, X_te = eval_split(splits[X], lab_X, seed=CARVE_SEED, carve=carve)
    test_rows = [(X, int(i)) for i in X_te]
    train_rows = []
    for d, cap in spec:
        if d == X:                                  # ID cell: X's own train split
            idx = list(X_tr if cap is None else np.asarray(X_tr)[:cap])
        else:                                       # OOD source
            idx = list(sampled_fn(splits[d], seed, cap))
            if carve == "all-rows":                 # never train on a row we cannot label
                m = np.asarray(labelled[d], dtype=bool)
                idx = [int(i) for i in idx if m[i]]
        # BUILD-TIME GUARD: a NAMED source contributing ZERO rows is the silent-admission bug --
        # a pool label that overstates its contents. Fail loud rather than train on a
        # smaller-than-labelled pool.
        if d != X and len(idx) == 0:
            raise SystemExit(f"build_rows: source '{d}' for eval '{X}' contributed 0 rows "
                             f"(cap={cap}). A named source must draw rows or the pool label lies.")
        train_rows += [(d, int(i)) for i in idx]
    return train_rows, test_rows


def coverage(labelled):
    """(n_labelled, n_rows, fraction) — reported as a first-class number, never a footnote."""
    lab = np.asarray(labelled, dtype=bool)
    return int(lab.sum()), len(lab), float(lab.mean())
