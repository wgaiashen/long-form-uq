"""Grounding tests for the benchmark definition in `src/probe_drift_long`.

The grid decides which examples go where, so every reported number depends on it. A change here
would move results silently rather than raising, which is why these assert the definition itself
rather than any method built on it. Everything below is arithmetic over the frozen constants: no
dataset, no cache, no model and no network.
"""
import numpy as np
import pytest

import probe_drift_long as P


# ---- the frozen constants ------------------------------------------------------------------
def test_eight_long_targets_in_three_families():
    assert len(P.LONG_DATASETS) == 8
    assert set(P.FINE_FAMILIES) == set(P.LONG_DATASETS)
    assert set(P.FINE_FAMILIES.values()) == {"correctness_qa", "factuality", "summ"}


def test_budget_and_carve_constants():
    assert P.XL_TOTAL == 1800
    assert P.XL_TEST_FRAC == 0.30
    assert P.CARVE_SEED == 0


# ---- the rungs -----------------------------------------------------------------------------
def test_grid_is_eight_targets_by_five_rungs():
    cells = P.cells_long(list(P.LONG_SRC), list(P.LONG_DATASETS))
    assert len(cells) == 8 * 5
    assert {tag for tag, _, _ in cells} == set(P.LONG_RUNGS)


def test_a_target_is_never_a_source_for_its_own_evaluation():
    """Training on the target would leak the test set into the training pool."""
    for tag, X, spec in P.cells_long(list(P.LONG_SRC), list(P.LONG_DATASETS)):
        if tag == "ID":
            continue
        assert X not in [d for d, _ in spec], f"{X} is a source of its own {tag} cell"


def test_multi_source_rungs_split_the_budget_evenly():
    for tag, X, spec in P.cells_long(list(P.LONG_SRC), list(P.LONG_DATASETS)):
        if tag == "ID" or len(spec) == 1:
            continue
        caps = {cap for _, cap in spec}
        assert caps == {P.XL_TOTAL // len(spec)}, (tag, X, spec)


def test_one_dataset_rung_gets_the_whole_budget_from_one_source():
    for tag, X, spec in P.cells_long(list(P.LONG_SRC), list(P.LONG_DATASETS)):
        if tag != "1ds-Diff-long":
            continue
        assert len(spec) == 1 and spec[0][1] == P.XL_TOTAL


def test_one_dataset_rung_takes_the_first_different_family_source_in_list_order():
    """The rule the ordering comment in `dataset_configs` describes, asserted rather than
    described. Note that the `sources` argument is a filter, not an ordering: the iteration is
    always over `LONG_SRC`, so a caller cannot change the grid by passing a reordered list."""
    for tag, X, spec in P.cells_long(list(P.LONG_SRC), list(P.LONG_DATASETS)):
        if tag != "1ds-Diff-long":
            continue
        expected = next(d for d in P.LONG_SRC
                        if d != X and P.FINE_FAMILIES[d] != P.FINE_FAMILIES[X])
        assert spec[0][0] == expected


def test_the_one_dataset_assignment_is_frozen():
    """Pinned to the values the reported results were produced under. Sorting, de-duplicating or
    otherwise tidying `LONG_SRC` would change which single dataset trains the narrowest
    out-of-distribution setting on every target, silently, so it fails here instead."""
    assert {X: spec[0][0]
            for tag, X, spec in P.cells_long(list(P.LONG_SRC), list(P.LONG_DATASETS))
            if tag == "1ds-Diff-long"} == {
        "pubmed_qa": "xsum",
        "xsum": "pubmed_qa",
        "cnn_dailymail": "pubmed_qa",
        "med_quad": "xsum",
        "samsum": "pubmed_qa",
        "expertqa": "pubmed_qa",
        "asqa": "xsum",
        "factscore": "pubmed_qa",
    }


# ---- the carve -----------------------------------------------------------------------------
def _splitless(n):
    return np.array(["train"] * n)


def test_carve_is_deterministic_at_a_fixed_seed():
    a = P.eval_split(_splitless(948), carve="legacy")
    b = P.eval_split(_splitless(948), carve="legacy")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_carve_holds_out_the_configured_fraction():
    _, test = P.eval_split(_splitless(1000), carve="legacy")
    assert len(test) == round(1000 * P.XL_TEST_FRAC)


def test_train_and_test_are_disjoint_and_cover_every_row():
    train, test = P.eval_split(_splitless(500), carve="legacy")
    assert not set(train) & set(test)
    assert len(train) + len(test) == 500


def test_a_baked_in_split_is_used_untouched():
    split = np.array(["train"] * 30 + ["test"] * 70)
    train, test = P.eval_split(split, carve="legacy")
    assert len(train) == 30 and len(test) == 70


def test_both_carve_modes_agree_when_every_row_is_labelled():
    n = 600
    legacy = P.eval_split(_splitless(n), carve="legacy")
    allrows = P.eval_split(_splitless(n), np.ones(n, dtype=bool), carve="all-rows")
    assert np.array_equal(legacy[1], allrows[1])


def test_all_rows_carve_refuses_to_guess_a_label_mask():
    """Defaulting the mask would silently score a different population, so it raises instead."""
    with pytest.raises(ValueError):
        P.eval_split(_splitless(100), carve="all-rows")
    with pytest.raises(ValueError):
        P.eval_split(_splitless(100), np.ones(100, dtype=bool), carve="legacy")


def test_unlabelled_rows_are_dropped_from_both_sides_not_reassigned():
    n = 300
    labelled = np.array([i % 3 != 0 for i in range(n)])
    train, test = P.eval_split(_splitless(n), labelled, carve="all-rows")
    assert labelled[train].all() and labelled[test].all()
    assert len(train) + len(test) == int(labelled.sum())


# ---- the row-order guard -------------------------------------------------------------------
def _records(n, split="train"):
    return [{"split": split, "idx": i} for i in range(n)]


def test_canonical_order_accepts_contiguous_rows():
    P.assert_canonical_order(_records(50), "synthetic")


def test_canonical_order_rejects_duplicates_and_reordering():
    dupes = _records(10) + [{"split": "train", "idx": 0}]
    with pytest.raises(SystemExit):
        P.assert_canonical_order(dupes, "synthetic")
    shuffled = _records(10)
    shuffled[2], shuffled[7] = shuffled[7], shuffled[2]
    with pytest.raises(SystemExit):
        P.assert_canonical_order(shuffled, "synthetic")


# ---- sampling and coverage -----------------------------------------------------------------
def test_source_sampling_is_deterministic_and_respects_the_cap():
    split = np.array(["train"] * 1800)
    a = P.sampled_train_idx(split, 1, 360)
    b = P.sampled_train_idx(split, 1, 360)
    assert len(a) == 360 and np.array_equal(a, b)
    assert not np.array_equal(a, P.sampled_train_idx(split, 2, 360))


def test_a_cap_at_or_above_the_pool_size_takes_everything():
    split = np.array(["train"] * 100)
    assert len(P.sampled_train_idx(split, 1, 5000)) == 100


def test_coverage_reports_the_labelled_fraction():
    n_lab, n_rows, frac = P.coverage([True, True, False, True])
    assert (n_lab, n_rows) == (3, 4)
    assert frac == pytest.approx(0.75)
