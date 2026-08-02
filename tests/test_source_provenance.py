"""The source-provenance guard: which probe_drift built a cache.

These pin the three behaviours the guard depends on. The middle one is the subtle one and is the
reason the guard is written the way it is: a MISSING stamp must compare equal, because every cache
written before this existed has none, and treating "unknown" as "different" would refuse to touch
the entire existing grid.
"""
from luq import cache


def test_provenance_has_the_identifying_fields():
    p = cache.source_provenance()
    assert p["package"] == "probe_drift"
    # path + a content hash of dataset_configs.py are what identify a checkout. `version` is NOT
    # identifying (0.1.0 in both checkouts), which is why it is excluded from PROVENANCE_KEYS.
    assert p["path"] and p["dataset_configs_sha256"]
    assert "version" not in cache.PROVENANCE_KEYS
    assert set(cache.PROVENANCE_KEYS) == {"path", "dataset_configs_sha256"}


def test_missing_stamp_is_unknown_not_different():
    """No stamp -> no complaint. Absence and disagreement must not be confusable."""
    assert cache.provenance_mismatch(None, cache.source_provenance()) is None


def test_identical_provenance_matches_and_a_changed_checkout_does_not():
    cur = cache.source_provenance()
    assert cache.provenance_mismatch(dict(cur), cur) is None

    other_path = dict(cur, path="/somewhere/else/ProbeDrift/probe_drift")
    why = cache.provenance_mismatch(other_path, cur)
    assert why and "path" in why

    edited = dict(cur, dataset_configs_sha256="0" * 64)
    why = cache.provenance_mismatch(edited, cur)
    assert why and "dataset_configs_sha256" in why

    # A version bump alone must NOT trip the guard: both checkouts report 0.1.0, so keying on it
    # would let a real library swap pass as a match.
    assert cache.provenance_mismatch(dict(cur, version="9.9.9"), cur) is None


def test_round_trip_through_disk(tmp_path):
    prov = cache.source_provenance()
    cache.save_source_provenance(prov, tmp_path, "somekey")
    assert cache.load_source_provenance(tmp_path, "somekey") == prov
    assert cache.load_source_provenance(tmp_path, "absent_key") is None
