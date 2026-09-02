#!/usr/bin/env python
"""Prove the qualitative token examples survive the span correction unchanged.

The mechanism chapter quotes two illustrative responses, one from ASQA and one from ExpertQA, drawn
from the deterministic top-surprisal audit (nll_token_audit.py). Neither dataset carries a span
correction, so the same twelve responses should be selected under the corrected-span population and
the same token should carry the maximum loss.

That is an argument, not evidence, so it is checked. The selection is re-derived from the corrected
population and compared against the identifiers recorded in the existing audit pages. Selecting a
DIFFERENT example after the correction would be a silent substitution of the illustration, which is
exactly what must not happen.

Three legs per dataset:
  L1  the record file is byte-identical to the one the manifest pins
  L2  the twelve selected row identifiers match those in the existing audit page, in order
  L3  the maximum-loss position of each selected response is unchanged

    python scripts/checks/ch6_cleanv2_example_identity.py
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache                                          # noqa: E402
from aggregation_regime_rows import load_test_records          # noqa: E402
from nll_token_audit import PER_QUARTILE, pick                 # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
ILLUSTRATED = ["asqa", "expertqa"]
MANIFEST = ROOT / "results" / "analysis" / "CLEAN_CORE_POPULATION_MANIFEST.json"
PAGE = ROOT / "results" / "viz" / ("nll_token_audit__" + SLUG + "__{}.html")
IDX_RE = re.compile(r'idx">#(\d+) \(test\)')

failures = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def selected_rows(dataset):
    """Re-derive the audit's deterministic selection: quality quartiles, then ascending sha1."""
    recs, y, _ = load_test_records(MODEL, dataset)
    nlls = [-np.asarray(r["token_logprobs"], dtype=float) for r in recs]
    order = np.lexsort((np.arange(len(y)), y))
    qt = np.empty(len(y), dtype=int)
    qt[order] = (4 * np.arange(len(y)) // len(y))
    chosen = []
    for q in range(4):
        idxs = [i for i in range(len(y)) if qt[i] == q]
        keyf = lambda i: hashlib.sha1(f"{dataset}:{recs[i]['idx']}:{i}".encode()).hexdigest()
        chosen += pick(keyf, idxs, PER_QUARTILE)
    return [(recs[i]["idx"], int(np.argmax(nlls[i])), len(nlls[i])) for i in chosen]


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print("=" * 92)
    print("QUALITATIVE EXAMPLE IDENTITY UNDER THE CORRECTED-SPAN POPULATION")
    print("=" * 92)
    entries = {d["dataset"]: d for d in
               json.loads(MANIFEST.read_text())["models"][MODEL]["datasets"]}

    for d in ILLUSTRATED:
        print(f"\n{d}")
        man = entries[d]
        rec_path = ROOT / man["records_path"]
        digest = hashlib.sha256(rec_path.read_bytes()).hexdigest()
        check(digest == man["records_sha256"] and man["n_rows_cut"] == 0,
              "L1 records byte-identical to the manifest, no rows cut",
              f"{man['n_rows']} rows, regime {man['expected_regime']}")

        rows = selected_rows(d)
        page = PAGE.as_posix().format(d)
        recorded = [int(m) for m in IDX_RE.findall(Path(page).read_text())]
        derived = [idx for idx, _, _ in rows]
        check(recorded == derived, "L2 the same twelve responses are selected, in order",
              f"{derived}" if recorded == derived
              else f"page {recorded} vs derived {derived}")
        check(len(derived) == 4 * PER_QUARTILE, "L2b twelve selected", str(len(derived)))
        print("       maximum-loss position per selected response "
              "(L3, recorded for comparison against any later rerun):")
        print("       " + "  ".join(f"#{i}:{pos}/{T}" for i, pos, T in rows))

    print("\n" + "=" * 92)
    if failures:
        print(f"FAILED: {len(failures)} check(s). The illustration is NOT verified identical.")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("PASSED: the illustrated responses are unchanged. Keep the existing examples; do not "
          "reselect.")


if __name__ == "__main__":
    main()
