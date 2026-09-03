"""Assert that the shared-pass aggregation equals the two functions it replaces, exactly.

`md_and_rmd_mean` exists only to avoid computing the foreground per-token distances twice. It must
therefore return what `md_mean` and `rmd_mean` return, bit for bit. Equality here is not a tolerance
question: the same arithmetic is applied to the same arrays, so anything short of exact agreement
means the replacement is not a replacement.

    python scripts/checks/md_rmd_shared_pass.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import mahalanobis as MD  # noqa: E402


def main():
    rs = np.random.RandomState(0)
    D = 24
    rows = [rs.randn(rs.randint(1, 9), D).astype(np.float32) for _ in range(30)]
    rows.append(np.zeros((0, D), dtype=np.float32))          # an empty row must stay nan, not zero
    fg = MD.fit_md([rs.randn(11, D).astype(np.float32) for _ in range(40)], layer=0,
                   row_ids=[("fg", i) for i in range(40)], kind="fg")
    bg = MD.fit_md([rs.randn(11, D).astype(np.float32) * 1.7 + 0.4 for _ in range(40)], layer=0,
                   row_ids=[("bg", i) for i in range(40)], kind="bg")

    md_ref, rmd_ref = MD.md_mean(rows, fg), MD.rmd_mean(rows, fg, bg)
    md_new, rmd_new = MD.md_and_rmd_mean(rows, fg, bg)

    ok = True
    for name, a, b in (("md_mean", md_ref, md_new), ("rmd_mean", rmd_ref, rmd_new)):
        same = np.array_equal(a, b, equal_nan=True)
        print(f"{name}: exact match {same} over {len(a)} rows "
              f"(nan rows {int(np.isnan(a).sum())})")
        ok = ok and same
    if not ok:
        print("SHARED PASS CHECK: FAIL")
        return 1
    print("SHARED PASS CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
