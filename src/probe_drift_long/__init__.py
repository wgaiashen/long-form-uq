"""ProbeDriftLong — the long-form generalisation benchmark.

Sibling to `probe_drift` (joestacey/ProbeDrift), which covers short-form and mixed tasks. This
package covers the LONG-FORM grid: 8 evaluation targets x 5 rungs, with the task-family taxonomy,
the rung definitions, and the train/test carving that define the benchmark.

Like `probe_drift`, this is a pure data/grid provider: numpy is its only dependency. It does NOT
generate, judge, extract features, train probes, or compute PRR — those stay in `luq`. It also
does not own generation budgets (`max_new_tokens`), for the same reason `probe_drift` stopped
owning them: they are a property of the pipeline, not of the benchmark.

    from probe_drift_long import cells_long, eval_split, label_of, LONG_DATASETS
"""

__version__ = "0.1.0"

from probe_drift_long.dataset_configs import (
    LONG_DATASETS,
    LONG_SRC,
    SHORT_DATASETS,
    FINE_FAMILIES,
    PARTIALLY_LABELLED,
    DEFAULT_LABEL,
    label_of,
    different_label_projection,
)
from probe_drift_long.ood_settings import (
    LONG_RUNGS,
    TRANSFER_RUNG,
    XL_TOTAL,
    cells_long,
    rung_sources_long,
    get_training_spec_long,
)
from probe_drift_long.splits import (
    XL_TEST_FRAC,
    CARVE_SEED,
    assert_canonical_order,
    build_rows,
    coverage,
    eval_split,
    remap_to_filtered,
    sampled_train_idx,
)

__all__ = [
    "__version__",
    # taxonomy
    "LONG_DATASETS", "LONG_SRC", "SHORT_DATASETS", "FINE_FAMILIES", "PARTIALLY_LABELLED",
    "DEFAULT_LABEL", "label_of", "different_label_projection",
    # rungs
    "LONG_RUNGS", "TRANSFER_RUNG", "XL_TOTAL", "cells_long", "rung_sources_long",
    "get_training_spec_long",
    # splits
    "XL_TEST_FRAC", "CARVE_SEED", "assert_canonical_order", "build_rows", "coverage",
    "eval_split", "remap_to_filtered", "sampled_train_idx",
]
