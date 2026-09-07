"""The five long-form rungs, and the cell grid they generate.

Mirrors `probe_drift.ood_settings` in shape: pure dict/list arithmetic over the taxonomy in
`dataset_configs`, no I/O and no ML dependencies.

The rungs, for an eval target X:

    ID              train on X's own train split
    SameTask-long   the other long sets in X's FINE family
    DiffTask-long   every long set in a DIFFERENT fine family
    LOO-long        all seven other long sets
    1ds-Diff-long   the FIRST different-family set only, at the full budget

`DiffTask-long` splits on the fine family rather than the broad one. This differs from the
sibling short-form benchmark, whose different-task setting is a broad split between question
answering and summarisation. Here factuality is its own family, so "different task" is a
three-way distinction. Widening it to the broad split would change every cross-task cell.

`1ds-Diff-long` selects the first different-family source, so it depends on the order of
`LONG_SRC`. See `dataset_configs` for why that order is fixed.
"""

from .dataset_configs import (FINE_FAMILIES, LONG_DATASETS, LONG_SRC, SHORT_DATASETS)

# Matched training budget per multi-source rung, split evenly across that rung's sources.
XL_TOTAL = 1800

LONG_RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
TRANSFER_RUNG = "Long->Short"


def rung_sources_long(X, sources=None):
    """The candidate source datasets for each rung of eval target X, before capping.

    `sources` restricts to what is actually available (default: the full LONG_SRC pool).
    X is always excluded from its own sources, which is what makes it safe for the samplers to
    draw from a source's full row set.
    """
    pool = LONG_SRC if sources is None else [d for d in LONG_SRC if d in sources]
    same = [d for d in pool if d != X and FINE_FAMILIES[d] == FINE_FAMILIES[X]]
    diff = [d for d in pool if d != X and FINE_FAMILIES[d] != FINE_FAMILIES[X]]
    loo = [d for d in pool if d != X]
    return {"SameTask-long": same, "DiffTask-long": diff, "LOO-long": loo,
            "1ds-Diff-long": diff[:1]}


def cells_long(sources, evals):
    """[(rung_tag, X, spec)] for every eval in `evals` that has data, where spec = [(source, cap)].

    `cap=None` on the ID cell means "all of X's train rows". Multi-source rungs split XL_TOTAL
    evenly; `1ds-Diff-long` gives its single source the whole budget. The eval target's TEST rows
    do not come from here -- they come from `splits.eval_split`.

    A rung with no available sources is omitted rather than emitted empty. An empty cell that
    still appears in the grid is indistinguishable downstream from a cell that was measured and
    scored zero, so it must not be produced in the first place.
    """
    out = []
    for X in evals:
        if X not in sources:
            continue
        if X in LONG_DATASETS:
            out.append(("ID", X, [(X, None)]))
            rs = rung_sources_long(X, sources)
            for tag in ("SameTask-long", "DiffTask-long", "LOO-long"):
                srcs = rs[tag]
                if srcs:
                    cap = max(1, XL_TOTAL // len(srcs))
                    out.append((tag, X, [(d, cap) for d in srcs]))
            one = rs["1ds-Diff-long"]
            if one:
                out.append(("1ds-Diff-long", X, [(one[0], XL_TOTAL)]))
        elif X in SHORT_DATASETS:                      # long -> short transfer
            srcs = [d for d in LONG_SRC if d in sources]
            if srcs:
                cap = max(1, XL_TOTAL // len(srcs))
                out.append((TRANSFER_RUNG, X, [(d, cap) for d in srcs]))
    return out


def get_training_spec_long(X, rung, sources=None):
    """The [(source, n_samples)] spec for ONE (eval, rung) pair -- the ProbeDrift-shaped accessor.

    `cells_long` is what the ladders drive off; this is the single-cell convenience form, and it
    raises rather than returning an empty spec so a missing rung can never be mistaken for a
    measured-but-empty one.
    """
    pool = set(LONG_SRC if sources is None else sources)
    for tag, ev, spec in cells_long(pool | {X}, [X]):
        if tag == rung:
            return spec
    raise ValueError(f"no '{rung}' rung for eval '{X}' (available: "
                     f"{[t for t, _, _ in cells_long(pool | {X}, [X])]})")
