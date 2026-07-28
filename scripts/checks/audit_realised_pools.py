"""PHASE-0 AUDIT (Round 3): realised-vs-labelled training-pool sizes, via the REAL build_rows code path.

WHY. Pool labels like `…+expertqa:360+factscore:360` may overstate the realised pool: `sampled_train_idx`
filters `split=="train"`, and the eval-only sets (expertqa/factscore/asqa) are 100% `split="test"`, so they
contribute ZERO rows as sources. This turns a code-read inference into a LOGGED FACT before Task A acts.

FAITHFUL BUT LIGHT. `load_per_token` builds its split array as `[r["split"] for r in records]` with no label
filtering (attn_pool.py:93), so the split the drivers feed to `build_rows` is reproduced EXACTLY by reading
records alone -- no heavy per-token states, login-node-safe. We import the drivers' OWN
`build_rows`/`cells`/`cells_long`/`sampled_train_idx` (not a re-implementation), so realised counts match the
drivers cell-for-cell.

WHAT IT DECIDES (pre-committed rule). Both drivers guard `if not train_rows: continue`
(probedriftlong.py:183, contribution_ladder.py:169) -- a cell whose sources ALL contribute 0 is DROPPED (no
PRR from an untrained model); a mixed cell is silently SMALLER. So the audit classifies every cell as
exact / SMALLER / DROPPED and flags any cell that (against expectation) emitted a PRR on 0 realised rows
(the only Option-2 escalation trigger).

    python scripts/checks/audit_realised_pools.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

# the drivers' OWN pool-construction code (module-level import; main() is __main__-guarded so nothing runs)
from xl_rungs import build_rows, eval_split, cells as xl_cells                      # noqa: E402
from probedriftlong import cells_long, sampled_train_idx as pdl_sampled, LONG, LONG_SRC  # noqa: E402
from contribution_ladder import sampled_train_idx as cl_sampled                     # noqa: E402

MODEL_SLUG = "meta-llama_Meta-Llama-3.1-8B"
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}
LABEL = {"expertqa": "factuality", "factscore": "factuality"}   # rest: correctness (only used to note usable rows)


def load_split(dataset):
    """(split_array, records) exactly as load_per_token would produce -- records order, no filtering."""
    base = ROOT / "cache" / REGIME[dataset] / "records" if dataset in REGIME else ROOT / "cache" / "records"
    hits = [h for h in base.glob(f"*__{dataset}__ID.jsonl") if "Meta-Llama-3.1-8B" in h.name]
    if len(hits) != 1:
        return None
    recs = [json.loads(l) for l in open(hits[0])]
    return np.array([r["split"] for r in recs]), recs


def parse_spec(spec):
    """spec = [(dataset, cap)] -> labelled total (cap None on ID = 'all X's train')."""
    return {d: cap for d, cap in spec}


def audit(driver, cells_fn, sampled_fn, sources_list, evals):
    print(f"\n{'='*90}\n{driver}\n{'='*90}")
    PT = {}
    for d in sorted(set(sources_list) | set(evals)):
        sr = load_split(d)
        if sr is None:
            print(f"  (no single Llama record file for {d} -- skipped)")
            continue
        PT[d] = (None, sr[0], None, sr[1])
    sources = set(PT)
    rows = []
    for rung, X, spec in cells_fn(sources, [e for e in evals if e in sources]):
        # realised via the REAL build_rows (seed fixed; realised COUNT is seed-independent for capped sources)
        train_rows, test_rows = build_rows(X, spec, PT, 1, sampled_fn)
        labelled = parse_spec(spec)
        realised = {}
        for d, _i in train_rows:
            realised[d] = realised.get(d, 0) + 1
        # status
        dropped = (not train_rows) or (not test_rows)
        anomalies = []          # per-source lines worth showing (zero or shrunk)
        shrunk = False
        for d, cap in spec:
            got = realised.get(d, 0)
            lab = "all" if cap is None else cap
            if d == X:
                continue                                      # the ID self-source (X_tr carve) is expected
            if got == 0:
                shrunk = True; anomalies.append(f"{d}:{got}/{lab}  <ZERO ROWS")
            elif cap is not None and got < cap:
                shrunk = True; anomalies.append(f"{d}:{got}/{lab}  (short)")
        lab_tot = sum(c for _d, c in spec if isinstance(c, int))
        real_tot = sum(realised.values())
        status = "DROPPED (no PRR)" if dropped else ("SMALLER" if shrunk else "exact")
        rows.append((X, rung, status, lab_tot, real_tot, anomalies))
        flag = "  <==" if status != "exact" else ""
        print(f"  {X:13s} {rung:18s} {status:16s} labelled~{lab_tot:5d} realised={real_tot:5d}{flag}")
        for ps in anomalies:
            print(f"        {ps}")
    return rows


def main():
    ALL9 = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa", "asqa"]

    # ---- §C.2 / §B.3 : contribution_ladder (STANDARD ladder) ----
    c2 = audit("§C.2/§B.3  contribution_ladder (STANDARD)", xl_cells, cl_sampled, ALL9, ALL9)

    # ---- §C.3 : probedriftlong (LONG ladder) ----
    c3 = audit("§C.3  probedriftlong (LONG)", cells_long, pdl_sampled, LONG_SRC, LONG)

    # ---- summary: which cells are SMALLER vs DROPPED, and any 0-row cell that still emitted (escalation) ----
    print(f"\n{'='*90}\nSUMMARY\n{'='*90}")
    for tag, rows in [("§C.2/§B.3", c2), ("§C.3", c3)]:
        smaller = [(X, r) for X, r, s, *_ in rows if s == "SMALLER"]
        dropped = [(X, r) for X, r, s, *_ in rows if s.startswith("DROPPED")]
        print(f"\n{tag}: {len(rows)} cells -> {len(smaller)} SMALLER-than-labelled, {len(dropped)} DROPPED, "
              f"{len(rows)-len(smaller)-len(dropped)} exact")
        if smaller:
            print("  SMALLER (documentation defect -- pool trained on fewer rows than the label claims):")
            for X, r in smaller:
                print(f"    {X} {r}")
        if dropped:
            print("  DROPPED (source list all eval-only -> empty train_rows -> cell skipped, no PRR):")
            for X, r in dropped:
                print(f"    {X} {r}")
    print("\nESCALATION CHECK: a DROPPED cell emits no PRR (both drivers guard `if not train_rows: continue`),")
    print("so no untrained-model number can exist -> pre-committed rule resolves to OPTION 1 (fix-forward)")
    print("UNLESS a cell above is marked SMALLER with realised=0 (would mean the guard failed). None expected.")

    # ---- V3: asqa ID training-row count (does asqa ID get rows via eval_split carve, or 0?) ----
    print(f"\n{'='*90}\nV3 CHECK: asqa ID realised training rows (wMSP/pooler ID path via build_rows)\n{'='*90}")
    sr = load_split("asqa")
    if sr is not None:
        PT = {"asqa": (None, sr[0], None, sr[1])}
        tr, te = build_rows("asqa", [("asqa", None)], PT, 1, pdl_sampled)
        print(f"  asqa ID: realised train rows = {len(tr)}, test rows = {len(te)}  "
              f"(all-test split; eval_split carves ~70/30). "
              f"{'-> NONZERO: V3 zero-rows hypothesis DISCONFIRMED; degeneracy is regularisation-collapse.' if len(tr) else '-> ZERO: V3 closed with zero-rows mechanism.'}")


if __name__ == "__main__":
    main()
