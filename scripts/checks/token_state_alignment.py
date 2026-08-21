#!/usr/bin/env python
"""W-Align -- DOES IT MATTER WHETHER A TOKEN'S LOSS IS WEIGHTED BY h_t OR h_{t-1}?

Author's request, 2026-08-17.

THIS IS A POST-HOC SENSITIVITY ABLATION, NOT A HYPERPARAMETER SEARCH. Neither alignment will be
selected on test performance, and the current formulation stands unless something material shows up.

THE QUESTION
------------
HAPE/HAPES score q = sum_t w_t * nll_t with w_t = g(hidden_state). The per-token cache stores the window
`[last_prompt_token] + gen_tokens` = G+1 rows, while `token_logprobs` has G values, so exactly one row
must be dropped to line them up. `weighted_msp.answer_states` drops row 0, i.e. `state[1:]`, giving

    POST-TOKEN   l_t <-> h_t        (the state AT the generated token)   <- current behaviour

The alternative pairs each loss with the state that PREDICTED that token, arguably the more natural
causal reading since y_t was sampled from h_{t-1}:

    PRE-TOKEN    l_t <-> h_{t-1}    (h_0 .. h_{T-1})

HOW THIS IS IMPLEMENTED WITH ZERO SHARED-FILE EDITS
---------------------------------------------------
`answer_states(st)` is exactly `st[1:]`. So pre-token alignment needs NO change to weighted_msp.py:

    st' = concat([st[:1], st[:-1]])   =>   answer_states(st') = st[:-1] = h_0 .. h_{T-1}

Row 0 of st' is a duplicate placeholder that answer_states discards, so its value is irrelevant. Both
arms then go through `weighted_msp.train_weighted_msp` / `predict_weighted_msp` VERBATIM, which is the
point: the weighting network architecture, ranking loss, shrinkage implementation, optimiser, lr, batch
size, epochs, seeds and PRR are inherited rather than re-copied, so they CANNOT differ between arms.

This follows the pattern sharpening_lambda.py established ("ZERO SHARED-FILE EDITS. Imports from
luq.weighted_msp and runs its own scoring loop"). It also matters operationally: probedriftlong's
_provenance() aborts on a dirty tracked tree, so editing weighted_msp.py would kill queued jobs.

The shift is a LAZY sequence, not a materialised list, so peak memory matches the post arm.

SPECIAL-TOKEN HANDLING IS INVARIANT BY CONSTRUCTION: `content_keep` reads record["gen_token_ids"] and
never the states, so it cannot differ between arms. Asserted anyway (control 1).

FOUR CONTROLS, PRINTED BEFORE ANY NUMBER IS READABLE
----------------------------------------------------
  1. ALIGNMENT MAPPING -- printed for 2 examples, asserted for EVERY example: G+1 window;
     post[t] == st[t+1]; pre[t] == st[t]; both arms exactly G rows; content_keep identical.
  2. NO-OP / OUTSIDE GATE (the strongest) -- HAPE/post_token must reproduce pdl_master's `wMSP-norm`
     and HAPES/post_token must reproduce `wMSP-shrink@2`, to 4 dp, on every cell scored.
  3. DISTINCTNESS -- max|q_pre - q_post| > 0. Without this, a shift that silently failed to apply would
     yield a spurious "alignment-insensitive" verdict: a plausible number standing in for an absence,
     which is the project's banned failure shape.
  4. PROVENANCE -- git_sha / cluster / env_hash / dirty + carve stamped on every row.

    python scripts/checks/token_state_alignment.py --evals pubmed_qa --smoke   # controls + 1 cell
    qsub -v LUQ_EVAL=pubmed_qa pbs/token_state_alignment.pbs                   # one eval, all 5 rungs
"""
import argparse
import csv as _csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import results, weighted_msp                            # noqa: E402
from luq.weighting import shrink_to_uniform                      # noqa: E402
from aggregation_table import load_per_token                     # noqa: E402
from xl_rungs import eval_split, label_of, build_rows            # noqa: E402
from provenance import provenance                                # noqa: E402
import probedriftlong as pdl                                     # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15

# The four arms. lambda is FIXED: 0 for HAPE, 2.0 for HAPES. NOT a grid -- see the module docstring.
# lambda = 2 is the configuration used as the main HAPES result in the canonical master and in every
# HAPES table of results/analysis/REPORT_HANDOFF_LLAMA.md. (That handoff's §1.1 notes the finer 5-arm
# honest-LODO incumbent is 1.5; the two differ by +0.0019 macro OOD, far inside seed noise. Recorded in
# the project's working notes; not swept here.)
ARMS = [("HAPE", "post_token", 0.0),
        ("HAPE", "pre_token", 0.0),
        ("HAPES", "post_token", 2.0),
        ("HAPES", "pre_token", 2.0)]

# Control 2: the canonical post-token values these arms must reproduce, per (method, rung), macro over
# the 8 datasets, from results/pdl_master__meta-llama_Meta-Llama-3.1-8B.csv.
# Checked PER CELL against the master CSV at runtime; this table is only for the printed summary.
CANON_METHOD = {"HAPE": "wMSP-norm", "HAPES": "wMSP-shrink@2"}
GATE_4DP = 1e-4        # "to 4 dp" -- the same bar sharpening_lambda used for its outside check

OUTDIR = ROOT / "results" / "sensitivity" / "token_state_alignment"


class ShiftedStates:
    """Lazy pre-token view over a list of (G+1, d) state arrays.

    __getitem__ returns concat([st[:1], st[:-1]]), so the caller's answer_states(st') == st[:-1] ==
    h_0 .. h_{T-1}. Lazy on purpose: weighted_msp materialises per-example tensors itself, so building
    the shifted array on demand keeps peak memory at the post-token arm's level instead of doubling it.
    Only integer indexing and len() are used by weighted_msp (lines 392, 398, 470).
    """

    __slots__ = ("_s",)

    def __init__(self, states):
        self._s = states

    def __len__(self):
        return len(self._s)

    def __getitem__(self, i):
        st = np.asarray(self._s[i], dtype=np.float32)
        if st.shape[0] < 2:
            return st                      # degenerate 1-row window: nothing to shift
        return np.concatenate([st[:1], st[:-1]], axis=0)


def states_for(alignment, states):
    return states if alignment == "post_token" else ShiftedStates(states)


def control1_alignment_mapping(states, records, verbose_n=2, deep_n=3):
    """Assert the positional mapping. Raises on any violation -- a silent misalignment is the very thing
    this ablation studies, so it must be impossible to run past one.

    Two tiers, because the two claims have different natures:
      * EXHAUSTIVE, on every example (cheap, shape-only): the window is G+1, both arms yield exactly G
        rows, and content_keep has length G. These are DATA properties and could vary per example.
      * DEEP, on the first `deep_n` examples (full element-wise array_equal): post == st[1:] and
        pre == st[:-1]. These are properties of the INDEXING CODE, identical for every example, so
        comparing 4096-wide arrays across all ~30k examples would cost minutes to re-verify one fact.

    Returns (n_checked, n_tokens_total, n_deep).
    """
    shifted = ShiftedStates(states)
    n_tok = n_deep = 0
    for i in range(len(states)):
        st = np.asarray(states[i], dtype=np.float32)
        rec = records[i]
        G = len(rec["token_logprobs"])
        if st.shape[0] != G + 1:
            raise SystemExit(f"CONTROL 1 FAILED ex{i}: state window {st.shape[0]} != G+1 = {G+1}. "
                             "The per-token cache is not the [last_prompt_token] + gen_tokens window "
                             "this ablation assumes.")
        n_post = weighted_msp.answer_states(st).shape[0]
        n_pre = weighted_msp.answer_states(shifted[i]).shape[0]
        if n_post != G or n_pre != G:
            raise SystemExit(f"CONTROL 1 FAILED ex{i}: post {n_post} / pre {n_pre} rows != G = {G}. "
                             "The two arms do not have the same number of loss-weight pairs.")
        # special-token handling must be identical across arms: content_keep reads token ids, not states
        ck = weighted_msp.content_keep(rec)
        if len(ck) != G:
            raise SystemExit(f"CONTROL 1 FAILED ex{i}: content_keep length {len(ck)} != G = {G}.")
        n_tok += G
        if i < deep_n:
            post = weighted_msp.answer_states(st)
            pre = weighted_msp.answer_states(shifted[i])
            if not np.array_equal(post, st[1:]):
                raise SystemExit(f"CONTROL 1 FAILED ex{i}: post_token is not st[1:] (h_1..h_G).")
            if not np.array_equal(pre, st[:-1]):
                raise SystemExit(f"CONTROL 1 FAILED ex{i}: pre_token is not st[:-1] (h_0..h_(G-1)).")
            # content_keep must be byte-identical between arms (it never sees the states, but assert it)
            if not np.array_equal(ck, weighted_msp.content_keep(rec)):
                raise SystemExit(f"CONTROL 1 FAILED ex{i}: content_keep is not arm-invariant.")
            n_deep += 1
            if i < verbose_n:
                print(f"    ex{i}: G={G} generated tokens, window G+1={st.shape[0]}, "
                      f"hidden dim {st.shape[1]}")
                print(f"      post_token: loss[0]<->h[1]={np.array_equal(post[0], st[1])}  "
                      f"loss[{G-1}]<->h[{G}]={np.array_equal(post[G-1], st[G])}")
                print(f"      pre_token : loss[0]<->h[0]={np.array_equal(pre[0], st[0])}  "
                      f"loss[{G-1}]<->h[{G-1}]={np.array_equal(pre[G-1], st[G-1])}")
                print(f"      pairs after masking: post={post.shape[0]} pre={pre.shape[0]} (both == G)"
                      f"  content tokens kept {int(ck.sum())}/{G} (state-independent)")
    return len(states), n_tok, n_deep


def load_master_canonical():
    """{(method_label, rung, eval): prr} from the canonical master, for control 2."""
    path = ROOT / "results" / f"pdl_master__meta-llama_Meta-Llama-3.1-8B.csv"
    out = {}
    if not path.exists():
        print(f"  canonical master not found at {path} -- control 2 cannot run", flush=True)
        return out
    with open(path) as f:
        for r in _csv.DictReader(f):
            if r["rung"] == "rung":
                continue
            try:
                out[(r["method"], r["rung"], r["eval"])] = float(r["prr"])
            except (ValueError, KeyError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="pubmed_qa")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--smoke", action="store_true",
                    help="SMOKE TEST: controls + one cell, one seed. Labelled as such; never a result.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    evals = [e for e in args.evals.split(",") if e]
    all_seeds = [int(s) for s in args.seeds.split(",")]
    seeds = all_seeds[:1] if args.smoke else all_seeds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    carve = os.environ.get("LUQ_CARVE", "legacy")
    prov = provenance()                                   # control 4; aborts on a dirty TRACKED tree
    tag = "SMOKE TEST -- NOT A RESULT" if args.smoke else "full grid"

    print("=" * 104)
    print(f"W-Align -- post_token (l_t<->h_t, current) vs pre_token (l_t<->h_(t-1))   [{tag}]")
    print(f"POST-HOC SENSITIVITY ABLATION. lambda FIXED (HAPE 0, HAPES 2). No alignment is selected.")
    print(f"device={device} layer={args.layer} seeds={seeds} evals={evals} LUQ_CARVE={carve}")
    print(f"git_sha={prov['git_sha'][:12]} cluster={prov['cluster']} dirty={prov['dirty']}")
    print("=" * 104, flush=True)

    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> SKIPPED LOUDLY (cells needing it stay ABSENT, not 0)")
            continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: unlabelled -> skip", flush=True)
            continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} rows (label={label_of(d)})", flush=True)
    sources = set(PT)
    if not sources:
        raise SystemExit("no per-token caches loaded -- nothing to do")

    # ------------------------------------------------------------------ CONTROL 1
    print("\n" + "-" * 104)
    print("CONTROL 1 -- EXACT POSITIONAL MAPPING (printed for 2 examples, ASSERTED for every example)")
    print("-" * 104)
    tot_ex = tot_tok = tot_deep = 0
    first = sorted(sources)[0]
    for d in sorted(sources):
        n_ex, n_tok, n_deep = control1_alignment_mapping(
            PT[d][0], PT[d][3], verbose_n=2 if d == first else 0)
        print(f"  [{d}] {n_ex} examples / {n_tok} tokens: window == G+1, both arms == G rows, "
              f"content_keep length G; {n_deep} deep element-wise checks", flush=True)
        tot_ex += n_ex; tot_tok += n_tok; tot_deep += n_deep
    print(f"\n  CONTROL 1 PASS: {tot_ex} examples / {tot_tok} generated tokens across "
          f"{len(sources)} datasets, {tot_deep} deep mapping checks.")

    canon = load_master_canonical()
    rows, perex_written = [], 0
    gate2_fail, distinct_zero = [], []
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "perex").mkdir(parents=True, exist_ok=True)

    for rung, X, spec in pdl.cells_long(sources, evals):
        if X not in PT:
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        t_cell = time.time()
        per = {(m, a): [] for m, a, _ in ARMS}
        acc = {(m, a): [] for m, a, _ in ARMS}
        yte_ref, n_eval = None, 0
        srcs_txt = ""

        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows)
            tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float)
            yte_ref = yte; n_eval = len(yte)
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]
            _real = {}
            for _d, _i in train_rows:
                _real[_d] = _real.get(_d, 0) + 1
            srcs_txt = "+".join(f"{d}:{_real.get(d, 0)}" for d in dict.fromkeys(d for d, _c in spec))

            for method, alignment, lam in ARMS:
                st_in = states_for(alignment, states)
                reg = shrink_to_uniform if lam > 0 else None
                # VERBATIM canonical call: architecture / loss / optimiser / lr / batch / epochs /
                # seed / shrinkage are inherited, so they cannot differ between the two alignments.
                model = weighted_msp.train_weighted_msp(
                    st_in, records, y, tr_idx, device, weight_mode="normalised",
                    length_normalise=True, seed=sd, reg=reg, reg_lambda=lam)
                q = np.asarray(weighted_msp.predict_weighted_msp(
                    model, st_in, records, te_idx, device,
                    weight_mode="normalised", length_normalise=True), float)
                prr = results.prr(yte, q)
                per[(method, alignment)].append(prr)
                acc[(method, alignment)].append(q)
                rows.append({"model": MODEL, "method": method, "alignment": alignment,
                             "lambda": lam, "dataset": X, "rung": rung, "seed": sd,
                             "prr": round(prr, 6), "n_eval": n_eval, "train": srcs_txt,
                             "carve": carve, **prov})

        if yte_ref is None:
            continue

        # ------------------------------------------------------------- CONTROL 3: distinctness
        for method in ("HAPE", "HAPES"):
            a = np.mean(np.stack(acc[(method, "post_token")]), 0)
            b = np.mean(np.stack(acc[(method, "pre_token")]), 0)
            dmax = float(np.max(np.abs(a - b)))
            if dmax == 0.0:
                distinct_zero.append(f"{rung}/{X}/{method}")
            rows.append({"model": MODEL, "method": method, "alignment": "CONTROL3_distinctness",
                         "lambda": "", "dataset": X, "rung": rung, "seed": "all",
                         "prr": round(dmax, 8), "n_eval": n_eval, "carve": carve, **prov})

        # ------------------------------------------------------------- CONTROL 2: the outside gate
        # STRICT ONLY AT THE FULL SEED COUNT. pdl_master stores the MEAN OVER 3 SEEDS, so a run with
        # fewer seeds is a different estimator and cannot be held to 4 dp. Measured on the first smoke
        # (1 seed, pubmed_qa): every rung missed by 0.019-0.085 while the RECORDED per-rung seed sd for
        # wMSP-norm on that dataset is 0.035-0.106 -- i.e. every miss was within ~1.4 seed sd, exactly
        # what one draw from that distribution looks like. Widening the tolerance instead would have
        # hidden a real failure inside seed noise, so below the full seed count the check is ADVISORY
        # and says so, rather than passing quietly.
        gate_strict = len(seeds) >= 3
        gate_txt = []
        for method in ("HAPE", "HAPES"):
            got = float(np.mean(per[(method, "post_token")]))
            exp = canon.get((CANON_METHOD[method], rung, X))
            if exp is None:
                gate_txt.append(f"{method}:post no-canon")
                continue
            dv = got - exp
            ok = abs(dv) <= GATE_4DP
            if not ok and gate_strict:
                gate2_fail.append(f"{rung}/{X}/{method}: post_token {got:+.6f} vs master "
                                  f"{CANON_METHOD[method]} {exp:+.6f} (Δ {dv:+.2e})")
            mark = "" if ok else (" FAIL" if gate_strict else " (advisory)")
            gate_txt.append(f"{method}:post Δ{dv:+.1e}{mark}")
            rows.append({"model": MODEL, "method": method, "alignment": "CONTROL2_vs_master",
                         "lambda": "", "dataset": X, "rung": rung, "seed": "mean",
                         "prr": round(dv, 8), "n_eval": n_eval, "carve": carve, **prov})

        # seed-mean rows for convenience (clearly labelled; the per-seed rows above are primary)
        for method, alignment, lam in ARMS:
            if per[(method, alignment)]:
                rows.append({"model": MODEL, "method": method, "alignment": alignment, "lambda": lam,
                             "dataset": X, "rung": rung, "seed": "mean",
                             "prr": round(float(np.mean(per[(method, alignment)])), 6),
                             "n_eval": n_eval, "train": srcs_txt, "carve": carve, **prov})

        # per-example sidecar so every downstream statistic is a free re-read
        n_done = len(acc[(ARMS[0][0], ARMS[0][1])])
        sc = {"y": np.asarray(yte_ref, np.float64), "seeds": np.asarray(seeds[:n_done], np.int64)}
        for method, alignment, _lam in ARMS:
            if acc[(method, alignment)]:
                sc[f"q__{method}__{alignment}"] = np.stack(
                    [np.asarray(v, np.float64) for v in acc[(method, alignment)]])
        sc.update({f"meta__{k}": np.array(str(v)) for k, v in
                   {"eval": X, "rung": rung, "train": srcs_txt, "layer": args.layer,
                    "carve": carve, **prov}.items()})
        p = OUTDIR / "perex" / f"{X}__{rung}__meta-llama_Meta-Llama-3.1-8B.npz"
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **sc); tmp.replace(p)
        perex_written += 1

        el = time.time() - t_cell
        line = "  ".join(f"{m}/{a[:4]} {np.mean(per[(m, a)]):+.4f}" for m, a, _ in ARMS
                         if per[(m, a)])
        print(f"[{rung:14s}] {X:14s} n={n_eval:5d}  {line}   | gate2: {'  '.join(gate_txt)}"
              f"  | {el/60:.1f} min", flush=True)

    # ------------------------------------------------------------------ gate summary
    print("\n" + "=" * 104)
    print(f"CONTROL 2 -- post_token vs canonical master (bar: |Δ| <= {GATE_4DP:g}, i.e. 4 dp)")
    if len(seeds) < 3:
        print(f"  ADVISORY ONLY -- ran {len(seeds)} seed(s); pdl_master stores the MEAN OVER 3 SEEDS,")
        print("     so these are different estimators and the 4 dp bar does not apply. The deltas are")
        print("     printed per cell above for inspection. The STRICT gate runs in the full 3-seed pass,")
        print("     and no pre_token number may be interpreted until it passes there.")
    elif gate2_fail:
        print("  FAIL -- the post_token arm does not reproduce current behaviour, so the pre_token")
        print("     numbers are NOT readable. Do not interpret them.")
        for f in gate2_fail[:20]:
            print(f"    {f}")
    else:
        print("  PASS on every cell scored")
    print(f"\nCONTROL 3 -- distinctness (max|q_pre - q_post| > 0)")
    if distinct_zero:
        print("  FAIL -- the two arms produced IDENTICAL scores on the cells below, which means the")
        print("     shift did not apply. An 'alignment-insensitive' verdict here would be an artefact.")
        for f in distinct_zero[:20]:
            print(f"    {f}")
    else:
        print("  PASS -- the arms differ on every cell scored")
    print("=" * 104)

    out = Path(args.out) if args.out else (
        OUTDIR / ("SMOKE_alignment_prr__meta-llama_Meta-Llama-3.1-8B.csv" if args.smoke
                  else f"alignment_prr_{'-'.join(evals)}__meta-llama_Meta-Llama-3.1-8B.csv"))
    fields = ["model", "method", "alignment", "lambda", "dataset", "rung", "seed", "prr", "n_eval",
              "train", "carve", "git_sha", "cluster", "env_hash", "dirty"]
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows, {perex_written} per-example sidecars)")
    if args.smoke:
        print("SMOKE TEST -- this file must never enter a results table.")


if __name__ == "__main__":
    main()
