#!/usr/bin/env python
"""PR2 -- SOURCE-RELATIVE RANK SUPERVISION FOR WEIGHTED MSP.

Pre-registration: ../prereg/PR2_source_relative_wmsp.md (committed before any source-relative PRR).
Output:           results/method_dev/source_relative/

THE QUESTION
------------
Canonical weighted MSP shuffles the whole MIXED training pool and ranks every pair inside a batch of
32 (weighted_msp.py:418 randperm, :121-123 full pairwise matrix masked only on the diagonal). So a
6-source LOO rung trains on cross-dataset comparisons -- roughly 5.3 examples per source per batch --
while PRR is scored WITHIN one dataset. Does restricting the rank supervision to same-source
comparisons transfer better?

TWO ARMS (prereg §2)
--------------------
  masked  PRIMARY, clean isolation. Batch membership, permutation, step count, seed, optimiser and
          lambda are byte-identical to canonical. ONLY the rank term changes: computed within each
          source subgroup, subgroups of <2 skipped, averaged over valid subgroups. The shrink
          penalty stays averaged over the FULL batch, exactly as canonical does it.
          Acknowledged limitation: ~5.3 items per subgroup makes the rank targets noisy. That is the
          price of changing exactly one thing.
  pure    SECONDARY, better powered. Source-pure batches of 32, round-robin across sources so no
          source dominates by size. Confounds batching with the comparison set -- which is why
          `masked` is primary and why the interpretation table in prereg §4 is fixed in advance.

⚠️ THE CONTROL THE BRIEF ASKED FOR IS LOGICALLY IMPOSSIBLE, AND IS NOT CREATED.
`wmsp_sourcepure_global` -- source-pure batches with a global ranking loss -- cannot exist: inside a
source-pure batch every comparison is ALREADY within-source, so "source-pure batching" and
"within-source ranking" are the same intervention. Building it would produce a third name for the
`pure` arm and invite a comparison that is a tautology.

⚠️ TWO RUNGS CANNOT DIFFER, AND THAT IS USED AS A CONTROL.
`ID` and `1ds-Diff-long` are single-source pools, so within-source ranking IS global ranking there.
Both arms are constructed so that a single-source pool takes exactly the canonical RNG draws, and
those cells are run at seed 1 as an EXACT-INVARIANCE CONTROL: they must reproduce canonical to
<1e-6. Same control shape as the MedQuAD span sensitivity.

ZERO SHARED-FILE EDITS
----------------------
The training loop below is a deliberate COPY of weighted_msp.train_weighted_msp (the same precedent
as scripts/checks/anchor_msp_min.py:86-188), so `src/luq/weighted_msp.py` is not touched -- it is on
both the Qwen port's and the sharpening line's edit lists. The copy is kept honest by a NO-OP
FIDELITY GATE: with mode="canonical" it must reproduce the library's own `weighted_msp_unc` to
<1e-6 on the same cell and seed, or the run aborts.

    python scripts/checks/source_relative_wmsp.py --evals asqa --seeds 1 --smoke
"""
import argparse
import csv as _csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch                                                          # noqa: E402
from luq import msp, results, weighted_msp                            # noqa: E402
from luq.weighted_msp import (TokenWeightMLP, answer_states, per_token_nll, content_keep,  # noqa: E402
                              _seq_q, _soft_rank, _true_rank, predict_weighted_msp)
from luq.weighting import shrink_to_uniform                           # noqa: E402
from aggregation_table import load_per_token                          # noqa: E402
from xl_rungs import build_rows, eval_split, label_of, different_label_projection  # noqa: E402
import probedriftlong as pdl                                          # noqa: E402

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAMBDA = 2.0             # prereg §2: the registry's main regularised configuration. NOT tuned.
GATE_TOL = 1e-6
# Rungs whose training pool is single-source, so the method is mathematically identical to canonical.
INERT_RUNGS = {"ID", "1ds-Diff-long"}


def train_srcrel(states, records, y, tr_idx, sources, device, *, mode="canonical", seed=1,
                 n_epochs=5, batch_size=32, lr=1e-3, reg_lambda=LAMBDA, diag=None):
    """Copy of weighted_msp.train_weighted_msp with source-relative rank supervision.

    `sources` is a per-training-example dataset name, index-aligned to `tr_idx`. `mode`:
      "canonical" -- reproduce the library exactly (the no-op fidelity gate)
      "masked"    -- same batches, rank loss computed within each source subgroup
      "pure"      -- source-pure batches, round-robin across sources

    `diag` (optional dict) collects the prereg §5 mechanism diagnostics.
    """
    torch.manual_seed(seed)
    d = answer_states(states[tr_idx[0]]).shape[1]
    model = TokenWeightMLP(d).to(device)

    emb = [torch.from_numpy(answer_states(states[i])).to(device) for i in tr_idx]
    nll = [torch.from_numpy(per_token_nll(records[i])).to(device) for i in tr_idx]
    kep = [torch.from_numpy(content_keep(records[i])).to(device) for i in tr_idx]
    incorrect = torch.tensor([1.0 - float(y[i]) for i in tr_idx], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    n_seq = len(tr_idx)
    g = torch.Generator().manual_seed(seed)
    # positions 0..n_seq-1 grouped by source, preserving order (matters for the RNG equivalence below)
    groups = defaultdict(list)
    for pos, s in enumerate(sources):
        groups[s].append(pos)
    src_of = list(sources)

    model.train()
    for _ep in range(n_epochs):
        if mode == "pure":
            # ⚠️ RNG EQUIVALENCE. With ONE source this draws torch.randperm(n_seq, generator=g) and
            # slices it in 32s -- byte-identical to the canonical path. That is what makes the
            # single-source cells an exact-invariance control rather than an approximate one.
            chunked = []
            for s in sorted(groups):
                pos = groups[s]
                order = [pos[k] for k in torch.randperm(len(pos), generator=g).tolist()]
                chunked.append([order[i:i + batch_size] for i in range(0, len(order), batch_size)])
            batches = []
            for i in range(max(len(c) for c in chunked)):
                for c in chunked:                      # round-robin: equal turns per source
                    if i < len(c):
                        batches.append(c[i])
        else:
            perm = torch.randperm(n_seq, generator=g).tolist()
            batches = [perm[b:b + batch_size] for b in range(0, n_seq, batch_size)]

        for batch in batches:
            if len(batch) < 2:
                continue                       # a rank loss needs >=2 items (canonical skips these)
            qs, ws = [], []
            for j in batch:
                qj, wj = _seq_q(model(emb[j]), nll[j], "normalised", True, keep=kep[j], return_w=True)
                qs.append(qj)
                ws.append(wj)
            q = torch.stack(qs)
            penalty = torch.stack([shrink_to_uniform(w) for w in ws]).mean()   # FULL batch, as canonical

            if mode in ("masked", "masked_scaled"):
                # Rank only within each source subgroup, then average over the valid subgroups.
                #
                # ⚠️ WHY `masked_scaled` EXISTS (added 2026-08-13, prereg §7 amendment).
                # The rank loss is an MSE between soft and hard ranks, so its SCALE grows with the
                # number of items ranked: ranks run 1..m, and the MSE is O(m^2). Ranking inside
                # subgroups of ~5 instead of a batch of 32 therefore shrinks the rank term by
                # roughly an order of magnitude -- while lambda stays fixed at 2. The shrink penalty
                # then dominates and drives the weights UNIFORM, at which point weighted MSP IS
                # perplexity. That is not a null result about source-relative supervision; it is the
                # method quietly becoming a different method.
                # Measured on pubmed_qa, and it is unambiguous -- Omega(w) = mean((w-1)^2) falls
                # monotonically as the subgroups get smaller:
                #     2 sources -> Omega 0.97 | 5 sources -> 0.11 | 7 sources -> 0.05
                # against ~3.3-4.5 for the full-batch arms, with PRR landing on pubmed's perplexity
                # (-0.174) rather than anywhere near canonical (+0.19).
                # `masked_scaled` rescales each subgroup's ranks to span the range the FULL batch
                # would have spanned, so the rank term keeps its canonical magnitude and lambda's
                # effective strength is unchanged -- which is what the pre-registration promised.
                # This is a SCALE correction, not a hyperparameter tune: lambda, lr, batch size,
                # epochs, seeds, batch membership and permutation are all untouched.
                sub_losses = []
                by_src = defaultdict(list)
                for k, j in enumerate(batch):
                    by_src[src_of[j]].append(k)        # k indexes into q / the batch, not tr_idx
                for s in sorted(by_src):
                    ks = by_src[s]
                    if len(ks) < 2:
                        continue                       # prereg §2: subgroups of <2 are skipped
                    kt = torch.tensor(ks, dtype=torch.long, device=device)
                    tgt = _true_rank(incorrect[[batch[k] for k in ks]])
                    sq = (_soft_rank(q[kt]) - tgt) ** 2
                    if mode == "masked_scaled":
                        # Match the EXPECTED rank-MSE of a batch of size B: Var(uniform 1..m) is
                        # (m^2-1)/12, so scaling by (B^2-1)/(m^2-1) makes a subgroup of size m carry
                        # the same magnitude as the full batch, leaving lambda's effective strength
                        # unchanged. (The cruder ((B-1)/(m-1))^2 overshoots by 1.3-1.5x here.)
                        B, m_ = float(len(batch)), float(len(ks))
                        sq = sq * ((B * B - 1.0) / max(m_ * m_ - 1.0, 1.0))
                    sub_losses.append(sq.mean())
                if not sub_losses:
                    continue                           # no valid subgroup -> no rank signal this step
                rank_loss = torch.stack(sub_losses).mean()
            else:
                # canonical and pure: rank over the whole batch. For `pure` the batch IS one source,
                # so this is already a within-source ranking -- see the module docstring.
                rank_loss = ((_soft_rank(q) - _true_rank(incorrect[batch])) ** 2).mean()

            loss_val = rank_loss + reg_lambda * penalty
            opt.zero_grad()
            loss_val.backward()
            opt.step()

            if diag is not None:
                diag.setdefault("rank_loss", []).append(float(rank_loss.detach()))
                diag.setdefault("omega", []).append(float(penalty.detach()))
                diag["n_steps"] = diag.get("n_steps", 0) + 1
    return model


def srcrel_unc(states, records, y, tr_idx, te_idx, sources, device, mode, seed, diag=None):
    """Train one weighter and score the test rows. Scoring is source-independent, so the library's
    own `predict_weighted_msp` is reused unchanged (the test set is single-source anyway)."""
    m = train_srcrel(states, records, y, tr_idx, sources, device, mode=mode, seed=seed, diag=diag)
    return np.asarray(predict_weighted_msp(m, states, records, te_idx, device,
                                           weight_mode="normalised", length_normalise=True), float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--evals", default="")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--rungs", default="", help="base rung names; default = the multi-source scope")
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default="masked,pure",
                    help="comma-separated source-relative modes to run: masked | masked_scaled | pure. "
                         "The canonical comparator and the no-op fidelity gate ALWAYS run, so every "
                         "output file is self-contained and can be cross-checked against any other.")
    args = ap.parse_args()

    MODEL = args.model
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    want_rungs = set(s.strip() for s in args.rungs.split(",") if s.strip()) or None

    prov = pdl._provenance()
    print(f"PROVENANCE: git_sha={prov['git_sha'][:12]} cluster={prov['cluster']} "
          f"env_hash={prov['env_hash']} carve={prov['carve']}", flush=True)
    if args.smoke:
        print("⚠️  SMOKE TEST -- results are NOT reportable", flush=True)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    _valid = {"masked", "masked_scaled", "pure"}
    if set(arms) - _valid:
        raise SystemExit(f"--arms: unknown {sorted(set(arms) - _valid)}; valid are {sorted(_valid)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={MODEL} layer={args.layer} lambda={LAMBDA} arms={arms}", flush=True)
    evals = [e.strip() for e in args.evals.split(",") if e.strip()] or list(pdl.LONG_SRC)

    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: NO pertok cache -> cells touching {d} will be ABSENT", flush=True)
            continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            continue
        if not finite.all():
            k = np.where(finite)[0]
            states = [states[i] for i in k]; records = [records[i] for i in k]
            split = split[k]; y = y[k]
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} labelled rows", flush=True)

    out_rows, n_gate = [], 0
    for rung, X, spec in pdl.cells_long(set(PT), evals):
        if want_rungs is not None and rung.replace("-long", "") not in want_rungs:
            continue
        if X not in PT:
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue

        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            if args.smoke:
                # ⚠️ STRIDE, DO NOT TRUNCATE. `build_rows` returns SOURCE-BLOCKED rows (splits.py:180
                # concatenates one contiguous block per source), so train_rows[:96] would take every
                # row from the FIRST source and hand a silently SINGLE-SOURCE pool to a method whose
                # whole point is multi-source behaviour -- the smoke would pass while testing nothing.
                train_rows = train_rows[::max(1, len(train_rows) // 96)][:96]
                test_rows = test_rows[:60]
            n_tr = len(train_rows)
            tr_idx = list(range(n_tr))
            te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows

            # THE SOURCE LABEL. It was always here -- `build_rows` returns (dataset, idx) pairs and
            # the canonical driver simply discards the dataset before calling the method.
            sources = [d for d, _ in train_rows]
            n_src = len(set(sources))
            inert = (n_src == 1)

            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float)
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]

            v, diags = {}, {}
            # The canonical comparator, from the LIBRARY (not our copy) -- so the comparison is
            # against the real registry method, not against our reimplementation of it.
            v["wmsp_shrink2"] = np.asarray(weighted_msp.weighted_msp_unc(
                states, records, y, tr_idx, te_idx, device, length_normalise=True, seed=sd,
                weight_mode="normalised", reg=shrink_to_uniform, reg_lambda=LAMBDA), float)

            # NO-OP FIDELITY GATE: our copied loop in canonical mode must reproduce the library.
            v_noop = srcrel_unc(states, records, y, tr_idx, te_idx, sources, device, "canonical", sd)
            gap = float(np.abs(v_noop - v["wmsp_shrink2"]).max())
            if gap > GATE_TOL:
                raise SystemExit(
                    f"FATAL NO-OP GATE FAILURE at [{rung}/{X}/seed{sd}]: our copied training loop in "
                    f"canonical mode differs from luq.weighted_msp by {gap:.3e} > {GATE_TOL:g} on the "
                    f"per-example uncertainties. The copy has drifted from the library, so any "
                    f"source-relative delta measured against it would be confounded with that drift.")
            n_gate += 1

            for mode in arms:
                dg = {}
                v[f"wmsp_srcrel_{mode}_shrink2"] = srcrel_unc(
                    states, records, y, tr_idx, te_idx, sources, device, mode, sd, diag=dg)
                diags[f"wmsp_srcrel_{mode}_shrink2"] = dg
                if inert:
                    # prereg §1: single-source pools MUST be identical to canonical.
                    g2 = float(np.abs(v[f"wmsp_srcrel_{mode}_shrink2"] - v["wmsp_shrink2"]).max())
                    if g2 > GATE_TOL:
                        raise SystemExit(
                            f"FATAL INVARIANCE FAILURE at [{rung}/{X}/seed{sd}] arm={mode}: the "
                            f"training pool has ONE source, where within-source ranking is "
                            f"mathematically identical to global ranking, yet the arms differ by "
                            f"{g2:.3e}. The source-relative path is not reducing correctly.")

            v["floor_min"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "min")
                                       for i in te_idx])

            for m_name, vec in v.items():
                dg = diags.get(m_name, {})
                out_rows.append({
                    "model": MODEL, "layer": args.layer, "eval": X, "rung": rung, "seed": sd,
                    "method": m_name, "prr": results.prr(yte, vec),
                    "n_train": n_tr, "n_test": len(te_idx), "n_sources": n_src, "inert": inert,
                    "final_rank_loss": (float(np.mean(dg["rank_loss"][-10:])) if dg.get("rank_loss") else ""),
                    "final_omega": (float(np.mean(dg["omega"][-10:])) if dg.get("omega") else ""),
                    "n_steps": dg.get("n_steps", ""),
                    "different_label_projection": different_label_projection(X),
                    "smoke": bool(args.smoke),
                    "git_sha": prov["git_sha"], "cluster": prov["cluster"],
                    "env_hash": prov["env_hash"], "carve": prov["carve"],
                })
            tag = " [INERT: must equal canonical]" if inert else f" [{n_src} sources]"
            print(f"  [{rung}/{X}/s{sd}] noop {gap:.1e}{tag}  "
                  f"canon={results.prr(yte, v['wmsp_shrink2']):+.4f}  "
                  + "  ".join(f"{m}={results.prr(yte, v[f'wmsp_srcrel_{m}_shrink2']):+.4f}"
                              for m in arms), flush=True)

    if not out_rows:
        raise SystemExit("no cells produced -- refusing to write an empty CSV")

    slug = MODEL.replace("/", "_")
    default = ROOT / "results" / "method_dev" / "source_relative" / (
        f"source_relative_{slug}" + ("__SMOKE" if args.smoke else "") + ".csv")
    out = Path(args.out) if args.out else default
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"FATAL: {out} exists, refusing to clobber")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n[source-relative] {len(out_rows)} rows, {n_gate} cells passed the no-op gate -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
