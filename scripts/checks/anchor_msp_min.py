#!/usr/bin/env python
"""F5 -- REGULARISE WEIGHTED MSP TOWARD msp_min INSTEAD OF TOWARD perplexity.

Pre-registration: prereg/F5_anchor_at_msp_min.md.  Results: ../STOCKTAKE_sharpening_axis.md §16.

WHY, IN ONE PARAGRAPH
---------------------
`weighting.shrink_to_uniform(w) = ((w-1)**2).mean()` pulls the learned weights to w = 1, and
`weighted_msp.py:522` states what w = 1 IS: `q = mean(nll) == msp 'perplexity'`. So weighted MSP is
anchored on `perplexity` (+0.117) when `msp_min` (+0.186) is the stronger floor and the project's
pre-registered bar. W5 (§15.2) then measured the consequence directly: the optimal lambda per dataset
tracks how good `perplexity` is there, Spearman +0.835, p = 0.0099, stable under leave-one-out. On
pubmed_qa and xsum the anchor is -0.174 and -0.172, i.e. shrinking imports NOISE, and those are
exactly the two datasets where shrinkage helps least. F5 changes the anchor, not the pooling.

⚠️ THE PENALTY IS IN PROBABILITY SPACE, AND THE OBVIOUS FORM IS REJECTED
The naive analogue, squared error toward `w* = n * onehot(argmax nll)`, evaluates to `n - 1` at
uniform weights, so it scales as O(n): one lambda would mean ~7x different things at pubmed_qa
(median 32 tokens) and expertqa (214). That is the LENGTH CONFOUND that killed the tau family in
round 1 (`max z <= sqrt(n-1)`), and it is not being reintroduced. Registered form:

    p       = softmax(raw) over the KEPT tokens        (sums to 1 -> length-independent)
    k       = argmax(nll) among the KEPT tokens        (the anchor)
    penalty = 1 - p[k]                                 (bounded [0,1], zero at the anchor)

⚠️ THE ANCHOR IS AMONG KEPT TOKENS. `_weights_from_raw` masks specials to -inf, so an anchor on an
excluded token could never be reached and the penalty could never reach 0. Consequence, stated
plainly: the lambda -> inf limit is `msp_min` restricted to CONTENT tokens (§12: 0.3478 vs 0.3710 on
pubmed_qa), not `msp_min` over all tokens. Both are reported.

⚠️ WHY THE TRAINING LOOP IS COPIED. `reg(w)` in `weighted_msp.train_weighted_msp` receives only the
weight vector, so it cannot know which token is the anchor, and `weighted_msp.py` is on the Qwen
port's edit list. The loop below is copied VERBATIM from `train_weighted_msp` with only the penalty
call changed, and CONTROL A proves the copy faithful.

FOUR CONTROLS, ALL ABORTING ON FAILURE
  A. NO-OP FIDELITY  -- this loop with `shrink_to_uniform` must reproduce `train_weighted_msp`
                        exactly at the same seed. A silently diverged copy would make every number
                        here incomparable with the master ladder.
  B. ANCHOR ENDPOINT -- forcing w = n_kept * onehot(k) must give exactly the content-token msp_min
                        PRR. This is what makes the lambda -> inf limit msp_min rather than perplexity.
  C. RANDOM ANCHOR   -- the same penalty anchored on a RANDOM kept token. If anchoring anywhere works
                        as well as anchoring at argmax(nll), the mechanism is not what is claimed.
  D. PENALTY BITES   -- mean p[k] must rise with lambda. The bounded form has gradient ~ p[k], which
                        is ~1/n at init (0.005 on expertqa), so it may barely act. If p[k] does not
                        move, that is a FAILURE OF THE OPTIMISATION, not of the idea, and is reported
                        as such (fallback `-log p[k]` is named in the prereg so it is not post-hoc).

    qsub -v LUQ_EVAL=pubmed_qa pbs/anchor_msp_min.pbs
"""
import argparse
import csv as _csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import msp, results, weighted_msp                           # noqa: E402
from luq.weighted_msp import (TokenWeightMLP, answer_states, content_keep,  # noqa: E402
                              per_token_nll, _seq_q, _true_rank, _soft_rank)
from luq.weighting import shrink_to_uniform                          # noqa: E402
from aggregation_table import load_per_token                         # noqa: E402
from xl_rungs import eval_split, label_of, build_rows                # noqa: E402
import probedriftlong as pdl                                         # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15
LAMBDAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]     # bounded penalty -> different scale from {2,10}
OUT = ROOT / "results"


def anchor_index(nll_np, keep_np):
    """argmax(nll) among KEPT tokens -- the anchor the penalty pulls toward."""
    kb = keep_np.astype(bool)
    if not kb.any():
        return int(np.argmax(nll_np))
    return int(np.flatnonzero(kb)[np.argmax(nll_np[kb])])


def train(states, records, y, tr_idx, device, *, lam, mode, seed, rng=None, penalty_form="linear",
          pretrain_epochs=0):
    """Copied VERBATIM from weighted_msp.train_weighted_msp; only the penalty differs.

    mode: 'uniform'  -> shrink_to_uniform(w)          [CONTROL A: must reproduce the library]
          'anchor'   -> penalty on p[argmax nll among kept]  [the F5 penalty]
          'random'   -> same penalty on a random kept token  [CONTROL C]

    penalty_form (F5b, the fallback NAMED IN THE PREREG before any run):
          'linear'   -> 1 - p[k]        bounded, but gradient ~ p[k] ~ 1/n at init -- Control D
                        showed it never bites (mean p[k] flat across lambda 0..20 on the first evals)
          'log'      -> -log(p[k])      gradient ~ 1/p[k]: LARGE exactly when the anchor mass is
                        small, so it cannot stall at initialisation. Unbounded, so the lambda scale
                        differs; clamp p at 1e-6 for numerical safety.
    """
    torch.manual_seed(seed)
    d = answer_states(states[tr_idx[0]]).shape[1]
    model = TokenWeightMLP(d).to(device)
    emb = [torch.from_numpy(answer_states(states[i])).to(device) for i in tr_idx]
    nll = [torch.from_numpy(per_token_nll(records[i])).to(device) for i in tr_idx]
    kep = [torch.from_numpy(content_keep(records[i])).to(device) for i in tr_idx]
    # the anchor index per training example, fixed before training so it cannot drift
    anc = []
    for i in tr_idx:
        kn = content_keep(records[i])
        if mode == "random":
            kb = np.flatnonzero(kn.astype(bool))
            anc.append(int(rng.choice(kb)) if len(kb) else 0)
        else:
            anc.append(anchor_index(per_token_nll(records[i]), kn))
    incorrect = torch.tensor([1.0 - float(y[i]) for i in tr_idx], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_seq = len(tr_idx)
    g = torch.Generator().manual_seed(seed)
    model.train()
    # F5c WARM-START (prereg F5c §2a): push p[k] up on the PENALTY ALONE before the rank loss enters,
    # so the anchor is actually reachable (F5b: the penalty never bit on 5 of 8 evals without this).
    # Control A calls this with pretrain_epochs=0, so its library-exactness is untouched.
    if pretrain_epochs and mode in ("anchor", "random", "combo") and lam > 0:
        for _ in range(pretrain_epochs):
            perm = torch.randperm(n_seq, generator=g).tolist()
            for b in range(0, n_seq, 32):
                batch = perm[b:b + 32]
                if len(batch) < 2:
                    continue
                ps = []
                for j in batch:
                    _, wj = _seq_q(model(emb[j]), nll[j], "normalised", True,
                                   keep=kep[j], return_w=True)
                    nk = torch.clamp(kep[j].sum(), min=1.0)
                    pkv = wj[anc[j]] / nk
                    ps.append(-torch.log(torch.clamp(pkv, min=1e-6)) if penalty_form == "log"
                              else 1.0 - pkv)
                opt.zero_grad()
                (lam * torch.stack(ps).mean()).backward()
                opt.step()
    for _ in range(5):
        perm = torch.randperm(n_seq, generator=g).tolist()
        for b in range(0, n_seq, 32):
            batch = perm[b:b + 32]
            if len(batch) < 2:
                continue
            use_reg = lam > 0
            if use_reg:
                qs, ps = [], []
                for j in batch:
                    qj, wj = _seq_q(model(emb[j]), nll[j], "normalised", True,
                                    keep=kep[j], return_w=True)
                    qs.append(qj)
                    if mode == "uniform":
                        ps.append(shrink_to_uniform(wj))
                    elif mode == "combo":
                        # prereg F5c §2b: BOTH pressures, uniform coefficient FIXED at the incumbent's
                        # 2.0 (never tuned here); lam sweeps only the anchor term.
                        nk = torch.clamp(kep[j].sum(), min=1.0)
                        pkv = wj[anc[j]] / nk
                        at = (-torch.log(torch.clamp(pkv, min=1e-6)) if penalty_form == "log"
                              else 1.0 - pkv)
                        ps.append((2.0 / max(lam, 1e-9)) * shrink_to_uniform(wj) + at)
                    else:
                        # w = softmax(masked raw) * n_kept, so p = w / n_kept  (probability space)
                        nk = torch.clamp(kep[j].sum(), min=1.0)
                        pk = wj[anc[j]] / nk
                        if penalty_form == "log":
                            ps.append(-torch.log(torch.clamp(pk, min=1e-6)))
                        else:
                            ps.append(1.0 - pk)
                q = torch.stack(qs)
                penalty = torch.stack(ps).mean()
            else:
                q = torch.stack([_seq_q(model(emb[j]), nll[j], "normalised", True, keep=kep[j])
                                 for j in batch])
                penalty = None
            loss_val = ((_soft_rank(q) - _true_rank(incorrect[batch])) ** 2).mean()
            if penalty is not None:
                loss_val = loss_val + lam * penalty
            opt.zero_grad()
            loss_val.backward()
            opt.step()
    return model


def predict(model, states, records, te_idx, device):
    """Scoring is the library's own path -- unchanged, so scores stay comparable."""
    return weighted_msp.predict_weighted_msp(model, states, records, te_idx, device,
                                             weight_mode="normalised", length_normalise=True)


def mean_p_anchor(model, states, records, idx, device):
    """CONTROL D: mean p[k]. If this does not rise with lambda the penalty is not biting."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in idx:
            kn = content_keep(records[i])
            k = anchor_index(per_token_nll(records[i]), kn)
            nl = torch.from_numpy(per_token_nll(records[i])).to(device)
            raw = model(torch.from_numpy(answer_states(states[i])).to(device))
            _, w = _seq_q(raw, nl, "normalised", True,
                          keep=torch.from_numpy(kn).to(device), return_w=True)
            out.append(float(w[k].item()) / max(float(kn.sum()), 1.0))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="pubmed_qa")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pretrain-epochs", type=int, default=0,
                    help="F5c warm-start: N epochs on the penalty alone before the combined loss")
    ap.add_argument("--combo", action="store_true",
                    help="F5c: add the shrink@2 + anchor combo arms (prereg F5c §2b)")
    ap.add_argument("--penalty", choices=["linear", "log"], default="linear",
                    help="F5b fallback: 'log' = -log p[k], gradients that cannot stall (prereg §3 C3)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    evals = [e for e in args.evals.split(",") if e]
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.smoke:
        seeds = seeds[:1]
    global LAMBDAS
    if args.penalty == "log":
        # -log p[k] starts at ~log(n) (3-5), an order larger than the bounded form, so the grid shifts down
        LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    carve = os.environ.get("LUQ_CARVE", "legacy")
    print("=" * 100)
    print(f"F5 -- ANCHOR THE PENALTY AT msp_min, NOT perplexity  [penalty={args.penalty}]  "
          f"[{'SMOKE, NOT A RESULT' if args.smoke else 'full grid'}]")
    print(f"device={device} seeds={seeds} evals={evals} LUQ_CARVE={carve}  lambdas={LAMBDAS}")
    print("=" * 100, flush=True)

    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok -> SKIPPED LOUDLY")
            continue
        st, sp, y, _, rc = loaded
        f = np.isfinite(y)
        if not f.any():
            continue
        if not f.all():
            kk = np.where(f)[0]
            st = [st[k] for k in kk]; rc = [rc[k] for k in kk]; sp = sp[kk]; y = y[kk]
        PT[d] = (st, sp, y, rc)
        print(f"  {d}: {len(st)} rows", flush=True)
    sources = set(PT)

    rows = []
    for rung, X, spec in pdl.cells_long(sources, evals):
        if X not in PT:
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        acc = {("anchor", l): [] for l in LAMBDAS}
        acc.update({("random", l): [] for l in LAMBDAS if l > 0})
        if args.combo:
            acc.update({("combo", l): [] for l in LAMBDAS if l > 0})
        pk = {l: [] for l in LAMBDAS}
        fl = {"msp_min": [], "msp_min_kept": [], "perplexity": []}
        for sd in seeds:
            tr_rows, te_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not tr_rows or not te_rows:
                continue
            n_tr = len(tr_rows)
            tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(te_rows)))
            allr = tr_rows + te_rows
            y = np.array([PT[d][2][i] for d, i in allr], float)
            yte = np.array([y[i] for i in te_idx], float)
            states = [PT[d][0][i] for d, i in allr]
            records = [PT[d][3][i] for d, i in allr]

            fl["msp_min"].append(results.prr(yte, np.array(
                [msp.msp_uncertainty(records[i]["token_logprobs"], "min") for i in te_idx])))
            fl["perplexity"].append(results.prr(yte, np.array(
                [msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity") for i in te_idx])))
            vmk = []
            for i in te_idx:
                nl = per_token_nll(records[i]); kb = content_keep(records[i]).astype(bool)
                vmk.append(float(nl[kb].max()) if kb.any() else float(nl.max()))
            fl["msp_min_kept"].append(results.prr(yte, np.array(vmk)))

            # ---- CONTROL A: no-op fidelity of the copied loop ----
            m_mine = train(states, records, y, tr_idx, device, lam=2.0, mode="uniform", seed=sd)
            v_mine = predict(m_mine, states, records, te_idx, device)
            v_lib = weighted_msp.weighted_msp_unc(states, records, y, tr_idx, te_idx, device,
                                                  weight_mode="normalised", length_normalise=True,
                                                  seed=sd, reg=shrink_to_uniform, reg_lambda=2.0)
            dmax = float(np.max(np.abs(np.asarray(v_mine) - np.asarray(v_lib))))
            if dmax > 1e-6:
                raise SystemExit(f"CONTROL A FAILED [{rung}/{X}/seed{sd}]: copied loop differs from "
                                 f"train_weighted_msp by {dmax:.3e}. Numbers would be incomparable.")

            # ---- CONTROL B: the anchor endpoint IS the content-token floor ----
            v_end = []
            for i in te_idx:
                nl = per_token_nll(records[i]); kn = content_keep(records[i])
                v_end.append(float(nl[anchor_index(nl, kn)]))
            d_end = abs(results.prr(yte, np.array(v_end)) - fl["msp_min_kept"][-1])
            if d_end > 1e-9:
                raise SystemExit(f"CONTROL B FAILED [{rung}/{X}/seed{sd}]: anchor endpoint "
                                 f"{results.prr(yte, np.array(v_end)):+.6f} != content-token msp_min "
                                 f"{fl['msp_min_kept'][-1]:+.6f}. F5's rationale is false.")

            rng = np.random.RandomState(1000 + sd)
            for l in LAMBDAS:
                m = train(states, records, y, tr_idx, device, lam=l, mode="anchor", seed=sd,
                          penalty_form=args.penalty, pretrain_epochs=args.pretrain_epochs)
                acc[("anchor", l)].append(results.prr(yte, np.asarray(predict(m, states, records,
                                                                             te_idx, device), float)))
                pk[l].append(mean_p_anchor(m, states, records, te_idx, device))
                if l > 0:
                    mr = train(states, records, y, tr_idx, device, lam=l, mode="random", seed=sd,
                               rng=rng, penalty_form=args.penalty,
                               pretrain_epochs=args.pretrain_epochs)
                    acc[("random", l)].append(results.prr(
                        yte, np.asarray(predict(mr, states, records, te_idx, device), float)))
                    if args.combo:
                        mc = train(states, records, y, tr_idx, device, lam=l, mode="combo", seed=sd,
                                   penalty_form=args.penalty,
                                   pretrain_epochs=args.pretrain_epochs)
                        acc[("combo", l)].append(results.prr(
                            yte, np.asarray(predict(mc, states, records, te_idx, device), float)))
        if not acc[("anchor", 0.0)]:
            continue
        f = {k: float(np.mean(v)) for k, v in fl.items()}
        print(f"\n[{rung:16s} {X:14s}]  msp_min {f['msp_min']:+.4f} (content-tok "
              f"{f['msp_min_kept']:+.4f})  ppl {f['perplexity']:+.4f}")
        print("   CONTROL A (copied loop == library) PASS   CONTROL B (anchor endpoint) PASS")
        print(f"   {'lambda':>8s}{'anchor':>10s}{'random':>10s}{'combo':>10s}{'anc-rand':>10s}{'mean p[k]':>11s}")
        for l in LAMBDAS:
            a = float(np.mean(acc[("anchor", l)]))
            r = float(np.mean(acc[("random", l)])) if l > 0 and acc[("random", l)] else float("nan")
            c = (float(np.mean(acc[("combo", l)])) if args.combo and l > 0 and acc.get(("combo", l))
                 else float("nan"))
            print(f"   {l:>8g}{a:>10.4f}{r:>10.4f}{c:>10.4f}{a - r:>10.4f}{float(np.mean(pk[l])):>11.4f}")
            rows.append((rung, X, "anchor", l, f"{a:.4f}", f"{np.mean(pk[l]):.4f}", len(seeds), carve))
            if l > 0:
                rows.append((rung, X, "random", l, f"{r:.4f}", "", len(seeds), carve))
                if args.combo and acc.get(("combo", l)):
                    rows.append((rung, X, "combo", l, f"{c:.4f}", "", len(seeds), carve))
        for k, v in f.items():
            rows.append((rung, X, "floor", k, f"{v:.4f}", "", len(seeds), carve))
        print("   ⚠️ CONTROL D: mean p[k] must RISE with lambda. If it does not, the bounded penalty")
        print("      is not biting and that is an optimisation failure, not a verdict on the idea.")
        if args.smoke:
            print("\n⚠️ SMOKE -- one cell, one seed. NOT A RESULT.")
            break

    ev = evals[0] if len(evals) == 1 else "multi"
    suffix = "" if args.penalty == "linear" else ("__logws" if args.pretrain_epochs else "__logpen")
    outp = Path(args.out) if args.out else OUT / f"anchor_msp_min_{ev}{suffix}__meta-llama_Meta-Llama-3.1-8B.csv"
    if args.smoke:
        outp = outp.with_name(outp.stem + "__SMOKE" + outp.suffix)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["rung", "eval", "mode", "lambda", "prr", "mean_p_anchor", "n_seeds", "carve"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {outp}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
