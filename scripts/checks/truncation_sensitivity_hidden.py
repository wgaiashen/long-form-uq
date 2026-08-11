#!/usr/bin/env python
"""Step 2b, hidden-state half — FROZEN-PROBE EVAL-SIDE truncation sensitivity.

WHAT IT ISOLATES. The supervised methods aggregate over per-token hidden states, so the trailing
next-template continuation is inside their input too. This trains each method ONCE per cell on the
UNCHANGED full states, then scores the SAME trained model on (a) the full test spans and (b) the
test spans truncated at the frozen `luq.template_restart` boundary. Labels stay RAW throughout.
Nothing is trained on truncated data, so the two columns differ only in what the fixed model was
shown at scoring time.

⚠️ CALL IT WHAT IT IS: **FROZEN-PROBE EVAL-SIDE SENSITIVITY**. The training population is still
uncorrected, so the CLEAN column is not a corrected result for a supervised method — it is a
diagnostic of how much the junk was contributing at scoring time. Only after the training cells are
refreshed does a corrected number exist.

⛔ NOT A VALIDITY TEST. The boundary was fixed in `luq.template_restart` from the datasets' own
prompt templates and committed before any of this ran. No number here may be used to move it.

────────────────────────────────────────────────────────────────────────────────────────────────
⚠️ THE ALIGNMENT TRAP, AND HOW IT IS HANDLED.
The per-token cache window is `[last_prompt_token] + gen_tokens`, i.e. **G+1 rows for G generated
tokens** (implementation_notes §6 — the project has already paid for getting this wrong once). So
keeping `n_keep` GENERATED tokens means keeping `states[:n_keep + 1]`, not `states[:n_keep]`. The
off-by-one is asserted, not assumed: `len(states) == len(token_logprobs) + 1` is checked per record
and the run aborts loudly if any record violates it.
────────────────────────────────────────────────────────────────────────────────────────────────

    python scripts/checks/truncation_sensitivity_hidden.py
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache, results, weighted_msp, template_restart as TR      # noqa: E402
from luq.weighting import shrink_to_uniform                               # noqa: E402
from transformers import AutoTokenizer                                    # noqa: E402
from aggregation_table import load_per_token, conf_meanpool, attn_unc     # noqa: E402
from attn_pool import train_attn, select_temperature                      # noqa: E402
from xl_rungs import eval_split, label_of, build_rows                     # noqa: E402
import probedriftlong as pdl                                              # noqa: E402

QWEN = "Qwen/Qwen2.5-14B"
LLAMA = "meta-llama/Meta-Llama-3.1-8B"
HARDEST = ["DiffTask-long", "1ds-Diff-long"]
MIN_KEEP = 2

FIELDS = ["model", "rung", "eval", "method", "prr_raw_span", "prr_trunc_span", "delta",
          "n_seeds", "n_test", "pct_test_cut", "carve", "git_sha", "cluster", "env_hash", "provenance"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN)
    ap.add_argument("--evals", default=",".join(pdl.LONG))
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--skip-poolers", action="store_true",
                    help="skip uniform/attention (select_temperature is the slow step)")
    args = ap.parse_args()

    slug = cache._slug(args.model)
    out = ROOT / "results" / "analysis" / f"truncation_sensitivity_hidden__{slug}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    evals = [e for e in args.evals.split(",") if e]
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prov = pdl._provenance()
    carve = os.environ.get("LUQ_CARVE", "legacy")
    tok = AutoTokenizer.from_pretrained(args.model)

    print("=" * 112)
    print(f"STEP 2b HIDDEN-STATE — FROZEN-PROBE EVAL-SIDE SENSITIVITY   model={args.model}")
    print(f"layer={args.layer} seeds={seeds} rungs={HARDEST} device={device} carve={carve}")
    print("Trained ONCE per cell on FULL states; the same model scores full vs truncated test spans.")
    print("Labels RAW throughout. Boundary frozen in luq.template_restart before this ran.")
    print("=" * 112, flush=True)

    if args.model != LLAMA:
        from luq import token_subsets
        weighted_msp.set_special_ids(tok.all_special_ids)
        token_subsets.set_special_ids(tok.all_special_ids)
        print(f"special-token ids registered ({len(tok.all_special_ids)} ids)", flush=True)

    # ---------------- pools, plus the truncated view of each row ----------------
    PT, KEEP = {}, {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        loaded = load_per_token(args.model, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> cells needing it stay ABSENT, not 0")
            continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            continue
        if not finite.all():
            k = np.where(finite)[0]
            states = [states[i] for i in k]; records = [records[i] for i in k]
            split = split[k]; y = y[k]
        # THE G+1 ASSERTION -- abort rather than silently mis-slice
        for i, (s, r) in enumerate(zip(states, records)):
            if len(s) != len(r["token_logprobs"]) + 1:
                raise SystemExit(f"{d}[{i}]: pertok rows {len(s)} != gen tokens "
                                 f"{len(r['token_logprobs'])} + 1. The window assumption is wrong; "
                                 f"refusing to slice.")
        keep_n = []
        for s, r in zip(states, records):
            t = r.get("gen_text", "") or ""
            clean, ch, _, _ = TR.restart_cut(t, d)
            if ch is None or not clean:
                keep_n.append(len(s))                      # untouched
            else:
                nk = len(tok(clean, add_special_tokens=False).input_ids)
                nk = max(MIN_KEEP, min(nk, len(r["token_logprobs"])))
                keep_n.append(nk + 1)                      # +1 for the prompt-anchor row
        PT[d] = (states, split, y, records)
        KEEP[d] = np.array(keep_n)
        cutfrac = float(np.mean(KEEP[d] < np.array([len(s) for s in states])))
        print(f"  {d}: {len(states)} rows, {100*cutfrac:.1f}% would be cut", flush=True)
    sources = set(PT)

    rows = []
    for rung, X, spec in pdl.cells_long(sources, evals):
        if rung not in HARDEST or X not in PT:
            continue
        if len(eval_split(PT[X][1])[1]) == 0:
            continue
        acc, n_test, pct_cut = {}, 0, 0.0
        for sd in seeds:
            tr_rows, te_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not tr_rows or not te_rows:
                continue
            n_tr = len(tr_rows)
            tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(te_rows)))
            allrows = tr_rows + te_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float)
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]
            # the TRUNCATED view: train rows untouched, test rows sliced
            keeps = [KEEP[d][i] for d, i in allrows]
            st_tr = list(states)
            for j in te_idx:
                st_tr[j] = states[j][:keeps[j]]
            n_test = len(yte)
            pct_cut = 100 * float(np.mean([keeps[j] < len(states[j]) for j in te_idx]))

            def add(m, a, b):
                acc.setdefault((m, "raw"), []).append(a)
                acc.setdefault((m, "trunc"), []).append(b)

            Xm = np.stack([s.mean(axis=0) for s in states])
            Xm_t = np.stack([s.mean(axis=0) for s in st_tr])
            # ONE probe, two scorings: fit on the full-state train rows, then score both views.
            add("saplma",
                results.prr(yte, 1.0 - conf_meanpool(Xm, tr_idx, te_idx, y, sd)),
                results.prr(yte, 1.0 - conf_meanpool(np.vstack([Xm[:n_tr], Xm_t[n_tr:]]),
                                                     tr_idx, te_idx, y, sd)))
            for nm, kw in [("wmsp_norm", {}),
                           ("wmsp_shrink2", {"reg": shrink_to_uniform, "reg_lambda": 2.0})]:
                mdl = weighted_msp.train_weighted_msp(states, records, y, tr_idx, device,
                                                      weight_mode="normalised",
                                                      length_normalise=True, seed=sd, **kw)
                a = weighted_msp.predict_weighted_msp(mdl, states, records, te_idx, device,
                                                      weight_mode="normalised", length_normalise=True)
                b = weighted_msp.predict_weighted_msp(mdl, st_tr, records, te_idx, device,
                                                      weight_mode="normalised", length_normalise=True)
                add(nm, results.prr(yte, np.asarray(a, float)), results.prr(yte, np.asarray(b, float)))
            if not args.skip_poolers:
                select_temperature(states, y, tr_idx, device, sd, False, False)
                unif = train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True)
                add("uniform", results.prr(yte, np.asarray(attn_unc(unif, states, te_idx, device), float)),
                    results.prr(yte, np.asarray(attn_unc(unif, st_tr, te_idx, device), float)))
                at = train_attn(states, y, tr_idx, device, seed=sd)
                add("attention", results.prr(yte, np.asarray(attn_unc(at, states, te_idx, device), float)),
                    results.prr(yte, np.asarray(attn_unc(at, st_tr, te_idx, device), float)))
        if not acc:
            continue
        line = []
        for m in sorted({k[0] for k in acc}):
            a, b = np.mean(acc[(m, "raw")]), np.mean(acc[(m, "trunc")])
            rows.append({"model": args.model, "rung": rung, "eval": X, "method": m,
                         "prr_raw_span": f"{a:.6f}", "prr_trunc_span": f"{b:.6f}",
                         "delta": f"{b-a:.6f}", "n_seeds": len(acc[(m, 'raw')]),
                         "n_test": n_test, "pct_test_cut": round(pct_cut, 2), "carve": carve,
                         "provenance": "frozen-probe-eval-side-sensitivity", **prov})
            line.append(f"{m} {a:+.4f}->{b:+.4f} ({b-a:+.4f})")
        print(f"  [{rung:15s} {X:14s}] cut {pct_cut:4.1f}%  " + "  ".join(line), flush=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")
    print("⚠️ FROZEN-PROBE EVAL-SIDE ONLY. Training populations are uncorrected, so the truncated")
    print("   column is a diagnostic of scoring-time contribution, not a corrected method score.")


if __name__ == "__main__":
    main()
