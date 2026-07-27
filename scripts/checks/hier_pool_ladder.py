"""W4 driver — does HIERARCHICAL (two-level) pooling beat the FLAT attention pooler on long-form?

Runs, on the long-form ladder (ID + the ProbeDriftLong long-only rungs), the same head/optimiser/seeds/
splits for every row so the ONLY thing that varies is the pooling:

    fair_floor   the PRE-REGISTERED msp_min bar (2026-07-24 meeting; all three variants still computed)
    saplma       mean-pool + MLP                     -- the standard supervised probe
    uniform      frozen-q pooler                     -- mean-pool through the SAME torch head
    attention    flat learned pooler                 -- the current best aggregator (the thing to beat)
    hier         two-level: attend within sentence, then across sentences   <- the W4 method
    hier_seg     level 2 only  (q_tok frozen -> uniform within a sentence)  -- "which SENTENCE matters"
    hier_tok     level 1 only  (q_seg frozen -> uniform across sentences)   -- "which TOKEN in a sentence"

The two frozen ablations are the point of the design: if `hier` wins, they say WHICH level did the work,
and if it loses they say whether one level was actively harmful. `uniform` is kept as the head-confound
guard -- a torch-vs-sklearn head difference once masqueraded as a +0.345 aggregation "win", so every row
here goes through the identical torch head.

Segment ids come from `sar._token_sentence_ids`, the SAME helper the weighted-MSP segment variant uses, so
"sentence" means the same thing on both tracks.

CPU-only (cached L15 per-token states).
"""

import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, msp, results, hier_pool  # noqa: E402
from luq.features import sar  # noqa: E402
from aggregation_table import (load_per_token, attn_unc, paired_bootstrap,  # noqa: E402
                               conf_meanpool)
from attn_pool import train_attn, select_temperature  # noqa: E402
from xl_rungs import label_of, eval_split, build_rows  # noqa: E402
import probedriftlong as pdl  # noqa: E402
import xl_rungs  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
METHODS = ["fair_floor", "saplma", "uniform", "attention", "hier", "hier_seg", "hier_tok"]
FLOORS = ["floor_sum", "floor_ppl", "floor_min"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--evals", default=",".join(pdl.LONG))
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--ladder", default="long", choices=["long", "standard"],
                    help="long = ProbeDriftLong rungs (train on long-form ONLY, rungs suffixed -long). "
                         "standard = the ProbeDrift(XL) ladder (ID/SameTask/LOO/1ds-Diff/DiffTask) with the "
                         "MIXED short+long training pool, i.e. the same rungs every other method is on.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    evals = args.evals.split(",")
    # Ladder selection. Both paths use the SAME build_rows/sampled_train_idx, so the only thing that
    # changes is which training pool and which rung spec -- the method itself is untouched.
    if args.ladder == "standard":
        SOURCE_POOL, CELLS = xl_rungs.SOURCE_POOL, xl_rungs.cells
    else:
        SOURCE_POOL, CELLS = pdl.LONG_SRC, pdl.cells_long

    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"device {device} | seeds {seeds} | evals {evals}", flush=True)

    # ---- load per-token states + build sentence ids once per dataset (string ops, no model) ----
    PT, SEG = {}, {}
    for d in sorted(set(SOURCE_POOL) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip", flush=True); continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: fully unlabelled -> skip", flush=True); continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]
        segs = []
        for r, st in zip(records, states):
            # NB the record field is `gen_text` (NOT `generation`). Passing an empty string here would make
            # _token_sentence_ids return all-zeros = ONE segment per response, which silently collapses the
            # hierarchical pooler into the flat one and would look like a clean null. Same call as
            # weighted_msp_keep_variants so "sentence" is identical across the two tracks.
            sid, _ = sar._token_sentence_ids(
                tok, list(r["gen_token_ids"]),
                r.get("gen_text") or tok.decode(r["gen_token_ids"], skip_special_tokens=True))
            sid = np.asarray(sid, dtype=np.int64)
            # ALIGNMENT: the per-token cache window is [last_prompt_token] + gen_tokens = G+1 rows (see
            # weighted_msp.answer_states), but the sentence splitter returns one id per GENERATED token = G.
            # We extend the ids UP to G+1 (assigning the leading last-prompt token to the first sentence)
            # rather than dropping row 0 of the states: every method in this driver -- uniform, attention,
            # hier -- must pool over the SAME token set, or a hier-vs-attention difference would partly be
            # "one of them also saw the prompt token", which is a confound, not a pooling result.
            g = int(st.shape[0])
            if len(sid) == g - 1:
                lead = sid[:1] if len(sid) else np.zeros(1, dtype=np.int64)
                sid = np.concatenate([lead, sid])
            if len(sid) != g:
                raise SystemExit(f"ABORT {d}: segment ids ({len(sid)}) do not align with per-token states "
                                 f"({g}) even after the +1 last-prompt adjustment. Refusing to run rather "
                                 f"than pad/trim blindly.")
            segs.append(sid)
        PT[d] = (states, split, y, records)
        SEG[d] = segs
        nseg = float(np.mean([s.max() + 1 for s in segs])) if segs else 0.0
        print(f"  {d}: {len(states)} rows | mean {nseg:.1f} sentences", flush=True)
        # FAIL LOUD -- but only for an EVAL TARGET. If the target's responses are single-sentence the
        # two-level pooler IS the flat pooler and its numbers would be a meaningless (and very tidy-looking)
        # null, so refuse. A single-sentence SOURCE is harmless: it just contributes flat training examples,
        # and on the STANDARD ladder the pool legitimately contains short-form sets (sciq/trivia ~1 sentence),
        # so aborting on those would make --ladder standard impossible. Scope fixed 2026-07-23.
        if nseg < 1.5:
            if d in evals:
                raise SystemExit(
                    f"ABORT: EVAL TARGET {d} segments into only {nseg:.2f} sentences/response -- the "
                    f"hierarchy is vacuous on it (it reduces to the flat pooler). Not a meaningful cell.")
            print(f"     note: {d} is single-sentence; fine as a training SOURCE, not usable as a hier eval",
                  flush=True)
    sources = set(PT)

    out_rows = []
    for rung, X, spec in CELLS(sources, evals):
        if X not in PT:
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        per = {m: [] for m in METHODS + FLOORS}
        unc_acc = {m: [] for m in METHODS + FLOORS}
        yte_ref = None
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows)
            tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float); yte_ref = yte
            states = [PT[d][0][i] for d, i in allrows]
            recs = [PT[d][3][i] for d, i in allrows]
            segs = [SEG[d][i] for d, i in allrows]
            v = {}
            v["floor_sum"] = np.array([msp.msp_uncertainty(recs[i]["token_logprobs"], "sum") for i in te_idx])
            v["floor_ppl"] = np.array([msp.msp_uncertainty(recs[i]["token_logprobs"], "perplexity") for i in te_idx])
            v["floor_min"] = np.array([msp.msp_uncertainty(recs[i]["token_logprobs"], "min") for i in te_idx])
            Xmean = np.stack([s.mean(axis=0) for s in states])
            v["saplma"] = 1.0 - conf_meanpool(Xmean, tr_idx, te_idx, y, sd)
            bestT, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
            v["uniform"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd,
                                                          freeze_query=True), states, te_idx, device), float)
            v["attention"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd,
                                                            temperature=bestT), states, te_idx, device), float)
            # --- the W4 rows: full two-level, then each level ablated to uniform ---
            for name, kw in [("hier", {}), ("hier_seg", {"freeze_tok": True}), ("hier_tok", {"freeze_seg": True})]:
                m = hier_pool.train_hier(states, segs, y, tr_idx, device, seed=sd, **kw)
                v[name] = np.asarray(hier_pool.hier_unc(m, states, segs, te_idx, device), float)
            for m in v:
                per[m].append(results.prr(yte, v[m])); unc_acc[m].append(v[m])
        if yte_ref is None:
            continue
        stats = {m: (float(np.mean(per[m])), float(np.std(per[m]))) for m in per if per[m]}
        fair = "floor_min" if "floor_min" in stats else max(FLOORS, key=lambda f: stats[f][0])  # PRE-REGISTERED msp_min bar (2026-07-24)
        stats["fair_floor"] = stats[fair]
        avg = {m: np.mean(np.stack(unc_acc[m]), 0) for m in unc_acc if unc_acc[m]}
        avg["fair_floor"] = avg[fair]
        srcs = "+".join(f"{d}:{c}" if c else d for d, c in spec)
        print(f"\n[{rung:16s}] eval={X} train={srcs}  fair_floor={fair} {stats['fair_floor'][0]:+.3f}", flush=True)
        for m in METHODS:
            if m in stats:
                print(f"    {m:12s} {stats[m][0]:+.3f} +/- {stats[m][1]:.3f}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": m,
                                 "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                                 "n_seeds": len(per[m])})
        # the decisive verdicts: two-level vs flat, and vs the honest floor
        for vk, a, b in [("hier_vs_attention", "hier", "attention"),
                         ("hier_vs_fairfloor", "hier", "fair_floor"),
                         ("hier_vs_uniform", "hier", "uniform")]:
            if a in avg and b in avg:
                mg, lo, hi, p, sig = paired_bootstrap(yte_ref, avg[a], avg[b])
                print(f"    [verdict] {vk:20s} margin {mg:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} "
                      f"{'SIG' if sig else 'ns'}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"VERDICT:{vk}",
                                 "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                 "boot_p": round(p, 4), "significant": bool(sig), "n_seeds": len(seeds)})

    out = Path(args.out) if args.out else (ROOT / "results" / f"hier_pool_ladder__{cache._slug(MODEL)}.csv")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "train", "method", "prr_mean", "prr_std",
                                           "n_seeds", "ci_lo", "ci_hi", "boot_p", "significant"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
