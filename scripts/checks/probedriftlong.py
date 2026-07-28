"""ProbeDriftLong (W3 / H4): train EXCLUSIVELY on long-form datasets and evaluate on long-form, plus a
long->short transfer cell. Joe's steer: the current ProbeDrift ID/LOO settings let probes lean on easy
short-form training data (long_form_loo §J showed SAPLMA's long-form OOD collapses to the floor once
short-form is removed). This driver removes that crutch by construction and asks: in the realistic
long-only setting, does weighted-MSP / shrink overtake the probes, measured against the HONEST fair floor
(max of msp_sum, perplexity, msp_min)?

LONG UNIVERSE: pubmed_qa, xsum, cnn_dailymail, med_quad, samsum, expertqa, asqa.
  UPDATE 2026-07-22 (author's decision): ExpertQA and ASQA are ORDINARY TRAINING SOURCES, so the long pool
  is MIXED-LABEL by default (ExpertQA = claim-precision, the rest = reference-agreement). Cells are tagged
  `different_label_projection`. `--label-homogeneous` drops ExpertQA from the sources to reproduce the
  pre-2026-07-22 baseline as the control.
  FINE families: correctness_qa = {pubmed_qa, med_quad, asqa};  factuality = {expertqa, factscore};
  summ = {xsum, cnn_dailymail, samsum}.  (factuality split from correctness_qa on 2026-07-27.)

RUNGS (long eval X): ID | SameTask-long (other long sets in X's family) | LOO-long (all other long sets)
  | DiffTask-long (opposite long family) | 1ds-Diff-long (one opposite-family long set).
  For a SHORT eval (sciq/trivia): the single "Long->Short" cell (train on the full long pool).

METHODS (the KEEP set): fair floor (msp_sum/perplexity/msp_min -> max), uniform + attention poolers,
  mean-pool SAPLMA (MLP), weighted-MSP {normalised, shrink@2, shrink@10, Blondel}. Paired bootstrap:
  best-wMSP vs fair-floor, best-wMSP vs best-pooler, attention vs fair-floor.

    python scripts/checks/probedriftlong.py --seeds 1,2,3
    python scripts/checks/probedriftlong.py --evals pubmed_qa,xsum --seeds 1   # smoke
"""
import argparse, csv as _csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "checks"))
import torch  # noqa: E402
from luq import cache, msp, results, weighted_msp, probe  # noqa: E402
from luq.features import sar  # noqa: E402  (shared sentence splitter)
from transformers import AutoTokenizer  # noqa: E402
from luq.weighting import shrink_to_uniform  # noqa: E402
from aggregation_table import load_per_token, attn_unc, paired_bootstrap, conf_meanpool, prr_from_conf  # noqa: E402
from attn_pool import train_attn, select_temperature  # noqa: E402
from xl_rungs import build_rows, eval_split, label_of, different_label_projection  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa", "asqa", "factscore"]
# Training sources for the long-only ladder. ASQA and ExpertQA are ordinary sources here, same as the rest
# (author's decision 2026-07-22) -- so the long pool is MIXED-LABEL by default (ExpertQA = faithfulness,
# the others = correctness). Cells are still tagged `different_label_projection` so they stay identifiable. A dataset is
# always excluded from its OWN eval's sources by cells_long(), so this never leaks train into test.
LONG_SRC = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa", "asqa", "factscore"]
SHORT = ["sciq", "trivia_qa"]
# FINE families (2026-07-27 FACTUALITY-FAMILY SPLIT): correctness-QA (correctness vs a gold answer) is kept
# SEPARATE from factuality (claim-support vs an external reference), so SameTask means same PROPERTY. asqa
# stays with pubmed/med_quad (correctness QA, per author); expertqa pairs with factscore (factuality).
# ⚠️ The factuality family is {expertqa, factscore}, so a factuality eval's SameTask-long is EMPTY until
# factscore's cache lands on RCS (post-DoC-rsync). Re-run §C.3 for the affected long evals (pubmed/med_quad/
# asqa lose expertqa from SameTask; expertqa/factscore gain each other) once factscore is cached.
FINE = {"pubmed_qa": "correctness_qa", "med_quad": "correctness_qa", "asqa": "correctness_qa",
        "expertqa": "factuality", "factscore": "factuality",
        "xsum": "summ", "cnn_dailymail": "summ", "samsum": "summ"}
XL_TOTAL = 1800
EVALS = LONG + SHORT
# wMSP KEEP variants: (col name, kwargs to weighted_msp_unc)  [all length_normalise=True]
WMSP = [("wmsp_norm", {"weight_mode": "normalised"}),
        # per-segment (sentence) wMSP -- FIRST run under ProbeDriftLong (Joe #7; was standard-ladder
        # only). `_seg_ids` is a sentinel: the loop replaces it with this cell's sentence ids.
        ("wmsp_seg_flat", {"weight_mode": "normalised", "segment_ids": "_seg_ids"}),
        ("wmsp_seg_softmax", {"weight_mode": "normalised", "segment_ids": "_seg_ids",
                              "segment_mode": "softmax"}),
        ("wmsp_shrink2", {"weight_mode": "normalised", "reg": shrink_to_uniform, "reg_lambda": 2.0}),
        ("wmsp_shrink10", {"weight_mode": "normalised", "reg": shrink_to_uniform, "reg_lambda": 10.0}),
        ("wmsp_blondel", {"weight_mode": "normalised", "loss": "blondel"}),
        # W2: Blondel loss on the KEEP shrink variants -> paired loss-only comparison vs wmsp_shrink2/10 above.
        ("wmsp_shrink2_blondel", {"weight_mode": "normalised", "reg": shrink_to_uniform,
                                  "reg_lambda": 2.0, "loss": "blondel"}),
        ("wmsp_shrink10_blondel", {"weight_mode": "normalised", "reg": shrink_to_uniform,
                                   "reg_lambda": 10.0, "loss": "blondel"})]
POOLERS = ["uniform", "attention"]
FLOORS = ["floor_sum", "floor_ppl", "floor_min"]
METHODS = FLOORS + ["fair_floor", "saplma"] + POOLERS + [w[0] for w in WMSP]


def rung_sources_long(X):
    same = [d for d in LONG_SRC if d != X and FINE[d] == FINE[X]]
    diff = [d for d in LONG_SRC if d != X and FINE[d] != FINE[X]]
    loo = [d for d in LONG_SRC if d != X]
    return {"SameTask-long": same, "DiffTask-long": diff, "LOO-long": loo, "1ds-Diff-long": diff[:1]}


def cells_long(sources, evals):
    out = []
    for X in evals:
        if X not in sources:
            continue
        if X in LONG:
            out.append(("ID", X, [(X, None)]))
            rs = rung_sources_long(X)
            for tag in ("SameTask-long", "DiffTask-long", "LOO-long"):
                srcs = [d for d in rs[tag] if d in sources]
                if srcs:
                    cap = max(1, XL_TOTAL // len(srcs))
                    out.append((tag, X, [(d, cap) for d in srcs]))
            one = [d for d in rs["1ds-Diff-long"] if d in sources]
            if one:
                out.append(("1ds-Diff-long", X, [(one[0], XL_TOTAL)]))
        elif X in SHORT:                                   # long -> short transfer
            srcs = [d for d in LONG_SRC if d in sources]
            if srcs:
                cap = max(1, XL_TOTAL // len(srcs))
                out.append(("Long->Short", X, [(d, cap) for d in srcs]))
    return out


def sampled_train_idx(split, seed, cap):
    # Round-3 Task A (2026-07-27): the defect was this filter returning EMPTY for eval-only sources (all
    # split=="test"), so a listed source like `expertqa:360` silently contributed 0 rows (V-A0). A dataset used
    # as a SOURCE for ANOTHER eval leaks nothing regardless of split label (source != eval is enforced by
    # cells_long), so draw from ALL its rows when it has no dedicated train split. Core sets (with a train
    # split) are UNCHANGED -- they still draw train-only.
    tr = np.where(split == "train")[0]
    if len(tr) == 0:                                  # eval-only source: use its full row set as source rows
        tr = np.arange(len(split))
    if cap is None or cap >= len(tr):
        return tr
    return tr[np.random.RandomState(seed).permutation(len(tr))[:cap]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--evals", default=",".join(EVALS))
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--include-expertqa", action="store_true",
                    help="DEPRECATED / no-op as of 2026-07-22: ExpertQA is now an ordinary training source by "
                         "default, so the mixed-label 'universal' pool IS the default. Kept so existing job "
                         "scripts keep working.")
    ap.add_argument("--label-homogeneous", action="store_true",
                    help="drop ExpertQA (faithfulness) from the TRAINING sources so every training label is "
                         "correctness. This reproduces the pre-2026-07-22 committed PART VII baseline and is "
                         "the control for 'does mixing label semantics help or hurt?'. ExpertQA remains an EVAL.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-wmsp", action="store_true",
                    help="skip the 10 wMSP KEEP variants (the training bottleneck). Poolers + floors + saplma "
                         "still run -- enough for §B.3 / P2a / Task D. Use when only the pooler numbers are "
                         "needed; wMSP (§C.4) is re-run separately.")
    args = ap.parse_args()
    # active method set: wMSP is the per-cell training bottleneck (10 variants); drop it when only the poolers
    # and floors are needed. best_w / the wmsp verdicts are guarded below when wMSP is off.
    active_wmsp = [] if args.skip_wmsp else WMSP
    active_methods = FLOORS + ["fair_floor", "saplma"] + POOLERS + [w[0] for w in active_wmsp]
    if args.skip_wmsp:
        print("SKIP-WMSP: running poolers + floors + saplma only (wMSP variants skipped)", flush=True)
    global LONG_SRC
    if args.label_homogeneous:
        LONG_SRC = [d for d in LONG_SRC if d != "expertqa"]
        print("LABEL-HOMOGENEOUS mode: ExpertQA removed from training sources (correctness labels only)",
              flush=True)
    else:
        print("default MIXED-LABEL pool: ExpertQA (faithfulness) is an ordinary training source", flush=True)
    evals = args.evals.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | seeds {seeds} | evals {evals}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    PT, SEG = {}, {}
    for d in sorted(set(LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip", flush=True); continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: fully unlabelled ({label_of(d)}) -> skip", flush=True); continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]
        segs = []
        for r, st in zip(records, states):
            sid, _ = sar._token_sentence_ids(tok, list(r['gen_token_ids']),
                                             r.get('gen_text') or tok.decode(r['gen_token_ids'], skip_special_tokens=True))
            sid = np.asarray(sid, dtype=np.int64)
            g = int(np.asarray(st).shape[0])   # per-token window is G+1; wMSP uses answer_states (G) -> ids length G
            if len(sid) == g:                  # already G (state has the +1 anchor); trim to answer tokens
                sid = sid
            segs.append(sid)
        PT[d] = (states, split, y, records); SEG[d] = segs
        print(f"  {d}: {len(states)} rows (label={label_of(d)})", flush=True)
    sources = set(PT)

    out_rows = []
    for rung, X, spec in cells_long(sources, evals):
        if X not in PT:
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        xlbl = different_label_projection(X)
        per = {m: [] for m in active_methods}; unc_acc = {m: [] for m in active_methods}; yte_ref = None
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows); tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float); yte_ref = yte
            states = [PT[d][0][i] for d, i in allrows]; records = [PT[d][3][i] for d, i in allrows]
            v = {}
            # unsupervised floors (per-example vectors)
            v["floor_sum"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "sum") for i in te_idx])
            v["floor_ppl"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity") for i in te_idx])
            v["floor_min"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "min") for i in te_idx])
            # SAPLMA mean-pool + MLP
            Xmean = np.stack([s.mean(axis=0) for s in states])
            v["saplma"] = 1.0 - conf_meanpool(Xmean, tr_idx, te_idx, y, sd)
            # poolers
            best_T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
            v["uniform"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True),
                                               states, te_idx, device), float)
            v["attention"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T),
                                                 states, te_idx, device), float)
            # wMSP KEEP variants (skipped under --skip-wmsp -- the training bottleneck)
            seg_cell = [SEG[d][i] for d, i in allrows]
            for name, kw in active_wmsp:
                kw2 = dict(kw)
                if kw2.get('segment_ids') == '_seg_ids':
                    kw2['segment_ids'] = seg_cell
                v[name] = np.asarray(weighted_msp.weighted_msp_unc(states, records, y, tr_idx, te_idx, device,
                                     length_normalise=True, seed=sd, **kw2), float)
            for m in v:
                per[m].append(results.prr(yte, v[m])); unc_acc[m].append(v[m])
        if yte_ref is None:
            continue
        stats = {m: (float(np.mean(per[m])), float(np.std(per[m]))) for m in per if per[m]}
        # PRE-REGISTERED primary bar = floor_min (msp_min), FIXED across datasets (2026-07-24 meeting);
        # replaces the rejected max-of-three ("three shots for the baseline"). All three floors are still
        # persisted (METHODS) for the 3-variant table. `strongest` is tracked only for the DUAL-REPORT note
        # on datasets where a different variant is the strongest free score (cnn/samsum -> floor_ppl).
        _FLOOR_KEY = {"sum": "floor_sum", "perplexity": "floor_ppl", "min": "floor_min"}
        primary_name = _FLOOR_KEY[msp.PRIMARY_FLOOR_AGG]
        fair_name = primary_name if primary_name in stats else max(FLOORS, key=lambda f: stats[f][0])
        strongest = max(FLOORS, key=lambda f: stats[f][0])
        stats["fair_floor"] = stats[fair_name]
        avg = {m: np.mean(np.stack(unc_acc[m]), 0) for m in unc_acc if unc_acc[m]}
        avg["fair_floor"] = avg[fair_name]
        best_w = max((w[0] for w in active_wmsp), key=lambda m: stats[m][0]) if active_wmsp else None
        best_p = max(POOLERS, key=lambda m: stats[m][0])
        # HONEST LABEL (Round-3 Task A): `train` from REALISED per-source counts (last seed; seed-stable), not
        # requested caps -- so the label can never overstate the pool (the V-A0 defect). realised==labelled.
        _real = {}
        for _d, _i in train_rows:
            _real[_d] = _real.get(_d, 0) + 1
        srcs = "+".join(f"{d}:{_real.get(d, 0)}" for d in dict.fromkeys(d for d, _c in spec))
        xf = "  [CROSS-LABEL]" if (xlbl and rung != "ID") else ""
        dual = "" if strongest == fair_name else f"  (strongest free = {strongest} {stats[strongest][0]:+.3f}; DUAL-REPORT)"
        # LENGTH COLUMNS (Round-3 Task C): domain-shift vs length-shift made visible, over the REALISED pool.
        _tl = [len(PT[d][3][i]["token_logprobs"]) for d, i in train_rows]
        _el = [len(PT[X][3][i]["token_logprobs"]) for _d, i in test_rows]
        _lens = {"eval_med_len": round(float(np.median(_el)), 1) if _el else "",
                 "train_med_len": round(float(np.median(_tl)), 1) if _tl else "",
                 "train_max_len": int(max(_tl)) if _tl else ""}
        print(f"\n[{rung:14s}] eval={X} ({label_of(X)}) train={srcs}{xf}  primary_floor={fair_name} {stats['fair_floor'][0]:+.3f}{dual}", flush=True)
        for m in active_methods:
            if m in stats:
                print(f"    {m:14s} {stats[m][0]:+.3f} +/- {stats[m][1]:.3f}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": m,
                                 "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                                 "n_seeds": len(per[m]), "different_label_projection": bool(xlbl and rung != "ID"),
                                 **_lens})
        for vk, a, b in [("bestw_vs_fairfloor", best_w, "fair_floor"),
                         ("bestw_vs_bestpooler", best_w, best_p),
                         ("attention_vs_fairfloor", "attention", "fair_floor")]:
            if a in avg and b in avg:
                mg, lo, hi, p, sig = paired_bootstrap(yte_ref, avg[a], avg[b])
                print(f"    [verdict] {vk:24s} ({a} vs {b}) margin {mg:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} {'SIG' if sig else 'ns'}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"VERDICT:{vk}",
                                 "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                 "boot_p": round(p, 4), "significant": bool(sig), "n_seeds": len(seeds)})

    out = Path(args.out) if args.out else (ROOT / "results" / f"probedriftlong__{cache._slug(MODEL)}.csv")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "train", "method", "prr_mean", "prr_std",
                                           "n_seeds", "different_label_projection", "eval_med_len",
                                           "train_med_len", "train_max_len", "ci_lo", "ci_hi", "boot_p",
                                           "significant"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
