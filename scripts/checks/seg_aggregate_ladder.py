"""W-B1 — decompose-and-aggregate on the LONG-FORM ladder: does a LEARNED aggregator beat the fixed rules?

Pipeline per cell: split each response into sentences -> mean-pool the L15 token states within each
sentence -> train the SAPLMA MLP on sentence vectors (each carrying its response's label) -> get one
P(correct) per sentence -> AGGREGATE to one instance score.

Rows (identical probe, identical splits/seeds -- only the aggregation differs):
    fair_floor    the PRE-REGISTERED msp_min bar (2026-07-24 meeting; all three variants still computed)
    saplma        mean-pool over ALL tokens + MLP            the no-decomposition reference
    seg_mean      per-sentence probe, MEAN aggregation       the existing fixed rule
    seg_min       per-sentence probe, MIN  (weakest-link)    the existing fixed rule
    seg_geomean   per-sentence probe, GEOMEAN (soft conj.)   the existing fixed rule
    seg_learned   per-sentence probe, LEARNED power-mean     <- the W-B1 method (alpha fitted on TRAIN)

Why this is the interesting comparison: `seg_learned` sits in a family that CONTAINS the three fixed rules
(alpha = 1 -> mean, 0 -> geomean, -inf -> min), so it cannot lose to them by more than fitting noise, and the
FITTED ALPHA is itself the deliverable -- it says where a task sits on the dilution <-> weakest-link axis.
Reported per cell.

Two honesty constraints, both enforced here:
  * alpha is fitted on the TRAINING instances only (never on the test set we then report).
  * the per-sentence probe is the SAME `train_probe_mlp` used everywhere else, so a seg-vs-saplma difference
    is aggregation, not a different classifier.

Run on the long-form ladder (segments only mean something for multi-sentence output). CPU only.
    python scripts/checks/seg_aggregate_ladder.py --evals pubmed_qa --seeds 1,2,3
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402

from luq import cache, msp, probe, results, seg_aggregate as SA  # noqa: E402
from luq.features import sar  # noqa: E402
from aggregation_table import load_per_token, paired_bootstrap, conf_meanpool  # noqa: E402
from xl_rungs import build_rows, eval_split, label_of  # noqa: E402
import probedriftlong as pdl  # noqa: E402
import xl_rungs  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
METHODS = ["fair_floor", "saplma", "seg_mean", "seg_min", "seg_geomean", "seg_learned"]


def sentence_vectors(states, sent_ids):
    """(n_sent, d): mean-pool the token states within each sentence.

    The per-token cache window is [last_prompt_token] + gen_tokens = G+1 while the splitter returns G ids,
    so the leading prompt token is attached to the first sentence (same convention as hier_pool_ladder --
    keeping one definition of "sentence" across the whole project).
    """
    g = states.shape[0]
    sid = np.asarray(sent_ids, dtype=np.int64)
    if len(sid) == g - 1:
        sid = np.concatenate([sid[:1] if len(sid) else np.zeros(1, dtype=np.int64), sid])
    if len(sid) != g:
        raise SystemExit(f"segment ids ({len(sid)}) != states ({g}) after the +1 adjustment")
    return np.stack([states[sid == s].mean(axis=0) for s in np.unique(sid)])


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
    tok = AutoTokenizer.from_pretrained(MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | seeds {seeds} | evals {evals}", flush=True)

    PT, SENT = {}, {}
    for d in sorted(set(SOURCE_POOL) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok -> skip", flush=True); continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: unlabelled -> skip", flush=True); continue
        if not finite.all():
            k = np.where(finite)[0]
            states = [states[i] for i in k]; records = [records[i] for i in k]
            split = split[k]; y = y[k]
        sv = []
        for r, st in zip(records, states):
            sid, _ = sar._token_sentence_ids(
                tok, list(r["gen_token_ids"]),
                r.get("gen_text") or tok.decode(r["gen_token_ids"], skip_special_tokens=True))
            sv.append(sentence_vectors(np.asarray(st), sid))
        PT[d] = (states, split, y, records)
        SENT[d] = sv
        ns = float(np.mean([len(v) for v in sv]))
        print(f"  {d}: {len(states)} rows | mean {ns:.1f} sentences", flush=True)
        if ns < 1.5:
            print(f"     NOTE: {d} is near-single-sentence -> decomposition is near-vacuous here", flush=True)
    sources = set(PT)

    out_rows = []
    for rung, X, spec in CELLS(sources, evals):
        if X not in PT:
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        per = {m: [] for m in METHODS}
        acc = {m: [] for m in METHODS}
        alphas, yte_ref = [], None
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows)
            tr_i = list(range(n_tr)); te_i = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_i], float); yte_ref = yte
            recs = [PT[d][3][i] for d, i in allrows]
            sv = [SENT[d][i] for d, i in allrows]
            v = {}
            v["fair_floor"], _fname = msp.primary_floor([recs[i] for i in te_i])  # PRE-REGISTERED msp_min bar (2026-07-24)
            Xmean = np.stack([np.asarray(PT[d][0][i]).mean(axis=0) for d, i in allrows])
            v["saplma"] = 1.0 - conf_meanpool(Xmean, tr_i, te_i, y, sd)

            # one per-sentence probe, shared by every aggregator (so only aggregation varies)
            Xtr = np.concatenate([sv[i] for i in tr_i], axis=0)
            ytr = np.concatenate([np.full(len(sv[i]), y[i]) for i in tr_i])
            clf = probe.train_probe_mlp(Xtr, ytr, seed=sd)
            p_tr = [np.clip(clf.p_correct(sv[i]), 1e-6, 1 - 1e-6) for i in tr_i]
            p_te = [np.clip(clf.p_correct(sv[i]), 1e-6, 1 - 1e-6) for i in te_i]

            v["seg_mean"] = 1.0 - SA.aggregate(p_te, 1.0)
            v["seg_geomean"] = 1.0 - SA.aggregate(p_te, 0.0)
            v["seg_min"] = 1.0 - SA.aggregate(p_te, -2000.0)
            ytr_inst = np.array([y[i] for i in tr_i], float)
            v["seg_learned"], alpha = SA.learned_aggregate(p_tr, ytr_inst, p_te, results.prr)
            alphas.append(alpha)

            for m in v:
                per[m].append(results.prr(yte, v[m])); acc[m].append(v[m])
        if yte_ref is None:
            continue
        stats = {m: (float(np.mean(per[m])), float(np.std(per[m]))) for m in per if per[m]}
        avg = {m: np.mean(np.stack(acc[m]), 0) for m in acc if acc[m]}
        srcs = "+".join(f"{d}:{c}" if c else d for d, c in spec)
        a_str = ",".join(f"{a:g}" for a in alphas)
        print(f"\n[{rung:16s}] eval={X} train={srcs}  fitted_alpha=[{a_str}]", flush=True)
        for m in METHODS:
            if m in stats:
                print(f"    {m:12s} {stats[m][0]:+.3f} +/- {stats[m][1]:.3f}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": m,
                                 "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                                 "n_seeds": len(per[m]), "fitted_alpha": a_str})
        for vk, a, b in [("learned_vs_mean", "seg_learned", "seg_mean"),
                         ("learned_vs_min", "seg_learned", "seg_min"),
                         ("learned_vs_saplma", "seg_learned", "saplma"),
                         # ITEM 5: seg_min is the aggregator the weakest-link theory predicts, and its few
                         # nominal wins sit at the hardest shift rungs. Adding its paired test so the
                         # "decomposition never significantly beats whole-response" claim covers it too,
                         # not just the fitted seg_learned (STOCKTAKE XXI.4).
                         ("min_vs_saplma", "seg_min", "saplma"),
                         ("mean_vs_saplma", "seg_mean", "saplma"),
                         ("learned_vs_fairfloor", "seg_learned", "fair_floor")]:
            if a in avg and b in avg:
                mg, lo, hi, p, sig = paired_bootstrap(yte_ref, avg[a], avg[b])
                print(f"    [verdict] {vk:22s} {mg:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} "
                      f"{'SIG' if sig else 'ns'}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"VERDICT:{vk}",
                                 "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                 "boot_p": round(p, 4), "significant": bool(sig), "n_seeds": len(seeds)})

    out = Path(args.out) if args.out else (ROOT / "results" / f"seg_aggregate_ladder__{cache._slug(MODEL)}.csv")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "train", "method", "prr_mean", "prr_std",
                                           "n_seeds", "fitted_alpha", "ci_lo", "ci_hi", "boot_p",
                                           "significant"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
