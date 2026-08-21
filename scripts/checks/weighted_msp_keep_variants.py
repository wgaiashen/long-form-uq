"""Constrain the learned weighting to meaningful tokens (design note 4) — the keep-set sweep.

Runs weighted-MSP (normalised) with the softmax RESTRICTED to different token subsets, via the `keep=`
channel, across ID + the 5-rung ladder × all eval datasets. The subsets stack (each excludes more):
  special        drop only special/reserved tokens (the EOS fix; the post-fix baseline)
  special_punct  + drop punctuation / whitespace pieces
  content        + drop stop-words (negation carved back)  <- the "exclude stop words" idea
The learner can only place weight on the kept tokens (the softmax is over that subset only, pre-softmax
-inf on the rest), so it cannot fixate on EOS/punctuation/filler -- the failure the visualiser exposed.

Compared to plain MSP (floor). Same seeds/pools/harness as weighted_msp_all_variants (reused). CPU only.

    python scripts/checks/weighted_msp_keep_variants.py --seeds 1,2,3
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from luq import cache, msp, results, weighted_msp, token_subsets  # noqa: E402
from luq.features import sar  # noqa: E402  (sentence segmentation for the segment variant)
from aggregation_table import load_per_token  # noqa: E402
from weighted_msp_all_variants import EVALS, CANDIDATES, cells, sampled  # reuse the exact ladder (XL-aware)
import xl_rungs  # noqa: E402
from xl_rungs import label_of  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAB = "correctness"
# special/special_punct/content restrict the softmax to a token subset (keep=); segment learns one weight
# per SENTENCE and broadcasts it (segment_ids=, design note 7). All keep the EOS fix (exclude_special).
MODES = ["special", "special_punct", "content", "segment"]


def main():
    global MODES
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--modes", default=",".join(MODES),
                    help="comma-sep subset of special,special_punct,content,segment (run only these).")
    ap.add_argument("--loss", default="pairwise", choices=["pairwise", "blondel"],
                    help="ranking loss for the weighter. Default pairwise (what the committed keep-variant "
                         "numbers used); 'blondel' runs the same masks under the differentiable-rank loss "
                         "so the mask x loss interaction can be read off against those numbers (W2).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    MODES = [m for m in MODES if m in args.modes.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"device {device} | seeds {seeds} | modes {MODES}", flush=True)

    PT, KEEP = {}, {}
    for d in sorted(set(CANDIDATES) | set(EVALS)):    # sources + eval targets (e.g. ExpertQA)
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok -> skip"); continue
        states, split, y, _, records = loaded
        finite = np.isfinite(np.asarray(y, float))
        if not finite.any():
            print(f"  {d}: unlabelled ({label_of(d)}) -> skip"); continue
        if not finite.all():                          # keep labelled rows (masks recomputed over them below)
            keep_i = np.where(finite)[0]
            states = [states[k] for k in keep_i]; records = [records[k] for k in keep_i]
            split = split[keep_i]; y = np.asarray(y)[keep_i]
        PT[d] = (states, split, y, records)
        # precompute the keep masks + segment ids for every record
        km = {m: [] for m in ("special_punct", "content")}
        segids = []
        for r in records:
            pieces = tok.convert_ids_to_tokens(r["gen_token_ids"])
            km["special_punct"].append(token_subsets.keep_mask(r["gen_token_ids"], pieces, "special_punct"))
            km["content"].append(token_subsets.keep_mask(r["gen_token_ids"], pieces, "content"))
            sid, _ = sar._token_sentence_ids(tok, list(r["gen_token_ids"]),
                                             r.get("gen_text") or tok.decode(r["gen_token_ids"], skip_special_tokens=True))
            segids.append(np.asarray(sid, dtype=np.int64))
        km["segment"] = segids
        KEEP[d] = km
        fr = np.mean([m.mean() for m in km["content"]])
        nseg = np.mean([int(s.max()) + 1 for s in segids])
        print(f"  {d}: {len(states)} rows | content keeps {fr:.0%} | mean {nseg:.1f} sentences", flush=True)
    sources = set(PT)

    out_rows = []
    for rung, X, spec in cells(sources):
        spec = [(d, c) for d, c in spec if d in PT]
        if not spec:
            continue
        _, te0 = xl_rungs.eval_split(PT[X][1])         # baked-in for core; carved for XL evals
        if len(te0) == 0:
            continue
        yte = np.array([PT[X][2][i] for i in te0], float)
        # FAIR floor, not bare msp_sum (fixed 2026-07-22): the `floor` column is what every keep-variant
        # margin in the project's working notes is measured against, and msp_sum is not length-normalised, so on sets
        # where perplexity or msp_min is stronger every "beats the floor" count was overstated.
        _fv, _fname = msp.primary_floor([PT[X][3][i] for i in te0])  # PRE-REGISTERED msp_min bar (2026-07-24)
        floor = results.prr(yte, _fv)

        mode_prr = {m: [] for m in MODES}
        for sd in seeds:
            train_rows, test_rows = xl_rungs.build_rows(X, spec, PT, sd, sampled)
            if not train_rows:
                continue
            allrows = train_rows + test_rows
            n_tr = len(train_rows)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            states = [PT[d][0][i] for d, i in allrows]
            recs = [PT[d][3][i] for d, i in allrows]
            for m in MODES:
                keep, seg = None, None
                if m in ("special_punct", "content"):
                    keep = [KEEP[d][m][i] for d, i in allrows]
                elif m == "segment":
                    seg = [KEEP[d]["segment"][i] for d, i in allrows]
                # m == "special": keep=None -> the default special-token exclusion (the EOS fix)
                u = np.asarray(weighted_msp.weighted_msp_unc(
                    states, recs, y, tr_idx, te_idx, device, weight_mode="normalised",
                    length_normalise=True, seed=sd, keep=keep, segment_ids=seg, loss=args.loss), float)
                mode_prr[m].append(results.prr(yte, u))

        line = f"[{rung:18s}] {X:14s} floor {floor:+.3f}"
        for m in MODES:
            if mode_prr[m]:
                mean, std = float(np.mean(mode_prr[m])), float(np.std(mode_prr[m]))
                out_rows.append({"eval": X, "rung": rung, "mode": m, "prr_mean": round(mean, 4),
                                 "prr_std": round(std, 4), "floor": round(floor, 4)})
                line += f"  {m} {mean:+.3f}"
        print(line, flush=True)

    out = Path(args.out) if args.out else (ROOT / "results" / f"weighted_msp_keep_variants__{cache._slug(MODEL)}.csv")
    cols = ["eval", "rung", "mode", "prr_mean", "prr_std", "floor"]
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\nwrote {out} ({len(out_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
