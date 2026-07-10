"""The Orgad important-token method as ONE method across ID + the full OOD ladder, all datasets.

Question: does restricting weighted-MSP to the LLM-identified important tokens (Orgad, leak-free: the
model's own exact answer for QA; the key fact-bearing terms for summarisation) help, ID and OOD?

Same cell machinery as weighted_msp_blondel.py (get_training_spec splits, per-token cache, seeds,
paired_bootstrap) -- the ONLY change is we also run weighted-MSP with masks= restricting w*NLL to the
important tokens. For each ladder cell we report, mean+/-std over seeds:
  weighted_msp_pairwise            (unmasked -- the baseline being beaten)
  weighted_msp_pairwise_orgad      (masked to important tokens -- the method)
  msp_sum                          (the unsupervised floor both must clear)
plus paired bootstraps (orgad vs unmasked, and each vs floor). Task-adaptive locate: QA exact answer
(str) or summary key spans (list). Mask fallback when nothing is located = ALL tokens (so an unlocated
row is just plain weighted-MSP -- matches build_answer_masks). CPU only. Needs cache/orgad_llm/*.json.

    python scripts/checks/weighted_msp_orgad_ladder.py --seeds 1,2,3
"""
import argparse
import csv as _csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from luq import cache, msp, results, weighted_msp  # noqa: E402
from luq.features import orgad_llm  # noqa: E402
from probe_drift.ood_settings import get_training_spec  # noqa: E402
from aggregation_table import load_per_token, paired_bootstrap  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAB = "correctness"
LAYER = 15
EVALS = ["sciq", "trivia_qa", "pubmed_qa", "xsum"]
CANDIDATE_SOURCES = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "med_quad", "samsum"]
SETTINGS = [("SameTask", "OOD_ONE_DATASET_SAME_TASK"), ("LOO", "OOD_LEAVE_ONE_OUT"),
            ("OneDatasetDiffTask", "OOD_ONE_DATASET_DIFF_TASK"), ("DiffTask", "OOD_DIFF_TASK")]


def build_masks(tok, records):
    """Per-record important-token mask over the generated tokens (1.0 on located important tokens; ALL
    tokens if nothing located, matching build_answer_masks). Returns (masks_list, located_bool)."""
    d = records[0].get("_dataset")  # set by caller
    ex_path = ROOT / "cache" / "orgad_llm" / f"{cache._slug(MODEL)}__{d}__ID.json"
    ex = json.loads(ex_path.read_text()) if ex_path.exists() else {}
    masks, located = [], np.zeros(len(records), bool)
    for i, r in enumerate(records):
        g = len(r["gen_token_ids"])
        val = ex.get(f"{r['split']}:{r['idx']}", "NO ANSWER")
        rows, found = orgad_llm.locate_important_rows(tok, r["gen_token_ids"], val)
        m = np.zeros(g, np.float32)
        for row in rows:
            if 0 <= row - 1 < g:
                m[row - 1] = 1.0
        located[i] = found and m.sum() > 0
        if m.sum() == 0:
            m[:] = 1.0
        masks.append(m)
    return masks, located


def sampled_train_idx(split, seed, cap):
    tr = np.where(split == "train")[0]
    if cap is None or cap >= len(tr):
        return tr
    return tr[np.random.RandomState(seed).permutation(len(tr))[:cap]]


def cells(sources):
    out = [("ID", X, [(X, None)]) for X in EVALS]
    for X in EVALS:
        for tag, setting in SETTINGS:
            spec = [(s, n) for s, n in get_training_spec(X, setting) if s in sources and s != X]
            if spec:
                out.append((tag, X, spec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--sources", default=",".join(CANDIDATE_SOURCES))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"device {device} | seeds {seeds}", flush=True)

    PT, MASKS = {}, {}
    for d in args.sources.split(","):
        loaded = load_per_token(MODEL, d, LAYER, LAB)
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip", flush=True); continue
        states, split, y, _, records = loaded
        if np.isnan(y).any():
            print(f"  {d}: unlabelled -> skip", flush=True); continue
        for r in records:
            r["_dataset"] = d
        masks, located = build_masks(tok, records)
        PT[d] = (states, split, y, records)
        MASKS[d] = masks
        # leak re-check: does "located" still track correctness? (the gold-mask bug was 99% vs 0.2%)
        strm = [r.get("correctness_strmatch") for r in records]
        corr = (np.array([1 if isinstance(x, (int, float)) and x >= 0.5 else 0 for x in strm])
                if any(isinstance(x, (int, float)) for x in strm) else (y >= 0.5).astype(int))
        lc = 100 * located[corr == 1].mean() if (corr == 1).any() else float("nan")
        li = 100 * located[corr == 0].mean() if (corr == 0).any() else float("nan")
        print(f"  {d}: {len(states)} rows | located {located.mean()*100:.0f}% | "
              f"leak loc|corr {lc:.0f}% vs loc|incorr {li:.0f}% (want CLOSE)", flush=True)
    sources = set(PT)

    out_rows = []
    for rung, X, spec in cells(sources):
        if X not in PT:
            continue
        prr = {"unmasked": [], "orgad": [], "msp_sum": []}
        unc = {"unmasked": [], "orgad": []}
        yte_ref = None
        for sd in seeds:
            train_rows = [(d, i) for d, cap in spec for i in sampled_train_idx(PT[d][1], sd, cap)]
            test_rows = [(X, i) for i in np.where(PT[X][1] == "test")[0]]
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], dtype=float)
            yte = np.array([y[i] for i in te_idx], dtype=float)
            yte_ref = yte
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]
            mk = [MASKS[d][i] for d, i in allrows]
            for tag, use_mask in [("unmasked", None), ("orgad", mk)]:
                u = np.asarray(weighted_msp.weighted_msp_unc(
                    states, records, y, tr_idx, te_idx, device, weight_mode="normalised",
                    length_normalise=True, seed=sd, loss="pairwise", masks=use_mask), dtype=float)
                prr[tag].append(results.prr(yte, u)); unc[tag].append(u)
            floor = np.asarray([msp.msp_uncertainty(records[i]["token_logprobs"], "sum") for i in te_idx])
            prr["msp_sum"].append(results.prr(yte, floor))

        if yte_ref is None:
            continue
        srcs = "+".join(f"{d}:{c}" if c else d for d, c in spec)
        m = {k: (float(np.mean(v)), float(np.std(v))) for k, v in prr.items() if v}
        print(f"\n[{rung:18s}] eval={X} train={srcs}", flush=True)
        for k in ("unmasked", "orgad", "msp_sum"):
            if k in m:
                print(f"    {k:10s} {m[k][0]:+.3f} +/- {m[k][1]:.3f}", flush=True)
        avg = {k: np.mean(np.stack(unc[k]), axis=0) for k in unc if unc[k]}
        floor_vec = np.asarray([msp.msp_uncertainty(PT[X][3][i]["token_logprobs"], "sum")
                                for i in np.where(PT[X][1] == "test")[0]], dtype=float)
        for tag, av, bv in [("orgad_vs_unmasked", avg.get("orgad"), avg.get("unmasked")),
                            ("orgad_vs_floor", avg.get("orgad"), floor_vec),
                            ("unmasked_vs_floor", avg.get("unmasked"), floor_vec)]:
            if av is not None and bv is not None:
                mg, lo, hi, p, sig = paired_bootstrap(yte_ref, av, bv)
                print(f"    [verdict] {tag:20s} margin {mg:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} "
                      f"{'SIG' if sig else 'ns'}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"VERDICT:{tag}",
                                 "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                 "boot_p": round(p, 4), "significant": sig})
        for k in ("unmasked", "orgad", "msp_sum"):
            if k in m:
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"weighted_msp_{k}",
                                 "prr_mean": round(m[k][0], 4), "prr_std": round(m[k][1], 4),
                                 "n_seeds": len(prr[k])})

    out = ROOT / "results" / f"weighted_msp_orgad_ladder__{cache._slug(MODEL)}.csv"
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "train", "method", "prr_mean", "prr_std",
                                           "n_seeds", "ci_lo", "ci_hi", "boot_p", "significant"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
