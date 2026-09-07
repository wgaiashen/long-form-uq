#!/usr/bin/env python
"""Like-for-like comparison: learned token weighting against fixed and externally specified rules.

THE QUESTION
------------
All six scores below are computed from the SAME per-token log-probabilities of the SAME generation.
They differ only in WHICH tokens count and HOW MUCH each counts:

  sequence NLL          every token, equal weight, not length-normalised   (the published MSP)
  mean token NLL        every token, equal weight, length-normalised
  minimum token prob.   all the weight on the single least confident token
  TokenSAR              weights set by an external relevance model, unsupervised
  answer-span restricted  a binary mask from an external span extractor, then aggregated
  CAWSA lambda=2        weights LEARNED from correctness labels

So the comparison isolates the weighting rule, which is the thing the learned method claims to
contribute. It is not a comparison against probes, which read hidden states and are a different family.

TWO INDEX BASES, AND WHY THAT MATTERS
-------------------------------------
The ladder drops unlabelled rows, so its indices are positions in a FILTERED array, while the relevance
cache is a plain array in ORIGINAL record order and the span cache is keyed by the record's own
split and idx. ExpertQA drops 292 rows and FActScore 45, so indexing one with the other's basis
silently shifts those datasets by up to 292 places and still returns a plausible number. The mapping is
therefore explicit here and asserted, never inferred.

UNLOCATED SPANS
---------------
When the extractor finds no span in a generation the score falls back to all tokens, which is what the
existing masked implementation does. That makes the restricted score equal to the unrestricted one on
those rows, so the located rate is reported per dataset and must be read alongside the result.

    python scripts/checks/likeforlike_table.py
    python scripts/checks/likeforlike_table.py --model Qwen/Qwen2.5-14B
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

from luq import cache, msp as msp_mod, results        # noqa: E402
from luq.config import Config                          # noqa: E402
from luq.features import orgad_llm                     # noqa: E402
from probe_drift_long import LONG_SRC                  # noqa: E402
# The ladder's OWN wrappers, not the library functions underneath them. The wrapper fixes the carve
# seed and the carve rule that produced the master; calling the library directly with a different seed
# silently carves a DIFFERENT test set, which is what the floor gate caught on the first attempt.
from xl_rungs import eval_split, label_of              # noqa: E402

RUNGS = ["ID", "LOO-long", "SameTask-long", "DiffTask-long", "1ds-Diff-long"]
FLOOR_KEY = {"floor_sum": "sequence NLL (published MSP)",
             "floor_ppl": "mean token NLL",
             "floor_min": "minimum token probability"}

# Per-model resolution, taken verbatim from `clean_core_manifest.py`'s CORE dict (the frozen
# report-facing regime map), so this script and the manifest can never silently disagree about which
# cache namespace a dataset reads. SAR/ORGAD suffix only exists for Llama, where an OLDER (pre-2026-08-24
# uncorrected) cache sat next to the corrected one under the same base name and had to be disambiguated;
# Qwen's and gemma's SAR/answer-span caches are computed for the first time directly on the corrected
# population, so there is nothing to disambiguate and no suffix is used.
MODEL_CFG = {
    "meta-llama/Meta-Llama-3.1-8B": dict(
        master_rel="cleanv2/pdl_cleanv2_master__{slug}.csv",
        regime={"pubmed_qa": "", "xsum": "", "cnn_dailymail": "", "samsum": "",
                "med_quad": "cleanv2", "asqa": "asqa_rp12",
                "expertqa": "expertqa_rp12", "factscore": "factscore_rp12"},
        sar_suffix={"med_quad": "__cleanv2"}, orgad_suffix={"med_quad": "__cleanv2"}),
    "Qwen/Qwen2.5-14B": dict(
        master_rel="analysis/pdl_master_qwenclean__{slug}.csv",
        regime={"pubmed_qa": "", "xsum": "", "cnn_dailymail": "", "asqa": "asqa_rp12",
                "samsum": "trunc_v1", "med_quad": "trunc_v1",
                "expertqa": "expertqa_rp12_trunc_v1", "factscore": "factscore_rp12_trunc_v1"},
        sar_suffix={}, orgad_suffix={}),
    "google/gemma-2-9b": dict(
        master_rel="cleanv2/wmodels_sens8_cleanv2_master__{slug}.csv",
        regime={"pubmed_qa": "", "xsum": "", "cnn_dailymail": "", "samsum": "",
                "med_quad": "cleanv2", "asqa": "asqa_rp12",
                "expertqa": "expertqa_rp12", "factscore": "factscore_rp12"},
        sar_suffix={}, orgad_suffix={}),
}


def load_master(master_path):
    out = {}
    for r in _csv.DictReader(open(master_path)):
        try:
            out[(r["method"], r["eval"], r["rung"])] = float(r["prr_mean"])
        except (ValueError, KeyError):
            pass
    return out


def orgad_scores(tok, records, spans, agg):
    """Aggregate token NLL over the extracted spans only, falling back to all tokens when none is found."""
    vals, located = [], 0
    for r in records:
        lp = np.asarray(r["token_logprobs"], dtype=float)
        key = f"{r['split']}:{r['idx']}"
        rows, found = orgad_llm.locate_important_rows(tok, list(r["gen_token_ids"]),
                                                      spans.get(key, "NO ANSWER"))
        # locate_important_rows returns positions in the per-token window [P-1 : P+G], where window row
        # j corresponds to generated token j-1. Row 0 is the last PROMPT position and has no
        # log-probability of its own, so it is dropped rather than shifted onto a generated token.
        sel = [j - 1 for j in rows if 1 <= j <= len(lp)]
        if found and sel:
            located += 1
            use = lp[sel]
        else:
            use = lp
        vals.append(-use.sum() if agg == "sum" else -use.mean())
    return np.array(vals), located / max(len(records), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B", choices=list(MODEL_CFG))
    args = ap.parse_args()

    MODEL = args.model
    SLUG = cache._slug(MODEL)
    cfg = MODEL_CFG[MODEL]
    MASTER = ROOT / "results" / cfg["master_rel"].format(slug=SLUG)
    REGIME = cfg["regime"]
    SAR_SUFFIX = cfg["sar_suffix"]
    ORGAD_SUFFIX = cfg["orgad_suffix"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    master = load_master(MASTER)

    rows_out, gate_fail = [], []
    print(f"population: corrected-span eight-dataset grid, {MODEL}\n")
    print(f"{'dataset':14s}{'n_test':>7s}{'located':>9s}   gate vs master (floors)")
    for d in LONG_SRC:
        cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID", prompt_regime=REGIME[d])
        recs_all = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, d, "ID"))
        field = label_of(d)          # expertqa and factscore do NOT score on `correctness`
        y_all = np.array([r.get(field, np.nan) for r in recs_all], dtype=float)
        keep = np.where(np.isfinite(y_all))[0]                 # FILTERED basis = positions in `keep`
        recs = [recs_all[i] for i in keep]
        y = y_all[keep]
        split = np.array([r["split"] for r in recs])
        _, te = eval_split(split)    # the ladder's fixed carve seed and rule
        te = np.asarray(te)
        recs_te = [recs[i] for i in te]
        y_te = y[te]

        scores = {
            "sequence NLL (published MSP)": np.array([msp_mod.msp_uncertainty(r["token_logprobs"], "sum") for r in recs_te]),
            "mean token NLL": np.array([msp_mod.msp_uncertainty(r["token_logprobs"], "perplexity") for r in recs_te]),
            "minimum token probability": np.array([msp_mod.msp_uncertainty(r["token_logprobs"], "min") for r in recs_te]),
        }

        # TokenSAR: a plain array in ORIGINAL record order, so map filtered -> original explicitly.
        sp = ROOT / "cache" / "sar" / f"{SLUG}__{d}__ID__sentence{SAR_SUFFIX.get(d, '')}.npz"
        if sp.exists():
            z = np.load(sp, allow_pickle=True)
            ts = np.asarray(z["tokensar"], dtype=float)
            if len(ts) != len(recs_all):
                sys.exit(f"FATAL {d}: relevance cache has {len(ts)} rows but the record file has "
                         f"{len(recs_all)} -- refusing to align two different bases by guess.")
            scores["TokenSAR"] = ts[keep[te]]
        else:
            print(f"  {d}: no relevance cache -> TokenSAR left BLANK for this dataset")

        # Answer spans: keyed by the record's own split:idx, so the filtered records index themselves.
        op = ROOT / "cache" / "orgad_llm" / f"{SLUG}__{d}__ID__broad{ORGAD_SUFFIX.get(d, '')}.json"
        loc = float("nan")
        if op.exists():
            spans = json.loads(op.read_text())
            s_sum, loc = orgad_scores(tok, recs_te, spans, "sum")
            s_mean, _ = orgad_scores(tok, recs_te, spans, "mean")
            scores["answer-span sequence NLL"] = s_sum
            scores["answer-span mean NLL"] = s_mean
        else:
            print(f"  {d}: no span cache -> answer-span scores left BLANK for this dataset")

        # GATE. The three unweighted floors must reproduce the master exactly; if they do not, this
        # script is not scoring the master's test cohort and nothing else it computes is comparable.
        line = ""
        for k, disp in FLOOR_KEY.items():
            got = results.prr(y_te, scores[disp])
            exp = master.get((k, d, "ID"))
            ok = exp is not None and abs(got - exp) < 5e-4
            line += f"  {k}:{'ok' if ok else f'MISMATCH {got:+.4f} vs {exp}'}"
            if not ok:
                gate_fail.append((d, k, got, exp))
        print(f"{d:14s}{len(te):7d}{loc:9.2f}  {line}")

        for disp, vec in scores.items():
            rows_out.append({"dataset": d, "method": disp, "prr": round(results.prr(y_te, vec), 6),
                             "n_test": len(te), "located_rate": round(loc, 4) if loc == loc else ""})
        for rg in RUNGS:
            v = master.get(("wmsp_shrink2", d, rg))
            if v is not None:
                rows_out.append({"dataset": d, "method": f"CAWSA lambda=2 [{rg}]", "prr": v,
                                 "n_test": len(te), "located_rate": ""})

    if gate_fail:
        print("\nGATE FAILED -- the recomputed floors do not match the master, so this script is not "
              "scoring the same test cohort. Refusing to write a comparison table.")
        for d, k, got, exp in gate_fail:
            print(f"  {d}/{k}: {got:+.6f} vs master {exp}")
        sys.exit(1)
    print("\nGATE PASS: all three unweighted floors reproduce the master on every dataset.")

    out = ROOT / "results" / "hybrids" / f"likeforlike__{SLUG}.csv"
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["dataset", "method", "prr", "n_test", "located_rate"])
        w.writeheader(); w.writerows(rows_out)
    print(f"wrote {out.relative_to(ROOT)} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
