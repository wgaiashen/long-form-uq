#!/usr/bin/env python
"""§3 of the Qwen validity audit: the DETERMINISTIC stratified spot-check sample.

WHY DETERMINISTIC. The brief is explicit: "Do not hand-select examples based on which outputs or UQ
results look interesting." Every stratum here is defined by a rule fixed in advance, ties break on a
stable hash of (dataset, idx), and re-running produces a byte-identical CSV. Nothing is chosen
because it looked bad — the point of the audit is to find out whether things look bad, which a
hand-picked sample cannot answer.

THE STRATA (per the brief), ~40 per dataset and ~60 for expertqa/factscore, which have the most
complicated generation and labelling history:

    length/cap   5 shortest · 5 nearest the median · 5 longest NON-capped · 5 capped
    label        5 lowest quality quartile · 5 highest quality quartile
    msp-disagree 5 where msp_min ranks it far MORE uncertain than perplexity
                 5 where perplexity ranks it far more uncertain than msp_min
    (expertqa/factscore also get an extra 10 flagged + 10 clean, since the whole question there is
     whether the flagged and clean populations differ)

Duplicate ids are removed BETWEEN strata, keeping the first stratum that claimed them, and the
`selection_strata` column records every stratum an example qualified for — so a row that appears
once is not silently hiding that it was also, say, both capped and low-label. Short strata fall back
to deterministic hash order rather than to nothing.

    python scripts/tools/qwen_audit_sample.py
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, degeneracy, msp                    # noqa: E402
from luq.config import Config                             # noqa: E402
from attn_pool import PROMPT_REGIME                       # noqa: E402
from xl_rungs import label_of                             # noqa: E402

QWEN = "Qwen/Qwen2.5-14B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
BUDGET = {"pubmed_qa": 128, "med_quad": 128, "asqa": 256, "xsum": 56,
          "cnn_dailymail": 128, "samsum": 56, "expertqa": 384, "factscore": 256}
DEEP = {"expertqa", "factscore"}                          # the two with the messiest history
BLEED = re.compile(
    r"^\s*(?:Question:|Answer:|Available choices:|\(\d+\)\.|Summary:|Article:|Story:|Abstract:"
    r"|Text:|Document:|Dialogue:|Conversation:|A single-select problem|Is the question answered)",
    re.MULTILINE)
K = 5                                                     # per stratum, as the brief specifies


def h(dataset, idx):
    """Stable tiebreak/fallback order. Not Python's hash() -- that is salted per process and would
    make the 'deterministic' sample differ between runs, which is the whole thing this avoids."""
    return int(hashlib.sha256(f"{dataset}:{idx}".encode()).hexdigest()[:12], 16)


def load(model, dataset):
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    p = Path(cfg.cache_dir) / "records" / f"{cache.run_key(model, dataset, 'ID')}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def take(order, k, taken):
    """First k ids from `order` not already taken. Deterministic by construction."""
    out = []
    for i in order:
        if i in taken:
            continue
        out.append(i); taken.add(i)
        if len(out) == k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN)
    ap.add_argument("--out", default="results/analysis/qwen_generation_audit_sample.csv")
    args = ap.parse_args()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'dataset':15s}{'n':>7s}{'sampled':>9s}   strata coverage")
    for d in LONG:
        recs = load(args.model, d)
        if recs is None:
            print(f"{d:15s}  RECORDS MISSING -- reported, not skipped silently")
            continue
        bud = BUDGET[d]
        lf = label_of(d)
        idx = np.array([r.get("idx", i) for i, r in enumerate(recs)])
        n_gen = np.array([len(r["gen_token_ids"]) for r in recs])
        y = np.array([r.get(lf, np.nan) for r in recs], float)
        capped = n_gen >= bud
        u_min = np.array([msp.msp_uncertainty(r["token_logprobs"], "min") for r in recs])
        u_ppl = np.array([msp.msp_uncertainty(r["token_logprobs"], "perplexity") for r in recs])
        # rank-space disagreement: both are "higher = more uncertain", so the difference of ranks
        # is the only scale-free way to say "these two orderings disagree about THIS example".
        r_min = np.argsort(np.argsort(u_min)); r_ppl = np.argsort(np.argsort(u_ppl))
        dis = r_min - r_ppl
        flagged = np.array([degeneracy.is_severe(r.get("gen_text", "") or "")
                            or bool(BLEED.search(r.get("gen_text", "") or "")) for r in recs])
        tie = np.array([h(d, i) for i in idx])

        def order_by(key, desc=False):
            # lexsort: primary key, hash as the deterministic tiebreak
            o = np.lexsort((tie, -key if desc else key))
            return list(idx[o])

        taken, strata = set(), {}
        med = np.median(n_gen)
        picks = [
            ("shortest",        order_by(n_gen)),
            ("median_length",   order_by(np.abs(n_gen - med))),
            ("longest_uncapped", order_by(np.where(capped, -1, n_gen), desc=True)),
            ("capped",          order_by(np.where(capped, 0, 1))),
            ("label_low_q1",    order_by(np.where(np.isfinite(y), y, np.inf))),
            ("label_high_q4",   order_by(np.where(np.isfinite(y), y, -np.inf), desc=True)),
            ("mspmin_gt_ppl",   order_by(dis, desc=True)),
            ("ppl_gt_mspmin",   order_by(dis)),
        ]
        if d in DEEP:
            picks += [("flagged", order_by(np.where(flagged, 0, 1))),
                      ("clean",   order_by(np.where(flagged, 1, 0)))]
        k_deep = 10 if d in DEEP else K
        for name, order in picks:
            k = k_deep if name in ("flagged", "clean") else K
            for i in take(order, k, taken):
                strata.setdefault(i, []).append(name)
        # record EVERY stratum an id qualifies for, not just the one that claimed it -- otherwise a
        # capped low-label example looks like a plain capped one.
        pos = {v: p for p, v in enumerate(idx)}
        for i in sorted(strata, key=lambda z: h(d, z)):
            p = pos[i]
            also = []
            if capped[p]:
                also.append("is_capped")
            if flagged[p]:
                also.append("is_flagged")
            if not np.isfinite(y[p]):
                also.append("judge_declined")
            rows.append({"model": args.model, "dataset": d, "example_id": int(i),
                         "selection_strata": "|".join(strata[i]),
                         "also": "|".join(also), "n_gen": int(n_gen[p]), "budget": bud,
                         "capped": int(capped[p]), "flagged": int(flagged[p]),
                         "label": "" if not np.isfinite(y[p]) else round(float(y[p]), 4),
                         "label_field": lf})
        got = sum(1 for r in rows if r["dataset"] == d)
        print(f"{d:15s}{len(recs):>7d}{got:>9d}   {len(picks)} strata x {K}"
              f"{' (+10 flagged/clean)' if d in DEEP else ''}")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")
    print("Re-running this produces a byte-identical file: every order is a lexsort with a sha256")
    print("tiebreak on (dataset, idx), and nothing depends on Python's salted hash() or on time.")


if __name__ == "__main__":
    main()
