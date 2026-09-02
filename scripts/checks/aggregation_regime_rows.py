#!/usr/bin/env python
"""A2 — one row per scored TEST response: NLL-shape / concentration features for the
aggregation-regime audit. DESCRIPTIVE AUDIT, not a feature sweep: the feature set below is
predeclared (PLAN, Workstream A2) and no alternative definitions are added after correlations
are seen.

Records-only (token_logprobs + gen_text + judge label): no GPU, no per-token hidden states, no
training. Model-agnostic via --model (Llama by default; the same script builds the Qwen table
from its synced records — the caches stay wherever they are).

INTEGRATED A1 GATES (fail loud BEFORE any row is written):
  G-align  len(gen_token_ids) == len(token_logprobs) on every record read;
  G-b0     PRR(Lehmer beta=0) == PRR(perplexity ranking) to 1e-9 per dataset;
  G-binf   PRR(Lehmer beta->inf) == PRR(msp_min ranking) to 1e-9 per dataset;
  G-master PRR(perplexity_score) and PRR(msp_min_score) recomputed FROM THE TABLE'S OWN ROWS
           must reproduce the published master's floor cells to 4 dp (the master publishes 4 dp).

Loading mirrors lehmer_qwen.load_light's traps verbatim (explicit model pin + slug assert,
namespaced cache resolution via Config(prompt_regime), the finite-label filter, and the
DECLINED-vs-UNJUDGED guard) — duplicated rather than imported because this loader must also
return the records themselves (gen_text, token ids) which load_light deliberately does not.

EXPLICIT EDGE-CASE POLICY (predeclared, recorded here once):
  * token set = ALL generated tokens, raw logprobs, no special-token mask — the SAME policy as
    the canonical floors (special_token_audit.py: floors INCLUDE specials/EOS). wMSP's masked
    policy is a documented confound handled in the A6 matched-mask sensitivity, not here.
  * T == 1: second_max_nll, max_minus_second, nll_entropy_norm are NaN (undefined; log(T)=0).
  * T < 5: top5_mass_share uses the min(5, T) largest values (== 1.0 when T <= 5); the
    n_tokens column makes these rows identifiable, no separate flag needed.
  * sum(nll) == 0 (fully deterministic generation): all mass shares + entropy are NaN, never 0.
  * std == 0: max_z uses eps = 1e-12 in the denominator (plan definition).
  * claim_count is OMITTED ENTIRELY: records persist only aggregate ratios (factuality,
    uncovered), no per-claim states or counts exist anywhere (verified 2026-08-10), and a blank
    column invites silent zero-fill. sentence_count (regex [.!?]+ boundary) is the only proxy.
  * wmsp_score is OMITTED: per-example wMSP predictions are not persisted (pdl_perex sidecars
    carry floors+SAPLMA only); A6 trains the canonical normalised wMSP itself and reports
    weight-vs-NLL agreement separately.

    python scripts/checks/aggregation_regime_rows.py                       # Llama
    python scripts/checks/aggregation_regime_rows.py --model Qwen/Qwen2.5-14B
    qsub -v LUQ_CMD="scripts/checks/aggregation_regime_rows.py" pbs/audit_cpu.pbs
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache, msp, results                      # noqa: E402
from luq.config import Config                            # noqa: E402
from luq.data import MAX_NEW_TOKENS                      # noqa: E402
from xl_rungs import eval_split, label_of                # noqa: E402
from attn_pool import PROMPT_REGIME                      # noqa: E402
from sharpening_family import score_lehmer               # noqa: E402  (THE registered scorer)
from lehmer_qwen import JUDGE_SIBLINGS, MAX_UNJUDGED_FRAC  # noqa: E402  (same guard thresholds)

MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
BETAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
GATE_TOL = 1e-9
MASTER_TOL = 5e-5                                        # masters publish 4 dp
EPS = 1e-12
SENT_RE = re.compile(r"[.!?]+(?:\s|$)")

MASTER = {
    "meta-llama/Meta-Llama-3.1-8B": (ROOT / "results" / "pdl_master__meta-llama_Meta-Llama-3.1-8B.csv",
                                     {"min": "msp_min", "ppl": "perplexity"}),
    "Qwen/Qwen2.5-14B": (ROOT / "results" / "pdl_master__Qwen_Qwen2.5-14B.csv",
                         {"min": "floor_min", "ppl": "floor_ppl"}),
}

# REGIME-AWARE GATE REFERENCE. MASTER above is the ORIGINAL-span reference. When a dataset is
# redirected to a corrected-span cache root (LUQ_REGIME="med_quad=cleanv2"), its floors legitimately
# differ, and checking them against the original master would be a false failure -- the situation
# sharpening_family.py already handles with PUBLISHED_CLEAN_MEDQUAD.
#
# THE GATE IS NOT WEAKENED, AND DOING IT PER DATASET IS THE POINT. Only redirected datasets get the
# corrected reference; every dataset still reading its original cache is still checked against the
# original master. That residual check IS the invariance test: a dataset whose population did not
# change must still reproduce the number it always produced.
CORRECTED_MASTER = {
    "meta-llama/Meta-Llama-3.1-8B": (
        ROOT / "results" / "cleanv2" / "pdl_cleanv2_master__meta-llama_Meta-Llama-3.1-8B.csv",
        {"min": "floor_min", "ppl": "floor_ppl"}),
}
CORRECTED_REGIMES = {"cleanv2"}


def load_test_records(model, dataset):
    """TEST-row records + labels, with lehmer_qwen.load_light's traps (see module docstring)."""
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    key = cache.run_key(model, dataset, "ID")
    path = Path(cfg.cache_dir) / "records" / f"{key}.jsonl"
    slug = cache._slug(model)
    if not path.exists():
        raise SystemExit(f"FAIL [{dataset}]: no record file at {path}")
    if slug not in path.name:
        raise SystemExit(f"FAIL [{dataset}]: {path.name} does not carry slug {slug}")
    records = cache.load_records(cfg.cache_dir, key)
    lf = label_of(dataset)
    y = np.array([r.get(lf, np.nan) for r in records], dtype=float)
    finite = np.isfinite(y)
    touched = np.array([(lf in r) or any(s in r for s in JUDGE_SIBLINGS) for r in records])
    if (~touched).sum() / max(len(y), 1) > MAX_UNJUDGED_FRAC:
        raise SystemExit(f"LABELLING INCOMPLETE [{dataset}]: {(~touched).sum()}/{len(y)} unjudged")
    keep = np.where(finite)[0]
    records = [records[k] for k in keep]
    split = np.array([r["split"] for r in records])
    _, te = eval_split(split)
    if len(te) == 0:
        raise SystemExit(f"FAIL [{dataset}]: empty test split")
    return [records[i] for i in te], y[keep][te], lf


def features(nll, gen_text, dataset):
    """The predeclared A2 feature dict for one response. nll: 1-D array, all generated tokens."""
    T = len(nll)
    s = float(nll.sum())
    srt = np.sort(nll)[::-1]
    mx = float(srt[0])
    second = float(srt[1]) if T > 1 else np.nan
    mean, std, med = float(nll.mean()), float(nll.std()), float(np.median(nll))
    q75, q90, q95, q99 = (float(np.percentile(nll, q)) for q in (75, 90, 95, 99))
    if s > 0:
        top1 = mx / s
        top5 = float(srt[:min(5, T)].sum()) / s
        k10 = int(np.ceil(0.10 * T))
        top10p = float(srt[:k10].sum()) / s
        p = nll / s
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        ent_norm = ent / np.log(T) if T > 1 else np.nan
    else:
        top1 = top5 = top10p = ent_norm = np.nan
    return {
        "n_tokens": T,
        "hit_generation_cap": int(T >= MAX_NEW_TOKENS[dataset]),
        "mean_nll": mean, "std_nll": std, "median_nll": med,
        "max_nll": mx, "second_max_nll": second,
        "q75_nll": q75, "q90_nll": q90, "q95_nll": q95, "q99_nll": q99,
        "top1_mass_share": top1, "top5_mass_share": top5, "top10pct_mass_share": top10p,
        "nll_entropy_norm": ent_norm,
        "max_minus_second": mx - second if T > 1 else np.nan,
        "max_z": (mx - mean) / (std + EPS),
        "sentence_count": len(SENT_RE.findall(gen_text)) or (1 if gen_text.strip() else 0),
    }


def _read_master_floors(path, nm, prr_col, seed_regime=None):
    """(dataset -> {min: prr, ppl: prr}) from one master CSV's matched-setting rows."""
    out = {}
    for r in csv.DictReader(open(path)):
        if r["rung"] != "ID":
            continue
        if seed_regime is not None and r.get("seed_regime") != seed_regime:
            continue
        for k, name in nm.items():
            if r["method"] == name:
                out.setdefault(r["eval"], {})[k] = float(r[prr_col])
    return out


def load_master_floors(model):
    """Gate reference per dataset: original master, overlaid by the corrected one where redirected."""
    path, nm = MASTER[model]
    if model.startswith("meta-llama"):
        out = _read_master_floors(path, nm, "prr", seed_regime="3seed")
    else:
        out = _read_master_floors(path, nm, "prr_mean")

    redirected = [d for d, rg in PROMPT_REGIME.items() if rg in CORRECTED_REGIMES]
    if redirected:
        if model not in CORRECTED_MASTER:
            raise SystemExit(f"{model} has datasets redirected to a corrected-span cache "
                             f"({redirected}) but no corrected-span master is registered. Refusing "
                             "to gate corrected data against the original reference.")
        cpath, cnm = CORRECTED_MASTER[model]
        corrected = _read_master_floors(cpath, cnm, "prr_mean")
        for d in redirected:
            if d not in corrected:
                raise SystemExit(f"{d} is redirected to a corrected-span cache but the corrected "
                                 f"master {cpath.name} has no matched-setting floors for it.")
            out[d] = corrected[d]
            print(f"[gate reference] {d}: corrected-span master {cpath.name}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--out", default=None,
                    help="output CSV path. Give one when running under a cache-root override, so a "
                         "corrected-span table can never overwrite the original-span one.")
    args = ap.parse_args()
    slug = cache._slug(args.model)
    out_csv = (Path(args.out) if args.out
               else ROOT / "results" / "analysis" / f"aggregation_regime_rows__{slug}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    master_floors = load_master_floors(args.model)
    feat_names = None
    all_rows = []
    print("=" * 100)
    print(f"A2 AGGREGATION-REGIME ROWS  model={args.model}")
    print("Token set = ALL generated tokens, raw logprobs (canonical floor policy, no mask).")
    print("=" * 100)
    for d in LONG:
        recs, y, lf = load_test_records(args.model, d)
        nlls = []
        for r in recs:
            # G-align: the cache's own invariant, asserted on every record read
            if len(r["gen_token_ids"]) != len(r["token_logprobs"]):
                raise SystemExit(f"G-align FAIL [{d}] idx={r['idx']}: "
                                 f"{len(r['gen_token_ids'])} ids vs {len(r['token_logprobs'])} logprobs")
            nlls.append(-np.asarray(r["token_logprobs"], dtype=float))

        # scores per response
        v_min = np.array([msp.msp_uncertainty(-a, "min") for a in nlls])
        v_ppl = np.array([msp.msp_uncertainty(-a, "perplexity") for a in nlls])
        lehmer = {b: np.array([score_lehmer(a, b) for a in nlls]) for b in BETAS}

        # G-b0 / G-binf: endpoint identities (rankings, so PRR-identical)
        p_min, p_ppl = results.prr(y, v_min), results.prr(y, v_ppl)
        p_b0 = results.prr(y, np.array([score_lehmer(a, 0.0) for a in nlls]))
        p_binf = results.prr(y, np.array([score_lehmer(a, np.inf) for a in nlls]))
        if abs(p_b0 - p_ppl) > GATE_TOL or abs(p_binf - p_min) > GATE_TOL:
            raise SystemExit(f"G-b0/G-binf FAIL [{d}]: {abs(p_b0-p_ppl):.1e} / {abs(p_binf-p_min):.1e}")

        # G-master: the table's own floor scores must reproduce the published master cells
        mf = master_floors.get(d)
        if mf is None or "min" not in mf or "ppl" not in mf:
            raise SystemExit(f"G-master FAIL [{d}]: floor cells missing from the master CSV")
        d_min, d_ppl = abs(p_min - mf["min"]), abs(p_ppl - mf["ppl"])
        ok = d_min < MASTER_TOL and d_ppl < MASTER_TOL
        print(f"[{d:14s}] n={len(y):5d}  msp_min {p_min:+.4f} (master {mf['min']:+.4f}, d {d_min:.1e})  "
              f"perplexity {p_ppl:+.4f} (master {mf['ppl']:+.4f}, d {d_ppl:.1e})  "
              f"endpoints PASS  master {'PASS' if ok else 'FAIL <=='}", flush=True)
        if not ok:
            raise SystemExit(f"G-master FAIL [{d}]: table floors do not reproduce the published "
                             f"master — population or convention drift; nothing downstream is safe")

        for i, (r, a) in enumerate(zip(recs, nlls)):
            f = features(a, r.get("gen_text", ""), d)
            row = {"eval": d, "example_id": r["idx"], "quality_label": float(y[i]),
                   "label_field": lf, **f,
                   "msp_min_score": float(v_min[i]), "perplexity_score": float(v_ppl[i])}
            for b in BETAS:
                row[f"lehmer_beta_{b:g}"] = float(lehmer[b][i])
            if feat_names is None:
                feat_names = list(row.keys())
            all_rows.append(row)

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=feat_names)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: ("" if isinstance(v, float) and not np.isfinite(v) else v)
                        for k, v in row.items()})
    print(f"\nALL GATES PASS. wrote {out_csv}  ({len(all_rows)} rows)")
    print("claim_count: UNAVAILABLE on this project (records persist aggregate ratios only) — "
          "the A4.4 claim-count stratification is therefore marked unavailable, by design.")


if __name__ == "__main__":
    main()
