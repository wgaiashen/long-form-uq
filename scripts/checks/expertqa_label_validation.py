"""Validate the ExpertQA factuality-over-covered label before trusting Role-C (field/cluster
domain-shift OOD). Read-only over the existing expertqa_rp12 labelled records + ExpertQA's native
expert annotations. No regeneration.

CHECK 1 — coverage-by-field(/cluster) symmetry. Role C assumes the ~56% blind spot is SYMMETRIC
across the split, so it cancels. But the split variable is field/cluster, and coverage (= fraction
of output the gold can actually check = 1 - uncovered) may vary by field. If it does, training on
high-coverage groups and testing on low-coverage ones changes what the LABEL MEANS across the split,
confounding any PRR drop. We compute per-group mean coverage and the spread, and PASS/FAIL on it.

Coverage note: `uncovered` was force-set to 0.0 for distrust records (quarantined-SEVERE or
coherent=false), so their coverage is an ARTIFACT, not a measurement — excluded from the coverage
stats. all-uncovered records (uncovered=1.0, coverage 0) ARE real measurements and are kept.

CHECK 2 / 3 are added once the coverage spread is seen (the caller stops after CHECK 1).

    python scripts/checks/expertqa_label_validation.py --stage 1
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, expertqa          # noqa: E402
from luq.config import Config             # noqa: E402

OUT = Path("results/expertqa/label_validation.txt")


def load_labelled_with_group():
    """Return list of dicts joining each labelled record to its field/cluster (positional by idx)."""
    cfg = Config(dataset="expertqa", ood_setting="ID", prompt_regime="expertqa_rp12")
    key = cache.run_key("meta-llama/Meta-Llama-3.1-8B", "expertqa", "ID")
    recs = cache.load_records(cfg.cache_dir, key)
    pool = expertqa.load_records()                       # positional idx alignment (verified: 2016==2016)
    assert len(pool) == max(r["idx"] for r in recs) + 1, "pool/record idx misaligned"
    out = []
    for r in recs:
        src = pool[r["idx"]]
        out.append({**r, "field": src["field"], "cluster": src["cluster"]})
    return out


def is_real_coverage(r):
    """True if `uncovered` is a genuine judge measurement (not a distrust artifact)."""
    if r.get("factuality_quarantined"):               # SEVERE: uncovered force-set 0.0
        return False
    if r.get("coherent") is False:                      # incoherent marginal: uncovered force-set 0.0
        return False
    return r.get("uncovered") is not None


def group_table(rows, keyname, min_n=30):
    """Per-group mean coverage (=1-uncovered) + n, sorted by mean. Returns (lines, spread_bign)."""
    by = defaultdict(list)
    for r in rows:
        by[r[keyname]].append(1.0 - r["uncovered"])     # coverage fraction
    stats = []
    for g, cov in by.items():
        stats.append((g, float(np.mean(cov)), float(np.std(cov)), len(cov)))
    stats.sort(key=lambda t: t[1])
    lines = [f"  {'group':<26} {'n':>5} {'mean_cov':>9} {'std':>7}"]
    for g, m, s, n in stats:
        flag = "" if n >= min_n else "  (small-n, excluded from spread)"
        lines.append(f"  {g:<26} {n:>5} {m:>9.3f} {s:>7.3f}{flag}")
    big = [(g, m, n) for g, m, s, n in stats if n >= min_n]
    if big:
        lo = min(big, key=lambda t: t[1])
        hi = max(big, key=lambda t: t[1])
        spread = hi[1] - lo[1]
        lines.append(f"  spread over groups with n>={min_n}: "
                     f"{lo[1]:.3f} ({lo[0]}) .. {hi[1]:.3f} ({hi[0]})  =>  {spread:.3f}")
        return lines, spread, (lo, hi)
    return lines, float("nan"), None


def check1():
    rows = load_labelled_with_group()
    real = [r for r in rows if is_real_coverage(r)]
    n_art = len(rows) - len(real)
    header = [
        "=" * 78,
        "CHECK 1 — coverage-by-field/cluster symmetry (is Role C confounded?)",
        "=" * 78,
        f"records: {len(rows)} | usable coverage measurements: {len(real)} "
        f"(excluded {n_art} distrust-artifact uncovered=0.0)",
        f"overall mean coverage (1-uncovered): {np.mean([1-r['uncovered'] for r in real]):.3f}",
        "",
        "Role C splits on CLUSTER (6 groups); raw FIELD (32) reported too but many fields have <30 n.",
        "",
        "--- by CLUSTER (the actual Role-C split unit) ---",
    ]
    cl_lines, cl_spread, cl_ext = group_table(real, "cluster")
    fl_header = ["", "--- by FIELD (32 raw fields; the check as literally specified) ---"]
    fl_lines, fl_spread, fl_ext = group_table(real, "field")

    # Verdict on the split-relevant unit (cluster). Tight band ~<=0.12 (the user's 50-60% example is
    # a 10pt band) = symmetry holds. Wide (>~0.20, e.g. 30-70) = confounded.
    def verdict(spread):
        if spread != spread:                            # nan
            return "INCONCLUSIVE (no group met min-n)"
        if spread <= 0.12:
            return f"PASS (spread {spread:.3f} <= 0.12: coverage ~symmetric, Role C clean on this axis)"
        if spread <= 0.20:
            return f"MARGINAL (spread {spread:.3f} in 0.12-0.20: mild coverage drift, note it)"
        return f"FAIL (spread {spread:.3f} > 0.20: coverage varies by group => Role C inherits a label-meaning confound)"

    tail = [
        "",
        "-" * 78,
        f"VERDICT (cluster = the Role-C split unit): {verdict(cl_spread)}",
        f"   (field-level spread for reference: {fl_spread:.3f} -> {verdict(fl_spread)})",
        "-" * 78,
    ]
    lines = header + cl_lines + fl_header + fl_lines + tail
    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n\n[CHECK 2 and 3 pending — held for review of the coverage spread]\n")
    print(text)


# ---- CHECK 2 & 3: native-annotation validity/replacement ------------------------------------

CORR = {"Definitely correct": 1.0, "Probably correct": 0.75, "Unsure": 0.5,
        "Likely incorrect": 0.25, "Definitely incorrect": 0.0}          # expert per-claim factuality
SUP = {"Complete": 1.0, "Partial": 0.5, "Incomplete": 0.5, "Missing": 0.0, "None": 0.0}  # attribution


def load_native_aligned():
    """Rebuild the SAME factual-core filtered pool but keep each answer's native annotation object,
    so raw_ann[idx] joins to our record idx (verified: 0 positional question mismatches)."""
    import json
    out = []
    for line in open(expertqa.DEFAULT_JSONL, encoding="utf-8"):
        rec = json.loads(line)
        a = next(iter(rec["answers"].values()))
        gold = (a.get("revised_answer_string") or "").strip()
        if not gold:
            continue
        types = {t.strip() for t in rec["metadata"]["question_type"].split("|") if t.strip()}
        if not (types & expertqa.FACTUAL_CORE):
            continue
        claims = a.get("claims", [])
        cn = [CORR[c["correctness"]] for c in claims if c.get("correctness") in CORR]
        sn = [SUP[c["support"]] for c in claims if c.get("support") in SUP]
        out.append({"question": rec["question"], "answer_string": (a.get("answer_string") or "").strip(),
                    "gold": gold, "claims": claims,
                    "corr": float(np.mean(cn)) if cn else None,
                    "sup": float(np.mean(sn)) if sn else None})
    return out


def check23():
    rows = load_labelled_with_group()
    native = load_native_aligned()
    pool = expertqa.load_records()
    assert len(native) == len(rows) == len(pool), "native/record/pool count mismatch"
    # Join-key soundness (the check asked to confirm no collisions). The authoritative join is
    # positional idx: native[idx] and pool[idx] are the SAME filtered stream, so their questions
    # must match 1:1. (Not parsed from r['prompt'] — the few-shot prompt reformats the question.)
    mism = sum(1 for i in range(len(pool)) if native[i]["question"] != pool[i]["question"])

    # Which TEXT do the native annotations describe? claim_strings drawn from answer_string vs our gen.
    def claim_hit(claims, text):
        if not claims:
            return None
        return sum(1 for c in claims if c.get("claim_string", "")[:60] and
                   c["claim_string"][:60] in text) / len(claims)
    from_ans, from_gen, gen_eq_ans = [], [], 0
    for r in rows:
        a = native[r["idx"]]
        h_a = claim_hit(a["claims"], a["answer_string"])
        h_g = claim_hit(a["claims"], r["gen_text"])
        if h_a is not None:
            from_ans.append(h_a)
        if h_g is not None:
            from_gen.append(h_g)
        gen_eq_ans += (r["gen_text"].strip() == a["answer_string"])

    corr = [a["corr"] for a in native if a["corr"] is not None]
    sup = [a["sup"] for a in native if a["sup"] is not None]

    L = [
        "", "=" * 78,
        "CHECK 2 — validate our label against ExpertQA's native expert annotations",
        "=" * 78,
        f"join key soundness: {len(native)} native rows, positional idx join, "
        f"{mism} question mismatches (0 = clean 1:1 join).",
        "",
        "WHICH TEXT DO THE NATIVE ANNOTATIONS DESCRIBE? (decides if a join is even meaningful)",
        f"  native claim_strings found in answer_string (original annotated answer): mean {np.mean(from_ans):.2f}",
        f"  native claim_strings found in OUR Llama generation:                      mean {np.mean(from_gen):.2f}",
        f"  our gen == answer_string: {gen_eq_ans}/{len(rows)}",
        "",
        "  => The native expert labels annotate `answer_string` (the ORIGINAL system's answer,",
        "     e.g. BingChat/GPT-4), NOT our Llama generation. A per-instance join of our",
        "     factuality label to them compares labels of DIFFERENT texts that share only a",
        "     question -> uninterpretable, so it is NOT run. Same reason blocks 'AlignScore vs",
        "     expert' as a direct join (AlignScore is on our gen, expert is on the original answer).",
        "",
        "  VALID alternative (not yet run, costs judge calls): run OUR three-state judge on",
        "  `answer_string` vs the same gold+evidence, then correlate with the native expert",
        "  correctness/support. Same text on both sides -> a real test of the JUDGE (labeller),",
        "  which transfers to our Llama labels (same judge+prompt). Caveat: validates the judge on",
        "  GPT-4/BingChat answers, a different distribution from Llama outputs.",
        "",
        "=" * 78,
        "CHECK 3 — can native gold REPLACE our factuality-over-covered label?",
        "=" * 78,
        f"native instance-level CORRECTNESS (expert factuality, mean over ~5.8 claims/answer):",
        f"  available for {len(corr)}/{len(native)} instances ({len(corr)/len(native):.0%})  "
        f"mean {np.mean(corr):.3f}  median {np.median(corr):.3f}",
        f"native instance-level SUPPORT (attribution): {len(sup)}/{len(native)} ({len(sup)/len(native):.0%})  "
        f"mean {np.mean(sup):.3f}",
        "  native labels cover the WHOLE original answer (every claim) -> NO 56/64% blind spot.",
        "",
        "  VERDICT: NO. The native gold is a whole-answer factuality label (no blind spot), but it",
        "  describes the ORIGINAL answer, not our Llama generation. Our probe reads Llama's hidden",
        "  states, so the label MUST describe Llama's output. Native gold therefore CANNOT be the",
        "  probe label; it can only (a) validate our JUDGE via the CHECK-2 re-judge of answer_string,",
        "  or (b) report reference quality. (Confirms the plan's assertion, now verified on the data.)",
        "-" * 78,
    ]
    text = "\n".join(L)
    with open(OUT, "a") as f:
        f.write(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    args = ap.parse_args()
    if args.stage == 1:
        check1()
    elif args.stage == 23:
        check23()
    else:
        raise SystemExit("stage must be 1 or 23")


if __name__ == "__main__":
    main()
