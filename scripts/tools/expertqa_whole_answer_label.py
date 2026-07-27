"""Whole-answer factuality label for ExpertQA — evaluate a proposed UPGRADE of the current
covered-claims label. Instead of factuality over only the ~36% of Llama's answer the gold can
adjudicate (three-state judge, abstains on UNCOVERED), score Llama's WHOLE answer against the
expert-revised reference (xsum-style consistency), so there is ~0 blind spot.

FRAMING (Gaia's, kept honest): this is FAITHFULNESS to an expert reference, NOT factuality. It does
not turn ExpertQA into a factuality task; it makes the existing factuality label whole-answer
instead of 36%-visible. Reuse the mini judge (r=0.86 vs gpt-5). No regeneration.

Claude's reservation (see worklog 2026-07-07): the blind spot is a real property (Llama answers
open expert Qs more broadly than one expert's ~200-word answer), so a MATCH-based whole-answer judge
may penalise correct-but-different answers — the reference-coverage trap the three-state design (and
the AlignScore r=0.19 result) flagged. So STEP 3 adds a BREADTH-BIAS diagnostic, not just Spearman.

STEP 1 (gate, stop): reference coverage + provenance + 5 eyeball pairs.
STEP 2 (after OK): run the whole-answer judge -> label_faithful_to_revised.
STEP 3: compare vs covered-claims label + the breadth-bias check.

    python scripts/tools/expertqa_whole_answer_label.py --step 1
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, expertqa          # noqa: E402
from luq.config import Config             # noqa: E402

OUT = Path("results/expertqa/whole_answer_label.txt")


def load_source_aligned():
    """raw_src[idx] joins to record idx (positional over the SAME factual-core filtered stream).
    Keeps the reference (prefer revised_answer_string, fall back to answer_string) + provenance."""
    src = []
    for line in open(expertqa.DEFAULT_JSONL, encoding="utf-8"):
        rec = json.loads(line)
        model, a = next(iter(rec["answers"].items()))       # exactly one answer per record (verified)
        revised = (a.get("revised_answer_string") or "").strip()
        raw = (a.get("answer_string") or "").strip()
        if not revised:
            continue                                         # dropped from our pool
        types = {t.strip() for t in rec["metadata"]["question_type"].split("|") if t.strip()}
        if not (types & expertqa.FACTUAL_CORE):
            continue
        ref = revised if revised else raw
        src.append({"question": rec["question"], "model": model,
                    "reference": ref, "used_revised": bool(revised), "raw_only": (not revised and bool(raw))})
    return src


def load_records():
    cfg = Config(dataset="expertqa", ood_setting="ID", prompt_regime="expertqa_rp12")
    key = cache.run_key("meta-llama/Meta-Llama-3.1-8B", "expertqa", "ID")
    return cache.load_records(cfg.cache_dir, key)


def step1():
    recs = load_records()
    src = load_source_aligned()
    pool = expertqa.load_records()
    assert len(src) == len(recs) == len(pool), "src/record/pool count mismatch"
    # Join-key soundness: positional idx over the same filtered stream; questions must match 1:1
    # (there is no separate 'split' dimension — ExpertQA here is one ID pool).
    collisions = sum(1 for i in range(len(pool)) if src[i]["question"] != pool[i]["question"])

    n_rev = sum(1 for s in src if s["used_revised"])
    n_raw = sum(1 for s in src if s["raw_only"])
    n_none = sum(1 for s in src if not s["reference"])
    models = Counter(s["model"] for s in src)

    L = [
        "=" * 78,
        "STEP 1 — reference coverage gate (whole-answer factuality label)",
        "=" * 78,
        f"records: {len(recs)} | join: positional idx (no 'split' dim, all ID), "
        f"{collisions} question mismatches (0 = clean 1:1, no key collisions).",
        "",
        "REFERENCE COVERAGE (per Llama gen):",
        f"  (a) non-empty REVISED expert answer : {n_rev} ({n_rev/len(src):.0%})",
        f"  (b) only raw answer_string fallback : {n_raw} ({n_raw/len(src):.0%})",
        f"  (c) neither (no reference)          : {n_none} ({n_none/len(src):.0%})",
        "  => every instance has a real expert-REVISED full answer; no sparse-revision risk.",
        "",
        "REFERENCE PROVENANCE (original system, all expert-revised):",
    ] + [f"  {m:<22} {c:>5} ({c/len(src):.0%})" for m, c in models.most_common()]

    L += ["", "5 EXAMPLE (Llama answer  |  expert reference) PAIRS — eyeball the reference is sensible:"]
    for i in (0, 400, 800, 1200, 1600):
        r, s = recs[i], src[i]
        L += [
            "-" * 78,
            f"[idx {i}]  Q: {s['question'][:100]}",
            f"  LLAMA ({len(r['gen_text'])} ch): {r['gen_text'].strip()[:280].replace(chr(10),' ')}",
            f"  REF   ({len(s['reference'])} ch, {s['model']}{', revised' if s['used_revised'] else ', RAW'}): "
            f"{s['reference'][:280].replace(chr(10),' ')}",
        ]
    L += ["-" * 78,
          "GATE: PASS (100% revised references). Decision to Gaia before STEP 2 — the reference is",
          "solid; the open question is whether whole-answer MATCH penalises correct-but-different",
          "Llama answers (STEP 3 breadth-bias check), not whether the reference exists."]
    text = "\n".join(L)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n\n[STEP 2 & 3 pending — held for OK after the coverage gate]\n")
    print(text)


# ---- STEP 2: whole-answer MATCH judge on a stratified ~200 sample ---------------------------

SAMPLE_JSON = Path("results/expertqa/whole_answer_sample.json")

# Faithful adaptation of llm_judge._summary_prompt (the xsum MATCH prompt): summarisation->QA,
# Summary->Answer, + the Question. Deliberately the match wording (the trap-prone one we test).
WHOLE_ANSWER_PROMPT = """
Only respond with a number between 0.0 to 1.0. Do not write any explanation.

The task below is a question-answering task. You will see a Question, a Reference Answer written by an expert, and a Model Answer. Score how well the Model Answer matches the Reference Answer from 0.0 to 1.0, where 1.0 means it conveys the same information, 0.0 means it is completely different or irrelevant, and scores in between reflect partial overlap in the key points covered.

Question: {q}
Reference Answer: {ref}
Model Answer: {ans}
Score:
"""


def category(r):
    if r.get("factuality_quarantined") or r.get("coherent") is False:
        return "distrust"
    if r.get("factuality") is None:
        return "all_uncovered"
    return "covered"


def build_sample(recs):
    """Deterministic stratified ~200: 100 covered (25/quartile of old uncovered) + 60 all_uncovered
    + 40 distrust. seed=1 (project convention). Returns a list of idx."""
    import numpy as np
    rng = np.random.RandomState(1)
    by = {"covered": [], "all_uncovered": [], "distrust": []}
    for i, r in enumerate(recs):
        by[category(r)].append(i)

    def take(pool, n):
        pool = list(pool)
        rng.shuffle(pool)
        return pool[:n]

    # covered: stratify across old-uncovered quartiles for breadth-bias range
    cov = by["covered"]
    u = np.array([recs[i]["uncovered"] for i in cov])
    edges = np.percentile(u, [0, 25, 50, 75, 100])
    chosen = []
    for lo, hi, k in zip(edges[:-1], edges[1:], [25, 25, 25, 25]):
        band = [i for i in cov if lo <= recs[i]["uncovered"] <= hi]
        chosen += take(band, k)
    chosen = list(dict.fromkeys(chosen))            # dedup band overlaps at the edges
    sample = chosen + take(by["all_uncovered"], 60) + take(by["distrust"], 40)
    return sorted(set(sample))


def step2(judge_model="gpt-5-mini"):
    from luq.labels import llm_judge
    recs = load_records()
    src = load_source_aligned()
    idxs = build_sample(recs)
    print(f"STEP 2: whole-answer MATCH judge ({judge_model}) on {len(idxs)} sampled instances", flush=True)

    out = json.loads(SAMPLE_JSON.read_text()) if SAMPLE_JSON.exists() else {}
    n_new = 0
    for j, i in enumerate(idxs):
        if str(i) in out:                            # resume
            continue
        r, s = recs[i], src[i]
        prompt = WHOLE_ANSWER_PROMPT.format(q=s["question"], ref=s["reference"], ans=r["gen_text"].strip())
        score = llm_judge.parse_score(llm_judge._gpt_response(prompt, judge_model))
        out[str(i)] = {
            "idx": i, "category": category(r),
            "new_label": score,                      # whole-answer match factuality (None if unparseable)
            "old_factuality": r.get("factuality"),
            "old_uncovered": r.get("uncovered"),
            "old_quarantined": bool(r.get("factuality_quarantined")),
            "old_coherent": r.get("coherent"),
            "gen_len": len(r["gen_text"].strip()),
            "ref_len": len(s["reference"]),
        }
        n_new += 1
        if n_new % 25 == 0:
            SAMPLE_JSON.parent.mkdir(parents=True, exist_ok=True)
            SAMPLE_JSON.write_text(json.dumps(out))
            print(f"  labelled {n_new} new (at {j+1}/{len(idxs)})", flush=True)
    SAMPLE_JSON.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_JSON.write_text(json.dumps(out))
    print(f"STEP 2 done: {n_new} new, {len(out)} total -> {SAMPLE_JSON}", flush=True)


# ---- STEP 3: comparison + the BREADTH-BIAS diagnostic (the real gate) -----------------------

def _corr(a, b):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or not a.std() or not b.std():
        return float("nan"), float("nan")
    p = float(np.corrcoef(a, b)[0, 1])
    s = float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
    return p, s


def _hist(vals, bins=10):
    import numpy as np
    v = np.asarray([x for x in vals if x is not None], float)
    h, _ = np.histogram(v, bins=bins, range=(0, 1))
    return "  ".join(f"{lo:.1f}:{c}" for lo, c in zip(np.linspace(0, 1, bins, endpoint=False), h))


def step3():
    import numpy as np
    recs = load_records()
    src = load_source_aligned()
    data = list(json.loads(SAMPLE_JSON.read_text()).values())
    scored = [d for d in data if d["new_label"] is not None]

    L = ["", "=" * 78, "STEP 3 — whole-answer vs covered-claims + BREADTH-BIAS diagnostic",
         "=" * 78,
         f"sampled: {len(data)} | new-label parseable: {len(scored)} "
         f"(blind spot on new label: {1 - len(scored)/len(data):.0%}, target ~0)"]

    # coverage / recovery
    by = lambda c: [d for d in data if d["category"] == c]
    au, dt, cov = by("all_uncovered"), by("distrust"), by("covered")
    au_lab = [d for d in au if d["new_label"] is not None]
    L += ["",
          f"NEW label overall: mean {np.mean([d['new_label'] for d in scored]):.3f}  "
          f"median {np.median([d['new_label'] for d in scored]):.3f}",
          f"  hist(0..1): {_hist([d['new_label'] for d in scored])}",
          "",
          f"RECOVERY of previously-unlabelable all_uncovered: {len(au_lab)}/{len(au)} now labelled "
          f"(mean {np.mean([d['new_label'] for d in au_lab]):.3f})",
          f"DISTRUST subset under new label: n={len(dt)} mean "
          f"{np.mean([d['new_label'] for d in dt if d['new_label'] is not None]):.3f} "
          f"(should be LOW — a derailed answer isn't faithful either)"]

    # Spearman new vs old covered-claims factuality (both defined, covered subset)
    pairs = [(d["new_label"], d["old_factuality"]) for d in cov
             if d["new_label"] is not None and d["old_factuality"] is not None]
    if pairs:
        a, b = zip(*pairs)
        p, s = _corr(a, b)
        L += ["", f"NEW vs OLD covered-claims factuality (covered subset, n={len(pairs)}): "
                  f"Pearson {p:.2f} Spearman {s:.2f}  (agreement where both see the answer)"]

    # ---- THE GATE: breadth-bias diagnostics ----
    L += ["", "-" * 78, "BREADTH-BIAS DIAGNOSTIC (the gate — a biased label looks like a win above):"]
    # (1) new label vs OLD UNCOVERED fraction (covered subset). Negative => penalises divergence.
    bu = [(d["new_label"], d["old_uncovered"]) for d in cov
          if d["new_label"] is not None and d["old_uncovered"] is not None]
    if bu:
        a, b = zip(*bu)
        p, s = _corr(a, b)
        L += [f"(1) new_label vs old UNCOVERED fraction (n={len(bu)}): Pearson {p:.2f} Spearman {s:.2f}",
              "    strong NEGATIVE => new label falls as the answer diverges from the reference => "
              "penalising breadth (BIAS)."]
        bias1 = s
    else:
        bias1 = float("nan")
    # (2) new label vs generation LENGTH (all scored). Negative => longer answers scored lower.
    bl = [(d["new_label"], d["gen_len"]) for d in scored]
    a, b = zip(*bl)
    p2, s2 = _corr(a, b)
    L += [f"(2) new_label vs Llama answer LENGTH (n={len(bl)}): Pearson {p2:.2f} Spearman {s2:.2f}",
          "    strong NEGATIVE => longer answers penalised regardless of quality (breadth artifact)."]

    # (3) eyeball: low-new / high-old-covered — bias (correct-but-different) or real (bad answer)?
    susp = sorted([d for d in cov if d["new_label"] is not None and d["old_factuality"] is not None
                   and d["new_label"] <= 0.3 and d["old_factuality"] >= 0.7],
                  key=lambda d: d["new_label"])
    L += ["", f"(3) low-new(<=0.3) but high-old-covered(>=0.7): {len(susp)} cases "
              "(each = old covered-claims judged the answer faithful, new match judged it unfaithful)."]
    for d in susp[:4]:
        i = d["idx"]
        L += [f"    [idx {i}] new {d['new_label']:.2f} / old {d['old_factuality']:.2f} / "
              f"uncov {d['old_uncovered']:.2f} | Q: {src[i]['question'][:80]}",
              f"        LLAMA: {recs[i]['gen_text'].strip()[:150].replace(chr(10),' ')}",
              f"        REF  : {src[i]['reference'][:150].replace(chr(10),' ')}"]

    # verdict
    def verdict(b1, b2):
        strong = [x for x in (b1, b2) if x == x and x <= -0.30]
        mild = [x for x in (b1, b2) if x == x and -0.30 < x <= -0.15]
        if strong:
            return "BREADTH-BIASED (>= one diagnostic strongly negative) — NOT a clean upgrade. " \
                   "Keep covered-claims, or use whole-answer only as a complementary contradiction signal."
        if mild:
            return "MILD breadth drift — usable but note the confound; a consistency/non-contradiction " \
                   "prompt would likely reduce it. Compare before adopting."
        return "NO breadth bias detected on the sample — whole-answer label is an upgrade candidate; " \
               "scale to 2016 and re-confirm."
    L += ["", "-" * 78, f"VERDICT: {verdict(bias1, s2)}", "-" * 78]

    text = "\n".join(L)
    with open(OUT, "a") as f:
        f.write(text + "\n")
    print(text)


# ---- STEP 4/5: CONSISTENCY (non-contradiction) prompt on the SAME sample --------------------

# Same structure as the match prompt, but the semantics the STEP-3 diagnostic pointed to: extra
# valid content and omissions are NEUTRAL; only contradictions/errors are penalised. Tests whether
# the whole-answer blind-spot fix survives once we stop rewarding similarity-to-this-reference.
CONS_PROMPT = """
Only respond with a number between 0.0 to 1.0. Do not write any explanation.

The task below is a question-answering task. You will see a Question, a Reference Answer written by an expert, and a Model Answer. Judge the Model Answer ONLY for whether it is consistent with the expert Reference Answer. Score from 0.0 to 1.0, where 1.0 means nothing in the Model Answer contradicts the Reference Answer or is factually wrong (it is completely fine for the Model Answer to add extra correct detail, or to cover different valid points, or to omit points that are in the reference), 0.0 means the Model Answer directly contradicts the Reference Answer or is off-topic or incorrect, and scores in between reflect partial contradiction. Do NOT lower the score for missing or extra information; lower it ONLY for contradictions or errors.

Question: {q}
Reference Answer: {ref}
Model Answer: {ans}
Score:
"""


def step4(judge_model="gpt-5-mini"):
    """Label the SAME 200 sample with the consistency prompt; store as `cons_label` alongside."""
    from luq.labels import llm_judge
    recs = load_records()
    src = load_source_aligned()
    out = json.loads(SAMPLE_JSON.read_text())          # must exist (step2 ran)
    idxs = [int(k) for k in out]
    print(f"STEP 4: CONSISTENCY judge ({judge_model}) on the same {len(idxs)} sampled instances", flush=True)
    n_new = 0
    for j, i in enumerate(idxs):
        e = out[str(i)]
        if e.get("cons_label", "MISS") != "MISS":       # resume (None is a valid cached value)
            continue
        r, s = recs[i], src[i]
        prompt = CONS_PROMPT.format(q=s["question"], ref=s["reference"], ans=r["gen_text"].strip())
        e["cons_label"] = llm_judge.parse_score(llm_judge._gpt_response(prompt, judge_model))
        n_new += 1
        if n_new % 25 == 0:
            SAMPLE_JSON.write_text(json.dumps(out))
            print(f"  labelled {n_new} new (at {j+1}/{len(idxs)})", flush=True)
    SAMPLE_JSON.write_text(json.dumps(out))
    print(f"STEP 4 done: {n_new} new -> {SAMPLE_JSON}", flush=True)


def step5():
    """Diagnostic on the consistency label + head-to-head vs the match label on the same sample."""
    import numpy as np
    recs = load_records()
    src = load_source_aligned()
    data = list(json.loads(SAMPLE_JSON.read_text()).values())
    scored = [d for d in data if d.get("cons_label") is not None]
    by = lambda c: [d for d in data if d["category"] == c]
    cov = [d for d in by("covered") if d.get("cons_label") is not None]

    L = ["", "=" * 78, "STEP 5 — CONSISTENCY (non-contradiction) label: breadth-bias + discrimination",
         "=" * 78,
         f"sampled: {len(data)} | consistency parseable: {len(scored)} "
         f"(blind spot: {1-len(scored)/len(data):.0%})",
         f"CONSISTENCY overall: mean {np.mean([d['cons_label'] for d in scored]):.3f}  "
         f"median {np.median([d['cons_label'] for d in scored]):.3f}  std {np.std([d['cons_label'] for d in scored]):.3f}",
         f"  hist(0..1): {_hist([d['cons_label'] for d in scored])}"]

    # discrimination guardrail: did it collapse to all-high (uninformative)?
    hi = np.mean([d["cons_label"] >= 0.9 for d in scored])
    L += [f"  fraction >=0.9: {hi:.0%}  (if ~all high, the label is uninformative — the OPPOSITE failure)"]

    # recovery + distrust
    au = [d for d in by("all_uncovered") if d.get("cons_label") is not None]
    dt = [d for d in by("distrust") if d.get("cons_label") is not None]
    L += [f"RECOVERY all_uncovered: {len(au)}/{len(by('all_uncovered'))} (mean {np.mean([d['cons_label'] for d in au]):.3f})",
          f"DISTRUST subset under consistency: n={len(dt)} mean {np.mean([d['cons_label'] for d in dt]):.3f} "
          "(should still be LOW — a derailed/off-topic answer is inconsistent/incorrect)"]

    # agreement with old covered-claims factuality (should stay POSITIVE — still catches errors)
    pc = [(d["cons_label"], d["old_factuality"]) for d in cov if d["old_factuality"] is not None]
    a, b = zip(*pc); p, s = _corr(a, b)
    L += ["", f"CONSISTENCY vs OLD covered-claims factuality (n={len(pc)}): Pearson {p:.2f} Spearman {s:.2f} "
              "(want POSITIVE — both should flag genuine errors)"]

    # THE GATE — breadth-bias must now be ~0 (not negative like the match label's -0.27)
    L += ["", "-" * 78, "BREADTH-BIAS DIAGNOSTIC (must be ~0 now, vs the match label's -0.27):"]
    bu = [(d["cons_label"], d["old_uncovered"]) for d in cov if d["old_uncovered"] is not None]
    a, b = zip(*bu); p1, s1 = _corr(a, b)
    bl = [(d["cons_label"], d["gen_len"]) for d in scored]
    a, b = zip(*bl); p2, s2 = _corr(a, b)
    L += [f"(1) consistency vs old UNCOVERED fraction (n={len(bu)}): Pearson {p1:.2f} Spearman {s1:.2f}",
          f"(2) consistency vs answer LENGTH (n={len(bl)}): Pearson {p2:.2f} Spearman {s2:.2f}"]

    # did the match label's 24 correct-but-different flips get rescued?
    flips = [d for d in cov if d.get("new_label") is not None and d["old_factuality"] is not None
             and d["new_label"] <= 0.3 and d["old_factuality"] >= 0.7]
    resc = [d for d in flips if d["cons_label"] >= 0.6]
    L += ["", f"MATCH's correct-but-different flips ({len(flips)}): now {len(resc)} score >=0.6 under consistency "
              "(rescued = no longer wrongly penalised for being different)."]
    for d in sorted(flips, key=lambda d: d["cons_label"], reverse=True)[:4]:
        i = d["idx"]
        L += [f"    [idx {i}] match {d['new_label']:.2f} -> consistency {d['cons_label']:.2f} | "
              f"old-covered {d['old_factuality']:.2f} | Q: {src[i]['question'][:70]}"]

    def verdict(bias_s, hi_frac, agree_s):
        if bias_s == bias_s and bias_s <= -0.20:
            return "STILL breadth-biased — consistency wording did not fix it; keep covered-claims."
        if hi_frac >= 0.85:
            return "UNINFORMATIVE — collapsed to all-high (no discrimination); keep covered-claims."
        if agree_s == agree_s and agree_s < 0.15:
            return "LOOSE — no breadth bias but barely tracks real errors; weak label, keep covered-claims."
        return ("CLEAN UPGRADE candidate — breadth bias removed, keeps discrimination + error-tracking. "
                "Scale to 2016 and re-confirm on the full set.")
    L += ["", "-" * 78, f"VERDICT: {verdict(s1, hi, s)}", "-" * 78]

    text = "\n".join(L)
    with open(OUT, "a") as f:
        f.write(text + "\n")
    print(text)


# ---- STEP 6/7: scale the consistency label to the full 2016 as a SECONDARY label ------------

def step6(judge_model="gpt-5-mini"):
    """Write `consistency` (+ consistency_model, consistency_quarantined) into the records cache for
    ALL 2016, as a SECONDARY whole-answer label ALONGSIDE the covered-claims `factuality` (never
    overwrites it). Same distrust rule as covered-claims (SEVERE or coherent=false -> 0, skip judge)
    so both labels quarantine the identical set. Reuses the 200-sample non-distrust scores. Resumable
    via the `consistency_model` stamp."""
    from luq import degeneracy
    from luq.labels import llm_judge
    cfg = Config(dataset="expertqa", ood_setting="ID", prompt_regime="expertqa_rp12")
    key = cache.run_key("meta-llama/Meta-Llama-3.1-8B", "expertqa", "ID")
    recs = cache.load_records(cfg.cache_dir, key)
    src = load_source_aligned()
    assert len(src) == len(recs)
    # pre-seed reuse map from the 200 sample (non-distrust, defined) -> idx: cons_label
    sample = json.loads(SAMPLE_JSON.read_text()) if SAMPLE_JSON.exists() else {}
    reuse = {int(k): v["cons_label"] for k, v in sample.items()
             if v.get("category") != "distrust" and v.get("cons_label") is not None}

    def save():
        cache.save_records(recs, cfg.cache_dir, key)

    done = sum(1 for r in recs if r.get("consistency_model"))
    print(f"STEP 6: consistency -> full 2016 ({judge_model}). already done: {done}, "
          f"reusable from sample: {len(reuse)}", flush=True)
    n_new = n_judge = 0
    for r in recs:
        if r.get("consistency_model"):
            continue                                     # resume
        i = r["idx"]
        if degeneracy.is_severe(r["gen_text"]) or r.get("coherent") is False:
            r.update(consistency=0.0, consistency_quarantined=True)   # distrust, no judge call
        elif i in reuse:
            r.update(consistency=reuse[i], consistency_quarantined=False)  # reuse sample score
        else:
            s = src[i]
            prompt = CONS_PROMPT.format(q=s["question"], ref=s["reference"], ans=r["gen_text"].strip())
            r.update(consistency=llm_judge.parse_score(llm_judge._gpt_response(prompt, judge_model)),
                     consistency_quarantined=False)
            n_judge += 1
        r["consistency_model"] = judge_model
        n_new += 1
        if n_new % 25 == 0:
            save()
            print(f"  {n_new} new ({n_judge} judged, rest reused/quarantined) at idx {i}", flush=True)
    save()
    print(f"STEP 6 done: {n_new} new ({n_judge} judge calls) -> records `consistency`", flush=True)


def step7():
    """Full-2016 summary + re-confirm the breadth-bias finding on the whole set (not just n=200)."""
    import numpy as np
    cfg = Config(dataset="expertqa", ood_setting="ID", prompt_regime="expertqa_rp12")
    key = cache.run_key("meta-llama/Meta-Llama-3.1-8B", "expertqa", "ID")
    recs = cache.load_records(cfg.cache_dir, key)
    lab = [r for r in recs if r.get("consistency_model")]
    defd = [r for r in lab if r.get("consistency") is not None]
    quar = [r for r in lab if r.get("consistency_quarantined")]
    # non-quarantined, genuinely judged (for the bias check we need real judge scores)
    judged = [r for r in lab if not r.get("consistency_quarantined") and r.get("consistency") is not None]
    # recovered: previously all-uncovered (old factuality None, not quarantined) now have consistency
    recovered = [r for r in judged if r.get("factuality") is None]

    def _corr(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if len(a) < 3 or not a.std() or not b.std():
            return float("nan"), float("nan")
        return (float(np.corrcoef(a, b)[0, 1]),
                float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]))

    # breadth bias on the full set: consistency vs old-uncovered, over records with a defined uncovered
    bu = [(r["consistency"], r["uncovered"]) for r in judged if r.get("uncovered") is not None]
    a, b = zip(*bu); p1, s1 = _corr(a, b)
    # agreement with covered-claims factuality where both defined & not quarantined
    fj = [(r["consistency"], r["factuality"]) for r in judged
          if r.get("factuality") is not None and not r.get("factuality_quarantined")]
    a, b = zip(*fj); p2, s2 = _corr(a, b)
    cons = np.array([r["consistency"] for r in defd])

    L = ["", "=" * 78, "STEP 7 — consistency SECONDARY label on the full 2016", "=" * 78,
         f"labelled: {len(lab)}/2016 | consistency defined: {len(defd)} | quarantined (=0): {len(quar)}",
         f"consistency mean {cons.mean():.3f} median {np.median(cons):.3f} std {cons.std():.3f}  "
         f"| >=0.9: {np.mean(cons>=0.9):.0%}  <=0.1: {np.mean(cons<=0.1):.0%}",
         f"  hist(0..1): {_hist(list(cons))}",
         f"RECOVERED (old all-uncovered, now labelled): {len(recovered)} (mean "
         f"{np.mean([r['consistency'] for r in recovered]):.3f})",
         "",
         f"breadth bias (consistency vs old-uncovered, judged n={len(bu)}): Pearson {p1:.2f} Spearman {s1:.2f} "
         f"(want ~0; sample was -0.07)",
         f"agreement w/ covered-claims factuality (n={len(fj)}): Pearson {p2:.2f} Spearman {s2:.2f} "
         f"(sample was 0.67/0.63)",
         "",
         "SECONDARY label stamped as `consistency` (+consistency_model, consistency_quarantined); the",
         "covered-claims `factuality` is untouched. Both quarantine the identical distrust set.",
         "-" * 78]
    text = "\n".join(L)
    with open(OUT, "a") as f:
        f.write(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--judge", default="gpt-5-mini")
    args = ap.parse_args()
    steps = {1: step1, 3: step3, 5: step5, 7: step7}
    if args.step in steps:
        steps[args.step]()
    elif args.step == 2:
        step2(args.judge)
    elif args.step == 4:
        step4(args.judge)
    elif args.step == 6:
        step6(args.judge)
    else:
        raise SystemExit("step must be 1..7")


if __name__ == "__main__":
    main()
