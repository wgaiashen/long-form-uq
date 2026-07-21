"""ASQA closed-book PILOT (W8 gate). Generate ~100 CLOSED-BOOK Llama-3.1-8B answers on ASQA and report the
Str-EM (coverage) label distribution. Decision:
  * HEALTHY spread (labels not piled at 0 or 1, std reasonable) -> GO: build the full ASQA factuality set.
  * DEGENERATE (nearly all 0 or all 1) -> NO-GO: switch to WildHallucinations (per the programme).

WHY closed-book: keeps ASQA a FACTUALITY test (world knowledge), not a RAG/faithfulness test. WHY Str-EM: it is
FREE (no API, no retrieval) -- for each disambiguated qa_pair, does ANY gold short answer appear (normalised) in
the generation. Str-EM = coverage/recall of the gold disambiguations. (Disambig-F1 via RoBERTa-SQuAD2 is the
heavier version; Str-EM is enough to judge the label DISTRIBUTION for the gate.)

Run on DoC (A100) via slurm/asqa_pilot.sbatch. First thing it prints is the dataset schema + one example, so any
field-name mismatch is caught immediately (ASQA HF ports vary slightly).

    python scripts/asqa_pilot.py --n 100
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "meta-llama/Meta-Llama-3.1-8B"
# Closed-book: the ambiguous question only, asking for a paragraph that resolves the ambiguity. No context.
PROMPT = ("Answer the following question with a detailed, self-contained paragraph. If the question is "
          "ambiguous, cover each distinct correct interpretation.\n\nQuestion: {q}\nAnswer:")


def _norm(s):
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _short_answers(qa):
    """Robust to field-name variants across ASQA HF ports."""
    for k in ("short_answers", "short_answer", "answers", "answer"):
        v = qa.get(k)
        if v:
            return v if isinstance(v, list) else [v]
    return []


def _qa_pairs(ex):
    for k in ("qa_pairs", "qa", "annotations"):
        v = ex.get(k)
        if v:
            return v
    return []


def _question(ex):
    for k in ("ambiguous_question", "question", "ambiguous_q"):
        if ex.get(k):
            return ex[k]
    return None


def str_em(gen, qa_pairs):
    """Fraction of qa_pairs whose ANY gold short answer appears (normalised substring) in the generation."""
    g = _norm(gen)
    hits = n = 0
    for qa in qa_pairs:
        sa = _short_answers(qa)
        if not sa:
            continue
        n += 1
        if any(_norm(a) and _norm(a) in g for a in sa):
            hits += 1
    return hits / n if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", default="dev", help="ASQA split (dev=948, train=4353)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", default="results/asqa/asqa_pilot.jsonl")
    args = ap.parse_args()

    print("loading din0s/asqa ...", flush=True)
    ds = load_dataset("din0s/asqa", split=args.split)
    print(f"SCHEMA: {list(ds.features.keys())}", flush=True)
    print(f"EXAMPLE[0] keys+types: { {k: type(v).__name__ for k, v in ds[0].items()} }", flush=True)
    ex0 = ds[0]
    print(f"  question: {str(_question(ex0))[:200]}", flush=True)
    print(f"  #qa_pairs: {len(_qa_pairs(ex0))}; first short answers: "
          f"{_short_answers(_qa_pairs(ex0)[0]) if _qa_pairs(ex0) else 'NONE'}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows, ems = [], []
    n = min(args.n, len(ds))
    for i in range(n):
        ex = ds[i]
        q = _question(ex); qa = _qa_pairs(ex)
        if q is None or not qa:
            continue
        prompt = PROMPT.format(q=q)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        em = str_em(gen, qa)
        ems.append(em)
        rows.append({"i": i, "question": q, "gen": gen, "str_em": em, "n_qa": len(qa)})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n} done (running mean Str-EM = {np.nanmean(ems):.3f})", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    arr = np.array([e for e in ems if np.isfinite(e)])
    print("\n================ ASQA PILOT — Str-EM label distribution ================")
    print(f"n={len(arr)}  mean={arr.mean():.3f}  std={arr.std():.3f}  "
          f"frac@0={np.mean(arr == 0):.2f}  frac@1={np.mean(arr == 1):.2f}  median={np.median(arr):.3f}")
    # coarse histogram
    hist, edges = np.histogram(arr, bins=[0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.001])
    labels = ["=0", "(0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1)", "=1"]
    print("  histogram: " + "  ".join(f"{l}:{c}" for l, c in zip(labels, hist)))
    # GO/NO-GO heuristic (report both the numbers and the verdict; the human decides)
    degenerate = (np.mean(arr == 0) > 0.7) or (np.mean(arr == 1) > 0.7) or (arr.std() < 0.12)
    print(f"\n  VERDICT: {'NO-GO (degenerate -> use WildHallucinations)' if degenerate else 'GO (healthy spread -> build ASQA)'}")
    print(f"  wrote {args.out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
