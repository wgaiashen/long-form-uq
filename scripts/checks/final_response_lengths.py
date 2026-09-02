#!/usr/bin/env python
"""Descriptive response-length statistics for the final scored long-form population.

WHAT THIS MEASURES, AND WHY THE POPULATION HAS TO BE RESOLVED RATHER THAN ASSUMED
--------------------------------------------------------------------------------
Several populations of the same eight datasets sit side by side on disk for each model: an original
namespace, a truncated one, and a corrected-span one. A length table built from the wrong namespace
is not an error anyone would notice, because every namespace returns a plausible number. So this
script does not glob for generations. It reads the population manifest written by
`clean_core_manifest.py`, which records, per model and dataset, the exact records file the
report-facing master was built from, its row count, its label field and its SHA-256, and it re-checks
that hash before measuring anything.

THE TEXT MEASURED is each record's `gen_text` field in the resolved namespace. In every namespace
that field is `tokenizer.decode(gen_token_ids)` for the RETAINED span: in the truncated and
corrected-span namespaces the cut is made in token space and the stored text is the decode of the
retained prefix, so the text, the token log-probabilities and the per-token hidden states all
describe the same characters. `gen_text_raw`, where present, holds the pre-correction text and is
deliberately NOT measured here.

ROWS MEASURED are the rows the evaluation actually scores: those with a finite label in the
dataset's label field. The scoring driver drops unlabelled rows before it carves any split, so a
length table over all rows would describe a larger population than any reported result.

TWO ROW POPULATIONS are reported, because both are defensible and they are not the same size.
`all_scored` is every labelled row: the complete set of responses the benchmark holds for that
dataset, which serve as evaluation targets in that dataset's own cells and as training-pool rows in
others. `eval_test` is the held-out test carve that a reported PRR is actually computed on,
reproduced by calling the ladder's own `eval_split` on the same rows in the same order. The first is
the wider descriptive population and is the primary table here; the second is given so the numbers
can be compared against analyses that report the test rows only.

TWO LENGTH DEFINITIONS
----------------------
1. Model-native tokens: the count of NON-SPECIAL tokens in `gen_token_ids`, which is the generation's
   own tokenisation of exactly the characters in `gen_text`, not a re-tokenisation of that string.
   Every row is checked to satisfy `tokenizer.decode(gen_token_ids, skip_special_tokens=True) ==
   gen_text`, so this count and the measured text describe the same span by construction.
   The end-of-text token is excluded because it contributes no characters; a response stopped at the
   generation cap carries no such token, so counting it would add one token to naturally terminated
   responses only and make the two kinds of row not comparable. The full `len(gen_token_ids)`, which
   is the window the token log-probabilities and so the probability scores aggregate over, is
   reported alongside as `scored_window`.
   The script additionally re-tokenises `gen_text` and reports any disagreement, as a check.
2. Whitespace words: `len(gen_text.strip().split())`. Deterministic and tokenizer-independent, so it
   is comparable between model families with different vocabularies.

    python scripts/checks/final_response_lengths.py
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xl_rungs import eval_split  # noqa: E402

MANIFEST = ROOT / "results" / "analysis" / "CLEAN_CORE_POPULATION_MANIFEST.json"
OUTDIR = ROOT / "results" / "report"

# The two generating models this table compares, and the display names used in the report.
MODELS = ["meta-llama/Meta-Llama-3.1-8B", "Qwen/Qwen2.5-14B"]
SHORT = {"meta-llama/Meta-Llama-3.1-8B": "Llama", "Qwen/Qwen2.5-14B": "Qwen"}

# Dataset order and display names, fixed here so every emitted table agrees.
DATASETS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum",
            "expertqa", "factscore"]
DISPLAY = {"pubmed_qa": "PubMedQA", "med_quad": "MedQuAD", "asqa": "ASQA", "xsum": "XSum",
           "cnn_dailymail": "CNN/DailyMail", "samsum": "SAMSum", "expertqa": "ExpertQA",
           "factscore": "FActScore"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pertok_row_count(path):
    """Number of rows in a per-token cache, read from the .npy header inside the .npz.

    The states array is an object array of per-example matrices and is several gigabytes, so it is
    not loaded. Reading the header gives the row count, which is what has to match the records file.
    """
    import zipfile
    import numpy.lib.format as fmt
    with zipfile.ZipFile(path) as z:
        with z.open("states.npy") as fh:
            version = fmt.read_magic(fh)
            shape, _, _ = fmt._read_array_header(fh, version)
    return int(shape[0])


def describe(values):
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "sd": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
            "min": float(a.min()), "max": float(a.max())}


def measure(model, entry, tok):
    """Length statistics and integrity checks for one model/dataset population."""
    rec_path = ROOT / entry["records_path"]
    field = entry["label_field"]
    special = set(tok.all_special_ids)

    native, window, retok, words, splits = [], [], [], [], []
    n_rows = 0
    seen_key, dup_key = set(), 0
    n_decode_mismatch = 0
    n_retok_mismatch = 0
    n_logprob_mismatch = 0
    n_prompt_leak = 0
    n_empty = 0
    n_raw_present = 0
    n_special_total = 0
    special_positions = {}

    with open(rec_path) as fh:
        for line in fh:
            r = json.loads(line)
            n_rows += 1
            # `idx` restarts within each split, so a row's identity is the (split, idx) pair.
            key = (r.get("split"), r.get("idx"))
            if key in seen_key:
                dup_key += 1
            seen_key.add(key)
            if r.get("gen_text_raw") is not None:
                n_raw_present += 1
            y = r.get(field)
            if y is None or not math.isfinite(float(y)):
                continue

            ids = r["gen_token_ids"]
            text = r["gen_text"]
            if len(r.get("token_logprobs", [])) != len(ids):
                n_logprob_mismatch += 1
            # The scored text is the special-token-free decode of the retained ids.
            if tok.decode(ids, skip_special_tokens=True) != text:
                n_decode_mismatch += 1
            sp = [i for i, t in enumerate(ids) if t in special]
            n_special_total += len(sp)
            for i in sp:
                # Recorded as an offset from the end, so a special token anywhere other than the
                # final position stays visible rather than being averaged away.
                k = i - len(ids)
                special_positions[k] = special_positions.get(k, 0) + 1
            # A prompt leak would show as the tail of the prompt reappearing at the head of the
            # response. Compared on the last 80 characters, which is long enough to be specific.
            tail = (r.get("prompt") or "")[-80:]
            if tail and text.startswith(tail):
                n_prompt_leak += 1

            w = len(text.strip().split())
            if w == 0:
                n_empty += 1
            n_native = len(ids) - len(sp)
            splits.append(r.get("split"))
            native.append(n_native)
            window.append(len(ids))
            words.append(w)
            rt = len(tok(text, add_special_tokens=False)["input_ids"])
            retok.append(rt)
            if rt != n_native:
                n_retok_mismatch += 1

    checks = {
        "rows_in_file": n_rows,
        "rows_scored": len(native),
        "rows_unlabelled_dropped": n_rows - len(native),
        "manifest_n_rows": entry["n_rows"],
        "manifest_n_labelled": entry["n_labelled"],
        "duplicate_split_idx_pairs": dup_key,
        "decode_roundtrip_mismatches": n_decode_mismatch,
        "retokenisation_mismatches": n_retok_mismatch,
        "logprob_length_mismatches": n_logprob_mismatch,
        "prompt_tail_at_start_of_response": n_prompt_leak,
        "empty_responses": n_empty,
        "rows_carrying_precorrection_text": n_raw_present,
        "special_tokens_in_retained_ids": n_special_total,
        "special_token_offsets_from_end": {str(k): v for k, v in sorted(special_positions.items())},
    }
    if entry.get("pertok_path"):
        checks["pertok_rows"] = pertok_row_count(ROOT / entry["pertok_path"])

    # The held-out test carve, reproduced with the ladder's own splitter on the same rows in the same
    # order the ladder loads them, so this subset is the one a reported PRR is computed on.
    _, test_idx = eval_split(np.array(splits))
    t = np.asarray(test_idx, dtype=int)
    checks["eval_test_rows"] = int(t.size)
    return {"native": describe(native), "words": describe(words),
            "scored_window": describe(window), "retok": describe(retok),
            "native_eval_test": describe(np.asarray(native)[t]),
            "words_eval_test": describe(np.asarray(words)[t]),
            "checks": checks}


def fmt_row(cells):
    return "| " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTDIR), help="directory for the CSV and markdown outputs")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    man = json.loads(MANIFEST.read_text())
    results, provenance = {}, {}

    for model in MODELS:
        blk = man["models"][model]
        by_ds = {r["dataset"]: r for r in blk["datasets"]}
        tok = AutoTokenizer.from_pretrained(model)
        results[model] = {}
        provenance[model] = {"layer": blk["layer"], "master": blk["master"],
                             "stocktake": blk["stocktake"], "datasets": {}}
        for ds in DATASETS:
            entry = by_ds[ds]
            rec_path = ROOT / entry["records_path"]
            got = sha256_file(rec_path)
            if got != entry["records_sha256"]:
                sys.exit(f"{model}/{ds}: records file has changed since the manifest was written "
                         f"({entry['records_sha256'][:16]} -> {got[:16]}). Refusing to measure a "
                         f"population that no longer matches the recorded one.")
            print(f"  {SHORT[model]:6s} {ds:14s} {entry['expected_regime']:24s} "
                  f"{entry['n_rows']:5d} rows", flush=True)
            results[model][ds] = measure(model, entry, tok)
            provenance[model]["datasets"][ds] = {
                "regime": entry["expected_regime"], "records_path": entry["records_path"],
                "pertok_path": entry.get("pertok_path"), "label_field": entry["label_field"],
                "records_sha256": entry["records_sha256"]}

    # ---- CSVs: one row per model x dataset, full descriptive statistics -------------------------
    for kind, fname in (("native", "final_response_lengths_native_tokens.csv"),
                        ("words", "final_response_lengths_words.csv")):
        # `mean_scored_window_tokens` appears only on the native-token file: it is the mean of
        # `len(gen_token_ids)`, the window the probability scores aggregate over, which exceeds the
        # native count by one on every naturally terminated response.
        head = "model,dataset,regime,label_field,n,mean,median,sd,min,max"
        if kind == "native":
            head += ",mean_scored_window_tokens"
        head += ",eval_test_n,eval_test_mean,eval_test_median"
        lines = [head]
        for model in MODELS:
            for ds in DATASETS:
                s = results[model][ds][kind]
                p = provenance[model]["datasets"][ds]
                row = (f"{model},{DISPLAY[ds]},{p['regime']},{p['label_field']},"
                       f"{s['n']},{s['mean']:.2f},{s['median']:.1f},{s['sd']:.2f},"
                       f"{s['min']:.0f},{s['max']:.0f}")
                if kind == "native":
                    row += f",{results[model][ds]['scored_window']['mean']:.2f}"
                e = results[model][ds][f"{kind}_eval_test"]
                row += f",{e['n']},{e['mean']:.2f},{e['median']:.1f}"
                lines.append(row)
        (outdir / fname).write_text("\n".join(lines) + "\n")
        print(f"wrote {(outdir / fname)}")

    (outdir / "final_response_lengths_full.json").write_text(
        json.dumps({"provenance": provenance, "statistics": results}, indent=2))
    print(f"wrote {(outdir / 'final_response_lengths_full.json')}")

    # ---- console summary tables ------------------------------------------------------------------
    for kind, unit in (("native", "tokens"), ("words", "words")):
        print(f"\n### {unit}")
        print(fmt_row(["Dataset", f"Llama mean {unit}", f"Llama median {unit}",
                       f"Qwen mean {unit}", f"Qwen median {unit}"]))
        print(fmt_row(["---", "---:", "---:", "---:", "---:"]))
        for ds in DATASETS:
            a = results[MODELS[0]][ds][kind]
            b = results[MODELS[1]][ds][kind]
            print(fmt_row([DISPLAY[ds], f"{a['mean']:.1f}", f"{a['median']:.0f}",
                           f"{b['mean']:.1f}", f"{b['median']:.0f}"]))

    print("\n### integrity checks")
    for model in MODELS:
        for ds in DATASETS:
            c = results[model][ds]["checks"]
            flags = []
            if c["rows_scored"] != c["manifest_n_labelled"]:
                flags.append(f"scored {c['rows_scored']} != manifest labelled {c['manifest_n_labelled']}")
            if c["rows_in_file"] != c["manifest_n_rows"]:
                flags.append("row count moved")
            if c.get("pertok_rows") is not None and c["pertok_rows"] != c["rows_in_file"]:
                flags.append(f"pertok {c['pertok_rows']} != records {c['rows_in_file']}")
            for k in ("duplicate_split_idx_pairs", "decode_roundtrip_mismatches", "logprob_length_mismatches",
                      "prompt_tail_at_start_of_response", "empty_responses"):
                if c[k]:
                    flags.append(f"{k}={c[k]}")
            status = "; ".join(flags) if flags else "clean"
            print(f"  {SHORT[model]:6s} {ds:14s} n={c['rows_scored']:5d}  "
                  f"retok_mismatch={c['retokenisation_mismatches']:5d}  {status}")


if __name__ == "__main__":
    main()
