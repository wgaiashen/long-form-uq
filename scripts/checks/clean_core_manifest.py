#!/usr/bin/env python
"""Build and re-verify the report-facing population manifest for the corrected-span core.

WHY THIS EXISTS
---------------
Several distinct populations of the same eight datasets now sit side by side on disk: the original
namespace, a truncated one, and a corrected-span one, per model. Resolution between them is by cache
namespace, and a namespace is chosen by an environment variable. That is an easy thing to get wrong
silently, and a wrong resolution produces a perfectly plausible number rather than an error.

This script writes down, for every model and dataset in the report-facing core, exactly which files a
run must read, how many rows they hold, which label field describes them, and a hash. `--verify` then
re-resolves everything and fails if anything has moved.

The manifest is descriptive, not authoritative: it records what the report-facing masters were built
from. It is written once and checked thereafter.

    python scripts/checks/clean_core_manifest.py --write
    python scripts/checks/clean_core_manifest.py --verify
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402

OUT_JSON = ROOT / "results" / "analysis" / "CLEAN_CORE_POPULATION_MANIFEST.json"
OUT_MD = ROOT / "results" / "analysis" / "CLEAN_CORE_POPULATION_MANIFEST.md"

# The report-facing resolution, per model. Written out in full rather than derived, because the whole
# point is to record what was actually used rather than what a default would produce today.
CORE = {
    "meta-llama/Meta-Llama-3.1-8B": {
        "layer": 15,
        "master": "results/cleanv2/pdl_cleanv2_master__meta-llama_Meta-Llama-3.1-8B.csv",
        "stocktake": "STOCKTAKE_cleanv2.md",
        "regimes": {"pubmed_qa": "", "xsum": "", "cnn_dailymail": "", "samsum": "",
                    "med_quad": "cleanv2", "asqa": "asqa_rp12",
                    "expertqa": "expertqa_rp12", "factscore": "factscore_rp12"},
    },
    "google/gemma-2-9b": {
        "layer": 20,
        "master": "results/cleanv2/wmodels_sens8_cleanv2_master__google_gemma-2-9b.csv",
        "stocktake": "STOCKTAKE_gemma2_9b.md",
        "regimes": {"pubmed_qa": "", "xsum": "", "cnn_dailymail": "", "samsum": "",
                    "med_quad": "cleanv2", "asqa": "asqa_rp12",
                    "expertqa": "expertqa_rp12", "factscore": "factscore_rp12"},
    },
    # The corrected Qwen population uses a DIFFERENT namespace set from the other two, and that is
    # deliberate and frozen. Its summarisation sets have separately built truncated siblings that the
    # report-facing master does NOT use, because the measured effect was too small to justify rerunning
    # the ladder. Substituting them would silently change the population.
    "Qwen/Qwen2.5-14B": {
        "layer": 23,
        "master": "results/analysis/pdl_master_qwenclean__Qwen_Qwen2.5-14B.csv",
        "stocktake": "STOCKTAKE_qwen.md",
        "regimes": {"pubmed_qa": "", "xsum": "", "cnn_dailymail": "", "asqa": "asqa_rp12",
                    "samsum": "trunc_v1", "med_quad": "trunc_v1",
                    "expertqa": "expertqa_rp12_trunc_v1", "factscore": "factscore_rp12_trunc_v1"},
    },
}

LABEL_FIELD = {"expertqa": "factuality", "factscore": "factuality"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def strong_identity(path):
    """Size, mtime and a hash of the first and last 4 MB.

    A full hash of every per-token cache would read about 33 GB. This is explicitly a PARTIAL hash and
    is labelled as one wherever it is reported: it detects truncation, replacement and most in-place
    edits, and it is not a proof of byte equality.
    """
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(1 << 22))
        if st.st_size > (1 << 23):
            fh.seek(-(1 << 22), os.SEEK_END)
            h.update(fh.read(1 << 22))
    return {"size_bytes": st.st_size, "mtime": int(st.st_mtime),
            "partial_sha256_first_last_4mb": h.hexdigest()}


def describe(model, dataset, regime, layer):
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID", prompt_regime=regime)
    key = cache.run_key(model, dataset, "ID")
    slug = cache._slug(model)
    rec_path = Path(cfg.cache_dir) / "records" / f"{key}.jsonl"
    pt_path = Path(cfg.cache_dir) / "pertok" / f"{slug}__{dataset}__ID__L{layer}.npz"
    rec = {"model": model, "slug": slug, "dataset": dataset,
           "expected_regime": regime or "(canonical)",
           "records_path": str(rec_path.relative_to(ROOT)) if rec_path.exists() else None,
           "pertok_path": str(pt_path.relative_to(ROOT)) if pt_path.exists() else None,
           "layer": layer}
    if not rec_path.exists():
        rec["error"] = "records absent"
        return rec
    rows = [json.loads(l) for l in open(rec_path)]
    field = LABEL_FIELD.get(dataset, "correctness")
    y = np.array([r.get(field, np.nan) for r in rows], dtype=float)
    spans = {r.get("label_span") for r in rows if r.get("label_span")}
    n_cut = sum(1 for r in rows if r.get("gen_text_raw") is not None
                and r.get("gen_text") != r.get("gen_text_raw"))
    rec.update({
        "n_rows": len(rows),
        "label_field": field,
        "n_labelled": int(np.isfinite(y).sum()),
        "clean_span_applied": bool(spans) or n_cut > 0,
        "clean_span_markers": sorted(spans) if spans else [],
        "n_rows_cut": n_cut,
        "records_sha256": sha256_file(rec_path),
    })
    if pt_path.exists():
        rec["pertok_identity"] = strong_identity(pt_path)
    else:
        rec["error"] = "pertok absent"
    return rec


def build():
    out = {"note": "Report-facing population resolution for the corrected-span three-model core. "
                   "The per-token identity is a PARTIAL hash (see strong_identity).",
           "models": {}}
    for model, cfg in CORE.items():
        recs = [describe(model, d, cfg["regimes"][d], cfg["layer"]) for d in sorted(cfg["regimes"])]
        out["models"][model] = {"layer": cfg["layer"], "master": cfg["master"],
                                "stocktake": cfg["stocktake"], "datasets": recs}
    return out


def render_md(man):
    L = ["# Corrected-span core — population manifest", "",
         "Machine-readable twin: `CLEAN_CORE_POPULATION_MANIFEST.json`. Per-token identities are "
         "PARTIAL hashes (size, mtime, and the first and last 4 MB), not proofs of byte equality.", ""]
    for model, blk in man["models"].items():
        L += [f"## {model}", "",
              f"Layer {blk['layer']} · master `{blk['master']}` · stocktake `{blk['stocktake']}`", "",
              "| dataset | regime | rows | labelled | label field | span corrected | rows cut | records sha256 |",
              "|---|---|---:|---:|---|---|---:|---|"]
        for r in blk["datasets"]:
            if r.get("error") and "n_rows" not in r:
                L.append(f"| {r['dataset']} | {r['expected_regime']} | — | — | — | — | — | **{r['error']}** |")
                continue
            L.append(f"| {r['dataset']} | `{r['expected_regime']}` | {r['n_rows']} | {r['n_labelled']} | "
                     f"{r['label_field']} | {'yes' if r['clean_span_applied'] else 'no'} | "
                     f"{r['n_rows_cut']} | `{r['records_sha256'][:16]}…` |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    man = build()

    missing = [(m, r["dataset"], r["error"]) for m, b in man["models"].items()
               for r in b["datasets"] if r.get("error")]
    for m, d, e in missing:
        print(f"  MISSING  {m} / {d}: {e}")
    n = sum(len(b["datasets"]) for b in man["models"].values())
    print(f"resolved {n - len(missing)}/{n} model-dataset populations")

    if args.verify:
        if not OUT_JSON.exists():
            sys.exit(f"no manifest at {OUT_JSON} to verify against.")
        old = json.loads(OUT_JSON.read_text())
        drift = []
        for model, blk in man["models"].items():
            ob = {r["dataset"]: r for r in old["models"].get(model, {}).get("datasets", [])}
            for r in blk["datasets"]:
                o = ob.get(r["dataset"])
                if o is None:
                    drift.append(f"{model}/{r['dataset']}: absent from the manifest")
                    continue
                for k in ("records_path", "pertok_path", "n_rows", "n_labelled", "label_field",
                          "records_sha256", "expected_regime", "layer"):
                    if r.get(k) != o.get(k):
                        drift.append(f"{model}/{r['dataset']}.{k}: {o.get(k)!r} -> {r.get(k)!r}")
        if drift:
            print("\nMANIFEST DRIFT:")
            for d in drift:
                print("  " + d)
            sys.exit(1)
        print("VERIFY: every resolved population matches the manifest.")
        return

    if args.write:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(man, indent=2))
        OUT_MD.write_text(render_md(man))
        print(f"wrote {OUT_JSON.relative_to(ROOT)}\nwrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
