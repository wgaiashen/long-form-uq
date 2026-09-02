"""FAIL-LOUD guard: the SAPLMA FEATURE cache must be the teacher-forced re-pool, matching the pertok cache.

Root cause it guards against (traced 2026-07-22): the SAPLMA feature cache can be produced two ways —
(a) INLINE autoregressive capture during 01_extract (prefill + per-token with the KV cache), or
(b) TEACHER-FORCED re-pool (01e_repool / the same single forward the pertok cache uses). They differ by a
~1e-4 relative (looks like ~8e-3 absolute on the large residual-stream states) — NOT precision. Some sets
were re-pooled (sciq/trivia/pubmed/xsum → match pertok at ~1e-6), others were not (med_quad/cnn/samsum/
ExpertQA/ASQA → inline capture → ~8e-3 gap). The ladder drivers use the teacher-forced PERTOK, so they are
safe; but code that reads the FEATURE cache directly (aggregation_table §A ID SAPLMA, canonical_ladder ID-gate,
ood_refpools, reference_table_anchors, expertqa_loco, aggregators) then mixes the two conventions.

This guard asserts, per dataset: feats[:, L15].mean-pool == pertok(L15).mean over the SAME window, to ~1e-6.
Run it after ANY extraction; a failure means the feature cache is inline-only -> re-pool it (01e_repool).

    python scripts/checks/feature_pertok_consistency.py --datasets sciq,med_quad,asqa,expertqa   # sample check
    python scripts/checks/feature_pertok_consistency.py --datasets asqa --prompt-regime asqa_rp12 --full
"""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B"
# Known regime-namespaced sets (so --datasets can list them without a per-dataset --prompt-regime).
# Imported from attn_pool rather than re-declared: this map was duplicated here, and a duplicated
# cache-root map is how one reader silently ends up on a different cache from every other. Importing it
# also means `LUQ_REGIME=...` redirects this guard to the v2 caches along with the ladders, so the guard
# checks the data that is actually being used.
sys.path.insert(0, str(ROOT / "scripts" / "checks"))
from attn_pool import PROMPT_REGIME as REGIME  # noqa: E402
TOL = 1e-4          # max abs Δ that still counts as "teacher-forced consistent" (fp32 match is ~1e-6)


def check(dataset, regime, layer=15, sample=60, full=False, model=DEFAULT_MODEL):
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID",
                 prompt_regime=regime or REGIME.get(dataset, ""))
    key = cache.run_key(model, dataset, "ID")
    ptp = cfg.cache_dir / "pertok" / f"{cache._slug(model)}__{dataset}__ID__L{layer}.npz"
    if not ptp.exists():
        return dataset, None, "no pertok cache"
    # TWO SHAPES OF FEATURE PLANE EXIST. An extract run stores every layer, (n, n_layers, hidden).
    # A plane rebuilt by re-pooling one per-token layer stores only that layer, (n, 1, hidden), and
    # names it in a `layer` field. Indexing the second by layer number raises; indexing it by 0
    # without reading that field would compare the window mean against whatever layer is stored and
    # could report agreement between two different layers. So the field is read and asserted.
    fpath = cfg.cache_dir / "features" / f"{key}__saplma.npz"
    if not fpath.exists():
        return dataset, None, "no feature cache"
    try:
        with np.load(fpath) as z:
            feats = z["feats"]
            stored_layer = int(z["layer"]) if "layer" in z.files else None
    except Exception as e:
        return dataset, None, f"no feature cache ({e})"
    if stored_layer is not None:
        if feats.shape[1] != 1:
            return dataset, False, (f"cache names a single layer ({stored_layer}) but holds "
                                    f"{feats.shape[1]}; self-inconsistent")
        if stored_layer != layer:
            return dataset, False, (f"cache holds layer {stored_layer}, checking layer {layer}; "
                                    "refusing to compare two different layers")
        feat_col = 0
    else:
        feat_col = layer
    pt = np.load(ptp, allow_pickle=True, mmap_mode="r")
    states = pt["states"]
    n = len(states)
    idx = range(n) if full else np.linspace(0, n - 1, min(sample, n)).astype(int)
    deltas = []
    for i in idx:
        wm = np.asarray(states[i], np.float32).mean(0)               # teacher-forced pertok window-mean
        deltas.append(np.abs(wm - feats[i, feat_col, :]).max())
    d = np.array(deltas)
    ok = bool(d.max() < TOL)     # Python bool, NOT numpy bool_ (else `ok is False` never matches in main())
    return dataset, ok, f"max|Δ|={d.max():.2e} mean|Δ|={d.mean():.2e} (n={len(idx)}{'/full' if full else ''})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sciq,trivia_qa,pubmed_qa,xsum,cnn_dailymail,med_quad,samsum,expertqa")
    ap.add_argument("--prompt-regime", default="", help="override regime for ALL listed datasets")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model whose caches to check (default keeps every existing invocation identical). "
                         "NOTE: --layer must be that model's pertok layer — the file is keyed __L{layer} "
                         "(mid = (n_hidden_layers+1)//2: Llama-3.1-8B -> 15/16 as dumped, Qwen2.5-14B -> 24).")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--full", action="store_true", help="check every example (heavier) instead of a sample")
    ap.add_argument("--require-checked", action="store_true",
                    help="exit non-zero if ANY dataset was skipped. Default off so existing callers "
                         "are unaffected, but every NEW caller should pass it: a guard that skipped "
                         "has verified nothing, and must not be mistaken for one that passed.")
    args = ap.parse_args()
    print(f"model = {args.model}")
    print(f"{'dataset':16s} {'verdict':10s} detail")
    any_fail = False
    skipped = []
    for d in args.datasets.split(","):
        name, ok, detail = check(d, args.prompt_regime, args.layer, full=args.full, model=args.model)
        v = "SKIP" if ok is None else ("PASS" if ok else "FAIL (inline-only -> repool)")
        if ok is False:
            any_fail = True
        if ok is None:
            skipped.append(name)
        print(f"{name:16s} {v:10s} {detail}")
    if any_fail:
        print("\nFAIL: one or more feature caches are inline-only (not teacher-forced). Re-pool with 01e_repool.")
        sys.exit(1)
    # A SKIP IS NOT A PASS. Until 2026-08-15 a run where every dataset skipped (typically because
    # the pertok cache did not exist yet) still printed "ALL consistent" and exited 0 -- a guard
    # reporting success for having checked nothing, which is the exact failure mode this guard exists
    # to prevent elsewhere. Found when a W-Models post-extract job ran the guard BEFORE 01h_pertoken:
    # it skipped every dataset and passed, so it could never have caught anything.
    if skipped:
        print(f"\nSKIPPED (nothing compared): {skipped}. A skip is NOT a pass — the pertok cache "
              f"must exist, so run this AFTER 01h_pertoken.")
        if args.require_checked:
            print("--require-checked: exiting non-zero because at least one dataset was not checked.")
            sys.exit(2)
    if not skipped:
        print("\nALL consistent (feature cache == teacher-forced pertok).")


if __name__ == "__main__":
    main()
