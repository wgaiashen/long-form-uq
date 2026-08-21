"""Preflight for running the WHOLE Qwen pipeline on DoC — check it, do not assume it.

WHY THIS EXISTS
---------------
The sync between the two clusters has always run DoC -> RCS: DoC does one GPU job, the small
artifact comes back, and RCS is the master. So RCS accumulates everything and DoC accumulates
almost nothing. Moving generation, extraction AND the ladder to DoC reverses that for the first
time, and the failure mode is not a crash on job 1 — it is a crash on job 6, hours in, because some
dataset loader needs a file nobody thought about.

So: check every input the pipeline touches, before spending GPU hours. Read-only, no GPU, no £,
seconds to run.

    python scripts/checks/doc_preflight_qwen.py
    python scripts/checks/doc_preflight_qwen.py --model Qwen/Qwen2.5-14B

A MISSING ITEM IS REPORTED, NEVER WORKED AROUND. This script does not create directories, does
not download, and does not fall back. It tells you what is absent and exits non-zero.
"""
import argparse
import importlib
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OK, BAD, WARN = "  ok", "  BAD", "  warn "
_fail = []
_warn = []


def check(name, fn, fatal=True):
    """Run one check. Never let an exception kill the sweep — a preflight that stops at the first
    problem hides the other five, and you then discover them one job at a time."""
    try:
        ok, detail = fn()
    except Exception as e:                                    # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    if ok:
        print(f"{OK} {name}: {detail}", flush=True)
    elif fatal:
        _fail.append(name); print(f"{BAD} {name}: {detail}", flush=True)
    else:
        _warn.append(name); print(f"{WARN}{name}: {detail}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B")
    ap.add_argument("--datasets", default="pubmed_qa,xsum,cnn_dailymail,med_quad,samsum,expertqa,asqa,factscore")
    args = ap.parse_args()
    datasets = args.datasets.split(",")

    print(f"DoC PREFLIGHT — {args.model}\n{'='*74}", flush=True)

    # ---- 1. environment ----------------------------------------------------------------------
    print("\n--- environment ---", flush=True)
    check("LUQ_CLUSTER", lambda: (os.environ.get("LUQ_CLUSTER") == "doc",
                                  f"{os.environ.get('LUQ_CLUSTER')!r} (expect 'doc'; source pbs/_env.sh)"),
          fatal=False)

    def _cache_root():
        from luq.config import CACHE_DIR, RESULTS_DIR
        writable = os.access(CACHE_DIR, os.W_OK)
        free = shutil.disk_usage(CACHE_DIR).free / 2**30
        # ~90GB of Qwen cache: 16GB pooled features + ~74GB per-token at 3 layers.
        return (writable and free > 120,
                f"{CACHE_DIR} | writable={writable} | {free:.0f} GiB free (need >120 for ~90GB of cache)"
                f"\n        results -> {RESULTS_DIR}")
    check("cache root (LUQ_CACHE_ROOT)", _cache_root)

    def _repo_vol():
        free = shutil.disk_usage(ROOT).free / 2**30
        return (free > 1, f"repo volume {free:.1f} GiB free — records (~150MB) + results live here")
    check("repo volume headroom", _repo_vol, fatal=False)

    # ---- 2. python packages ------------------------------------------------------------------
    print("\n--- packages ---", flush=True)
    for mod, why in [("torch", "generation + extraction"),
                     ("transformers", "model + tokenizer"),
                     ("probe_drift", "dataset prompts (luq.data imports it)"),
                     ("probe_drift_long", "the long-form grid — NEW, must be cloned + pip install -e"),
                     ("sklearn", "probe training in the ladder"),
                     ("scipy", "stats in the ladder")]:
        def f(m=mod, w=why):
            mm = importlib.import_module(m)
            return True, f"{getattr(mm, '__version__', '?')} ({w})"
        check(f"import {mod}", f)

    # ---- 3. the model ------------------------------------------------------------------------
    print("\n--- model ---", flush=True)

    def _model():
        from transformers import AutoConfig, AutoTokenizer
        cfg = AutoConfig.from_pretrained(args.model)
        tok = AutoTokenizer.from_pretrained(args.model)
        want = (48, 5120) if "Qwen2.5-14B" in args.model else None
        got = (cfg.num_hidden_layers, cfg.hidden_size)
        ok = want is None or got == want
        return ok, (f"{got[0]} layers, hidden {got[1]}, vocab {cfg.vocab_size}, "
                    f"tok vocab {len(tok)}, bos={tok.bos_token!r}"
                    + ("" if ok else f"  EXPECTED {want}"))
    check("model + tokenizer load from cache", _model)

    def _hf_home():
        h = os.environ.get("HF_HOME", "")
        free = shutil.disk_usage(h).free / 2**30 if h and Path(h).exists() else -1
        return (bool(h) and Path(h).exists(), f"{h or '(unset)'} | {free:.0f} GiB free")
    check("HF_HOME", _hf_home, fatal=False)

    # ---- 4. THE BIG ONE: can every dataset actually LOAD here? --------------------------------
    # This is what the DoC->RCS sync direction hides. ExpertQA needs a local JSONL, FActScore needs
    # a wiki reference directory, ASQA needs an HF download. Any one of them missing does not show
    # up until that dataset's job runs, hours in.
    print("\n--- dataset loaders (the ones the one-way sync tends to leave behind) ---", flush=True)
    from luq import data

    # ASYMMETRY FOUND BY THIS SCRIPT'S FIRST RUN (2026-08-08): factscore's SOURCE data master
    # lives on DoC (`/vol/gpudata/.../factscore_data`, 21 GB, the default in `luq/factscore.py:23`)
    # and does NOT exist on RCS at all. RCS only ever had the cached RECORDS, which is why the Llama
    # grid ran there without anyone noticing. So RCS cannot regenerate factscore prompts -- had Qwen
    # generation stayed on RCS, that dataset would have failed. It is the one dataset that is
    # DoC-native, which is an argument FOR the move, not against it.
    for ds in datasets:
        def f(d=ds):
            tr, ev = data.load(d, "ID")
            n = len(tr.x) + len(ev.x)
            if n == 0:
                return False, "loaded but EMPTY"
            budget = data.MAX_NEW_TOKENS.get(d)
            return True, f"{len(tr.x)} train + {len(ev.x)} eval = {n} prompts, budget {budget}"
        check(f"load {ds}", f)

    # ---- 5. judge labelling ------------------------------------------------------------------
    print("\n--- judge (needed only if labelling runs HERE rather than on RCS) ---", flush=True)
    check("OPENAI_API_KEY", lambda: (bool(os.environ.get("OPENAI_API_KEY")),
                                     "set" if os.environ.get("OPENAI_API_KEY") else
                                     "NOT set — label on RCS, or export it here"), fatal=False)

    def _net():
        # ANY HTTP RESPONSE MEANS REACHABLE, INCLUDING AN ERROR STATUS. A bare GET to
        # api.openai.com returns 421 Misdirected Request -- the server answered, so the network is
        # fine. Treating that as "no internet" is a false negative that would send someone hunting a
        # firewall that is not there (it did exactly that on the RCS smoke run).
        import urllib.request, urllib.error
        try:
            r = urllib.request.urlopen("https://api.openai.com", timeout=10)
            return True, f"api.openai.com reachable (HTTP {r.status})"
        except urllib.error.HTTPError as e:
            return True, f"api.openai.com reachable (HTTP {e.code} — a response, so the route works)"
        except Exception as e:
            return False, f"unreachable: {type(e).__name__}: {e}"
    check("outbound internet", _net, fatal=False)

    # ---- 6. GPU ------------------------------------------------------------------------------
    print("\n--- GPU (run this ON a compute node, not the jump box) ---", flush=True)

    def _gpu():
        import torch
        if not torch.cuda.is_available():
            return False, "no CUDA visible — expected on the jump box; re-run under srun"
        props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
        biggest = max(p.total_memory for p in props) / 2**30
        # fp32 Qwen-14B is ~59GB of weights.
        return (biggest >= 70,
                f"{len(props)}x {props[0].name}, biggest {biggest:.0f} GiB "
                f"({'fits fp32 on ONE card' if biggest >= 70 else 'TOO SMALL for single-card fp32'})")
    check("GPU memory", _gpu, fatal=False)

    # ---- 7. the pipeline's own guards exist --------------------------------------------------
    print("\n--- pipeline scripts present ---", flush=True)
    for s in ["01_extract.py", "01e_repool.py", "01h_pertoken.py", "01b_ptrue.py",
              "01c_lookback.py", "01g_ptrue_unsup.py", "02_label.py",
              "checks/feature_pertok_consistency.py", "checks/probedriftlong.py",
              "checks/generation_quality.py"]:
        check(f"scripts/{s}", lambda p=s: ((ROOT / "scripts" / p).exists(), "present"))

    # ---- verdict -----------------------------------------------------------------------------
    print("\n" + "=" * 74)
    if _warn:
        print(f"{len(_warn)} non-fatal: {', '.join(_warn)}")
    if _fail:
        print(f"PREFLIGHT FAILED — {len(_fail)} blocking: {', '.join(_fail)}")
        print("   Report these; do not work around them.")
        return 1
    print("PREFLIGHT PASSED — every input the Qwen pipeline touches is present here.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
