"""Which probe_drift built each cache -- including caches written before anything stamped it.

WHY. On 2026-08-01 the xsum arm of the xsum/cnn probe turned out not to be the one-variable change
it was designed as: it was generated under a different `probe_drift` from its own v1 cache, so the
prompt and the drawn examples moved along with the budget. Its prompts, prompt_token_ids and targets
matched v1 for 0/400 examples, against cnn's 400/400.

Nothing caught it, and nothing could have. `01_extract`'s prompt-hash guard only compares a cache
against ITS OWN stored hash, so a fresh `--prompt-regime` starts empty and has nothing to disagree
with. The comparison that mattered was across namespaces, v2-vs-v1, and no guard looked there.

`cache.source_provenance()` now stamps new caches. This script covers the other half:

  1. It REPORTS the stamp where one exists.
  2. Where none exists -- which is every cache built before today -- it RECONSTRUCTS the answer, by
     rebuilding each dataset's prompts under every probe_drift checkout it can find and seeing which
     one reproduces the cache's stored prompt hash. That is exactly the manual procedure that
     identified the problem, made repeatable.
  3. It COMPARES two namespaces and says plainly whether they are safe to read against each other.

The reconstruction is what makes this useful now rather than in six months: the entire existing grid
predates the stamp, and this can still date it.

    python scripts/checks/source_provenance.py                       # audit every dataset
    python scripts/checks/source_provenance.py --compare v2probe_xsum   # is that regime v1-comparable?

Loads datasets (CPU, no GPU, no model, no judge). One subprocess per checkout, because two copies of
the same package cannot be imported into one interpreter.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from attn_pool import PROMPT_REGIME  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
DATASETS = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum"]

# The worker that runs inside each candidate checkout. It must be a separate process: `probe_drift`
# is a single module name, so two checkouts cannot coexist in one interpreter.
_WORKER = r'''
import sys, json
sys.path.insert(0, %(src)r)
import probe_drift
from luq import data, cache
out = {"__path__": probe_drift.__file__}
for ds in %(datasets)r:
    try:
        tr, ev = data.load(ds, "ID")
        out[ds] = cache.prompt_hash(list(tr.x) + list(ev.x), list(tr.y) + list(ev.y))
    except Exception as e:
        out[ds] = "ERR:" + type(e).__name__
print("@@@" + json.dumps(out))
'''


def find_checkouts() -> list[Path]:
    """Every directory beside the repo that looks like a probe_drift checkout."""
    return sorted(p for p in PARENT.glob("*") if (p / "probe_drift" / "dataset_configs.py").exists())


def hashes_under(checkout: Path, datasets: list[str]) -> dict:
    """Rebuild each dataset's prompt hash with `checkout` forced onto the path."""
    code = _WORKER % {"src": str(ROOT / "src"), "datasets": datasets}
    env_path = f"{checkout}:{ROOT / 'src'}"
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**__import__("os").environ, "PYTHONPATH": env_path},
                           cwd=str(ROOT), timeout=1800)
    except subprocess.TimeoutExpired:
        return {}
    line = next((l for l in r.stdout.splitlines() if l.startswith("@@@")), None)
    return json.loads(line[3:]) if line else {}


def stored_for(dataset: str, regime: str | None):
    rg = regime if regime is not None else PROMPT_REGIME.get(dataset, "")
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID", prompt_regime=rg)
    key = cache.run_key(MODEL, dataset, "ID")
    return (cache.load_prompt_hash(cfg.cache_dir, key),
            cache.load_source_provenance(cfg.cache_dir, key))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--compare", default=None,
                    help="a regime to check against its v1 counterpart, e.g. v2probe_xsum")
    args = ap.parse_args()

    checkouts = find_checkouts()
    print("probe_drift checkouts found:")
    for c in checkouts:
        print(f"  {c}")
    cur = cache.source_provenance()
    print(f"\nthis process resolves probe_drift to:\n  {cur['path']}")
    print(f"  dataset_configs.py sha256 {str(cur['dataset_configs_sha256'])[:16]}")

    print("\ncomputing prompt hashes under each checkout (one subprocess each)...", flush=True)
    per = {c: hashes_under(c, args.datasets) for c in checkouts}

    print("\n" + "=" * 100)
    print("WHICH CHECKOUT BUILT EACH v1 CACHE  (reconstructed from the stored prompt hash)")
    print("=" * 100)
    print(f"{'dataset':14s} {'stamped?':>9s} {'v1 hash':>13s}  reproduced by")
    origin = {}
    for ds in args.datasets:
        st_hash, st_prov = stored_for(ds, "")
        hits = [c.name for c in checkouts if per[c].get(ds) == st_hash] if st_hash else []
        if st_hash is None:
            verdict = "(no prompt hash stored -- cannot reconstruct)"
        elif not hits:
            verdict = "*** NO checkout reproduces it: v1 predates both ***"
        elif len(hits) > 1:
            verdict = f"AMBIGUOUS, all agree: {', '.join(hits)}"
        else:
            verdict = hits[0]
        origin[ds] = hits[0] if len(hits) == 1 else None
        print(f"{ds:14s} {('yes' if st_prov else 'no'):>9s} "
              f"{(st_hash[:12] if st_hash else '--'):>13s}  {verdict}")

    groups = {}
    for ds, o in origin.items():
        if o:
            groups.setdefault(o, []).append(ds)
    if len(groups) > 1:
        print("\n⚠️ THE v1 GRID SPANS MORE THAN ONE CHECKOUT. Datasets built under different ones are")
        print("   not directly comparable on anything prompt-dependent, and a regeneration of one")
        print("   compared against the other confounds the library change with whatever was intended:")
        for o, dss in sorted(groups.items()):
            print(f"     {o:22s} {', '.join(dss)}")

    if not args.compare:
        return
    print("\n" + "=" * 100)
    print(f"IS REGIME {args.compare!r} COMPARABLE WITH ITS v1 COUNTERPART?")
    print("=" * 100)
    for ds in args.datasets:
        v2h, v2p = stored_for(ds, args.compare)
        if v2h is None:
            continue
        v1h, _ = stored_for(ds, "")
        v2o = next((c.name for c in checkouts if per[c].get(ds) == v2h), None)
        same = (v1h == v2h)
        print(f"  {ds:14s} v1 {str(v1h)[:12]}  {args.compare} {v2h[:12]}  "
              f"built by {v2o or '?'}")
        print(f"  {'':14s} -> {'SAME prompts, comparable' if same else 'DIFFERENT prompts: NOT a one-variable comparison'}")
        if not same and origin.get(ds) and v2o and origin[ds] != v2o:
            print(f"  {'':14s}    cause: v1 came from {origin[ds]}, this regime from {v2o}")


if __name__ == "__main__":
    main()
