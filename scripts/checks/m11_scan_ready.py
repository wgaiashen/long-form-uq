"""Which layers are ready to scan, and which are not, with the reason.

Scanning is the long pole of this campaign: about eight hours per layer, independent across layers,
so the throughput is set by how many can be launched at once. Launching one that is not ready wastes
a slot and produces a partial cell file, so the preconditions are checked here rather than discovered
inside the job.

A layer is ready when all eight datasets have per-token states at that layer AND the background
statistics for every generation budget exist at the window the scan will run under.

    python scripts/checks/m11_scan_ready.py                # human readable
    python scripts/checks/m11_scan_ready.py --shell        # ready layer numbers, one per line
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.data import MAX_NEW_TOKENS  # noqa: E402

DATASETS = ["pubmed_qa", "xsum", "cnn_dailymail", "samsum", "med_quad", "asqa", "expertqa",
            "factscore"]
REGIME = {"med_quad": "cleanv2", "asqa": "asqa_rp12", "expertqa": "expertqa_rp12",
          "factscore": "factscore_rp12"}
PUBLISHED_LAYERS = list(range(31)) + [32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--layers", default="", help="default: the full published set")
    ap.add_argument("--window", default="ref", choices=["project", "ref"])
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()

    slug = cache._slug(args.model)
    layers = ([int(x) for x in args.layers.split(",") if x.strip()] if args.layers
              else PUBLISHED_LAYERS)
    tag = "" if args.window == "project" else "__refwin"
    budgets = sorted({MAX_NEW_TOKENS[d] for d in DATASETS})

    ready, blocked = [], []
    for L in layers:
        missing_ds = []
        for d in DATASETS:
            cfg = Config(model_name=args.model, dataset=d, ood_setting="ID",
                         prompt_regime=REGIME.get(d, ""))
            if not (Path(cfg.cache_dir) / "pertok" /
                    f"{slug}__{d}__ID__L{L}.npz").exists():
                missing_ds.append(d)
        missing_bg = [b for b in budgets
                      if not (ROOT / "cache" / "background_c4" /
                              f"{slug}__bgstats__L{L}__b{b}{tag}.npz").exists()]
        if not missing_ds and not missing_bg:
            ready.append(L)
        else:
            blocked.append((L, missing_ds, missing_bg))

    if args.shell:
        for L in ready:
            print(L)
        return 0 if ready else 1

    print(f"window '{args.window}' | {len(layers)} layers considered")
    print(f"\nREADY TO SCAN ({len(ready)}): {ready}")
    if blocked:
        print(f"\nNOT READY ({len(blocked)}):")
        for L, md, mb in blocked:
            why = []
            if md:
                why.append(f"{len(md)} dataset(s) missing: {','.join(md[:4])}"
                           + (" ..." if len(md) > 4 else ""))
            if mb:
                why.append(f"background missing at budget(s) {mb}")
            print(f"  L{L:<3} {'; '.join(why)}")
    n_bg_missing = sum(1 for _, _, mb in blocked if mb)
    if n_bg_missing:
        print(f"\n{n_bg_missing} layer(s) are blocked ONLY or PARTLY by the background, which is "
              f"produced on the other cluster. Nothing here can fix that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
