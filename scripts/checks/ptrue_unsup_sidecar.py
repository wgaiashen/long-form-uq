#!/usr/bin/env python
"""Move unsupervised-P(True) scores between clusters via a SIDECAR, never by shipping records back.

WHY A SIDECAR
-------------
`01g_ptrue_unsup.py` writes `ptrue_unsup` / `ptrue_unsup_mass` back INTO the Tier-1 record. That is fine
when it runs where the records live. It is NOT fine across clusters: syncing the modified records back
would overwrite RCS's canonical copies — the ones carrying the £-paid judge labels and the ones hashed in
`prereg/CANONICAL_v1_MANIFEST.md`. A stale or partial remote copy landing on top of them is an
unrecoverable loss, and rsync protects against deletion, not against overwriting a same-named file.

So: DoC exports a tiny CSV keyed by row position, RCS merges it into its OWN records. ~50KB crosses the
wire instead of megabytes of records, and the canonical files are only ever written by the machine that
owns them.

    # on DoC, after 01g has run:
    python scripts/checks/ptrue_unsup_sidecar.py --export --datasets cnn_dailymail,med_quad,samsum

    # on RCS, after syncing the CSVs into results/sidecar_ptrue_unsup/:
    python scripts/checks/ptrue_unsup_sidecar.py --merge --datasets cnn_dailymail,med_quad,samsum

KEYED BY POSITION, AND THE MERGE PROVES THE ALIGNMENT. `idx` is NOT unique in these records (the
per-token loader documents this), so position is the only safe key — and position is only safe if both
sides hold the same file. The merge therefore refuses unless the row count matches AND a fingerprint of
the gold targets matches. Without that check a misaligned merge would attach every score to the wrong
example and still look perfectly well-formed.
"""
import argparse
import csv as _csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq.config import Config  # noqa: E402
from luq import cache  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}
SIDECAR_DIR = ROOT / "results" / "sidecar_ptrue_unsup"
FIELDS = ("ptrue_unsup", "ptrue_unsup_mass", "ptrue_unsup_model")


def _records(ds):
    cfg = Config(model_name=MODEL, dataset=ds, ood_setting="ID", prompt_regime=REGIME.get(ds, ""))
    return cfg, cache.run_key(MODEL, ds, "ID"), cache.load_records(cfg.cache_dir,
                                                                   cache.run_key(MODEL, ds, "ID"))


def _fingerprint(recs):
    """Cheap alignment proof: hash of the gold targets in order. Independent of anything 01g writes, so
    it cannot be accidentally satisfied by the very field being merged."""
    h = hashlib.sha256()
    for r in recs:
        h.update(repr(r.get("target", "")).encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def export(datasets):
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        _cfg, _key, recs = _records(ds)
        got = [r for r in recs if isinstance(r.get("ptrue_unsup"), (int, float))]
        if not got:
            print(f"  {ds:14s} SKIPPED LOUDLY: no ptrue_unsup on the records — run 01g first")
            continue
        out = SIDECAR_DIR / f"ptrue_unsup__{ds}.csv"
        with open(out, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["#n_rows", len(recs), "#fingerprint", _fingerprint(recs)])
            w.writerow(["pos", *FIELDS])
            for i, r in enumerate(recs):
                if isinstance(r.get("ptrue_unsup"), (int, float)):
                    w.writerow([i, r.get("ptrue_unsup"), r.get("ptrue_unsup_mass"),
                                r.get("ptrue_unsup_model", "")])
        print(f"  {ds:14s} exported {len(got)}/{len(recs)} rows -> {out.name} "
              f"({out.stat().st_size / 1024:.0f} KB)")


def merge(datasets):
    for ds in datasets:
        p = SIDECAR_DIR / f"ptrue_unsup__{ds}.csv"
        if not p.exists():
            print(f"  {ds:14s} SKIPPED LOUDLY: {p.name} not found")
            continue
        cfg, key, recs = _records(ds)
        with open(p) as fh:
            rows = list(_csv.reader(fh))
        hdr = rows[0]
        n_claimed, fp_claimed = int(hdr[1]), hdr[3]
        # REFUSE ON ANY DOUBT. A positional merge onto a different file attaches every score to the
        # wrong example and produces a perfectly well-formed, entirely wrong dataset.
        if n_claimed != len(recs):
            sys.exit(f"{ds}: sidecar says {n_claimed} rows, local records have {len(recs)}. "
                     "Refusing to merge by position onto a different file.")
        fp_local = _fingerprint(recs)
        if fp_claimed != fp_local:
            sys.exit(f"{ds}: target fingerprint {fp_claimed} != local {fp_local}. The two sides hold "
                     "DIFFERENT records — refusing to merge.")
        n = 0
        for row in rows[2:]:
            i = int(row[0])
            recs[i]["ptrue_unsup"] = float(row[1])
            recs[i]["ptrue_unsup_mass"] = float(row[2]) if row[2] not in ("", "None") else None
            recs[i]["ptrue_unsup_model"] = row[3]
            n += 1
        cache.save_records(recs, cfg.cache_dir, key)
        print(f"  {ds:14s} merged {n} rows into the CANONICAL records (row count + target "
              f"fingerprint {fp_local} both verified)")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--export", action="store_true", help="run WHERE 01g ran (DoC): records -> sidecar")
    g.add_argument("--merge", action="store_true", help="run WHERE the canonical records live (RCS)")
    ap.add_argument("--datasets", required=True, help="comma-separated")
    args = ap.parse_args()
    ds = [d.strip() for d in args.datasets.split(",") if d.strip()]
    (export if args.export else merge)(ds)


if __name__ == "__main__":
    main()
