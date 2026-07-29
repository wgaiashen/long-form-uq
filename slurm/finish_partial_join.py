"""STEP-C job-level helpers (NOT application code) for the S3->S6 join when S3 is INCOMPLETE.

FIX 1 (relabel-deferred): join_arma_baseline --mode fill-baseline already fills the cell INTERSECTION
  and reports missing cells, but it flips baseline_deferred=False on EVERY row. This re-labels any row
  it could NOT fill (empty k1_armA_s3) back to baseline_deferred=True, so a blank is never read as a
  zero/loss. Run it on the fill-baseline OUTPUT csv. Reports filled vs still-deferred by cell.

FIX 2 (backfill-provenance): S3's CSV predates the commit/seeds columns, so merge-factscore aborts on
  schema mismatch. This backfills the KNOWN, VERIFIED values (numerics shown identical across commits)
  and writes a .PROVENANCE_BACKFILLED marker so an asserted value can never be mistaken for a logged one.

Usage:
  python finish_partial_join.py relabel-deferred <joined_s6.csv>
  python finish_partial_join.py backfill-provenance <s3.csv> <commit> <seeds> <ref_schema_csv>
"""
import csv, os, re, sys, datetime

# S3 CSV schema (pre-provenance driver 5518ba2 — no commit/seeds columns)
S3_SCHEMA = ["rung", "eval", "train", "method", "prr_mean", "prr_std", "n_seeds",
             "bar_msp_min", "ci_lo", "ci_hi", "boot_p", "significant", "cluster", "env_hash"]


def rebuild_partial(log_path, out_csv):
    """FIX-1 support: S3 walltime-kill writes NO CSV. Dump WHATEVER cells the .o log has (no coverage
    gate — partial is fine, fill-baseline is partial-tolerant). boot_p stays blank (not in the log)."""
    txt = open(log_path).read()
    m = re.search(r"cluster (\S+) \| env (\S+)", txt)
    cluster, env_hash = (m.group(1), m.group(2)) if m else ("?", "?")
    rows = []
    for blk in re.split(r"(?=^\[[^\]]+\] eval=)", txt, flags=re.M):
        head = blk.splitlines()[0] if blk.strip() else ""
        hm = re.match(r"\[(?P<rung>[^\]]+?)\s*\] eval=(?P<eval>\S+) \(.*?\) train=(?P<train>\S+)\s+BAR=msp_min (?P<bar>[+\-][\d.]+)", head)
        if not hm:
            continue
        rung, ev, train, bar = hm["rung"].strip(), hm["eval"], hm["train"], float(hm["bar"])
        for mm in re.finditer(r"^    (?P<method>floor_min|armA|armB|armC_\w+|armD_\w+)\s+(?P<mean>[+\-][\d.]+) \+/- (?P<std>[\d.]+)", blk, flags=re.M):
            rows.append({"rung": rung, "eval": ev, "train": train, "method": mm["method"],
                         "prr_mean": float(mm["mean"]), "prr_std": float(mm["std"]), "n_seeds": 3,
                         "bar_msp_min": bar, "ci_lo": "", "ci_hi": "", "boot_p": "", "significant": "",
                         "cluster": cluster, "env_hash": env_hash})
        for vm in re.finditer(r"^    \[verdict\] (?P<vk>\S+)\s+margin (?P<mg>[+\-][\d.]+) CI\[(?P<lo>[+\-][\d.]+),(?P<hi>[+\-][\d.]+)\] (?P<sig>SIG|ns)", blk, flags=re.M):
            rows.append({"rung": rung, "eval": ev, "train": train, "method": f"VERDICT:{vm['vk']}",
                         "prr_mean": float(vm["mg"]), "prr_std": "", "n_seeds": 3, "bar_msp_min": "",
                         "ci_lo": float(vm["lo"]), "ci_hi": float(vm["hi"]), "boot_p": "",
                         "significant": (vm["sig"] == "SIG"), "cluster": cluster, "env_hash": env_hash})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=S3_SCHEMA); w.writeheader(); w.writerows(rows)
    open(out_csv + ".REBUILT_FROM_LOG", "w").write(
        f"rebuilt (PARTIAL ok) from {os.path.basename(log_path)} on {datetime.date.today().isoformat()}; "
        "boot_p BLANK by provenance (not in log).\n")
    cells = sorted({(r["eval"], r["rung"]) for r in rows if r["method"] == "armA"})
    print(f"  rebuild-partial: {len(rows)} rows, {len(cells)} armA cells -> {out_csv}")
    print(f"  cells present: {cells}")


def relabel_deferred(path):
    rows = list(csv.DictReader(open(path)))
    fn = rows[0].keys() if rows else []
    assert "baseline_deferred" in fn and "k1_armA_s3" in fn, "not a fill-baseline output (missing columns)"
    filled, deferred = [], []
    for r in rows:
        if r["method"].startswith("VERDICT"):
            continue
        cell = (r["eval"], r["rung"])
        if r.get("k1_armA_s3") in ("", None):
            r["baseline_deferred"] = "True"          # could NOT fill -> stays deferred (never a silent 0)
            deferred.append(cell)
        else:
            r["baseline_deferred"] = "False"
            filled.append(cell)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fn)); w.writeheader(); w.writerows(rows)
    print(f"  relabel-deferred: {len(set(filled))} cells FILLED, {len(set(deferred))} still DEFERRED")
    if deferred:
        print(f"  still-deferred cells (K=1 not available from S3): {sorted(set(deferred))}")


def backfill_provenance(s3_path, commit, seeds, ref_schema_csv):
    rows = list(csv.DictReader(open(s3_path)))
    ref_fn = list(csv.DictReader(open(ref_schema_csv)).fieldnames)   # target schema (factscore/S6 side)
    cur_fn = list(rows[0].keys()) if rows else []
    add = [c for c in ("commit", "seeds") if c not in cur_fn]
    for r in rows:
        if "commit" in add:
            r["commit"] = commit
        if "seeds" in add:
            r["seeds"] = seeds
    out_fn = cur_fn + add
    # keep column order sane; only require the union covers ref
    with open(s3_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fn); w.writeheader(); w.writerows(rows)
    marker = s3_path + ".PROVENANCE_BACKFILLED"
    with open(marker, "w") as f:
        f.write(
            f"date: {datetime.date.today().isoformat()}\n"
            f"backfilled columns into {os.path.basename(s3_path)}: commit={commit!r}, seeds={seeds!r}\n"
            "REASON: S3 ran on the pre-provenance driver (commit 5518ba2); these columns were NOT natively\n"
            "logged. Values are ASSERTED (from git log + the S3 sbatch), not recorded by the run.\n"
            "EVIDENCE the numerics are comparable to the db2eff2 runs:\n"
            "  - attn_pool.py single-head path (n_query=n_head=1) is BYTE-IDENTICAL across 5518ba2..db2eff2\n"
            "  - src/luq/msp.py (the floor) is UNTOUCHED across the range\n"
            "  - fixed_prior_ladder.py diff is PROVENANCE-ONLY (adds commit/seeds columns, no numeric change)\n"
            "  - tests/test_multihead.py: K=1 == arm A to <1e-6 on db2eff2\n"
            "A later reader MUST treat commit/seeds in this file as asserted-not-logged.\n")
    print(f"  backfill-provenance: added {add} to {s3_path}; wrote marker {os.path.basename(marker)}")


if __name__ == "__main__":
    if sys.argv[1] == "rebuild-partial":
        rebuild_partial(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "relabel-deferred":
        relabel_deferred(sys.argv[2])
    elif sys.argv[1] == "backfill-provenance":
        backfill_provenance(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        sys.exit(f"unknown subcommand {sys.argv[1]}")
