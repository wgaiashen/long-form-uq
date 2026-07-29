"""S6 job-config helper (NOT application code): make the S3 baseline robust to a walltime kill.

Order:
  1. If the real S3 CSV exists AND covers every expected (eval,rung) cell -> use it (source=csv).
  2. Else if the S3 .o log covers every expected cell -> REBUILD the CSV from the log (source=log,
     boot_p blank-by-provenance) and write it to the baseline path.
  3. Else -> ABORT (incomplete: neither CSV nor log has all cells). Do NOT degrade.
Then run the STEP-3 asserts (evals identical, seed VALUES identical) on whichever baseline we have.

Exit 0 = baseline ready + asserts pass. Exit 1 = abort (S6 must not start).
Usage: python rebuild_and_assert_baseline.py <base_csv> <s3_log> <evals_csv> <s6_seeds> <s3_seeds>
"""
import csv, os, re, sys

base, s3_log, evals_arg, s6_seeds_arg, s3_seeds_arg = sys.argv[1:6]
evals = evals_arg.split(",")
RUNGS = ["ID", "DiffTask-long", "LOO-long"]
expected = {(e, r) for e in evals for r in RUNGS}
SCHEMA = ["rung", "eval", "train", "method", "prr_mean", "prr_std", "n_seeds",
          "bar_msp_min", "ci_lo", "ci_hi", "boot_p", "significant", "cluster", "env_hash"]


def parse_log(path):
    """Return list of CSV-schema rows parsed from an S3 .o log (STEP-2-validated parser)."""
    if not os.path.exists(path):
        return []
    txt = open(path).read()
    m = re.search(r"cluster (\S+) \| env (\S+)", txt)
    cluster, env_hash = (m.group(1), m.group(2)) if m else ("?", "?")
    rows = []
    blocks = re.split(r"(?=^\[[^\]]+\] eval=)", txt, flags=re.M)
    for blk in blocks:
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
    return rows


def covers(rows):
    have = {(r["eval"], r["rung"]) for r in rows if r["method"] == "armA"}
    return expected <= have, sorted(expected - have)


# --- pick a source ---
source = None
rows = []
if os.path.exists(base) and os.path.getsize(base) > 0:
    rows = list(csv.DictReader(open(base)))
    ok, missing = covers(rows)
    if ok:
        source = "csv"
    else:
        print(f"  baseline CSV present but INCOMPLETE (missing {missing}) -> try log", flush=True)
if source is None:
    rows = parse_log(s3_log)
    ok, missing = covers(rows)
    if ok:
        # write the rebuilt baseline to the exact path multihead_ladder reads
        with open(base, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SCHEMA)
            w.writeheader(); w.writerows(rows)
        open(base + ".REBUILT_FROM_LOG", "w").write(
            "baseline was REBUILT from the S3 .o log after a walltime kill; boot_p is BLANK by "
            "provenance (not zero) — significance flag + CI carry the decision.\n")
        source = "log"
        print(f"  REBUILT baseline from log -> {base} (boot_p blank-by-provenance; marker written)", flush=True)
    else:
        sys.exit(f"ABORT: neither CSV nor log covers all cells (missing {missing}). S6 must not start.")

# --- STEP 3 asserts on the chosen baseline ---
armA = [r for r in rows if r["method"] == "armA"]
csv_evals = {r["eval"] for r in armA}
if csv_evals != set(evals):
    sys.exit(f"ABORT: eval mismatch baseline={sorted(csv_evals)} vs S6={sorted(evals)}")
# seed VALUES (not count): compare S3's --seeds to S6's --seeds
s3_seeds = sorted(int(x) for x in re.split(r"[,\s]+", s3_seeds_arg.strip()) if x)
s6_seeds = sorted(int(x) for x in re.split(r"[,\s]+", s6_seeds_arg.strip()) if x)
if s3_seeds != s6_seeds:
    sys.exit(f"ABORT: seed VALUES differ S3={s3_seeds} vs S6={s6_seeds} -> pairing invalid")
ns = {str(r["n_seeds"]) for r in armA}
if ns != {"3"}:
    sys.exit(f"ABORT: n_seeds != 3 in baseline: {ns}")

print(f"  STEP 3 PASS  source={source}  evals={sorted(csv_evals)}  seeds S3={s3_seeds}==S6={s6_seeds}  "
      f"cells={len(expected)}", flush=True)
