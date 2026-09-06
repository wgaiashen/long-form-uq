"""Assemble the M11 numbers into the tables a write-up needs, from the result files themselves.

Everything here is read from the CSVs on disk and nothing is retyped, so a number in the write-up can
be traced to a file rather than to a message. Coverage travels with every row: a method on a partial
grid is labelled, and its cell count is printed next to it, because a mean over 15 cells is not
comparable to a mean over 40.

    python scripts/checks/m11_report_tables.py > results/analysis/M11_REPORT_TABLES.md
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLUG = "meta-llama_Meta-Llama-3.1-8B"
RUNGS = ["ID", "LOO-long", "SameTask-long", "DiffTask-long", "1ds-Diff-long"]
SHIFTED = RUNGS[1:]


def load(path, keep=None):
    out = defaultdict(dict)
    p = ROOT / path
    if not p.exists():
        return out
    for r in csv.DictReader(open(p)):
        m = r["method"]
        if keep and m not in keep:
            continue
        if not r.get("prr_mean"):
            continue
        out[m][(r["eval"], r["rung"])] = float(r["prr_mean"])
    return out


def rung_means(d):
    by = defaultdict(list)
    for (ev, rung), v in d.items():
        by[rung].append(v)
    return by


def row(name, d, label=None):
    by = rung_means(d)
    cells = sum(len(v) for v in by.values())
    vals = []
    for rg in RUNGS:
        vals.append(f"{st.fmean(by[rg]):.3f}" if by.get(rg) else "  -  ")
    sh = [v for rg in SHIFTED for v in by.get(rg, [])]
    shm = f"{st.fmean(sh):.3f}" if sh else "  -  "
    return f"| {label or name} | " + " | ".join(vals) + f" | **{shm}** | {cells} |"


HDR = ("| method | ID | LOO | SameTask | DiffTask | 1ds-Diff | shifted mean | cells |\n"
       "|---|---:|---:|---:|---:|---:|---:|---:|")

ALL = load(f"results/hybrids/pdl_mdlayers_alllayer__{SLUG}.csv",
           {"satmd_alllayer", "satrmd_alllayer", "huq_satmd_alllayer", "huq_satrmd_alllayer",
            "msp_satmd_alllayer", "msp_satrmd_alllayer"})
MID = load(f"results/hybrids/pdl_hybrids_rmd__{SLUG}.csv",
           {"satmd_mid", "satrmd_mid", "huq_satmd_mid", "huq_satrmd_mid",
            "msp_satmd_mid", "msp_satrmd_mid", "md_mean_mid", "rmd_mean_mid", "hbo", "msp", "saplma"})
REF = load(f"results/hybrids/pdl_hybrids_refwin__{SLUG}.csv",
           {"hbo", "satmd_mid", "huq_satmd_mid", "md_mean_mid", "msp", "saplma"})

print("# M11 result tables — the full-layer reproduction of the published distance family")
print()
print("Model `meta-llama/Meta-Llama-3.1-8B`. Every value is a prediction-rejection ratio averaged")
print("over three seeds, then over the evaluation datasets of that rung. Generated from the result")
print("files by `scripts/checks/m11_report_tables.py`; nothing is retyped.")
print()
print("## 1. The reproduction, full published layer set (32 layers)")
print()
print("Reference window, `PCA(10)` over the 32 layer-wise distances, unconstrained ridge.")
print()
print(HDR)
for m in ("satmd_alllayer", "satrmd_alllayer", "huq_satmd_alllayer", "huq_satrmd_alllayer"):
    if ALL[m]:
        print(row(m, ALL[m], f"`{m}`"))
print()
print("**Partial coverage, NOT comparable to the rows above and not for the main table:**")
print()
print(HDR)
for m in ("msp_satmd_alllayer", "msp_satrmd_alllayer"):
    if ALL[m]:
        print(row(m, ALL[m], f"`{m}` (15 cells)"))
print()
print("## 2. The middle-layer adaptation, for the appendix fidelity ladder")
print()
print("Project window. These are the rows the report currently carries.")
print()
print(HDR)
for m in ("satmd_mid", "satrmd_mid", "huq_satmd_mid", "huq_satrmd_mid",
          "md_mean_mid", "rmd_mean_mid", "hbo"):
    if MID[m]:
        print(row(m, MID[m], f"`{m}`"))
print()
print("## 3. What the full layer set buys, WINDOW MATCHED")
print()
print("The reproduction uses the reference window, so the middle-layer side of this comparison must")
print("use it too. Comparing the middle layer under the project window against the full set under")
print("the reference window would change two things at once and attribute both to the layer set.")
print()
RMDREF = load(f"results/hybrids/pdl_hybrids_rmd_refwin__{SLUG}.csv",
              {"satmd_mid", "satrmd_mid", "huq_satmd_mid", "huq_satrmd_mid"})
print("| method | ID mid (ref) | ID all (ref) | delta ID | shifted mid (ref) | shifted all (ref) "
      "| delta shifted |")
print("|---|---:|---:|---:|---:|---:|---:|")
unmatched = []
for base in ("satmd", "satrmd", "huq_satmd", "huq_satrmd"):
    a = ALL[base + "_alllayer"]
    m = RMDREF.get(base + "_mid") or REF.get(base + "_mid")
    if not a:
        continue
    if not m:
        unmatched.append(base)
        continue
    bm, ba = rung_means(m), rung_means(a)
    mi, ai = st.fmean(bm["ID"]), st.fmean(ba["ID"])
    ms = st.fmean([v for rg in SHIFTED for v in bm.get(rg, [])])
    as_ = st.fmean([v for rg in SHIFTED for v in ba.get(rg, [])])
    print(f"| `{base}` | {mi:.3f} | **{ai:.3f}** | **{ai-mi:+.3f}** | {ms:.3f} | {as_:.3f} | "
          f"{as_-ms:+.3f} |")
if unmatched:
    print()
    print(f"**NOT WINDOW MATCHED and therefore not shown: {', '.join(unmatched)}.** No "
          f"reference-window middle-layer row exists for these yet, and a comparison against their "
          f"project-window rows would confound the layer set with the window. Do not substitute one.")
print()
print("## 4. The response window, measured at the middle layer")
print()
print("Same methods, same layer, reference window against project window.")
print()
print("| method | ID ref | shifted ref | ID project | shifted project |")
print("|---|---:|---:|---:|---:|")
for m in ("hbo", "satmd_mid", "huq_satmd_mid", "md_mean_mid", "msp", "saplma"):
    if not REF[m] or not MID[m]:
        continue
    br, bm = rung_means(REF[m]), rung_means(MID[m])
    rs = st.fmean([v for rg in SHIFTED for v in br.get(rg, [])])
    ms = st.fmean([v for rg in SHIFTED for v in bm.get(rg, [])])
    print(f"| `{m}` | {st.fmean(br['ID']):.3f} | {rs:.3f} | {st.fmean(bm['ID']):.3f} | {ms:.3f} |")
print()
print("`msp` and `saplma` are the control: neither reads the per-token distance window, so neither")
print("can move, and neither does.")
print()
print("## 5. Where the numbers live")
print()
for label, path in (
        ("the reproduction", f"results/hybrids/pdl_mdlayers_alllayer__{SLUG}.csv"),
        ("its diagnostics", f"results/hybrids/pdl_mdlayers_alllayer__{SLUG}__diagnostics.csv"),
        ("middle layer, project window", f"results/hybrids/pdl_hybrids_rmd__{SLUG}.csv"),
        ("middle layer, reference window", f"results/hybrids/pdl_hybrids_refwin__{SLUG}.csv"),
        ("consolidated master", f"results/hybrids/pdl_consolidated_master__{SLUG}.csv"),
        ("per-cell distances, all 32 layers", f"results/hybrids/mdscan_refwin__{SLUG}/"),
        ("entropy settings, as measured", f"results/analysis/m11_entropy_confirm__{SLUG}.csv"),
):
    print(f"- {label}: `{path}`")
