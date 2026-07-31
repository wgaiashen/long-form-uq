"""Emit per-dataset PRR tables for the deck, READ STRAIGHT from pdl_master (no recomputation).
Rows = the 12 requested methods; columns = the 5 settings in Hidden-Failures order (ID, LOO, SameTask,
DiffTask, 1D-DiffTask). Missing cells -> "—" (listed with reason). Flags any (eval,rung,method) that has
>1 disagreeing value in the master (that would be a bug to know before the deck)."""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "pdl_master__meta-llama_Meta-Llama-3.1-8B.csv"

ROWS = [("MSP-sum", "msp_sum"), ("Perplexity", "perplexity"), ("MSP-min", "msp_min"),
        ("Weighted-MSP (norm)", "wMSP-norm"), ("SAPLMA", "SAPLMA"),
        ("Mean-pool / uniform (armB)", "armB(mean-pool)"), ("Attention pooler (armA)", "armA(attention)"),
        ("Prior-init, learns away (armD, NLL)", "armD:NLL"), ("Fixed prior (armC, NLL)", "armC:NLL"),
        ("Multi-head (MH)", "multi-head(MH)"), ("Ensemble MSP + SAPLMA", "ens{MSP,SAPLMA}"),
        ("Ensemble weighted-MSP + SAPLMA", "ens{wMSP,SAPLMA}")]
RUNGS = [("ID", "ID"), ("LOO", "LOO-long"), ("SameTask", "SameTask-long"),
         ("DiffTask", "DiffTask-long"), ("1D-DiffTask", "1ds-Diff-long")]
EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
PROBES = {"SAPLMA", "armB(mean-pool)", "armA(attention)", "armD:NLL", "armC:NLL", "multi-head(MH)"}

vals = defaultdict(list)
for r in csv.DictReader(open(CSV)):
    try:
        vals[(r["eval"], r["rung"], r["method"])].append(float(r["prr"]))
    except (ValueError, KeyError):
        pass


def cell(ev, rk, mk):
    vs = vals.get((ev, rk, mk), [])
    if not vs:
        return None, False
    dup = len(vs) > 1 and (max(vs) - min(vs) > 1e-9)
    return vs[0], dup


def fmt(x):
    return "—" if x is None else f"{x:+.3f}"


missing, dups = [], []
wide = [["eval", "method"] + [c for c, _ in RUNGS]]
for ev in EVALS:
    print(f"\n### {ev}\n")
    print("| Method | " + " | ".join(c for c, _ in RUNGS) + " |")
    print("|" + "---|" * (len(RUNGS) + 1))
    for label, mk in ROWS:
        cells = []
        wrow = [ev, label]
        for col, rk in RUNGS:
            x, dup = cell(ev, rk, mk)
            if x is None:
                missing.append(f"{ev} / {label} / {col}")
            if dup:
                dups.append(f"{ev} / {label} / {col} -> {vals[(ev, rk, mk)]}")
            cells.append(fmt(x)); wrow.append("" if x is None else f"{x:+.3f}")
        print(f"| {label} | " + " | ".join(cells) + " |")
        wide.append(wrow)

    # analysis line
    def get(mk, rk):
        x, _ = cell(ev, rk, mk); return x
    id_scores = {label: get(mk, "ID") for label, mk in ROWS if get(mk, "ID") is not None}
    ood_avg = {}
    for label, mk in ROWS:
        oo = [get(mk, rk) for _, rk in RUNGS[1:] if get(mk, rk) is not None]
        if len(oo) == 4:
            ood_avg[label] = sum(oo) / 4
    best_id = max(id_scores, key=id_scores.get)
    best_ood = max(ood_avg, key=ood_avg.get)
    # msp_min beats every probe at any setting?
    beats = []
    for col, rk in RUNGS:
        mm = get("msp_min", rk)
        pv = [get(mk, rk) for mk in PROBES if get(mk, rk) is not None]
        if mm is not None and pv and mm >= max(pv):
            beats.append(col)
    bstr = ("YES at " + ", ".join(beats)) if beats else "no (some probe ≥ MSP-min at every setting)"
    print(f"\n> **{ev}** — strongest at ID: **{best_id}** ({id_scores[best_id]:+.3f}); strongest over the 4 OOD "
          f"(mean): **{best_ood}** ({ood_avg[best_ood]:+.3f}); MSP-min beats every hidden-state probe: **{bstr}**.")

out = ROOT / "results" / "deck_per_dataset_tables__meta-llama_Meta-Llama-3.1-8B.csv"
with open(out, "w", newline="") as fh:
    csv.writer(fh).writerows(wide)
print(f"\n\nWIDE CSV: {out}")
print(f"SOURCE: {CSV}")
print(f"\nMISSING CELLS ({len(missing)}): " + ("; ".join(missing) if missing else "none"))
print(f"DUPLICATE/DISAGREEING CELLS ({len(dups)}): " + ("; ".join(dups) if dups else "none — every cell unique"))
