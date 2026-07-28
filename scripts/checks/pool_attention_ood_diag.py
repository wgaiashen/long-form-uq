"""B-GATE / E2 diagnostic: is the pooler's attention SHARPER (mass dumped on a few tokens) OOD than ID?

This is the gate for Workstream C. C1 (entropy moderation) only makes sense if OOD attention is measurably
sharper than ID; if the win case and the collapse case look the SAME, sharpness is not the mechanism and C1
will not work regardless of direction (the single most informative comparison in the gate -- so the script
reports it explicitly and is allowed to conclude "ambiguous").

READS the viz sidecars written by dump_viz_attention.py: `cache/viz/<key>__attn[<suffix>].npz`, each holding
`record_pos_all` + `pool_w` (per-example softmax weight vectors, length G+1: row 0 = last PROMPT token). By
convention the ID sidecar has NO suffix; an OOD sidecar (pooler TRAINED on another source, applied to this
eval's test set) is suffixed `__<rung>` e.g. `..__attn__DiffTask.npz` (E1 writes these).

Per (dataset, rung) it computes, PER EXAMPLE (not just the mean -- a mean shift can hide "most fine, a tail
of catastrophic collapses"):
  - NORMALISED entropy H/log T and EFFECTIVE SUPPORT exp(H). NORMALISED is non-negotiable: raw H scales with
    sequence length and we compare sciq (~2 tok) against expertqa (~384), so raw H would find a length
    artefact and call it sharpness.
  - TOP-k mass (fraction of weight on the top 1/5/10 tokens) -- the most direct reading of "mass on a few".
  - PEAK position (relative, 0=first gen tok .. 1=last) and weight on row0 (the boundary artefact guard).
  - CONTENT vs PUNCT vs SPECIAL mass split (mass-weighted over the whole distribution, not just the peak).
Reports the DISTRIBUTION (p10/p50/p90), not only the mean. Then ID->OOD deltas, the rung-severity trend, and
a REQUIRED one-line verdict at the top: sharper / flatter / ambiguous.

Read-only.  Examples:
    python scripts/checks/pool_attention_ood_diag.py --datasets sciq,pubmed_qa,xsum   # ID panels (validate)
    python scripts/checks/pool_attention_ood_diag.py --datasets cnn_dailymail,expertqa --compare-outcome
"""
import argparse
import glob
import os
import sys
from pathlib import Path

# read-only diagnostic on CACHED data -- never needs the network; force offline so the tokenizer load does
# not hang checking the hub (this is what stalled it on the login node).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # so the sibling module imports directly
from luq import cache                              # noqa: E402
from pool_peak_tokens import classify              # reuse the SAME token classifier  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
MODEL_SLUG = "meta-llama_Meta-Llama-3.1-8B"
# §C.3 OOD outcome per eval (from the ladder): does the pooler WIN or COLLAPSE OOD? Used only to label the
# cross-outcome comparison -- NOT to influence any number.
OUTCOME = {"cnn_dailymail": "collapse", "pubmed_qa": "null", "expertqa": "win", "xsum": "win"}


def sidecars_for(dataset, cache_dir):
    """{rung: path} for every viz sidecar of this eval. ID = no suffix; OOD suffix = the rung."""
    key = cache.run_key(MODEL, dataset, "ID")
    out = {}
    for p in sorted(glob.glob(str(cache_dir / "viz" / f"{key}__attn*.npz"))):
        stem = Path(p).name[len(f"{key}__attn"):-len(".npz")]     # "" for ID, "__DiffTask" for OOD
        out["ID" if stem == "" else stem.lstrip("_")] = p
    return out


_CLASS_CACHE = {}


def classify_id(tid, tok, special_ids):
    """Memoised token-id -> class. A token id always maps to the same class, so decode ONCE (the per-token
    tok.decode was the bottleneck that hung the login node -- ~100k decodes -> a few thousand unique)."""
    c = _CLASS_CACHE.get(tid)
    if c is None:
        c = classify(tok.decode([tid]), tid, special_ids)
        _CLASS_CACHE[tid] = c
    return c


def panel(sidecar_path, records, special_ids, tok):
    """Per-example arrays of the sharpness/placement statistics for one sidecar."""
    z = np.load(sidecar_path, allow_pickle=True)
    pool = dict(zip(z["record_pos_all"].tolist(), z["pool_w"]))
    ne, eff, t1, t5, t10, ppos, row0 = [], [], [], [], [], [], []
    c_mass, p_mass, s_mass = [], [], []
    conc, effrac = [], []          # P1: length-comparable ratios (concentration vs uniform; eff-support fraction)
    for i, w in pool.items():
        w = np.asarray(w, float)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        gids = records[i]["gen_token_ids"]
        G = len(gids)
        if len(w) != G + 1:                          # alignment guard (G+1: row0 = last-prompt token)
            continue
        T = len(w)
        H = float(-(w * np.log(w + 1e-12)).sum())    # raw entropy (nats)
        ne.append(H / np.log(T))                      # NORMALISED entropy (length-invariant)
        eff.append(float(np.exp(H)))                  # effective support (#tokens the mass spreads over)
        sw = np.sort(w)[::-1]
        t1.append(float(sw[:1].sum())); t5.append(float(sw[:5].sum())); t10.append(float(sw[:10].sum()))
        conc.append(float(sw[0] * T))                 # top-1 mass / uniform (1/T): 1.0 = uniform, higher = peaked
        effrac.append(float(np.exp(H) / T))           # effective support as a FRACTION of T (length-comparable)
        row0.append(float(w[0]))
        gw = w[1:]
        j = int(np.argmax(gw))
        ppos.append(j / max(G - 1, 1))
        cm = pm = sm = 0.0
        for k, tid in enumerate(gids):               # mass-weighted content/punct/special split
            cls = classify_id(int(tid), tok, special_ids)
            if cls == "content":
                cm += gw[k]
            elif cls == "punct_space":
                pm += gw[k]
            elif cls == "special":
                sm += gw[k]
        c_mass.append(cm); p_mass.append(pm); s_mass.append(sm)
    return {k: np.array(v) for k, v in dict(
        norm_entropy=ne, eff_support=eff, top1=t1, top5=t5, top10=t10,
        peak_pos=ppos, row0=row0, content_mass=c_mass, punct_mass=p_mass, special_mass=s_mass,
        concentration=conc, eff_support_frac=effrac).items()}


def pct(a):
    return f"{np.mean(a):.3f} [p10 {np.percentile(a,10):.3f} p50 {np.percentile(a,50):.3f} p90 {np.percentile(a,90):.3f}]"


def show(tag, P):
    print(f"\n#### {tag}  (n={len(P['norm_entropy'])}) ####")
    print(f"  norm-entropy H/logT : {pct(P['norm_entropy'])}   (LOWER = sharper; NOTE: ceiling ~0.9, low resolution above)")
    print(f"  eff-support exp(H)  : {pct(P['eff_support'])}   (#tokens mass spreads over)")
    print(f"  eff-support / T     : {pct(P['eff_support_frac'])}   (fraction of tokens; length-comparable)")
    print(f"  concentration (t1*T): {pct(P['concentration'])}   (1.0 = uniform; higher = more peaked)")
    print(f"  top-1 mass          : {pct(P['top1'])}")
    print(f"  top-5 mass          : {pct(P['top5'])}")
    print(f"  top-10 mass         : {pct(P['top10'])}")
    print(f"  peak position (rel) : {pct(P['peak_pos'])}")
    print(f"  content / punct / special mass : "
          f"{np.mean(P['content_mass']):.2f} / {np.mean(P['punct_mass']):.2f} / {np.mean(P['special_mass']):.2f}")
    print(f"  weight on row0      : mean {np.mean(P['row0']):.3f}  (boundary-artefact guard, ~0 wanted)")


def verdict(id_ne, ood_ne):
    """one-line sharper/flatter/ambiguous on the ID->OOD normalised-entropy shift."""
    d = float(np.mean(ood_ne) - np.mean(id_ne))
    if d < -0.03:
        return f"OOD attention is SHARPER than ID (norm-entropy {np.mean(id_ne):.3f} -> {np.mean(ood_ne):.3f}, Δ{d:+.3f})"
    if d > 0.03:
        return f"OOD attention is FLATTER than ID (norm-entropy {np.mean(id_ne):.3f} -> {np.mean(ood_ne):.3f}, Δ{d:+.3f})"
    return f"AMBIGUOUS: OOD attention ~ ID sharpness (norm-entropy Δ{d:+.3f}, |Δ|<0.03)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sciq,trivia_qa,pubmed_qa,xsum")
    ap.add_argument("--compare-outcome", action="store_true",
                    help="contrast the ID->OOD entropy shift in the WIN case vs the COLLAPSE case (the gate)")
    args = ap.parse_args()
    cache_dir = ROOT / "cache"
    # regime-namespaced sets keep their records under cache/<regime>/records/ (sidecars are still in cache/viz).
    REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}
    tok = AutoTokenizer.from_pretrained(MODEL)
    special_ids = set(getattr(tok, "all_special_ids", []))

    per_ds = {}
    for d in args.datasets.split(","):
        scs = sidecars_for(d, cache_dir)
        if not scs:
            print(f"\n#### {d}: no sidecar found -- run dump_viz_attention first. skipped.")
            continue
        rec_dir = cache_dir / REGIME[d] if d in REGIME else cache_dir
        recs = cache.load_records(rec_dir, cache.run_key(MODEL, d, "ID"))
        panels = {rung: panel(p, recs, special_ids, tok) for rung, p in scs.items()}
        per_ds[d] = panels
        print(f"\n================ {d}  (rungs: {', '.join(panels)}) ================")
        for rung in panels:
            show(f"{d} / {rung}", panels[rung])
        if "ID" in panels:
            ood_rungs = [r for r in panels if r != "ID"]
            for r in ood_rungs:                       # ID->OOD verdict + rung trend
                print(f"\n  >>> {d}  ID -> {r}: {verdict(panels['ID']['norm_entropy'], panels[r]['norm_entropy'])}")
            if not ood_rungs:
                print(f"\n  (only the ID sidecar present for {d} -- OOD sidecars pending E1; ID panel validated)")

    if args.compare_outcome:
        print("\n================ CROSS-OUTCOME (the mechanism test) ================")
        print("If sharpness is the SAME in the win case and the collapse case, sharpness is NOT the mechanism.")
        for d, panels in per_ds.items():
            if "ID" in panels and len(panels) > 1:
                r = [x for x in panels if x != "ID"][0]
                dne = float(np.mean(panels[r]['norm_entropy']) - np.mean(panels['ID']['norm_entropy']))
                print(f"  {d:14s} [{OUTCOME.get(d,'?'):8s}]  ID->OOD norm-entropy Δ = {dne:+.3f}")


if __name__ == "__main__":
    main()
