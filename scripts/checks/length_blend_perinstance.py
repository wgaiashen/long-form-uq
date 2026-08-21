#!/usr/bin/env python
"""W7 -- the PER-INSTANCE length blend of msp_min and SAPLMA (the 7 Aug §6, prereg W7).

u_i = w_i*z(msp_min)_i + (1-w_i)*z(saplma)_i,  w_i = exp(-len_i/L).  ONE parameter, LODO-selected.
Reads results/pdl_perex/ (40 cells, 3 seeds). Join gate: recomputed msp_min must match the sidecar
bit-for-bit. Registered expectation: NULL (prereg W7).
"""
import sys, glob
from pathlib import Path
import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "checks"))
from luq import cache, msp, results
from luq.config import Config
from xl_rungs import eval_split, label_of
from attn_pool import PROMPT_REGIME

M = "meta-llama/Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
OOD = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
LGRID = [8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]


def z(v):
    s = np.std(v)
    return (v - np.mean(v)) / s if s > 0 else v * 0.0


def main():
    lens, cells = {}, {}
    for d in LONG:
        cfg = Config(model_name=M, dataset=d, ood_setting="ID", prompt_regime=PROMPT_REGIME.get(d, ""))
        recs = cache.load_records(cfg.cache_dir, cache.run_key(M, d, "ID"))
        lf = label_of(d)
        y = np.array([r.get(lf, np.nan) for r in recs], float)
        fin = np.isfinite(y)
        recs = [recs[i] for i in np.where(fin)[0]]
        _, te = eval_split(np.array([r["split"] for r in recs]))
        lens[d] = np.array([len(recs[i]["token_logprobs"]) for i in te], float)
        vmin = np.array([msp.msp_uncertainty(recs[i]["token_logprobs"], "min") for i in te])
        for rg in OOD:
            p = ROOT / "results" / "pdl_perex" / f"{d}__{rg}__meta-llama_Meta-Llama-3.1-8B.npz"
            if not p.exists():
                print(f"SKIP LOUDLY: {p.name} missing"); continue
            zf = np.load(p, allow_pickle=True)
            fm = zf["unc__floor_min"][0]
            # 1e-9, not bit-equality: the npz round-trip differs in the last ulp (max 1.1e-16,
            # measured) while order and values align. Bit-equality would reject a correct join.
            if fm.shape != vmin.shape or float(np.max(np.abs(fm - vmin))) > 1e-9:
                print(f"JOIN GATE FAIL {d}/{rg}: recomputed msp_min != sidecar -- cell skipped LOUDLY")
                continue
            cells[(d, rg)] = (vmin, zf["unc__saplma"], zf["y"])
    print(f"joined {len(cells)}/32 OOD cells (gate: bit-identical msp_min)")

    def prr_blend(d, rg, L):
        vmin, sap, y = cells[(d, rg)]
        w = np.exp(-lens[d] / L)
        vals = [results.prr(y, w * z(vmin) + (1 - w) * z(sap[s])) for s in range(sap.shape[0])]
        return float(np.mean(vals))

    per = {d: {L: np.mean([prr_blend(d, rg, L) for rg in OOD if (d, rg) in cells]) for L in LGRID}
           for d in LONG}
    base_s = {d: np.mean([np.mean([results.prr(cells[(d, rg)][2], cells[(d, rg)][1][s])
                                   for s in range(3)]) for rg in OOD if (d, rg) in cells])
              for d in LONG}
    base_m = {d: np.mean([results.prr(cells[(d, rg)][2], cells[(d, rg)][0])
                          for rg in OOD if (d, rg) in cells]) for d in LONG}
    print(f"\n{'eval':15s}{'msp_min':>9s}{'SAPLMA':>9s}" + "".join(f"L={int(L)}" .rjust(8) for L in LGRID)
          + f"{'LODO':>9s}{'pick':>7s}")
    lodo = {}
    for d in LONG:
        others = [o for o in LONG if o != d]
        Lp = max(LGRID, key=lambda L: np.mean([per[o][L] for o in others]))
        lodo[d] = (Lp, per[d][Lp])
        print(f"{d:15s}{base_m[d]:>+9.3f}{base_s[d]:>+9.3f}"
              + "".join(f"{per[d][L]:>+8.3f}" for L in LGRID) + f"{per[d][Lp]:>+9.3f}{int(Lp):>7d}")
    dd = np.array([lodo[d][1] - base_s[d] for d in LONG])
    p = st.wilcoxon(dd).pvalue if not np.allclose(dd, 0) else 1.0
    print(f"\nLODO blend mean {np.mean([v for _, v in lodo.values()]):+.4f}  "
          f"vs SAPLMA {np.mean(list(base_s.values())):+.4f}  vs msp_min {np.mean(list(base_m.values())):+.4f}")
    print(f"REGISTERED BAR vs SAPLMA: margin {dd.mean():+.4f} (> +0.010?)  "
          f"signs {int((dd > 0).sum())}/8 (>= 6?)  Wilcoxon p={p:.4f} (< 0.05?)")
    ok = dd.mean() > 0.010 and int((dd > 0).sum()) >= 6 and p < 0.05
    print(f"VERDICT: {'YES' if ok else 'NO'}")


if __name__ == "__main__":
    main()
