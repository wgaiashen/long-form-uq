#!/usr/bin/env python
"""M6 POST-HOC (2026-08-20) -- WHY DOES xsum LOSE, AND WHEN DOES COMBINING HELP?


EXPLORATORY, NOT PRE-REGISTERED. Run outside the registered set AFTER the M6 verdict. It explains that
result; it does not revise it. The registered outcome stands: not established under the bar.

TWO QUESTIONS
-------------
1. WHAT PREDICTS THE GAIN? Per dataset, regress the ensemble's gain over SAPLMA on candidate predictors.
   Finding: CAWSA's own OOD level predicts it at rho = +0.952; SAPLMA's level predicts nothing (-0.071).
   xsum is then not an anomaly: CAWSA is nearly dead there (+0.043 OOD, -0.065 at DiffTask), so
   rank-averaging a strong signal with a near-dead one drags it down.

2. IS THE COMBINATION DOING ANYTHING BEYOND INTERPOLATING? `delta vs SAPLMA` is partly MECHANICAL -- a
   rank-average sits near its components midpoint, so it must correlate with the weaker component level.
   The clean quantity is

       bonus = ensemble PRR - mean(component PRRs)

   which is exactly 0 for an ensemble that merely interpolates. Finding: +0.0405 on 8/8 datasets for
   {CAWSA, SAPLMA} vs +0.0179 for the {SAPLMA, attention} control, and the difference is 8/8.

   A positive bonus is the EXPECTED direction from ordinary ensemble variance reduction, so its
   positivity alone is not news -- the PRIMARY vs CONTROL difference is. And at n = 8, 8/8 in one
   direction always gives exactly p = 0.0078, the floor of the exact test.

Pure post-hoc read of results/pdl_perex_ens/. No training, no GPU.

    python scripts/checks/ensemble_when_it_helps.py
"""
import numpy as np, sys
from scipy.stats import rankdata, spearmanr, wilcoxon
sys.path.insert(0,"src"); sys.path.insert(0,"scripts/checks")
from luq import results
from orthogonality_map import self_agreement, cross_agreement
import os
S=os.environ.get("LUQ_SLUG","meta-llama_Meta-Llama-3.1-8B")
D=os.environ.get("LUQ_PEREX", "../results/pdl_perex_ens" if S.startswith("meta-llama") else f"../results/pdl_perex_ens_{S}")
LONG=["pubmed_qa","med_quad","asqa","xsum","cnn_dailymail","samsum","expertqa","factscore"]
RUNGS=["ID","LOO-long","SameTask-long","DiffTask-long","1ds-Diff-long"]; OOD=RUNGS[1:]
def rankavg(*us): return np.mean([rankdata(u) for u in us],axis=0)
def prr_ps(y,V): return float(np.mean([results.prr(y,V[s]) for s in range(V.shape[0])]))
def ens_ps(y,A,B): return float(np.mean([results.prr(y,rankavg(A[s],B[s])) for s in range(min(len(A),len(B)))]))

Z={}
for d in LONG:
    for rg in RUNGS:
        z=np.load(f"{D}/{d}__{rg}__{S}.npz",allow_pickle=True)
        Z[(d,rg)]=({k[5:]:z[k] for k in z.files if k.startswith("unc__")}, z["y"])

print("PER-DATASET, mean over the 4 OOD rungs")
print(f"{'dataset':15s}{'SAPLMA':>9s}{'CAWSA':>9s}{'gap':>8s}{'ens':>9s}{'delta':>9s}{'disatt r':>10s}")
rows=[]
for d in LONG:
    sa=np.mean([prr_ps(Z[(d,r)][1],Z[(d,r)][0]["saplma"]) for r in OOD])
    hp=np.mean([prr_ps(Z[(d,r)][1],Z[(d,r)][0]["wmsp_shrink2"]) for r in OOD])
    en=np.mean([ens_ps(Z[(d,r)][1],Z[(d,r)][0]["wmsp_shrink2"],Z[(d,r)][0]["saplma"]) for r in OOD])
    ra=np.mean([self_agreement(Z[(d,r)][0]["wmsp_shrink2"]) for r in OOD])
    rb=np.mean([self_agreement(Z[(d,r)][0]["saplma"]) for r in OOD])
    cr=np.mean([cross_agreement(Z[(d,r)][0]["wmsp_shrink2"],Z[(d,r)][0]["saplma"]) for r in OOD])
    dis=cr/np.sqrt(max(ra,1e-9)*max(rb,1e-9))
    rows.append((d,sa,hp,sa-hp,en,en-sa,dis))
    print(f"{d:15s}{sa:>+9.3f}{hp:>+9.3f}{sa-hp:>+8.3f}{en:>+9.3f}{en-sa:>+9.3f}{dis:>10.3f}")

import numpy as np
gap=np.array([r[3] for r in rows]); dl=np.array([r[5] for r in rows]); dis=np.array([r[6] for r in rows])
hp=np.array([r[2] for r in rows]); sa=np.array([r[1] for r in rows])
print(f"\nSpearman(component gap SAPLMA-CAWSA , ensemble delta) = {spearmanr(gap,dl).statistic:+.3f}")
print(f"Spearman(CAWSA OOD level            , ensemble delta) = {spearmanr(hp,dl).statistic:+.3f}")
print(f"Spearman(disattenuated corr         , ensemble delta) = {spearmanr(dis,dl).statistic:+.3f}")
print(f"Spearman(SAPLMA OOD level           , ensemble delta) = {spearmanr(sa,dl).statistic:+.3f}")

print("\nXSUM per rung")
print(f"{'rung':16s}{'SAPLMA':>9s}{'CAWSA':>9s}{'ens':>9s}{'delta':>9s}{'disatt':>9s}")
for rg in RUNGS:
    m,y=Z[("xsum",rg)]
    sa1=prr_ps(y,m["saplma"]); hp1=prr_ps(y,m["wmsp_shrink2"]); en1=ens_ps(y,m["wmsp_shrink2"],m["saplma"])
    dis1=cross_agreement(m["wmsp_shrink2"],m["saplma"])/np.sqrt(self_agreement(m["wmsp_shrink2"])*self_agreement(m["saplma"]))
    print(f"{rg:16s}{sa1:>+9.3f}{hp1:>+9.3f}{en1:>+9.3f}{en1-sa1:>+9.3f}{dis1:>9.3f}")

# ---------------- part 2: the non-tautological bonus ----------------
def ps(y,V): return float(np.mean([results.prr(y,V[s]) for s in range(V.shape[0])]))
def en(y,A,B): return float(np.mean([results.prr(y,rankavg(A[s],B[s])) for s in range(min(len(A),len(B)))]))
Z={}
for d in LONG:
    for rg in OOD:
        z=np.load(f"{D}/{d}__{rg}__{S}.npz",allow_pickle=True)
        Z[(d,rg)]=({k[5:]:z[k] for k in z.files if k.startswith("unc__")}, z["y"])
print("COMPLEMENTARITY BONUS = ensemble PRR - mean(component PRRs).")
print("This removes the mechanical part: an ensemble that merely interpolates scores 0.\n")
print(f"{'dataset':15s}{'PRIMARY bonus':>15s}{'CONTROL bonus':>15s}")
bp,bc=[],[]
for d in LONG:
    p=np.mean([en(Z[(d,r)][1],Z[(d,r)][0]["wmsp_shrink2"],Z[(d,r)][0]["saplma"])
               -0.5*(ps(Z[(d,r)][1],Z[(d,r)][0]["wmsp_shrink2"])+ps(Z[(d,r)][1],Z[(d,r)][0]["saplma"])) for r in OOD])
    c=np.mean([en(Z[(d,r)][1],Z[(d,r)][0]["saplma"],Z[(d,r)][0]["attention"])
               -0.5*(ps(Z[(d,r)][1],Z[(d,r)][0]["saplma"])+ps(Z[(d,r)][1],Z[(d,r)][0]["attention"])) for r in OOD])
    bp.append(p); bc.append(c); print(f"{d:15s}{p:>+15.4f}{c:>+15.4f}")
bp,bc=np.array(bp),np.array(bc)
for nm,v in [("PRIMARY {CAWSA,SAPLMA}",bp),("CONTROL {SAPLMA,attn}",bc)]:
    p=wilcoxon(v,alternative="two-sided",method="exact").pvalue
    print(f"\n{nm}: macro {v.mean():+.4f}  median {np.median(v):+.4f}  positive {int((v>0).sum())}/8  exact Wilcoxon p={p:.4f}")
d=bp-bc; p=wilcoxon(d,alternative="two-sided",method="exact").pvalue
print(f"\nPRIMARY bonus - CONTROL bonus: macro {d.mean():+.4f}  {int((d>0).sum())}/8  exact Wilcoxon p={p:.4f}")
