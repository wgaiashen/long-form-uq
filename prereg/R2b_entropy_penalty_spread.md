# PRE-REGISTRATION — R2b: B.3 (one-sided entropy penalty) on the untested SPREAD regime

**Written 2026-08-03, BEFORE the run.** Registered because it CORRECTS a justification that appears in
the approved plan, and a corrected prediction written after seeing the result would be worthless.

---

## The correction: the plan's reason for running this is backwards

`PLAN` §R2 says to run B.3 on a SPREAD dataset because SPREAD is *"the regime never tested, and the one
where a sharpness penalty has a mechanism."* **The second half is wrong**, and it is wrong in a way that
is checkable from unlabelled data before spending anything.

B.3's loss is `L = L_task + lambda * mean( relu(tau - H_norm)^2 )` — it penalises normalised attention
entropy **BELOW** tau. It therefore **broadens attention that is too SHARP**, and is **exactly zero** for
any example already broader than tau. Confirmed in the outcome we already have: pubmed's entropy went
**up** under the penalty, 0.6065 → 0.8737.

So the penalty has headroom where attention is **sharp**, not where it is spread. Baseline normalised
entropy (`results/pool_peaks_9dataset.csv`, ID rung):

| dataset | regime | normalised entropy | headroom for a broadening penalty |
|---|---|---|---|
| **pubmed_qa** | CONCENTRATED | **0.6065** | large — and it was tested, and it **cost 0.018 PRR** |
| samsum | SPREAD | 0.8387 | small |
| asqa | SPREAD | 0.8333 | small |
| xsum | NOT-IN-PROB | 0.8455 | small — tested, null (+0.0009) |
| **cnn_dailymail** | SPREAD | **0.9410** | **smallest of all 8** |

**The regime with the least room for this intervention is SPREAD.** The plan had it exactly inverted, and
so did the R2 write-up before this check. Running cnn expecting a win would have been running the arm
where the mechanism is weakest and reading the inevitable null as evidence about the method.

## REGISTERED PREDICTIONS

**P1 (primary).** On cnn_dailymail, B.3 is **null**: |delta PRR| ≤ 0.01 at the selected (tau, lambda).

**P2 (the mechanism, which is what makes P1 informative rather than vacuous).** The null is accompanied by
a **low `frac_bound_before`** — the penalty acts on few examples because most are already broader than
tau. Registered ordering, from the entropies above: **frac_bound(cnn) < frac_bound(samsum) ≈
frac_bound(xsum) < frac_bound(pubmed)**. pubmed's was 0.968 and xsum's 0.828 at tau=0.9.

**P3.** samsum (entropy 0.8387, closest to xsum's 0.8455) behaves like xsum: null, |delta| ≤ 0.01.

**What would falsify P1/P2 — and this is the outcome worth having:** cnn shows a delta beyond ±0.01
**with** a low frac_bound. That would mean a penalty acting on a small minority of examples moved the
whole dataset's PRR, which the "no headroom" story cannot explain, and the intervention would be doing
something other than what its loss says. A large delta WITH a high frac_bound would instead mean the
entropy summary statistic is hiding a sharp subpopulation, i.e. the dataset-level mean misled us.

## Why run it at all, given P1 predicts a null

Because the null is **mechanistically predicted and independently checkable** via `frac_bound`, it
converts "three Track B nulls" into a statement with a reason attached: *the intervention needs sharp
attention to act on, and only one of our datasets has it — the one where broadening the attention
destroyed PRR.* That is a finding about **when the method can help at all**, decidable from unlabelled
data before running, which is the shape of claim this project wants.

It does **not** test "does a sharpness penalty help SPREAD datasets" — nothing tests that, because
B.3 cannot act there. If we want an intervention for SPREAD, it must be one that **sharpens** broad
attention (the opposite sign), and that is a different experiment which is **not** registered here.

## Population and settings

v1 generations, `cache/records` + `cache/pertok`, `cnn_dailymail` and `samsum`, ID rung, layer 15,
seed 1, the **same** tau/lambda grid as the two tested datasets (`--taus 0.5,0.6,0.7,0.8,0.9
--lambdas 0.5,1.0,2.0`) — a different grid would make the comparison a tuning difference.
Output: `results/regime_R2b_entropy_penalty_spread__<slug>.csv`.
