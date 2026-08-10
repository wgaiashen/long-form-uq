# PRE-REGISTRATION — W6: Lehmer β = 1 on the Qwen2.5-14B grid (out-of-sample)

> **Status (2026-08-10):** run. Q1 fails (margin +0.0137 and 7/8 pass, Wilcoxon p = 0.195, dragged by expertqa, which carries a severe length confound on this population); Q2 replicates on 1 of 3 (cnn_dailymail, boot p = 0.0020); Q3 (secondary) p = 0.0443. Record: `STOCKTAKE_qwen.md` §11. *(Wording edited 2026-08-10, cosmetic only — registered claims, thresholds and bars are unchanged; git history is the proof.)*

**Written 2026-08-09, BEFORE any Qwen record, label or ladder number exists.** Qwen generation has
started on DoC; no per-token logprobs have been scored against any label. Every choice below is
derived from **Llama-3.1-8B data only**.

Origin: the sharpening-axis line on Llama (project log: `PLAN_sharpening_axis.md`; Llama results
in `STOCKTAKE_sharpening_axis.md` §7.1b–c). Runs on the **Qwen population** as defined by
`PLAN_execution_post7Aug.md` — its splits, its carve, its judge labels — with **nothing re-selected
on Qwen data**.

---

## 1. What is being tested, in one paragraph

The Lehmer family scores an answer as a surprisal-weighted mean of its own token NLLs,
`q = Σ nll^(β+1) / Σ nll^β`: β = 0 is `perplexity`, β → ∞ is `msp_min`'s ranking. On Llama, an
honestly-selected interior β beat `msp_min` on 6/8 datasets (+0.027) but **missed the registered
cross-dataset bar** (Wilcoxon p = 0.250, n = 8), while the per-dataset wins were **significant on
exactly the three summarisation datasets** (paired bootstrap p = 0.0005 each, Bonferroni-safe) and
the two non-significant losses were the two most concentrated sets, where the max endpoint is already
optimal. `prereg/W4` §0 registered that no Llama pass could be a claim until it replicated
out-of-sample. **This is that replication.**

## 2. THE PRE-COMMITTED VALUE: β = 1, global, fixed here

Justification, from Llama only: β = 1 is (a) the **argmax of the Llama cross-dataset mean**
(+0.233, vs +0.229 at β = 2 and +0.186 at the `msp_min` endpoint), and (b) the **modal
leave-one-dataset-out selection** (6 of 8 folds). No Qwen information enters this choice.

⚠️ **If β = 1 fails, that is a failure of the pre-committed value.** No other β is promoted
afterwards; the full curve is reported for the record only.

## 3. REGISTERED CLAIMS AND BARS

**Q1 — cross-dataset (the claim that failed on Llama, now genuinely out-of-sample).**
Lehmer β = 1 vs Qwen's own `msp_min`, per-dataset mean over the 8 Qwen long evals, all three of:
margin > **+0.010** · signs ≥ **6/8** · two-sided Wilcoxon **p < 0.05**.

**Q2 — the family-level claim (the one Llama established).** On **each** of Qwen's three
summarisation datasets (`xsum`, `cnn_dailymail`, `samsum`), Lehmer β = 1 beats `msp_min` by
per-example paired bootstrap (2000 resamples) at **p < 0.0167** (Bonferroni over the 3).
**Registered directional prediction:** NO significant win on the concentrated/factuality sets
(`pubmed_qa`, `factscore`) — a significant win there would be *against* the regime map and is to be
reported as a surprise, not pooled into a success.

**Q3 — secondary, pooled.** Wilcoxon over the 16 paired per-dataset differences (8 Llama + 8 Qwen),
reported alongside, never instead of Q1.

## 4. REGISTERED FAILURE READINGS

- Q1 fails, Q2 passes → the cross-dataset claim is dead; the **family-level claim replicates** and is
  the reportable result (regime map made actionable).
- Q1 and Q2 both fail → the Llama summarisation wins were population-specific; **the Lehmer line
  closes fully**, and that closure is final.
- Q2 passes only on some of the three → partial replication, reported per dataset, no pooling.

## 5. MECHANICS (why this can run early, and the trap it avoids)

Lehmer is training-free ⇒ **rung-invariant** ⇒ it needs only Qwen **records + judge labels** — no
probes, no per-token hidden states, no ladder. It can be computed the day the Qwen labelling lands,
before any probe trains. Convention identical to Llama: **all generated tokens** (Lehmer uses no
content mask, so the Llama-only `id ≥ 128000` special-token rule — the known Qwen port trap in
`weighted_msp.py:185-193` — is **not touched by this method**). NLLs from `token_logprobs` as cached.
Test rows: the Qwen replication's own `eval_split`/carve, unmodified. Driver: `sharpening_family.py`'s
`score_lehmer` with a `--model` pin — **fail loud on any model-agnostic glob.**

⚠️ **This pre-registration is committed and pushed before any Qwen ladder runs.** It changes
nothing in the Qwen replication plan (no new jobs, no new selection) — it only fixes, in advance, how one free
method will be read on data that does not exist yet.
