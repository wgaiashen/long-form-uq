# Pre-registration — are PubMedQA's punctuation tokens carrying the signal?

> **Recorded outcome:** run. The premise it was written to test was only weakly supported, so the causal reading that depended on it is stated as a hypothesis rather than as established.

**Written 2026-08-02, BEFORE the ablation is run.** Registered in advance because it predicts that a
premise the project has repeated for weeks is WRONG, and a prediction of that shape is worthless if
produced after seeing the result.

---

## The premise being tested

The project has said, repeatedly and in the slides, that **"pubmed's attention is spread on unhelpful
tokens"**, and three Track B experiments (B.1 auxiliary-loss supervision, B.2 selective heads, B.3
entropy penalty) were designed to move or sharpen that attention.

**We never showed the tokens are unhelpful. We showed they are punctuation.** From
`results/pool_peaks_9dataset.csv` (Llama-3.1-8B, layer 15, the trained attention pooler):

| pubmed_qa | content mass | punct mass | peak token is punct/space | peak relative position | entropy ratio |
|---|---|---|---|---|---|
| **ID** | 0.333 | **0.515** | **94.95%** | **0.114** | 0.606 |
| LOO | 0.577 | 0.112 | 10.6% | 0.434 | 0.945 |
| DiffTask | 0.574 | 0.113 | 11.2% | 0.464 | 0.961 |

**The configuration we called defective achieves the highest ID PRR in the study** (0.7127–0.7371),
while the content-attending OOD configurations score far lower. A delimiter's residual state summarising
the preceding clause is a standard probing phenomenon, so "punctuation" and "uninformative" are not the
same claim and only one of them has been measured.

## REGISTERED PREDICTIONS

**P1 (primary).** Removing the punctuation/space positions from pubmed's pooling window, retraining, will
**REDUCE** pubmed ID PRR — by more than the random-ablation control.

**P2.** The effect is **pubmed-specific in magnitude**: xsum (punct mass 0.069) and cnn (0.059) should
move much less, because there is far less punctuation mass to remove.

**P3.** If P1 holds, then **B.1 failed on pubmed because it pushed attention AWAY from the informative
tokens** (B.1's real−baseline on pubmed was −0.0449 for the NLL target and −0.0415 for content-mass —
both moving attention toward content and surprisal). That is a post-hoc explanation of an existing
result, and is labelled as such; it is not independent evidence.

**What would falsify P1:** punctuation ablation leaves pubmed's PRR flat or improves it. Then the
punctuation mass is genuinely inert, the original "unhelpful tokens" framing is vindicated, and the
attention-redirection line is alive after all. Either outcome is reportable and the write-up is decided
by the measurement, not by which one we prefer.

## THE CONFOUND, AND WHY THERE ARE FOUR ARMS

Pubmed's peak sits at relative position **0.114** and is punctuation **94.95%** of the time. **Punctuation
and position are confounded in the data**, so a two-arm test (baseline vs punct-ablated) cannot say which
one carries the signal. Four arms, on the SAME population:

| arm | what is removed |
|---|---|
| **baseline** | nothing |
| **punct** | every punctuation/space position |
| **posmatch** | the same NUMBER of NON-punctuation tokens, chosen at the same relative positions |
| **random** | the same number of tokens, uniformly at random |

**Registered reading, fixed before the run:**

| outcome | conclusion |
|---|---|
| punct hurts, posmatch does not | it is the **delimiters** (token identity) |
| punct and posmatch hurt about equally | it is the **position**, not punctuation |
| neither hurts beyond random | the signal is elsewhere; the premise was wrong in a third way |

**This distinction decides what R6's attention-supervision target should be**, which is why the two extra
arms are worth their cost.

## Controlled variables (fixed in advance so an arm difference is not a tuning difference)

- **Same population across arms.** Masks for all four arms are computed first; any example where ANY arm
  would leave fewer than 4 tokens is dropped from ALL arms. A per-arm population would make the
  comparison a population difference.
- **Temperature fixed at T=1.0 for every arm and every dataset.** Selecting T per arm would let the
  baseline and the ablations differ in two ways at once. Recorded as a deliberate constant; it means
  these PRRs are NOT comparable to the ladder's temperature-selected numbers, and they are never merged
  with them.
- **Masking applies at TRAIN and TEST.** Masking only at test measures robustness to a shift we
  introduced, not the contribution of the tokens.
- **3 seeds (1,2,3), ID rung, layer 15**, the project's standard.
- **Three aggregators**: `saplma` (mean-pool + MLP), `uniform` (frozen-query pooler), `attention`
  (learned-query pooler). If ablation hurts SAPLMA too, the tokens carry signal independently of the
  attention mechanism; if it hurts only `attention`, it is about what the pooler learned to key on.

## Population

v1 generations, `cache/records` + `cache/pertok` (pubmed_qa, xsum, cnn_dailymail — all three in the
canonical namespace, none regenerated). Label `correctness`, judge gpt-5-mini. **This is the v1 research
workstream and is independent of the med_quad regeneration running on DoC.**

Output: `results/regime_R0_punct_ablation__<slug>.csv` — the `regime_*` prefix keeps it outside every
master-assembler glob.
