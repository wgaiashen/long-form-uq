"""AlignScore correctness label: a FREE, local long-form correctness signal.

This is the eval metric the runnable Hidden Failures pipeline actually uses for long-form
PRR (`run_polygraph.py:377-382` returns AlignScore among the generation metrics; the LLM
judge is only the probe's TRAINING target). It is a RoBERTa-large NLI model run locally, so
unlike the OpenAI judge it costs NOTHING per call -- it just needs a GPU.

We score the MODEL OUTPUT against the GOLD reference, matching the direction exactly
(`Temp_robust_UQ_probes/utils/alignscore.py:67-70`): `scorer.score(claims=gold,
contexts=output)`, evaluation_mode "nli_sp". The scorer machinery is vendored verbatim in
`_alignscore_utils.py` (from the reference `lm_polygraph_lite/.../alignscore_utils.py`, itself
adapted from yuh-zha/AlignScore). Verify against the authors' code with
scripts/checks/alignscore_vs_authors.py.

Returns a float in [0, 1] (higher = output better supported by the gold), or None on error.
For multi-reference golds (e.g. trivia_qa alias lists) we take the MAX over references, the
same aggregation the AggregatedMetric uses.

Runs on the LOGIN node? No -- it's a GPU model. Run the AlignScore labelling pass via Slurm,
not the login node (the OpenAI judge is the login-node one; this is not).
"""
from typing import Optional

# the exact checkpoint + config.
_CKPT = "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt"
MODEL = "roberta-large"
EVAL_MODE = "nli_sp"

_scorer = None  # lazy singleton: importing this module needs no GPU/checkpoint; only score() does.


def _get_scorer(batch_size: int = 16):
    global _scorer
    if _scorer is None:
        import torch

        from ._alignscore_utils import AlignScorer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _scorer = AlignScorer(
            model=MODEL, batch_size=batch_size, device=device,
            ckpt_path=_CKPT, evaluation_mode=EVAL_MODE, verbose=False,
        )
    return _scorer


def _clean(s) -> str:
    s = str(s)
    return s if s.strip() else "(empty)"  # matches the empty-string guard


def score(record: dict, dataset: str = None) -> Optional[float]:
    """Score one record's gen_text against its gold target with AlignScore.

    `dataset` is accepted for interface parity with llm_judge.judge() but unused
    (AlignScore compares output vs gold directly; no prompt routing needed).
    """
    try:
        scorer = _get_scorer()
        output = _clean(record["gen_text"])
        gold = record["target"]
        golds = gold if isinstance(gold, list) else [gold]
        golds = [_clean(g) for g in golds]
        # score(contexts=output, claims=gold) -> output supports gold; max over references.
        vals = scorer.score(claims=golds, contexts=[output] * len(golds))
        return float(max(vals))
    except Exception:
        return None
