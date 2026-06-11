"""Long-form correctness label: Joe's LLM-as-a-judge (GPT-5), run post-hoc.

A thin adapter around the project's llm_as_a_judge_scoring_prompt.py. It runs on the
LOGIN NODE (it needs internet and OPENAI_API_KEY), not on a compute node, over the
cached Tier-1 records. The judge returns a graded 0.0-1.0 score, used as `correctness`.
"""
import json
from pathlib import Path

from ..data import JUDGE_NAME_MAP


def records_to_judge_jsonl(records: list[dict], dataset: str, out_dir: Path) -> Path:
    """Write records in the JSONL shape the judge expects, return the path.

    TODO:
      - Each line needs fields: ids, input_texts, target, answer, each a SINGLE-element
        list (the script asserts batch size 1).
          input_texts = the prompt (it already carries the Question:/Context:/Summary:
                        markers the judge parses)
          answer      = record["gen_text"]
          target      = record["target"]
          ids         = [record["idx"]]
      - The judge asserts the dataset name appears in the filename, so name the file
        using JUDGE_NAME_MAP[dataset] (e.g. pubmed_qa -> pubmed).
    """
    judge_name = JUDGE_NAME_MAP[dataset]
    out = Path(out_dir) / f"{judge_name}_for_judge.jsonl"
    raise NotImplementedError("write the judge JSONL — follow the TODO above")


def run_judge(jsonl_path: Path) -> list[float]:
    """Call llm_as_a_judge_scoring_prompt.py over the JSONL and read back the scores.

    TODO:
      - Point at the project's llm_as_a_judge_scoring_prompt.py, pass the JSONL, and
        collect the 0.0-1.0 judge_response per line. Return them in record order so
        they line up with the cached records.
      - Remember: login node only, OPENAI_API_KEY set, expect a per-call cost.
    """
    raise NotImplementedError("call the judge script — follow the TODO above")
