"""Open-source local judge: the SAME prompt as the GPT-5 judge, run through a frozen
HF instruct model on the GPU. Free per call (no API), deterministic (greedy decoding),
so once it is validated against GPT-5 it scales to any volume. The prompt comes from
llm_judge.build_prompt, so the ONLY difference from the GPT-5 judge is the model.

Needs a GPU (the login node can't run it). The model is frozen — nothing is trained
here; this is just the judge label generator, the open-source counterpart to the GPT-5
call in llm_judge.py.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .llm_judge import build_prompt, parse_score


class LocalJudge:
    """Load an instruct model once, then score records one at a time.

    Greedy decoding makes the judge deterministic (reproducible), which a judge should
    be — unlike the GPT-5 path's temperature=1, kept only to match the original.
    """

    def __init__(self, model_name: str, dtype=torch.float16):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # device_map="auto" puts the model on the visible GPU(s); fp16 so 7B fits.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map="auto"
        )
        self.model.eval()

    def _reply(self, user_prompt: str) -> str:
        """Run one prompt through the model and return its (short) decoded reply."""
        # Wrap in the model's chat template so the instruct model sees a real user turn.
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=10,           # the answer is a single number
                do_sample=False,             # greedy -> deterministic judge
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Keep only the newly generated tokens (drop the prompt).
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def judge(self, record: dict, dataset: str):
        """Score one record in [0, 1], or None if the reply has no parseable number.
        (Greedy is deterministic, so a parse failure would repeat — no retry loop.)"""
        return parse_score(self._reply(build_prompt(record, dataset)))
