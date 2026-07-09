"""STEP 2 of the token-selection audit: quantify the Orgad mask's LABEL LEAK.

The Orgad mask is applied iff the gold answer SUBSTRING-MATCHES the generation ("located"). But that is
almost exactly the `correctness_strmatch` label. So if `located` correlates with correctness, then the
located rows (which get the answer-span mask) and the unlocated rows (which fall back to ALL tokens =
plain weighted-MSP) are scored by DIFFERENT estimators *along the label axis*. That makes the Orgad
±mask PRR comparison UNINTERPRETABLE, not "neutral" -- correct and incorrect examples are being treated
differently by construction.

Per dataset we report:
  - located rate overall, and split by correct vs incorrect
  - agreement between the `located` indicator and `correctness_strmatch` (short-form only)
  - mean JUDGE correctness for located vs unlocated rows (the label-axis skew)

    python scripts/checks/orgad_label_leak.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import orgad  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
DATASETS = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "med_quad"]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"{'dataset':10s} {'n':>5s} {'loc%':>6s} {'loc|corr%':>9s} {'loc|incorr%':>11s} "
          f"{'agree_strm':>10s} {'judge|loc':>9s} {'judge|unloc':>11s}", flush=True)
    for d in DATASETS:
        cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID")
        recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, d, "ID"))
        located = np.zeros(len(recs), bool)
        for i, r in enumerate(recs):
            _, found = orgad.locate_answer_rows(tok, r["prompt_token_ids"], r["gen_token_ids"], r["target"])
            located[i] = found
        judge = np.array([r.get("correctness", np.nan) for r in recs], dtype=float)
        # binary correctness: strmatch if present, else threshold the judge at 0.5
        strm_raw = [r.get("correctness_strmatch") for r in recs]
        has_strm = any(isinstance(x, (int, float)) for x in strm_raw)
        if has_strm:
            corr = np.array([1 if (isinstance(x, (int, float)) and x >= 0.5) else 0 for x in strm_raw])
        else:
            corr = (judge >= 0.5).astype(int)

        loc = located.mean()
        loc_c = located[corr == 1].mean() if (corr == 1).any() else float("nan")
        loc_i = located[corr == 0].mean() if (corr == 0).any() else float("nan")
        agree = (located.astype(int) == corr).mean() if has_strm else float("nan")
        j_loc = np.nanmean(judge[located]) if located.any() else float("nan")
        j_unloc = np.nanmean(judge[~located]) if (~located).any() else float("nan")
        print(f"{d:10s} {len(recs):5d} {100*loc:6.1f} {100*loc_c:9.1f} {100*loc_i:11.1f} "
              f"{('%.2f'%agree) if has_strm else '   -   ':>10s} {j_loc:9.3f} {j_unloc:11.3f}", flush=True)

    print("\nREAD: if loc|corr >> loc|incorr (and judge|loc >> judge|unloc), then 'located' tracks the "
          "label -> the ±Orgad-mask comparison scores correct vs incorrect rows with DIFFERENT estimators "
          "-> the existing Orgad PRR is UNINTERPRETABLE, not neutral.", flush=True)


if __name__ == "__main__":
    main()
