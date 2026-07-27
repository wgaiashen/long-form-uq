# FActScore-Bio — asset fetch (blocked on internet egress) + locked build spec

**Decisions locked (2026-07-24, with the user):**
- Reference = **FActScore's shipped enwiki-20230401 DB** (deterministic, paper-faithful, one compact page/entity).
- Entities = the **500 unlabeled** (`data/unlabeled/prompt_entities.txt`), each line a person = Wikipedia title.
- Prompt = `"Tell me a bio of <entity>."` (add `Question:`/`Answer:` markers for the judge parser).
- Run = **RCS** (short source: entity name in, ≤256 out — fits the L40S/A40; DoC NOT needed).
- Label = ExpertQA-style support judge vs the entity's Wikipedia page (**£ — gated; confirm before running**).
- Task family: **factuality**, tagged to pair with ExpertQA (SameTask), distinct from ASQA/pubmed (`long_qa`).

## The blocker
The login node reaches **HuggingFace** but NOT GitHub-raw / Google-Drive. FActScore ships its entity lists
AND the enwiki DB via a Google-Drive download (`python -m factscore.download_data`), so they can't be pulled
from the login shell. Fetch on an internet-enabled node (the DoC a30 had internet; or an RCS interactive node
if it has egress), or locally then rsync in.

## Fetch recipe (run where there IS internet)
```bash
pip install factscore          # or: git clone https://github.com/shmsw25/FActScore
python -m factscore.download_data --data_dir <target>   # pulls prompt_entities + enwiki-20230401.db (~4GB)
# then place / rsync into:
#   data/factscore/prompt_entities_unlabeled.txt          (500 lines)
#   <big volume>/factscore/enwiki-20230401.db             (multi-GB -> HF_HOME-adjacent, NOT home quota)
```
Verify after: `wc -l prompt_entities_unlabeled.txt` == 500; open the DB and confirm the `documents` table
schema (`title`, `text`) BEFORE the loader is written — do not assume it (verify-don't-conclude).

## Build order once assets land (I do this)
1. `src/luq/factscore.py` (mirror `asqa.py`): read the 500 entities + the enwiki DB; `load_records()` →
   {entity, prompt, gold=wiki page text, sample_id}; frozen PROMPT + `MAX_NEW_TOKENS`; **fail loudly** if the
   DB schema/title lookup misses.
2. Register in `src/luq/data.py` (import, JUDGE_NAME_MAP, MAX_NEW_TOKENS, `_load_factscore` eval-only, factuality
   TASK_OF) + `probedriftlong.py` LONG/LONG_SRC/FINE (factuality tag = ExpertQA's).
3. Generation sbatch (RCS): `01_extract --repetition-penalty 1.2`.
4. Label (£, GATED): ExpertQA-style judge vs the wiki page.
5. `01h_pertoken` → `01e_repool` → `feature_pertok_consistency.py` guard.
