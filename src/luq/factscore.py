"""FActScore-Bio loader — a long-form FACTUALITY eval (ExpertQA's same-task partner).

FActScore (Min et al., EMNLP 2023): generate a biography for a person, then score its factual PRECISION
against that person's Wikipedia article. We run Llama-3.1-8B CLOSED-BOOK ("Tell me a bio of <entity>") and
judge the bio against the enwiki reference with the SAME three-state factuality judge as ExpertQA
(faithfulness = SUPPORTED/(SUPPORTED+CONTRADICTED) = factual precision) so the two are directly comparable.

Assets (fetched via FActScore's own downloader; master on DoC `/vol/gpudata/.../factscore_data/`, a working
copy in RCS ephemeral). Paths are ENV-configurable so the SAME code runs on DoC and RCS:
  - FACTSCORE_DIR  -> the factscore_data dir holding data/unlabeled/prompt_entities.txt (500 people =
                     Wikipedia titles) and data/labeled/prompt_entities.txt (183, for judge validation).
  - FACTSCORE_DB   -> the enwiki-20230401.db (sqlite `documents(title PRIMARY KEY, text)`; schema verified;
                     text is plain prose with a leading `<s>` sentinel). Defaults to FACTSCORE_DIR/<db>.

Eval-only (like ExpertQA / ASQA): a NEW cross-task factuality EVAL target — probes train on the existing
pool and test on FActScore-Bio — so there is no FActScore train split. Entities are used in file order so
record idx aligns to load_records() (no manifest needed). The Wikipedia reference is fetched at LABEL time
(wiki_reference), not embedded in the records, so the 22GB db never has to travel with the cache.
"""
import os
from pathlib import Path

DEFAULT_DIR = "/vol/gpudata/gs925-msc_project/factscore_data"   # DoC master; override with FACTSCORE_DIR
_ENTITIES = {"unlabeled": "data/unlabeled/prompt_entities.txt",
             "labeled": "data/labeled/prompt_entities.txt"}
_DB_BASENAME = "enwiki-20230401.db"

# FROZEN generation prompt (stabilises 01_extract's prompt_hash guard). Closed-book bio; the Question:/Answer:
# markers match the pipeline's parsers. FActScore's own prompt is "Tell me a bio of <entity>."
PROMPT = "Question: Tell me a bio of {entity}.\nAnswer:"

# Bio budget: FActScore paragraph bios run ~150-250 words. 256 tokens covers a paragraph (a cap, not a
# target). Pair with --repetition-penalty 1.2 (the base-Llama anti-loop fix used for asqa/expertqa).
MAX_NEW_TOKENS = 256


def _dir() -> Path:
    return Path(os.environ.get("FACTSCORE_DIR", DEFAULT_DIR))


def db_path() -> str:
    """The enwiki db path: $FACTSCORE_DB if set, else <FACTSCORE_DIR>/enwiki-20230401.db."""
    return os.environ.get("FACTSCORE_DB") or str(_dir() / _DB_BASENAME)


def load_entities(split: str = "unlabeled") -> list[str]:
    """The person names (= Wikipedia titles), in file order. 'unlabeled' = 500 (the eval set)."""
    p = _dir() / _ENTITIES[split]
    if not p.exists():
        raise FileNotFoundError(f"FActScore entities not found: {p}. Set FACTSCORE_DIR (fetch via FActScore's "
                                f"downloader; master on DoC /vol/gpudata/.../factscore_data).")
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def load_records(split: str = "unlabeled") -> list[dict]:
    """One dict per entity, in file order so idx is stable:
    {entity, gold (= entity; the judge fetches the Wikipedia reference by this title), sample_id}."""
    return [{"entity": e, "gold": e, "sample_id": e} for e in load_entities(split)]


def wiki_reference(title: str, path: str | None = None, max_chars: int = 6000) -> str | None:
    """The entity's Wikipedia article text (the judge's reference). Returns None if the title is absent.
    Strips the leading `<s>` sentinel and trims to a judge-sized window (lead + early sections carry the
    biographical facts; a full article can be huge). Read-only single-row lookup (fast, not a table scan)."""
    import sqlite3
    con = sqlite3.connect(path or db_path())
    try:
        row = con.execute("SELECT text FROM documents WHERE title=?", (title,)).fetchone()
    finally:
        con.close()
    if not row or not row[0]:
        return None
    t = row[0].lstrip()
    if t.startswith("<s>"):
        t = t[3:].lstrip()
    return t[:max_chars].strip()
