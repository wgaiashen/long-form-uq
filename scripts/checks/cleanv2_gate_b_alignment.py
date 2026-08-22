#!/usr/bin/env python
"""Gate B: every artifact on a clean-v2 row must describe the SAME retained span.

The whole point of this correction is that the canonical MedQuAD population had labels judged on
the clean answer span while its features were computed over the raw generation. Fixing that is
worth nothing unless the corrected population is itself internally consistent, so this gate checks
the four artifacts against each other row by row rather than trusting the builder:

    text       gen_text is the decode of the retained ids, cut at the correct TOKEN boundary
    tokens     gen_token_ids are a prefix of the canonical ids
    logprobs   token_logprobs is an exact prefix of canonical and has one entry per generated token
    states     the layer-15 pertok array is the G+1 window (last prompt token + generated tokens)
    label      correctness was judged on THIS span -- inherited only where the span is unchanged

THE TEXT LEG CHECKS THE BUILDER'S RULE, NOT A CHARACTER SLICE. answer_span returns a character
offset `ch`, but the cut has to land on a token boundary, so build_truncated_records keeps the
largest n with `len(tok.decode(ids[:n])) <= ch` and stores `tok.decode(ids[:n])`. The retained text
is therefore usually a character or two SHORT of the regex span -- on 611 of the 862 cut rows here --
because the token straddling the boundary would overshoot it. Asserting character equality against
the regex slice would fail all 611 and would be testing the wrong thing. The real invariant is
two-sided and is what is checked:

    len(decode(ids[:n]))   <= ch        never keeps text past the rule's cut
    len(decode(ids[:n+1]))  > ch        and could not have kept one token more

That is maximality at the boundary, and it is imported from the builder rather than reimplemented,
so the gate cannot drift away from the code that wrote the cache.

The label leg is the one that can silently regress. A row whose retained span moved under v2 but
kept its v1 label would reintroduce exactly the mismatch being removed, and it would look fine in
every other check. So `label_span` is not read as a claim: the span-equality test is recomputed
here and the field must agree with it.

    python scripts/checks/cleanv2_gate_b_alignment.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import answer_span as A                                    # noqa: E402
from luq import cache                                              # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "checks"))
from build_truncated_records import keep_n_tokens, MIN_KEEP        # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = cache._slug(MODEL)
DATASET = "med_quad"
LAYER = 15


def cut_char(text, prompt, version):
    """The rule's character offset, with build_truncated_records' own no-cut handling."""
    _, ch, reason = A.answer_span(text, DATASET, context=prompt, version=version)
    if reason.startswith("no-cut") or ch in (0, None) or ch >= len(text):
        return None
    return ch


def span_of(text, prompt, version):
    """The retained string the rule implies, for the label-equality test."""
    txt, _, rsn = A.answer_span(text, DATASET, context=prompt, version=version)
    return (text if rsn == "no-cut" else txt).rstrip()


def main():
    canon = [json.loads(l) for l in
             open(ROOT / "cache" / "records" / f"{SLUG}__{DATASET}__ID.jsonl")]
    cv = [json.loads(l) for l in
          open(ROOT / "cache" / "cleanv2" / "records" / f"{SLUG}__{DATASET}__ID.jsonl")]
    if len(canon) != len(cv):
        sys.exit(f"row count mismatch: canonical {len(canon)} vs clean-v2 {len(cv)}")

    pt = np.load(ROOT / "cache" / "cleanv2" / "pertok" /
                 f"{SLUG}__{DATASET}__ID__L{LAYER}.npz", allow_pickle=True)
    states = pt["states"]
    if int(pt["layer"]) != LAYER:
        sys.exit(f"pertok cache is layer {int(pt['layer'])}, expected {LAYER}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    fail = {k: [] for k in
            ("text", "decode", "untouched", "tok_prefix", "lp_prefix", "lp_len", "state_window",
             "label_span_field", "label_stale")}
    n_cut = 0

    for i, (a, b) in enumerate(zip(canon, cv)):
        changed = (span_of(a["gen_text"], a.get("prompt"), 2)
                   != span_of(a["gen_text"], a.get("prompt"), 1))   # -> label must be re-judged
        cut = len(b["gen_token_ids"]) < len(a["gen_token_ids"])
        n_cut += bool(cut)
        g = len(b["gen_token_ids"])
        ids = list(a["gen_token_ids"])
        ch = cut_char(a["gen_text"], a.get("prompt"), version=2)

        # 1. the cut is at the maximal token boundary that does not pass the rule's offset
        if ch is None:
            if g != len(ids):
                fail["text"].append(i)                  # no cut called, yet the row was truncated
        else:
            n = max(MIN_KEEP, min(keep_n_tokens(tok, ids, ch), len(ids)))
            if n >= len(ids):
                if g != len(ids):
                    fail["text"].append(i)              # boundary past the end: must be a no-op
            elif g != n or len(tok.decode(ids[:g])) > ch or (
                    g + 1 <= len(ids) and len(tok.decode(ids[:g + 1])) <= ch):
                fail["text"].append(i)

        # 2. CUT rows: the stored text is exactly the decode of the retained ids, which is the
        #    builder's own convention (`nr["gen_text"] = tok.decode(ids[:n])`).
        #    UNCUT rows are not re-derived at all -- the builder appends the record untouched -- so
        #    the invariant there is byte-identity with canonical, not a decode round-trip. Asserting
        #    the decode on them would fail 14 rows that end in <|end_of_text|>: their gen_text was
        #    written at generation time with skip_special_tokens, so a plain decode is exactly 15
        #    characters longer. That mismatch is present on the CANONICAL records too and predates
        #    this correction entirely, so it is out of scope here.
        if cut:
            if b["gen_text"] != tok.decode(b["gen_token_ids"]):
                fail["decode"].append(i)
        elif (b["gen_text"] != a["gen_text"]
              or list(b["gen_token_ids"]) != list(a["gen_token_ids"])
              or list(b["token_logprobs"]) != list(a["token_logprobs"])):
            fail["untouched"].append(i)
        if list(b["gen_token_ids"]) != list(a["gen_token_ids"][:g]):
            fail["tok_prefix"].append(i)

        # 3. logprobs: one per generated token, exact canonical prefix
        if len(b["token_logprobs"]) != g:
            fail["lp_len"].append(i)
        if list(b["token_logprobs"]) != list(a["token_logprobs"][:g]):
            fail["lp_prefix"].append(i)

        # 4. states: the G+1 window (last prompt token + generated tokens)
        if len(states[i]) != g + 1:
            fail["state_window"].append(i)

        # 5. the label leg -- recomputed, not taken on trust
        span_field = b.get("label_span")
        expect = "cleanv2" if changed else "cleanv2-inherited"
        if span_field != expect:
            fail["label_span_field"].append(i)
        if not changed and b.get("correctness") != b.get("correctness_clean"):
            fail["label_stale"].append(i)

    print(f"rows {len(cv)}   cut {n_cut}   uncut {len(cv) - n_cut}   layer {LAYER}\n")
    labels = {
        "text": "cut is not at the maximal token boundary within the rule's offset",
        "decode": "cut row: gen_text != tok.decode(gen_token_ids)",
        "untouched": "uncut row: not byte-identical to canonical",
        "tok_prefix": "gen_token_ids are not a canonical prefix",
        "lp_len": "len(token_logprobs) != len(gen_token_ids)",
        "lp_prefix": "token_logprobs are not a canonical prefix",
        "state_window": "pertok row is not the G+1 window",
        "label_span_field": "label_span disagrees with the recomputed span-equality test",
        "label_stale": "span unchanged but correctness != correctness_clean",
    }
    ok = True
    for k, msg in labels.items():
        bad = fail[k]
        mark = "ok  " if not bad else "FAIL"
        extra = "" if not bad else f"   rows {bad[:10]}{' ...' if len(bad) > 10 else ''}"
        print(f"  [{mark}] {msg}: {len(bad)}{extra}")
        ok &= not bad

    eos = sum(1 for b in cv if b["gen_text"] != tok.decode(b["gen_token_ids"]))
    print(f"\n  note: {eos} uncut rows end in <|end_of_text|>, so a plain decode runs 15 chars "
          f"longer than the stored text.\n        Pre-existing on the canonical records, not "
          f"introduced here.")

    n_rejudged = sum(1 for b in cv if b.get("label_span") == "cleanv2")
    print(f"\n  re-judged rows (label_span='cleanv2'): {n_rejudged}")
    print(f"\n  GATE B: " + ("PASS -- text, tokens, logprobs, states and label all describe the "
                             "same span on every row" if ok else "**FAIL -- STOP**"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
