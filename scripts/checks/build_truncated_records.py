#!/usr/bin/env python
"""Build a TRUNCATED record population in a parallel cache namespace. Non-destructive.

WHAT IT DOES. Reads the canonical records, applies the frozen `luq.template_restart` boundary, and
writes a COMPLETE parallel record set (cut rows truncated, untouched rows copied verbatim) into a
new prompt-regime namespace. The canonical records are opened read-only and never modified. Nothing
is promoted; the ladder is pointed at the new namespace via the established `LUQ_REGIME` override,
and if the correction is rejected the whole namespace is one directory to delete.

THE TOKEN/TEXT CONSISTENCY RULE, WHICH IS THE WHOLE DIFFICULTY.
`token_logprobs` is per generated token, so a truncated record must keep a genuine PREFIX of the
original token ids — otherwise the logprobs no longer describe the text. Re-tokenising the clean
string is NOT safe: BPE is not guaranteed prefix-stable, so `tok(clean).input_ids` can differ from
`ids[:n]` and the logprobs would silently describe different tokens.

So the cut is done in TOKEN space: binary-search the largest `n` with
`len(tok.decode(ids[:n])) <= cut_char`, then set

    gen_token_ids  = ids[:n]
    token_logprobs = logprobs[:n]
    gen_text       = tok.decode(ids[:n])          # the DECODE, not the regex slice

so text and ids agree by construction rather than by hope. The regex's character offset only
chooses `n`; it never becomes the stored text. Every written row is asserted to satisfy
`len(gen_token_ids) == len(token_logprobs)`.

PROVENANCE PRESERVED ON EVERY ROW: `gen_text_raw`, `n_gen_raw`, `trunc_reason`, `trunc_standing`.
Labels are NOT copied blindly — a truncated row's raw label describes text that no longer exists,
so `correctness`/`factuality` is DROPPED on cut rows and left for the relabeller. Untouched rows
keep their label, which is correct: identical judge input, identical label.

    python scripts/checks/build_truncated_records.py --model Qwen/Qwen2.5-14B --suffix trunc_v1
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import answer_span as A, cache, template_restart as TR    # noqa: E402
from luq.config import Config                                     # noqa: E402
from attn_pool import PROMPT_REGIME                               # noqa: E402
from xl_rungs import label_of                                     # noqa: E402
from transformers import AutoTokenizer                            # noqa: E402

QWEN = "Qwen/Qwen2.5-14B"
DATASETS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
MIN_KEEP = 2

# WHICH CUT RULE PER DATASET, AND WHY IT IS NOT UNIFORM.
# med_quad is the one dataset whose answer-span rule was already ACCEPTED AND APPLIED: Llama's
# canonical `correctness` IS `correctness_clean`, judged on `luq.answer_span`'s span (verified:
# correctness == correctness_clean on 1800/1800 rows). The whole point of the Qwen med_quad work is
# PROTOCOL PARITY, so it has to cut with the IDENTICAL rule — a near-identical one would put the two
# models on two slightly different spans and quietly recreate the asymmetry it is meant to remove.
# (They are close but not equal here: on Llama answer_span cuts 855 rows, template_restart 850.)
#
# xsum and pubmed_qa also HAVE answer_span rules, but those were never applied to the canonical
# labels — step 4 found xsum label-robust and it was deliberately not relabelled — and xsum's rule
# (first newline) is far more aggressive than a template restart. Using it here would be adopting an
# unpromoted rule by accident. So everything except med_quad cuts with the frozen template_restart
# boundary, which is the rule this audit actually justified from the prompt templates.
ANSWER_SPAN_DATASETS = {"med_quad"}


def keep_n_tokens(tok, ids, cut_char):
    """Largest n with len(decode(ids[:n])) <= cut_char. Binary search; decode is monotone in n."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tok.decode(ids[:mid])) <= cut_char:
            lo = mid
        else:
            hi = mid - 1
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN)
    ap.add_argument("--suffix", default="trunc_v1")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--span-version", type=int, default=1, choices=(1, 2),
                    help="luq.answer_span rule version, used only for the ANSWER_SPAN_DATASETS "
                         "(med_quad). Default 1 = the historical rule, so every earlier invocation "
                         "of this script reproduces exactly. 2 tolerates whitespace around the "
                         "'Question:'/'Answer:' colon and is what the clean-v2 population is built "
                         "with. Recorded per row in `trunc_standing`.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    slug = cache._slug(args.model)
    dsets = [d for d in args.datasets.split(",") if d]

    print("=" * 104)
    print(f"BUILD TRUNCATED RECORDS   model={args.model}   namespace suffix='{args.suffix}'")
    print("Canonical records are opened READ-ONLY. Nothing is promoted. The boundary is the frozen")
    print("luq.template_restart rule, committed before any of its PRR effects were computed.")
    print("=" * 104)
    print(f"\n{'dataset':14s}{'rule':16s}{'rows':>7s}{'cut':>7s}{'medTokKept':>12s}{'relabel':>9s}{'lbl kept':>10s}  namespace")
    regime_map = {}
    for d in dsets:
        base = PROMPT_REGIME.get(d, "")
        newns = f"{base}_{args.suffix}" if base else args.suffix
        regime_map[d] = newns
        src_cfg = Config(model_name=args.model, dataset=d, ood_setting="ID", prompt_regime=base)
        dst_cfg = Config(model_name=args.model, dataset=d, ood_setting="ID", prompt_regime=newns)
        key = cache.run_key(args.model, d, "ID")
        src = Path(src_cfg.cache_dir) / "records" / f"{key}.jsonl"
        if not src.exists():
            print(f"{d:14s}  RECORDS MISSING at {src} — reported, not skipped silently")
            continue
        recs = [json.loads(l) for l in open(src)]
        lf = label_of(d)
        out, n_cut, kept, dropped, kept_label = [], 0, [], 0, 0
        for r in recs:
            t = r.get("gen_text", "") or ""
            if d in ANSWER_SPAN_DATASETS:
                _, ch, reason = A.answer_span(t, d, context=r.get("prompt"),
                                              version=args.span_version)
                standing = f"answer_span(promoted,v{args.span_version})"
                if reason.startswith("no-cut") or ch in (0, None) or ch >= len(t):
                    ch = None
            else:
                _, ch, reason, standing = TR.restart_cut(t, d)
            ids = list(r["gen_token_ids"])
            lps = list(r["token_logprobs"])
            if len(ids) != len(lps):
                raise SystemExit(f"{d}: record idx {r.get('idx')} has {len(ids)} ids vs {len(lps)} "
                                 f"logprobs — refusing to slice an inconsistent record.")
            if ch is None:
                out.append(r)                                  # untouched: label stays valid
                continue
            n = keep_n_tokens(tok, ids, ch)
            n = max(MIN_KEEP, min(n, len(ids)))
            if n >= len(ids):
                out.append(r)                                  # boundary past the end: no-op
                continue
            nr = dict(r)
            nr["gen_text_raw"] = t
            nr["n_gen_raw"] = len(ids)
            nr["trunc_reason"] = reason
            nr["trunc_standing"] = standing
            nr["gen_token_ids"] = ids[:n]
            nr["token_logprobs"] = lps[:n]
            nr["gen_text"] = tok.decode(ids[:n])               # decode, never the regex slice
            # DOES THE EXISTING LABEL ALREADY DESCRIBE THE CUT SPAN?
            # Normally no: the raw label was judged on text that no longer exists, so it is dropped
            # and the relabeller re-judges. But Llama's med_quad was ALREADY promoted to answer-span
            # labels in July -- its canonical `correctness` IS `correctness_clean`, judged on exactly
            # the span this cut produces. Re-judging those rows would spend money to replace a
            # correct label with a noisier draw of the same judge, and would DE-align the two models
            # rather than align them. So keep the label when it is already the clean one.
            already_clean = (f"{lf}_clean" in r and r.get(lf) == r.get(f"{lf}_clean")
                             and d in ANSWER_SPAN_DATASETS)
            # ...BUT ONLY IF THE SPAN IT WAS JUDGED ON IS THE SPAN WE ARE NOW KEEPING.
            # The clean label was judged on the v1 answer_span slice. Under --span-version 2 the
            # boundary moves on some rows (on Llama med_quad: 7 newly cut, and 3 where v1 cut at the
            # `\nAnswer:` INSIDE the fabricated block and so kept the invented question stem). For
            # those the existing clean label describes MORE text than we are retaining, which is the
            # exact label/feature mismatch this whole correction exists to remove. So the retention
            # test is span equality, checked per row, not merely "a clean label exists".
            if already_clean and args.span_version != 1:
                v1_txt, _, v1_rsn = A.answer_span(t, d, context=r.get("prompt"), version=1)
                v1_span = v1_txt.rstrip() if v1_rsn != "no-cut" else t.rstrip()
                vN_txt, _, vN_rsn = A.answer_span(t, d, context=r.get("prompt"),
                                                  version=args.span_version)
                vN_span = vN_txt.rstrip() if vN_rsn != "no-cut" else t.rstrip()
                if v1_span != vN_span:
                    already_clean = False                      # -> label dropped, relabeller re-judges
            if already_clean:
                kept_label += 1
            else:
                # DROP THE JUDGE'S SIBLING OUTPUTS TOO, NOT JUST THE SCORE.
                # expertqa/factscore rows carry `uncovered` / `coherent` /
                # `factuality_quarantined` alongside `factuality`. Those describe the OLD text just
                # as much as the score does. Leaving them behind made a truncated row look like a
                # row the judge had SEEN AND DECLINED, when in fact it is a row awaiting re-judging
                # -- the two are indistinguishable downstream, and "declined" is data while
                # "awaiting" is an absence. That is precisely the confusion this project's rules
                # exist to prevent, so every field the judge wrote goes together.
                for f in (lf, f"{lf}_model", "uncovered", "coherent", f"{lf}_quarantined"):
                    nr.pop(f, None)
                dropped += 1
            assert len(nr["gen_token_ids"]) == len(nr["token_logprobs"])
            out.append(nr); n_cut += 1; kept.append(n / max(len(ids), 1))
        med = 100 * float(np.median(kept)) if kept else 100.0
        rule = "answer_span" if d in ANSWER_SPAN_DATASETS else "template_restart"
        print(f"{d:14s}{rule:16s}{len(recs):>7d}{n_cut:>7d}{med:>11.1f}%{dropped:>9d}{kept_label:>10d}  {newns}")
        if args.dry_run:
            continue
        dst = Path(dst_cfg.cache_dir) / "records"
        dst.mkdir(parents=True, exist_ok=True)
        with open(dst / f"{key}.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        # carry the meta sidecars so the prompt-hash / source-provenance guards still resolve
        msrc = Path(src_cfg.cache_dir) / "meta"
        mdst = Path(dst_cfg.cache_dir) / "meta"
        mdst.mkdir(parents=True, exist_ok=True)
        for p in msrc.glob(f"{key}.*"):
            shutil.copy2(p, mdst / p.name)

    print("\nPoint any driver at this population with the established per-dataset override:")
    print("\n  export LUQ_REGIME=\"" + ",".join(f"{d}={r}" for d, r in regime_map.items()) + "\"\n")
    print("Unset = the canonical mapping, byte-identical to before. attn_pool prints the override")
    print("in its own log, so a truncated number can never later be mistaken for a canonical one.")
    if args.dry_run:
        print("\nDRY RUN — nothing was written.")


if __name__ == "__main__":
    main()
