"""Orgad exact-answer-token probe on short-form QA, and: does our attention find that token?

Two questions, both on the existing per-token cache (L15), CPU only:

1. Baseline. Locate the gold answer's token span in each generation (faithful port of Orgad's
   `get_indices_of_exact_answer`), probe the exact-answer-LAST token's hidden state with the
   SAPLMA MLP, and compare its PRR to mean-pool and to the learned attention pooler. On short-form
   the generation is mostly just the answer, so this is expected to sit near last-token pooling.
   The point is to have the faithful Orgad reference, not to win.

2. The overlap test. Our attention pooler learns which tokens to weight WITHOUT the gold answer.
   For each located test example, measure how much attention mass lands on Orgad's gold span and
   whether argmax(attention) falls inside it, against the uniform (chance) level. High overlap
   means the learned attention finds Orgad's important token on its own.

Deviations, logged: we locate the GOLD answer by substring match and fall back to the last answer
token when it is absent (Orgad instead runs an LLM to extract the model's own answer for wrong
cases), and trivia uses the first matching alias. Run on sciq / trivia_qa.

    python scripts/checks/orgad_exact_token.py --datasets sciq,trivia_qa
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/vol/gpudata/gs925-msc_project/hf_cache")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import json  # noqa: E402

from luq import cache, probe, results  # noqa: E402
from luq.features import orgad, orgad_llm  # noqa: E402
from attn_pool import load_per_token, pad_batch, train_attn, SEED  # noqa: E402

MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"


def load_orgad_llm(model, dataset):
    """The leak-free locator's cache: the model's OWN answer extracted by an LLM (gpt-5-mini), keyed
    split:idx. Returns {"split:idx": extracted_string} or None."""
    p = ROOT / "cache" / "orgad_llm" / f"{cache._slug(model)}__{dataset}__ID.json"
    return json.loads(p.read_text()) if p.exists() else None


def locate_rows(locator, tok, record, orgad_json):
    """Return (rows, found) in the per-token-cache window [P-1:P+G] (row 0 = last prompt token, gen
    token j -> row j+1), the SAME convention for both locators.
      gold: substring-match the GOLD answer (Orgad's get_indices_of_exact_answer) -- LEAKS (located
            tracks correctness).
      llm:  locate the model's OWN LLM-extracted answer (exists for correct AND incorrect rows), so
            'located' no longer tracks correctness."""
    if locator == "gold":
        return orgad.locate_answer_rows(tok, record["prompt_token_ids"],
                                        record["gen_token_ids"], record.get("target"))
    extracted = orgad_json.get(f"{record['split']}:{record['idx']}") if orgad_json else None
    if isinstance(extracted, list):          # summarisation stores a phrase list; QA (here) is a string
        extracted = next((s for s in extracted if isinstance(s, str) and s.strip()), None)
    if not extracted:
        return [], False
    return orgad_llm.locate_extracted_rows(tok, record["gen_token_ids"], extracted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--datasets", default="sciq,trivia_qa")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--locator", default="llm", choices=["llm", "gold"],
                    help="llm = leak-free model-own-answer locator (default); gold = the old "
                         "substring locator that LEAKS (located tracks correctness).")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"device: {device} | locator: {args.locator}")

    for dataset in args.datasets.split(","):
        loaded = load_per_token(args.model, dataset, args.layer)
        if loaded is None:
            print(f"\n==== {dataset}: no per-token cache, skip ====")
            continue
        states, split, y, layer, records = loaded
        if np.isnan(y).any():
            print(f"\n==== {dataset}: missing labels, skip ====")
            continue
        tr = np.where(split == "train")[0]
        te = np.where(split == "test")[0]
        ytr, yte = y[tr], [y[i] for i in te]
        print(f"\n==== {dataset} (layer {layer}, train {len(tr)}, test {len(te)}) ====")

        orgad_json = load_orgad_llm(args.model, dataset) if args.locator == "llm" else None
        if args.locator == "llm" and orgad_json is None:
            print(f"  !!! no orgad_llm cache for {dataset} -- run 01o_orgad_llm_extract.py; skipping")
            continue

        # Locate the answer span (cache rows) for every record, keeping last-token row + span.
        last_rows, spans, found_flags = [], [], []
        for k, r in enumerate(records):
            rows, found = locate_rows(args.locator, tok, r, orgad_json)
            g = len(r["gen_token_ids"])
            last_rows.append(orgad.last_token_row(rows, g, found))
            spans.append(set(rows))
            found_flags.append(found)
        found_flags = np.array(found_flags)
        print(f"  answer located ({args.locator}) in {found_flags.sum()}/{len(records)} "
              f"({100 * found_flags.mean():.0f}%); the rest fall back to the last answer token")

        # LEAK CHECK: does 'located' track correctness? (the whole reason to prefer the LLM locator).
        corr = np.array(y) >= 0.5
        lc = 100 * found_flags[corr].mean() if corr.any() else float("nan")
        li = 100 * found_flags[~corr].mean() if (~corr).any() else float("nan")
        gap = lc - li
        print(f"  [leak check] located|correct {lc:.0f}% vs located|incorrect {li:.0f}% "
              f"(gap {gap:+.0f}pt -- gold locator ~99 vs 0; a small gap means the probe is interpretable)")

        # ---- 1. PRR: mean-pool vs Orgad exact-answer-last-token ----
        Xmean = np.stack([s.mean(axis=0) for s in states])
        Xorg = np.stack([states[k][min(last_rows[k], states[k].shape[0] - 1)]
                         for k in range(len(records))])
        Xlast = np.stack([s[-1] for s in states])
        prr_mean = results.prr(yte, probe.uncertainty(
            probe.train_probe_mlp(Xmean[tr], ytr), Xmean[te]))
        prr_org = results.prr(yte, probe.uncertainty(
            probe.train_probe_mlp(Xorg[tr], ytr), Xorg[te]))
        prr_last = results.prr(yte, probe.uncertainty(
            probe.train_probe_mlp(Xlast[tr], ytr), Xlast[te]))
        print(f"  {'mean-pool':28s} {prr_mean:+.3f}")
        print(f"  {'last answer token':28s} {prr_last:+.3f}")
        print(f"  {'orgad exact-answer-last':28s} {prr_org:+.3f}")

        # ---- 2. Attention overlap with the Orgad span ----
        # The three numbers are not independent confirmations. They answer three questions in order:
        #   weight entropy   -> is the distribution peaked at all? (vs the uniform/old-collapse ceiling)
        #   argmax-in-span   -> is it peaked ON the answer span specifically? (the one that settles it)
        #   mass vs chance   -> by how much (chance = the span's token share = uniform mass on it)
        # Reported twice: LOCATED-ONLY (gold span actually matched, the reported figure) and ALL-TEST
        # (fallback cases use the last answer token as the "span"), so the fallback share on trivia
        # (57% located) is visible and the reported row is located-only, not the fallback-diluted one.
        model = train_attn(states, y, list(tr), device, seed=SEED, temperature=0.5)
        model.eval()

        def _stats(a, target):
            tgt = np.array(sorted(i for i in target if i < len(a)))
            if len(tgt) == 0:
                return None
            return dict(mass=float(a[tgt].sum()),
                        argmax_in=int(a.argmax() in set(tgt.tolist())),
                        chance=len(tgt) / len(a),
                        ent=float(-(a * np.log(a + 1e-12)).sum()),
                        unif=float(np.log(len(a))))

        located, all_test = [], []
        with torch.no_grad():
            for k in te:
                X, m, pos = pad_batch([states[k]], device)
                _, a = model(X, m, pos)
                a = a[0, : states[k].shape[0]].cpu().numpy()
                if found_flags[k] and spans[k]:
                    s = _stats(a, spans[k])
                    if s:
                        located.append(s)
                # all-test: the actual probed target, the located span if found, else the fallback token
                target = spans[k] if (found_flags[k] and spans[k]) else {last_rows[k]}
                s2 = _stats(a, target)
                if s2:
                    all_test.append(s2)

        def _report(name, rows):
            if not rows:
                return
            mean = lambda key: float(np.mean([r[key] for r in rows]))
            print(f"  [{name}, n={len(rows)}]")
            print(f"    weight entropy   : {mean('ent'):.3f} (uniform {mean('unif'):.3f})   "
                  f"-> peaked (not uniform)")
            print(f"    argmax in span   : {100 * mean('argmax_in'):.0f}%   -> peaked ON the span")
            print(f"    mass on span     : {mean('mass'):.3f} (chance {mean('chance'):.3f})   "
                  f"-> magnitude")
        _report("LOCATED-ONLY (gold span matched)", located)
        _report("all test (fallback = last answer token)", all_test)


if __name__ == "__main__":
    main()
