"""R0 — do pubmed's PUNCTUATION tokens carry the probe's signal, or is it their POSITION?

WHY THIS EXISTS
---------------
The project has said for weeks that "pubmed's attention is spread on UNHELPFUL tokens", and three Track B
experiments were designed to move that attention. We never measured "unhelpful" -- we measured
"punctuation" (pool_peaks_9dataset.csv: pubmed ID punct mass 0.515, peak token is punct/space in 94.95% of
rows). And that configuration has the HIGHEST ID PRR in the study. A delimiter's residual state summarising
the preceding clause is a standard probing phenomenon, so the premise needs a measurement, not a re-reading.

This script removes token positions from the pooling window, RETRAINS, and re-scores PRR.

THE CONFOUND (why four arms and not two)
----------------------------------------
Pubmed's attention peak sits at relative position 0.114 AND is punctuation 94.95% of the time, so the two
explanations are confounded in the data. A baseline-vs-punct comparison cannot separate them:

    baseline   nothing removed
    punct      every punctuation/space position removed
    posmatch   the same NUMBER of NON-punct tokens, taken at the same RELATIVE POSITIONS   <- the control
    random     the same number of tokens, uniformly at random                              <- the floor

  punct hurts, posmatch does not -> the DELIMITERS carry it (token identity)
  both hurt about equally        -> the POSITION carries it, punctuation is incidental
  neither beats random           -> the signal is elsewhere entirely

Pre-registered, with the predictions and the reading rule fixed in advance, at
`prereg/R0_punctuation_ablation.md`.

CONTROLLED VARIABLES (an arm difference must not be a tuning difference)
-----------------------------------------------------------------------
  * SAME POPULATION across arms. All four masks are built first; an example where ANY arm would leave
    fewer than MIN_TOKENS tokens is dropped from ALL arms. (A per-arm population would turn the
    comparison into a population difference -- the P0 bug class.)
  * TEMPERATURE FIXED at 1.0 everywhere. Selecting T per arm would vary two things at once. Consequence,
    stated rather than hidden: these PRRs are NOT comparable to the ladder's temperature-selected numbers
    and must never be merged into a ladder table.
  * MASKING APPLIES AT TRAIN AND TEST. Masking only at test would measure robustness to a shift we
    introduced, not the contribution of the tokens.
  * Three aggregators, because they answer different questions: `saplma` (mean-pool + MLP) tests whether
    the TOKENS carry signal at all; `uniform` (frozen-query pooler) is the mean-pool control inside the
    pooler code path; `attention` (learned query) tests what the POOLER keyed on.

    python scripts/checks/punct_ablation.py --datasets pubmed_qa,xsum,cnn_dailymail --seeds 1,2,3
"""
import argparse
import csv as _csv
import string
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from luq import cache  # noqa: E402
from aggregation_table import load_per_token, attn_unc, conf_meanpool, prr_from_conf  # noqa: E402
from attn_pool import train_attn  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = cache._slug(MODEL)
ARMS = ["baseline", "punct", "posmatch", "random"]
MIN_TOKENS = 4          # an example must keep at least this many positions in EVERY arm, or it is dropped
FIXED_T = 1.0           # see CONTROLLED VARIABLES -- deliberately not selected per arm

# Same punctuation set and the same classify() rule as pool_peak_tokens.py, so the ablation removes
# EXACTLY the tokens whose mass that script measured. Importing it would drag in its argparse/main, so the
# two constants are mirrored here and asserted equal at import time below.
PUNCT = set(string.punctuation) | {"…", "–", "—", "’", "“", "”"}


def _assert_classifier_matches_the_measurement():
    """The whole point of this experiment is to ablate the tokens that pool_peak_tokens.py COUNTED. If the
    two definitions ever drift, the ablation stops testing the claim -- so fail loud rather than quietly
    ablate a different token set."""
    import pool_peak_tokens as ppt
    if ppt.PUNCT != PUNCT:
        raise SystemExit("punct_ablation: PUNCT set differs from pool_peak_tokens.PUNCT -- the ablation "
                         "would remove a DIFFERENT token set from the one whose mass was measured. "
                         f"only-here={sorted(PUNCT - ppt.PUNCT)} only-there={sorted(ppt.PUNCT - PUNCT)}")
    return ppt


def window_token_ids(rec):
    """Token ids for the per-token cache window, which is [last_prompt_token] + gen_tokens (length G+1).

    This alignment is a documented trap: token-level annotations are length G while the cached states are
    G+1. Reconstructed from the record that WROTE the cache rather than assumed, and length-asserted by the
    caller."""
    return [int(rec["prompt_token_ids"][-1])] + [int(t) for t in rec["gen_token_ids"]]


def classify_window(records, states, ppt, tok, special_ids):
    """Per example, an array of class strings aligned to the state rows. Aborts on a length mismatch --
    never pads or trims, because a silent off-by-one would shift every mask by one token."""
    out = []
    for i, r in enumerate(records):
        ids = window_token_ids(r)
        if len(ids) != states[i].shape[0]:
            raise SystemExit(f"row {i}: window {len(ids)} tokens != {states[i].shape[0]} state rows. "
                             "The pertok cache and the record disagree; refusing to guess an alignment.")
        # EXPLICIT dtype "<U11". numpy sizes a string array to its longest element, so a window with no
        # punctuation would infer "<U7" (the width of "content"/"subword"/"special") and then
        # `cls == "punct_space"` compares against a TRUNCATED literal and is all-False. That happens to be
        # the right answer here, but only by luck -- a correctness-critical comparison must not depend on
        # dtype inference. (Caught by the unit test below, which hit exactly this truncation.)
        out.append(np.array([ppt.classify_id(t, tok, special_ids) for t in ids], dtype="<U11"))
    return out


PUNCT_CLASS = np.array("punct_space", dtype="<U11")   # the single spelling every mask compares against


def build_masks(classes, seed):
    """Four boolean keep-masks per example, on the SAME example set.

    posmatch is the control that separates token identity from position: for each punct token removed at
    relative position p, remove the NON-punct token whose relative position is closest to p (greedy,
    without replacement). That matches the ablation on count AND on positional profile, leaving token
    identity as the only difference.
    """
    rng = np.random.RandomState(seed)
    masks = {a: [] for a in ARMS}
    for cls in classes:
        cls = np.asarray(cls, dtype="<U11")     # re-assert the width: a caller-built "<U7" array would
        n = len(cls)                            # compare against a truncated literal and silently ablate
        is_punct = cls == PUNCT_CLASS           # NOTHING, which reads as "punctuation does not matter"
        k = int(is_punct.sum())
        rel = np.arange(n) / max(n - 1, 1)

        m_base = np.ones(n, dtype=bool)
        m_punct = ~is_punct

        m_pos = np.ones(n, dtype=bool)
        cand = np.where(~is_punct)[0]                       # only NON-punct tokens are eligible
        targets = rel[is_punct]
        taken = set()
        for t in targets:
            free = [c for c in cand if c not in taken]
            if not free:
                break
            j = min(free, key=lambda c: abs(rel[c] - t))
            taken.add(j); m_pos[j] = False

        m_rand = np.ones(n, dtype=bool)
        if k and n:
            m_rand[rng.permutation(n)[:min(k, n)]] = False

        for a, m in zip(ARMS, (m_base, m_punct, m_pos, m_rand)):
            masks[a].append(m)
    return masks


def common_population(masks, n_rows):
    """Rows kept in EVERY arm. An example whose punct-ablation empties it cannot be scored in that arm, and
    scoring the arms on different examples would make the comparison a population difference rather than an
    ablation. So the intersection is used everywhere and the loss is reported."""
    ok = np.ones(n_rows, dtype=bool)
    for a in ARMS:
        ok &= np.array([int(m.sum()) >= MIN_TOKENS for m in masks[a]])
    return ok


def apply_mask(states, masks):
    return [s[m] for s, m in zip(states, masks)]


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="pubmed_qa,xsum,cnn_dailymail",
                    help="pubmed is the target; xsum/cnn are the DATASET controls (punct mass 0.069/0.059, "
                         "so a pubmed-sized effect there would mean the mechanism is not punctuation).")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: keep only the first N rows. Output is "
                                                         "stamped SMOKE and must never enter a table.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ppt = _assert_classifier_matches_the_measurement()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    special_ids = set(getattr(tok, "all_special_ids", []))
    sha = _git_sha()
    smoke = args.limit > 0
    print(f"R0 punct ablation | device={device} seeds={seeds} T={FIXED_T} (FIXED) layer={args.layer} "
          f"git={sha[:12]}{'  [SMOKE]' if smoke else ''}", flush=True)

    rows = []
    for ds in args.datasets.split(","):
        ds = ds.strip()
        loaded = load_per_token(MODEL, ds, args.layer)
        if loaded is None:
            print(f"  {ds}: no pertok cache -> SKIPPED LOUDLY (left absent, never zero)", flush=True)
            continue
        states, split, y, _, records = loaded
        keep = np.where(np.isfinite(y))[0]
        if smoke:
            # STRATIFY BY SPLIT. Taking the first N labelled rows gives an all-`train` prefix and an EMPTY
            # test split (found by the first smoke: 186/0), so the smoke aborted without ever exercising
            # the pooler path it exists to exercise. Take N/2 from each side instead.
            per = max(args.limit // 2, 1)
            keep = np.concatenate([keep[split[keep] == s][:per] for s in ("train", "test")])
        states = [states[k] for k in keep]; records = [records[k] for k in keep]
        split = split[keep]; y = y[keep]

        classes = classify_window(records, states, ppt, tok, special_ids)
        punct_frac = float(np.mean([np.mean(c == "punct_space") for c in classes]))
        masks = build_masks(classes, seed=0)                # mask RNG fixed at 0: identical across seeds,
        ok = common_population(masks, len(states))          # so seed varies only the probe, not the data
        n_drop = int((~ok).sum())
        idx = np.where(ok)[0]
        print(f"  {ds}: {len(states)} labelled rows, punct fraction {punct_frac:.3f}; "
              f"{n_drop} dropped for the COMMON population ({100*n_drop/max(len(states),1):.1f}%)", flush=True)

        states = [states[i] for i in idx]; split = split[idx]; y = y[idx]
        masks = {a: [masks[a][i] for i in idx] for a in ARMS}
        tr_idx = np.where(split == "train")[0]
        te_idx = np.where(split == "test")[0]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            raise SystemExit(f"{ds}: empty train/test split ({len(tr_idx)}/{len(te_idx)})")

        for arm in ARMS:
            st = apply_mask(states, masks[arm])
            kept = float(np.mean([len(s) for s in st]))
            per_method = {m: [] for m in ("saplma", "uniform", "attention")}
            for sd in seeds:
                Xmean = np.stack([s.mean(axis=0) for s in st])
                per_method["saplma"].append(
                    prr_from_conf(y[te_idx], conf_meanpool(Xmean, tr_idx, te_idx, y, sd)))
                u = train_attn(st, y, tr_idx, device, seed=sd, freeze_query=True)
                per_method["uniform"].append(
                    prr_from_conf(y[te_idx], 1.0 - np.asarray(attn_unc(u, st, te_idx, device), float)))
                a = train_attn(st, y, tr_idx, device, seed=sd, temperature=FIXED_T)
                per_method["attention"].append(
                    prr_from_conf(y[te_idx], 1.0 - np.asarray(attn_unc(a, st, te_idx, device), float)))
            for m, vs in per_method.items():
                rows.append({"dataset": ds, "arm": arm, "method": m,
                             "prr_mean": round(float(np.mean(vs)), 4),
                             "prr_std": round(float(np.std(vs)), 4), "n_seeds": len(vs),
                             "mean_tokens_kept": round(kept, 2), "punct_fraction": round(punct_frac, 4),
                             "n_test": len(te_idx), "n_dropped_for_common_pop": n_drop,
                             "temperature": FIXED_T, "smoke": smoke, "git_sha": sha})
                print(f"    {ds:14s} {arm:9s} {m:10s} PRR {np.mean(vs):+.4f} "
                      f"(sd {np.std(vs):.4f}, {kept:.1f} tokens)", flush=True)

    if not rows:
        raise SystemExit("no rows produced -- every dataset was skipped. Refusing to write an empty CSV.")
    out = Path(args.out) if args.out else ROOT / "results" / (
        f"regime_R0_punct_ablation{'_SMOKE' if smoke else ''}__{SLUG}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}", flush=True)

    # A SMOKE RUN MUST NOT PRINT A VERDICT. The first successful smoke (100 train / 100 test) printed
    # "premise SURVIVES" from a probe fitted on 100 examples whose baseline PRR was 0.186 against a true
    # pubmed ID value of ~0.71 -- a line that READS like the answer, on data that cannot carry one. Same
    # failure family as a silent default returning a plausible number in place of an absence.
    if smoke:
        print(f"\nSMOKE RUN -- NO VERDICT. The reading rule is deliberately not applied: these PRRs come "
              f"from a probe fitted on ~{args.limit // 2} examples and are PLUMBING EVIDENCE ONLY.",
              flush=True)
        return

    # The registered reading rule, applied mechanically so the verdict is not a matter of eyeballing.
    #
    # THE COMPARATOR IS `random`, NOT `baseline`. All three ablation arms remove the SAME NUMBER of
    # tokens, so they share a sequence-length change the baseline does not have. Comparing an ablation arm
    # to the baseline confounds "these tokens mattered" with "the sequence got shorter" -- and the smoke
    # showed that confound is real and large (every arm, random included, scored ABOVE baseline). The
    # pre-registration names random as the control for exactly this reason; baseline is context only.
    print("\nREGISTERED READING (attention pooler). COMPARATOR = RANDOM ablation (same token count):",
          flush=True)
    for ds in sorted({r["dataset"] for r in rows}):
        g = {r["arm"]: r["prr_mean"] for r in rows if r["dataset"] == ds and r["method"] == "attention"}
        if not set(ARMS) <= set(g):
            print(f"  {ds:14s} INCOMPLETE ARMS {sorted(g)} -> NO VERDICT (never inferred from a subset)",
                  flush=True)
            continue
        punct_vs_rand = g["punct"] - g["random"]
        pos_vs_rand = g["posmatch"] - g["random"]
        if punct_vs_rand >= -0.01:
            verdict = "punct ablation did NOT hurt beyond random -> PREMISE SURVIVES"
        elif punct_vs_rand < pos_vs_rand - 0.01:
            verdict = "DELIMITERS carry it (token identity)"
        else:
            verdict = "POSITION carries it (punctuation incidental)"
        print(f"  {ds:14s} punct-random {punct_vs_rand:+.4f} | posmatch-random {pos_vs_rand:+.4f}"
              f"   [context: baseline {g['baseline']:+.4f}, random {g['random']:+.4f}]\n"
              f"  {'':14s} -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
