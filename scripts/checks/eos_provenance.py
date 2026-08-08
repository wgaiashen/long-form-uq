#!/usr/bin/env python
"""EOS PROVENANCE -- is the end-of-text token a DECISION or an ARTEFACT of truncation?

Results: ../STOCKTAKE_sharpening_axis.md §13.

WHY THIS RUNS BEFORE ANY METHOD USES THE EOS NLL
------------------------------------------------
The special-token mask in weighted MSP (§12) kills the EOS token in BOTH slots of
`q = sum_t w_t * nll_t`: it cannot receive weight, AND its NLL cannot enter the value. Only the first
is justified -- the documented failure mode is the weighter latching onto a positional artefact
(~70% of xsum/cnn generations end in EOS). Nothing about that argues the EOS SURPRISAL should be
discarded from the value, where it is plausibly a completeness signal: "did the model think it was
done?".

The separable fix is one extra scalar:

    q = sum_{content t} w_t * nll_t   +   gamma * nll_EOS

The weighter still cannot concentrate on EOS, so the shortcut stays closed, while the completeness
signal becomes available and `gamma` says directly how much it is worth. gamma ~ 0 falsifies it.

⚠️ BUT THAT IS ONLY WELL-POSED IF THE EOS MEANS SOMETHING. If a generation was TRUNCATED at the token
budget rather than finishing, there may be no EOS at all -- or the last token may be an artefact of
the cut rather than a decision. med_quad v1 has a documented ~97.7% capping rate. A `gamma` fitted on
truncated data would be measuring "was this truncated", which is a different and much less
interesting finding than "did the model think it was done".

So this measures, per dataset, BEFORE anything is built:
  * does the generation END in a special token at all?
  * how often does it hit the token budget (i.e. was it cut)?
  * is EOS-presence essentially the complement of capping? If so the two are inseparable here.
  * and does EOS-presence ALONE predict correctness? (a one-bit floor on what gamma could capture)

Records only. No GPU, no training, seconds.

    python scripts/checks/eos_provenance.py
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, results                                       # noqa: E402
from luq.config import Config                                        # noqa: E402
from luq.weighted_msp import per_token_nll, content_keep             # noqa: E402
from xl_rungs import eval_split, label_of                            # noqa: E402
from attn_pool import PROMPT_REGIME                                  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
OUT = ROOT / "results" / "eos_provenance__meta-llama_Meta-Llama-3.1-8B.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    print("=" * 100)
    print("EOS PROVENANCE -- decision or truncation artefact?  Llama-3.1-8B, test rows.")
    print("=" * 100)
    print(f"\n{'eval':15s}{'ends in':>10s}{'at token':>10s}{'EOS & at':>10s}{'med':>7s}{'max':>7s}"
          f"{'PRR of':>9s}{'PRR of':>9s}")
    print(f"{'':15s}{'special':>10s}{'budget':>10s}{'budget':>10s}{'len':>7s}{'len':>7s}"
          f"{'has-EOS':>9s}{'nll_EOS':>9s}")
    rows = []
    for d in LONG:
        cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID",
                     prompt_regime=PROMPT_REGIME.get(d, ""))
        recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, d, "ID"))
        lf = label_of(d)
        y = np.array([r.get(lf, np.nan) for r in recs], float)
        fin = np.isfinite(y)
        recs = [recs[i] for i in np.where(fin)[0]]
        y = y[fin]
        _, te = eval_split(np.array([r["split"] for r in recs]))
        yte = y[te]

        lens, has_eos, nll_eos = [], [], []
        for i in te:
            k = content_keep(recs[i]).astype(bool)
            nl = per_token_nll(recs[i])
            lens.append(len(nl))
            # "ends in a special token" = the LAST generated token is special
            e = (not k[-1]) if len(k) else False
            has_eos.append(1.0 if e else 0.0)
            # the EOS surprisal if there is one; NaN otherwise (never 0 -- that would invent a value)
            nll_eos.append(float(nl[-1]) if e else np.nan)
        lens = np.array(lens, float)
        has_eos = np.array(has_eos)
        nll_eos = np.array(nll_eos)
        mx = float(lens.max())
        at_cap = (lens >= mx).astype(float)          # the realised budget is the max observed length
        both = float(np.mean((has_eos > 0) & (at_cap > 0)))

        # one-bit floor: does EOS-PRESENCE alone rank correctness? (uncertainty = 1 - has_eos)
        prr_has = results.prr(yte, 1.0 - has_eos) if 0 < has_eos.mean() < 1 else np.nan
        # and does the EOS surprisal rank it, on the subset that HAS one?
        m = ~np.isnan(nll_eos)
        prr_nll = results.prr(yte[m], nll_eos[m]) if m.sum() > 30 and np.std(nll_eos[m]) > 0 else np.nan

        print(f"{d:15s}{has_eos.mean():>9.1%}{at_cap.mean():>10.1%}{both:>10.1%}"
              f"{np.median(lens):>7.0f}{mx:>7.0f}{prr_has:>9.3f}{prr_nll:>9.3f}")
        rows.append((d, f"{has_eos.mean():.4f}", f"{at_cap.mean():.4f}", f"{both:.4f}",
                     f"{np.median(lens):.0f}", f"{mx:.0f}",
                     f"{prr_has:.4f}", f"{prr_nll:.4f}", f"{int(m.sum())}"))

    print("\n" + "=" * 100)
    print("READING")
    print("=" * 100)
    print("  'ends in special'  -- the generation finished with an end-of-text token: a DECISION.")
    print("  'at token budget'  -- it ran to the maximum observed length: probably TRUNCATED.")
    print("  If 'ends in special' ~= 1 - 'at budget', the two are complements and EOS-presence is")
    print("  just the inverse of truncation, so a gamma fitted on it would be measuring truncation.")
    print("  'PRR of has-EOS'   -- what a SINGLE BIT (did it finish?) is worth on its own. This is")
    print("                        the floor gamma has to beat to be about surprisal rather than")
    print("                        merely about completion.")
    print("  'PRR of nll_EOS'   -- the EOS surprisal itself, on the subset that has one.")
    print("\n  ⚠️ A dataset with a high capping rate cannot separate 'the model thought it was done'")
    print("     from 'we cut it off'. med_quad v1 is documented at ~97.7% capped.")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["eval", "frac_ends_special", "frac_at_budget", "frac_both", "median_len",
                    "max_len", "prr_has_eos", "prr_nll_eos", "n_with_eos"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
