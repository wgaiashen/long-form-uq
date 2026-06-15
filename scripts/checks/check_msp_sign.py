"""Check 1: is the MSP-negative-on-pubmed result a real finding, or a flipped sign? (CPU)

Two parts:
 (a) orientation sanity: a synthetic case where uncertainty is HIGH on the wrong items must give
     PRR > 0 (and the flipped case PRR < 0). Confirms results.prr treats its input as uncertainty
     (higher = reject first).
 (b) the real argument: recompute MSP PRR from the cached records on BOTH sciq and pubmed with the
     SAME harness. sciq must be POSITIVE (MSP is known to work on short QA), which proves the
     harness's MSP sign is right -> the NEGATIVE on pubmed is genuine (confident tokens
     anti-correlate with faithfulness), not a bug.

    python scripts/check_msp_sign.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from luq import cache, msp, results
from luq.config import Config

# (a) orientation sanity -------------------------------------------------------
corr = np.array([1.0, 1.0, 0.0, 0.0])
unc_good = np.array([0.1, 0.2, 0.8, 0.9])   # high uncertainty on the WRONG items
unc_bad = 1.0 - unc_good                     # high uncertainty on the RIGHT items
print("(a) orientation sanity (results.prr expects uncertainty: higher = reject):")
print(f"    good uncertainty -> PRR {results.prr(corr, unc_good):+.3f}   (expect > 0)")
print(f"    bad  uncertainty -> PRR {results.prr(corr, unc_bad):+.3f}   (expect < 0)")

# (b) same harness on sciq (expect +) and pubmed (expect -) --------------------
print("\n(b) MSP PRR recomputed from cached records, same harness, both datasets:")
for ds in ["sciq", "pubmed_qa"]:
    cfg = Config(dataset=ds, ood_setting="ID")
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    try:
        recs = cache.load_records(cfg.cache_dir, key)
    except FileNotFoundError:
        print(f"    {ds}: no cache found, skipped")
        continue
    test = [r for r in recs if r["split"] == "test" and r.get("correctness") is not None]
    y = [r["correctness"] for r in test]
    for agg in ["mean", "min", "sum"]:
        u = [msp.msp_uncertainty(r["token_logprobs"], agg) for r in test]
        print(f"    {ds:10s} MSP {agg:4s}: PRR {results.prr(y, u):+.3f}")

print("\nIf sciq is positive and pubmed negative under the SAME code, the negative is a real")
print("finding (high token-confidence is anti-correlated with long-form faithfulness).")
