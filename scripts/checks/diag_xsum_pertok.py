"""Diagnose WHY the xsum per-token L15 cache fails the alignment gate (raw mean-pool != SAPLMA L15).

Two competing hypotheses, distinct signatures:
  H1  bf16 cache (the documented RCS OOM fallback): error is UNIFORM ~0.4% across all rows, and the
      float32 states are bf16-QUANTISED (low 16 mantissa bits all zero).
  H2  stale saplma (xsum regenerated after the saplma feature): error is LOCALISED -- a subset of
      rows very wrong (different generation) -- and states are true fp32 (low mantissa bits nonzero).

Checks, using sciq (a gate-PASSING cache) as the fp32 control:
  1. bf16-quantisation fraction (low-mantissa-zero) for xsum vs sciq states.
  2. per-row |mean(states)-saplma| distribution: uniform (H1) vs bimodal (H2).
  3. window-length agreement: states[k].rows == len(gen_token_ids)+1 (a length mismatch => H2).
  4. does excluding the row-0 anchor change the picture (rule out a window-definition artefact).

Heavy (loads the 2.2GB xsum cache) -> compute job. Read-only; writes nothing.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from attn_pool import load_per_token  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"


def bf16_quantised_fraction(states, n_sample=200):
    """Fraction of float32 values whose low 16 mantissa bits are zero (== bf16 rounded).
    fp32 data ~0; bf16-derived data ~1.0."""
    rng = np.random.RandomState(0)
    vals = np.concatenate([states[k].ravel() for k in rng.choice(len(states), min(n_sample, len(states)), replace=False)])
    bits = vals.astype(np.float32).view(np.uint32)
    low16_zero = (bits & 0xFFFF) == 0
    return float(low16_zero.mean())


def per_row_error(states, saplma, records):
    """Per-row max-abs error of the full-window mean-pool vs the cached SAPLMA L15 feature,
    plus the window-length agreement and an anchor-excluded variant."""
    n = len(states)
    err_full = np.zeros(n); err_noanchor = np.zeros(n); len_ok = np.zeros(n, bool)
    for k in range(n):
        s = states[k]
        err_full[k] = np.abs(s.mean(0) - saplma[k]).max()
        err_noanchor[k] = np.abs(s[1:].mean(0) - saplma[k]).max() if s.shape[0] > 1 else err_full[k]
        len_ok[k] = (s.shape[0] == len(records[k]["gen_token_ids"]) + 1)
    return err_full, err_noanchor, len_ok


def summarise(name, states, saplma, records):
    print(f"\n===== {name} =====", flush=True)
    print(f"  states dtype={states[0].dtype}  n={len(states)}  saplma dtype={saplma.dtype}")
    q = bf16_quantised_fraction(states)
    print(f"  bf16-quantised fraction (low-mantissa-zero): {q:.3f}   "
          f"({'BF16' if q > 0.9 else 'fp32' if q < 0.1 else 'MIXED/uncertain'})")
    err_full, err_noanchor, len_ok = per_row_error(states, saplma, records)
    for tag, e in [("full-window mean", err_full), ("anchor-excluded mean", err_noanchor)]:
        pct = np.percentile(e, [50, 90, 99, 100])
        print(f"  {tag:22s} per-row maxerr: median={pct[0]:.2e} p90={pct[1]:.2e} "
              f"p99={pct[2]:.2e} max={pct[3]:.2e}  rows>0.01={int((e>0.01).sum())} rows>0.1={int((e>0.1).sum())}")
    print(f"  window-length agrees (rows==gen+1): {100*len_ok.mean():.1f}%  ({int((~len_ok).sum())} mismatched)")
    # uniform (H1) vs bimodal (H2): ratio of p50 to max
    e = err_full
    shape = "UNIFORM (-> bf16/H1)" if e.max() > 0 and pct_ratio(e) > 0.05 else "LOCALISED (-> stale/H2)"
    print(f"  error shape: {shape}   (median/max = {pct_ratio(e):.3f})")


def pct_ratio(e):
    return float(np.median(e) / e.max()) if e.max() > 0 else 0.0


def main():
    for ds in ["xsum", "sciq"]:               # xsum = suspect, sciq = fp32 control
        loaded = load_per_token(MODEL, ds, 15)
        states, split, y, lyr, records = loaded
        cfg = Config(model_name=MODEL, dataset=ds, ood_setting="ID")
        saplma = cache.load_features(cfg.cache_dir, cache.run_key(MODEL, ds, "ID"), "saplma")[:, lyr, :]
        summarise(ds, states, saplma, records)


if __name__ == "__main__":
    main()
