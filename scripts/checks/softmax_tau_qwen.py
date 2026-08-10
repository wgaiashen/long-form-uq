#!/usr/bin/env python
"""Fixed softmax-tau = 1 on the Qwen2.5-14B long-form grid — a POST-HOC cross-model transfer check.

STATUS, stated on its face: this is NOT a pre-registered replication. The complete Qwen master and
its Lehmer curve were inspected before this run (2026-08-10, results/analysis/
QWEN_REPLICATION_VERDICT.md), so nothing here can be pristine confirmation. What keeps it honest is
that the configuration was FIXED ON LLAMA and is not retuned here: tau = 1.0 is `TAU_A0`, the
a-priori primary of the W1 sharpening prereg (prereg/W1_sharpening_axis.md §4), committed
2026-08-08 before any tau result existed on either model. One arm, no grid, no selection.

WHAT IT COMPUTES. The training-free softmax-sharpened estimator from sharpening_family.py:
weights = softmax(tau * z(nll)) within the answer, score = sum(w * nll). tau = 0 is uniform
(perplexity's ranking); tau -> inf is one-hot on the max token (msp_min's ranking). Only the
endpoints (the gate) and tau = 1 (the arm) are scored.

Scoring functions are IMPORTED from sharpening_family.py and the record loading is IMPORTED from
lehmer_qwen.py (the same fail-loud, model-pinned, namespace-aware path W6 used) — no
re-implementation of either. Training-free, records-only, rung-invariant, no GPU.

    python scripts/checks/softmax_tau_qwen.py
    qsub -v LUQ_CMD="scripts/checks/softmax_tau_qwen.py" pbs/audit_cpu.pbs
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import results                                   # noqa: E402
from sharpening_family import TAU_A0, score_softmax       # noqa: E402  (THE registered scorer)
from lehmer_qwen import GATE_TOL, LONG, load_light        # noqa: E402  (W6's verified loader)

MODEL_DEFAULT = "Qwen/Qwen2.5-14B"
# Llama's tau=1 arm for the side-by-side panel (descriptive; populations NEVER pooled).
LLAMA_ROUND2 = ROOT / "results" / "sharpening_family__meta-llama_Meta-Llama-3.1-8B__round2.csv"


def llama_tau1():
    """dataset -> (prr at tau=1, prr at tau=0, prr at tau=inf) from the published Llama CSV."""
    out = {}
    with open(LLAMA_ROUND2) as f:
        for r in csv.DictReader(f):
            if r["family"] != "softmax_tau" or r["param"] not in ("0.0", "1.0", "inf"):
                continue
            out.setdefault(r["dataset"], {})[r["param"]] = float(r["prr"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DEFAULT,
                    help="EXPLICIT model pin. No glob fallback, ever.")
    args = ap.parse_args()

    from luq import cache  # noqa: E402  (for the slug only)
    slug = cache._slug(args.model)
    out_csv = ROOT / "results" / f"softmax_tau_transfer__{slug}.csv"

    print("=" * 100)
    print(f"SOFTMAX-TAU = {TAU_A0} TRANSFER  model={args.model}")
    print("POST-HOC cross-model transfer check (the Qwen master was already inspected).")
    print("tau fixed on Llama (W1 prereg A0). One arm. No grid. No selection on this population.")
    print("=" * 100)

    rows, diffs_min, diffs_ppl = [], [], []
    print(f"\n{'dataset':16s}{'n_test':>8s}{'tau=0(=ppl)':>13s}{'tau=1':>9s}{'tau=inf(=min)':>15s}"
          f"{'d_vs_min':>10s}{'d_vs_ppl':>10s}")
    for d in LONG:
        nlls, y, lf, _, _ = load_light(args.model, d)
        p = {}
        for tau in (0.0, TAU_A0, np.inf):
            v = np.array([score_softmax(a, tau) for a in nlls])
            p[tau] = results.prr(y, v)
        # endpoint gate vs the master's floors is a RANKING identity; here we only need the two
        # endpoints to be finite and the arm to sit on the same population as the floors it is
        # compared to (same loader, same split, same finite-filter as W6 — by construction).
        d_min, d_ppl = p[TAU_A0] - p[np.inf], p[TAU_A0] - p[0.0]
        diffs_min.append(d_min); diffs_ppl.append(d_ppl)
        med = float(np.median([len(a) for a in nlls]))
        rows.append((d, len(y), med, lf, p))
        print(f"{d:16s}{len(y):>8d}{p[0.0]:>+13.4f}{p[TAU_A0]:>+9.4f}{p[np.inf]:>+15.4f}"
              f"{d_min:>+10.4f}{d_ppl:>+10.4f}")

    dm, dp = np.array(diffs_min), np.array(diffs_ppl)
    _, p_min = wilcoxon(dm, alternative="two-sided")
    _, p_ppl = wilcoxon(dp, alternative="two-sided")
    print(f"\nQwen cross-dataset (n=8, descriptive): tau=1 mean {np.mean([r[4][TAU_A0] for r in rows]):+.4f}")
    print(f"  vs msp_min    mean {dm.mean():+.4f}  signs {int((dm > 0).sum())}/8  p(two-sided) {p_min:.4f}")
    print(f"  vs perplexity mean {dp.mean():+.4f}  signs {int((dp > 0).sum())}/8  p(two-sided) {p_ppl:.4f}")

    # ---------------- side-by-side with Llama (published), never pooled ----------------
    lt = llama_tau1()
    print("\nSide-by-side (descriptive; separate populations, no pooled statistic):")
    print(f"{'dataset':16s}{'Llama tau=1':>12s}{'Llama d_min':>12s}{'Qwen tau=1':>12s}{'Qwen d_min':>12s}")
    for d, _, _, _, p in rows:
        l = lt.get(d)
        ls = f"{l['1.0']:+.3f}" if l else "absent"
        ld = f"{l['1.0'] - l['inf']:+.3f}" if l else "—"
        print(f"{d:16s}{ls:>12s}{ld:>12s}{p[TAU_A0]:>+12.4f}{p[TAU_A0] - p[np.inf]:>+12.4f}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "tau", "prr", "n_test", "med_len", "label", "provenance"])
        for d, n, med, lf, p in rows:
            for tau in (0.0, TAU_A0, np.inf):
                w.writerow([args.model, d, tau, f"{p[tau]:.6f}", n, f"{med:.1f}", lf,
                            "posthoc_transfer_tau_fixed_on_llama_W1_A0"])
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
