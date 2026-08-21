#!/usr/bin/env python
"""Fixed softmax sharpening at tau = 1 on the Qwen2.5-14B grid -- a POST-HOC cross-model transfer.

WHAT tau = 1 IS, AND WHERE IT COMES FROM. It is the a-priori sharpening strength W1 committed to in
writing before the Llama run: the project's working notes records it as "A0 a priori, tau = 1
(PRIMARY) | committed in writing before the run". It is NOT an argmax over the tau grid, which
matters -- the project has one recorded case of a grid-argmax being mistaken for a result (the
Lehmer beta = 2 row), and this is not that.

THIS IS NOT A PRE-REGISTERED QWEN TEST, AND MUST NEVER BE REPORTED AS ONE.
The Qwen master was already visible when this was run. It is a descriptive transfer of a FIXED Llama
value onto a second population: no parameter is searched, no per-dataset tau, no selector, no bars,
no PASS/FAIL. The output carries provenance = post-hoc-transfer and lives OUTSIDE the M2 scorecard.

For context, tau = 1's own registered claim FAILED on Llama: +0.0209 against msp_min, 6/8, p = 0.148
(the project's working notes). So this transfers a value that did not work on its home
population -- the opposite of a favourable-result search.

Why this is a separate file rather than a --model flag on sharpening_family.py: that file was in
active use for the Llama sharpening runs, and adding a flag to it risked disturbing them. The
scorer itself is imported from it, so this is the same code path the pre-registration names rather
than a re-implementation, which would break the rule that a ported method is verified against the
original and never against a paraphrase. Same pattern as lehmer_qwen.py, whose loader and bootstrap
are imported here for the identical reason.

  score(tau) = sum_t w_t * nll_t,   w = softmax(tau * z),   z = NLL standardised WITHIN the answer
               tau = 0   -> uniform weights   -> ranks identically to perplexity
               tau = inf -> one-hot on max nll -> ranks identically to msp_min

Training-free, therefore RUNG-INVARIANT: it reads cached `token_logprobs` only, never a training
pool. There is ONE number per dataset (n = 8), not 40 cells. The CSV says `rung_invariant` in the
rung column so that can't be lost downstream.

    python scripts/checks/softmax_tau_qwen.py
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache, msp, results                          # noqa: E402
from sharpening_family import score_softmax, softmax_weights  # noqa: E402  (THE registered scorer)
from lehmer_qwen import load_light, paired_bootstrap, LONG    # noqa: E402  (the same guarded loader)

MODEL_DEFAULT = "Qwen/Qwen2.5-14B"
TAU = 1.0                       # PRE-COMMITTED on Llama (W1 A0). Not selected here, not swept.
TAUS_RECORD = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf]   # the curve, RECORD ONLY

GATE_INT_TOL = 1e-9             # the endpoint identity is exact, not approximate
GATE_EXT_TOL = 5e-4             # vs the master CSV, which is written at 6 dp
BOOT_B = 2000


def master_floors(slug):
    """The published Qwen floor PRRs per dataset, for the EXTERNAL gate leg.

    W6 could not run this leg -- no Qwen master existed when it ran, and it said so rather than
    quietly dropping it. One does now, so the leg runs: if this script's own msp_min / perplexity
    disagree with the ladder's `floor_min` / `floor_ppl`, then its NLL convention or its row
    population differs from the ladder's and nothing downstream is comparable to the master.

    The floors are rung-invariant (they do not depend on a training pool), so any rung's value is
    the dataset's value; ID is read for definiteness.
    """
    path = ROOT / "results" / f"pdl_master__{slug}.csv"
    if not path.exists():
        return None, f"no master at {path.name}"
    out = {}
    for r in csv.DictReader(open(path)):
        if r["rung"] != "ID" or r["method"] not in ("floor_min", "floor_ppl"):
            continue
        try:
            out.setdefault(r["eval"], {})[r["method"]] = float(r["prr_mean"])
        except (TypeError, ValueError):
            pass
    return out, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DEFAULT,
                    help="EXPLICIT model pin. There is no glob fallback and there must never be one.")
    ap.add_argument("--boot", type=int, default=BOOT_B)
    args = ap.parse_args()

    slug = cache._slug(args.model)
    out_path = ROOT / "results" / f"softmax_tau_qwen__{slug}.csv"

    print("=" * 100)
    print(f"FIXED SOFTMAX SHARPENING tau = {TAU}   model={args.model}")
    print("POST-HOC CROSS-MODEL TRANSFER -- not a pre-registered Qwen test. The Qwen master was")
    print("   already visible. No parameter is searched here; tau is fixed from Llama's a-priori A0.")
    print("   tau = 1's own registered claim FAILED on Llama (+0.0209 vs msp_min, 6/8, p = 0.148).")
    print("Unit of analysis: the DATASET (n = 8). Training-free => RUNG-INVARIANT, one value each.")
    print("V3 NLL convention: cached token_logprobs are natural-log logprobs (negative), one per")
    print("   generated token. nll = -logprob. Mask: ALL generated tokens, as the family registers.")
    print("=" * 100)

    # ---------------- load (the same guarded loader W6 uses) ----------------
    data = {}
    print(f"\n{'dataset':16s}{'n_test':>8s}{'med_len':>9s}{'label':>14s}{'declined':>10s}")
    for d in LONG:
        nlls, y, lf, n_declined, _ = load_light(args.model, d)
        med = float(np.median([len(a) for a in nlls]))
        data[d] = {"nll": nlls, "y": y, "label": lf, "n": len(y), "med_len": med,
                   "declined": n_declined}
        print(f"{d:16s}{len(y):>8d}{med:>9.1f}{lf:>14s}{n_declined:>10d}")

    # ---------------- the endpoint gate, BOTH legs ----------------
    floors, why_no_ext = master_floors(slug)
    print("\n" + "=" * 100)
    print("ENDPOINT GATE -- a RANKING identity, not a value identity (PRR is rank-based).")
    print("   internal: score(tau=0) must rank as perplexity; score(tau=inf) must rank as msp_min.")
    if floors:
        print("   external: this script's msp_min / perplexity must match the ladder's floor_min /")
        print(f"             floor_ppl in pdl_master__{slug}.csv to {GATE_EXT_TOL}. W6 could NOT run")
        print("             this leg (no master existed then); it runs now, and it is the check that")
        print("             says this script sees the same rows the ladder did.")
    else:
        print(f"   external: NOT RUN -- {why_no_ext}. Stated, not silently skipped.")
    print("=" * 100)
    gate_ok = True
    for d in LONG:
        nl, y = data[d]["nll"], data[d]["y"]
        v_min = np.array([msp.msp_uncertainty(-a, "min") for a in nl])       # back to logprobs
        v_ppl = np.array([msp.msp_uncertainty(-a, "perplexity") for a in nl])
        p_min, p_ppl = results.prr(y, v_min), results.prr(y, v_ppl)
        p_t0 = results.prr(y, np.array([score_softmax(a, 0.0) for a in nl]))
        p_ti = results.prr(y, np.array([score_softmax(a, np.inf) for a in nl]))
        d_i0, d_ii = abs(p_t0 - p_ppl), abs(p_ti - p_min)
        ok = d_i0 < GATE_INT_TOL and d_ii < GATE_INT_TOL
        ext = ""
        if floors and d in floors:
            d_em = abs(p_min - floors[d].get("floor_min", np.nan))
            d_ep = abs(p_ppl - floors[d].get("floor_ppl", np.nan))
            ok = ok and d_em < GATE_EXT_TOL and d_ep < GATE_EXT_TOL
            ext = f" | vs master min d{d_em:.5f} ppl d{d_ep:.5f}"
        gate_ok = gate_ok and ok
        data[d].update(prr_min=p_min, prr_ppl=p_ppl, unc_min=v_min)
        print(f"[{d:14s}] msp_min {p_min:+.4f} ppl {p_ppl:+.4f} | t0==ppl d{d_i0:.1e} "
              f"tInf==min d{d_ii:.1e}{ext}  {'PASS' if ok else 'FAIL <=='}", flush=True)
    if not gate_ok:
        raise SystemExit("\nGATE FAIL -- the NLL convention or the row population differs from what "
                         "the family (or the ladder) assumes. Nothing below is interpretable.")

    # ---------------- the curve, for the record only ----------------
    print("\n" + "=" * 100)
    print(f"THE FULL tau CURVE (RECORD ONLY -- the transfer is tau = {TAU} and only tau = {TAU}).")
    print("Do NOT read an argmax off this table. Picking the best column on test is precisely the")
    print("   move that made the Lehmer beta = 2 row unquotable (~1.2 hits expected by chance).")
    print("=" * 100)
    print(f"{'eval':16s}" + "".join(f"{t:>9.2f}" if np.isfinite(t) else f"{'inf':>9s}"
                                    for t in TAUS_RECORD))
    curve = {}
    for d in LONG:
        nl, y = data[d]["nll"], data[d]["y"]
        row = []
        for t in TAUS_RECORD:
            v = np.array([score_softmax(a, t) for a in nl])
            row.append(results.prr(y, v))
            if t == TAU:
                data[d]["unc_tau"] = v
        curve[d] = row
        print(f"{d:16s}" + "".join(f"{p:>+9.3f}" for p in row))
    print(f"{'mean':16s}" + "".join(f"{float(np.mean([curve[d][i] for d in LONG])):>+9.3f}"
                                    for i in range(len(TAUS_RECORD))))

    # ---------------- the transfer, descriptive ----------------
    diffs = np.array([results.prr(data[d]["y"], data[d]["unc_tau"]) - data[d]["prr_min"]
                      for d in LONG])
    print("\n" + "=" * 100)
    print(f"tau = {TAU} vs Qwen's OWN msp_min -- DESCRIPTIVE. No bar is attached and none may be")
    print("   invented after the fact. Llama's comparison for reference: +0.0209, 6/8, p = 0.148.")
    print("=" * 100)
    print(f"{'eval':16s}{'tau=1':>10s}{'msp_min':>10s}{'diff':>10s}{'boot p':>10s}")
    boot_rows = {}
    for d, df in zip(LONG, diffs):
        m, lo, hi, p = paired_bootstrap(data[d]["y"], data[d]["unc_tau"], data[d]["unc_min"],
                                        b=args.boot)
        boot_rows[d] = (m, lo, hi, p)
        print(f"{d:16s}{results.prr(data[d]['y'], data[d]['unc_tau']):>+10.4f}"
              f"{data[d]['prr_min']:>+10.4f}{df:>+10.4f}{p:>10.4f}")
    w_p = wilcoxon(diffs, alternative="two-sided").pvalue
    print(f"\n  mean {diffs.mean():+.4f}   signs {int((diffs > 0).sum())}/8   "
          f"two-sided Wilcoxon p {w_p:.4f}   [descriptive, no verdict]")

    # ---------------- the mechanism check I would run if this looked good ----------------
    # Llama's own diagnostic found a fixed tau behaves substantially as a LENGTH rule (ESS correlates
    # with answer length at rho >= 0.85 on three of eight datasets), because max z <= sqrt(n-1) bounds
    # how concentrated the weights can get on a short answer. If tau = 1 looks strong on Qwen, this is
    # the check that says whether it is reading per-answer shape or just length -- so it runs
    # unconditionally, BEFORE anyone forms a view, rather than being reached for afterwards.
    print("\n" + "=" * 100)
    print("MECHANISM: is tau = 1 reading per-answer SHAPE, or just LENGTH? (ESS = 1/sum w^2)")
    print("   Run unconditionally, not in response to the numbers above. Llama: rho >= 0.85 on 3/8.")
    print("=" * 100)
    print(f"{'eval':16s}{'ESS/n':>9s}{'rho(ESS,len)':>14s}{'rho(score,len)':>16s}")
    ess_rows = {}
    for d in LONG:
        nl = data[d]["nll"]
        ess = np.array([1.0 / np.sum(softmax_weights(a, TAU) ** 2) for a in nl])
        lens = np.array([len(a) for a in nl], dtype=float)
        r_el = spearmanr(ess, lens).statistic if len(nl) > 2 else np.nan
        r_sl = spearmanr(data[d]["unc_tau"], lens).statistic if len(nl) > 2 else np.nan
        ess_rows[d] = (float(np.mean(ess / lens)), float(r_el), float(r_sl))
        print(f"{d:16s}{np.mean(ess / lens):>9.3f}{r_el:>14.3f}{r_sl:>16.3f}")

    # ---------------- persist ----------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "method", "eval", "rung", "prr", "n_test", "med_len", "label",
                    "diff_vs_msp_min", "boot_p", "ess_over_len", "rho_ess_len", "provenance"])
        for d in LONG:
            m, lo, hi, p = boot_rows[d]
            e_n, r_el, _ = ess_rows[d]
            # rung_invariant, spelled out: this is ONE observation per dataset. Repeating it across
            # the four OOD rungs for display would make eight numbers look like forty.
            w.writerow([args.model, "softmax_tau1", d, "rung_invariant",
                        f"{results.prr(data[d]['y'], data[d]['unc_tau']):.6f}", data[d]["n"],
                        f"{data[d]['med_len']:.1f}", data[d]["label"],
                        f"{results.prr(data[d]['y'], data[d]['unc_tau']) - data[d]['prr_min']:.6f}",
                        f"{p:.6f}", f"{e_n:.4f}", f"{r_el:.4f}", "post-hoc-transfer"])
        w.writerow([])
        w.writerow(["# the full tau curve, RECORD ONLY -- do not read an argmax off it"])
        w.writerow(["model", "eval", "tau", "prr"])
        for d in LONG:
            for t, p in zip(TAUS_RECORD, curve[d]):
                w.writerow([args.model, d, t, f"{p:.6f}"])
    print(f"\nwrote {out_path}")
    print("\nprovenance = post-hoc-transfer on every row. This does NOT enter the M2 scorecard (§2)")
    print("and must not be pooled with Llama.")


if __name__ == "__main__":
    main()
