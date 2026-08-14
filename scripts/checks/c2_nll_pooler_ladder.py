"""C2 — token probabilities as a pooler INPUT (Lihu's idea), on the long-form OOD ladder.

C2 concatenates the model's own per-token NLL (= -logprob) onto each 4096-dim hidden state, so the attention
pooler reads a 4097-dim input. The learned query can weight tokens partly by the model's confidence, and the
linear head can read the pooled confidence directly -- this FUSES the supervised pooler with the unsupervised
MSP signal. Crucially this needs NO change to the shared pooler code: train_attn / pad_batch / AttnPool all read
`d = states[0].shape[1]`, so handing them 4097-dim states just works, and every validated arm stays byte-identical.

Isolation: C2 is compared against armA (the incumbent 4096-dim learned pooler) with the SAME seeds, splits, and
val-selected temperature -- the ONLY difference is the extra NLL channel. Also vs the free msp_min floor. See
`results/c2_nll_pooler_PREREG.md` for the pre-registered prediction and grounding gates.

    python scripts/checks/c2_nll_pooler_ladder.py --evals pubmed_qa --seeds 1,2,3 \
        --out results/c2_nll_fill_pubmed_qa__meta-llama_Meta-Llama-3.1-8B.csv

C2 ≠ arm-D-nll (which tilts the SCORES by the NLL, never the head) and ≠ score-side weighted-MSP (no probe).
"""
import argparse
import csv as _csv
import hashlib
import inspect
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, msp, results                                          # noqa: E402
from aggregation_table import attn_unc, paired_bootstrap, load_per_token     # noqa: E402
from attn_pool import train_attn, select_temperature                         # noqa: E402
from xl_rungs import build_rows, eval_split, label_of                        # noqa: E402
import probedriftlong as pdl                                                 # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
DEFAULT_EVALS = ["pubmed_qa", "med_quad", "asqa", "expertqa", "factscore", "xsum", "cnn_dailymail", "samsum"]
GATE_TOL = 0.05        # armA reproduction + constant-channel wiring tolerance (seed/env noise)


def nll_window(record, state_len):
    """The per-token NLL over the SAPLMA window (row 0 = last-prompt anchor, rows 1..G = gen tokens), length G+1.
    NLL = -logprob (higher = more surprising = less confident). The anchor gets the example's own mean NLL, so it
    is neutral. ABORTS if the window length != the state window (never pad/trim silently -- the G+1 vs G trap)."""
    lp = np.asarray(record["token_logprobs"], dtype=np.float32)
    nll = -lp
    w = np.empty(len(nll) + 1, dtype=np.float32)
    w[0] = float(nll.mean()) if len(nll) else 0.0        # anchor: neutral (no logprob for the prompt token)
    w[1:] = nll
    if len(w) != state_len:                              # alignment gate (project convention: assert, do not pad silently)
        raise SystemExit(f"C2 NLL window {len(w)} != state window {state_len} — alignment bug, HALT")
    return w


def build_nll_channels(states, records, tr_idx):
    """Return a list of z-scored NLL windows aligned to `states`. Standardised with TRAIN-token stats only
    (label-free: uses logprobs, never y) so absolute confidence is preserved but rescaled to ~unit variance to
    sit at the hidden-state scale. Same mu/sd applied to train AND test (no per-example renorm, which would
    destroy the cross-token absolute-confidence signal that MSP relies on)."""
    raw = [nll_window(records[i], states[i].shape[0]) for i in range(len(states))]
    train_vals = np.concatenate([raw[i] for i in tr_idx]) if tr_idx else np.concatenate(raw)
    mu = float(train_vals.mean())
    sd = float(train_vals.std()) or 1.0
    return [((w - mu) / sd).astype(np.float32) for w in raw]


def augment(states, channels, const=False):
    """Concatenate the NLL channel as an extra feature dim -> [T, d+1]. const=True replaces it with a single
    constant value (the information-free control for the wiring gate)."""
    out = []
    for s, c in zip(states, channels):
        col = np.zeros((s.shape[0], 1), dtype=np.float32) if const else c[:, None].astype(np.float32)
        out.append(np.concatenate([s, col], axis=1))
    return out


def provenance():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        commit = "nogit"
    host = socket.gethostname()
    cluster = "doc" if host.startswith("cloud-vm") else "rcs"
    env_hash = hashlib.sha256((sys.version + torch.__version__).encode()).hexdigest()[:8]
    return commit, host, cluster, env_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default=",".join(DEFAULT_EVALS))
    ap.add_argument("--rungs", default="", help="base-rung filter (e.g. DiffTask,LOO); '' = all long OOD + ID")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    evals = [e for e in args.evals.split(",") if e]
    seeds = [int(s) for s in args.seeds.split(",")]
    want = set(r for r in args.rungs.split(",") if r)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    commit, host, cluster, env_hash = provenance()

    # LABEL-FREE ASSERT: the NLL channel is a pure function of the record's token_logprobs, no y anywhere.
    assert "record" in inspect.signature(nll_window).parameters and \
           "y" not in inspect.signature(build_nll_channels).parameters, "C2 channel must be label-free"
    print(f"[C2] device {device} | host {host} | cluster {cluster} | commit {commit[:12]} | evals {evals} | "
          f"seeds {seeds}", flush=True)
    print("[C2] LABEL-FREE ASSERT PASSED: NLL channel built from token_logprobs only.", flush=True)

    # Load per-token caches for every source the rungs might draw from (widened pool), plus the evals.
    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(evals)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok -> skip", flush=True); continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: unlabelled -> skip", flush=True); continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} rows (label={label_of(d)})", flush=True)
    sources = set(PT)

    out_rows = []
    gate_done = False
    for rung, X, spec in pdl.cells_long(sources, evals):
        base_rung = rung.replace("-long", "")
        if X not in PT or (want and base_rung not in want and rung != "ID"):
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        acc = {m: [] for m in ("floor_min", "armA", "c2")}
        yte_ref = None
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, pdl.sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows); tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float); yte_ref = yte
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]

            # floor + armA (control): armA's val-temperature is reused for C2 so ONLY the channel differs.
            floor = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "min") for i in te_idx])
            best_T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
            mA = train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T)
            uA = np.asarray(attn_unc(mA, states, te_idx, device), float)

            # C2: augment states with the z-scored NLL channel, retrain the (now d+1) pooler at the SAME T.
            channels = build_nll_channels(states, records, tr_idx)
            states_c2 = augment(states, channels)
            mC2 = train_attn(states_c2, y, tr_idx, device, seed=sd, temperature=best_T)
            uC2 = np.asarray(attn_unc(mC2, states_c2, te_idx, device), float)

            acc["floor_min"].append(floor); acc["armA"].append(uA); acc["c2"].append(uC2)

            # WIRING GATE (once, pubmed_qa ID seed 1): a CONSTANT channel must reproduce armA within tol.
            if not gate_done and X == "pubmed_qa" and rung == "ID" and sd == seeds[0]:
                states_const = augment(states, channels, const=True)
                mConst = train_attn(states_const, y, tr_idx, device, seed=sd, temperature=best_T)
                pr_const = results.prr(yte, np.asarray(attn_unc(mConst, states_const, te_idx, device), float))
                pr_armA = results.prr(yte, uA)
                d_ = abs(pr_const - pr_armA)
                ok = d_ < GATE_TOL
                print(f"[C2] WIRING GATE (const channel vs armA, pubmed ID s{sd}): const {pr_const:+.3f} vs "
                      f"armA {pr_armA:+.3f} |Δ|={d_:.3f} {'PASS' if ok else 'FAIL <== HALT'}", flush=True)
                if not ok:
                    raise SystemExit("C2 wiring gate FAILED — a constant (info-free) channel changed the result "
                                     "beyond noise, so the channel is miswired/mis-scaled. HALT.")
                gate_done = True

        if yte_ref is None or not acc["c2"]:
            continue
        stats = {}
        for m in acc:
            prs = np.array([results.prr(yte_ref, v) for v in acc[m]])
            stats[m] = (float(prs.mean()), float(prs.std()))
        avg = {m: np.mean(acc[m], axis=0) for m in acc}   # seed-averaged uncertainty for the paired bootstrap
        for m in ("floor_min", "armA", "c2"):
            out_rows.append({"rung": rung, "eval": X, "method": m, "prr_mean": round(stats[m][0], 4),
                             "prr_std": round(stats[m][1], 4), "n_seeds": len(acc[m]),
                             "commit": commit, "cluster": cluster, "env_hash": env_hash})
        # verdicts: C2 vs armA (the isolation) and vs the floor
        for vk, other in [("c2_vs_armA", "armA"), ("c2_vs_floor", "floor_min")]:
            mg, lo, hi, pv, sig = paired_bootstrap(yte_ref, avg["c2"], avg[other])
            out_rows.append({"rung": rung, "eval": X, "method": vk, "margin": round(mg, 4),
                             "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "sig": int(sig),
                             "commit": commit, "cluster": cluster, "env_hash": env_hash})
        print(f"  [{rung:14s} {X:13s}] floor {stats['floor_min'][0]:+.3f} | armA {stats['armA'][0]:+.3f} | "
              f"c2 {stats['c2'][0]:+.3f}  (c2−armA {stats['c2'][0]-stats['armA'][0]:+.3f})", flush=True)

    out = Path(args.out) if args.out else (ROOT / "results" / f"c2_nll_pooler_ladder__{SLUG}.csv")
    cols = ["rung", "eval", "method", "prr_mean", "prr_std", "n_seeds", "margin", "ci_lo", "ci_hi", "sig",
            "commit", "cluster", "env_hash"]
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\n[C2] wrote {out}  ({len(out_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
