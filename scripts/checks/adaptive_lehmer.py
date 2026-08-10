#!/usr/bin/env python
"""W8 — the adaptive-Lehmer grid on the canonical Llama ProbeDriftLong population.

Prereg: prereg/W8_adaptive_lehmer.md (committed BEFORE any PRR from this script existed; the
preregistration discloses that all Llama baselines/Lehmer curves and the Qwen W6/master results
were inspected before the method was designed — this is a prospective test on a reused
development population, not independent confirmation).

METHODS (src/luq/adaptive_lehmer.py; all share the identical Lehmer scoring of raw token NLLs):
    lehmer_global_beta            one learned scalar beta per training cell (control)
    lehmer_nllshape_beta          beta_i from the 10 predeclared NLL-shape features (PRIMARY)
    lehmer_hs_beta                beta_i from the canonical mean-pooled L15 hidden state
    lehmer_hybrid_beta            g = g_h(hidden) + g_n(shape) + b (contributions logged)
    lehmer_hs_beta_shuffled_h     B6.4 control: train-pool hidden states permuted within source
    lehmer_hybrid_beta_shuffled_h same control for the hybrid

POPULATION: exactly the canonical ladder's — probe_drift_long's cells_long / build_rows /
sampled_train_idx, layer 15, seeds 1,2,3, carve legacy, finite-label filter FIRST then carve
(the §3.1a order), 1800-row supervised pools. Nothing regenerated, nothing re-selected.

HIDDEN STATE: the cached SAPLMA pooled feature plane (features __saplma.npz, layer 15) — the
canonical mean-pool representation. The repool guard (feature_pertok_consistency.py) holds it to
~1e-6 of the pertok window mean, so this is the same vector the ladder's SAPLMA reads, at 1/60th
the memory of the per-token caches (which the gates never need). ORIGINAL->FILTERED reindexing
and the length assertions mirror probedriftlong.py's POOLED join verbatim. Hidden states and
shape features are standardised on TRAIN-POOL stats only (the canonical standardize=True choice).

TRAINING: weighted_msp's exact recipe — AdamW lr=1e-3, 5 epochs, batch 32, Joe's pairwise
sigmoid soft-rank MSE against 1-y, batches < 2 skipped, torch.manual_seed(seed). No sweep.

B6.4 SHUFFLE: one fixed permutation of the TRAIN rows' hidden vectors within each source
dataset, seed = 100000*train_seed + crc32(dataset) % 9973, applied once (never per epoch).

SMOKE MODE (--smoke): integrity only — tiny sample, one cell, forward/backward, NaN and schema
checks. PRINTS NO PRR and WRITES NOTHING to results/ (B9: no partial-result peeking).

    python scripts/checks/adaptive_lehmer.py --smoke
    qsub -v LUQ_CMD="scripts/checks/adaptive_lehmer.py" pbs/audit_cpu.pbs        # full grid
    python scripts/checks/adaptive_lehmer.py --eval xsum                          # one eval
Output: results/adaptive_lehmer__meta-llama_Meta-Llama-3.1-8B.csv (+ __diag.csv). Never touches
pdl_master, the sharpening CSVs, or any Qwen file.
"""
import argparse
import csv
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache, msp, results                        # noqa: E402
from luq.adaptive_lehmer import (AdaptiveLehmerGate, BETA_MAX, NLL_FLOOR, ShapeStandardiser,
                                 nll_shape_vector)         # noqa: E402
from luq.config import Config                              # noqa: E402
from luq.weighted_msp import _soft_rank, _true_rank        # noqa: E402  (THE canonical loss)
from xl_rungs import eval_split, label_of                  # noqa: E402
from attn_pool import PROMPT_REGIME                        # noqa: E402
from probe_drift_long import cells_long, sampled_train_idx  # noqa: E402
from probe_drift_long.splits import build_rows             # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15
SEEDS = [1, 2, 3]
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
LONG_SRC = LONG                                            # the widened post-Task-A source pool
METHODS = ["lehmer_global_beta", "lehmer_nllshape_beta", "lehmer_hs_beta", "lehmer_hybrid_beta",
           "lehmer_hs_beta_shuffled_h", "lehmer_hybrid_beta_shuffled_h"]
MODE_OF = {"lehmer_global_beta": "global", "lehmer_nllshape_beta": "nllshape",
           "lehmer_hs_beta": "hs", "lehmer_hybrid_beta": "hybrid",
           "lehmer_hs_beta_shuffled_h": "hs", "lehmer_hybrid_beta_shuffled_h": "hybrid"}
N_EPOCHS, BATCH, LR = 5, 32, 1e-3                          # weighted_msp's recipe, unchanged


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()[:12]
    except Exception:
        return "unknown"


def load_dataset(d):
    """(feats_L15, split, y, records, orig) with the ladder's exact filter/reindex order."""
    cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(d, ""))
    key = cache.run_key(MODEL, d, "ID")
    records = cache.load_records(cfg.cache_dir, key)
    n_orig = len(records)
    arr = cache.load_features(cfg.cache_dir, key, "saplma")
    feats = np.ascontiguousarray(arr[:, LAYER, :], dtype=np.float32); del arr
    if len(feats) != n_orig:
        raise SystemExit(f"FATAL {d}: feature cache {len(feats)} rows vs {n_orig} records")
    lf = label_of(d)
    y = np.array([r.get(lf, np.nan) for r in records], dtype=float)
    finite = np.isfinite(y)
    keep = np.where(finite)[0]
    records = [records[k] for k in keep]
    feats = feats[keep]
    split = np.array([r["split"] for r in records])
    return feats, split, y[keep], records, keep


def precompute(records):
    """per-row (log-floored-NLL tensor, shape vector, nll vector) — computed once per dataset."""
    logl, shapes, nlls = [], [], []
    for r in records:
        nll = -np.asarray(r["token_logprobs"], dtype=np.float64)
        if len(nll) != len(r["gen_token_ids"]):
            raise SystemExit(f"G-align FAIL idx={r['idx']}")
        nlls.append(nll)
        logl.append(torch.log(torch.clamp(torch.from_numpy(nll), min=NLL_FLOOR)).float())
        shapes.append(nll_shape_vector(nll))
    return logl, np.stack(shapes), nlls


def batch_scores(gate, logl_rows, h, x):
    """Differentiable U_i for a batch. h/x: (B, d) tensors or None per the gate's mode."""
    beta = gate.beta(h, x, n=len(logl_rows))
    return torch.stack([torch.exp(torch.logsumexp((beta[i] + 1.0) * logl_rows[i], 0)
                                  - torch.logsumexp(beta[i] * logl_rows[i], 0))
                        for i in range(len(logl_rows))]), beta


def train_gate(mode, logl, H, Xs, y, tr_idx, seed):
    """The canonical recipe on the gate parameters only."""
    torch.manual_seed(seed)
    gate = AdaptiveLehmerGate(mode, d_hidden=H.shape[1] if H is not None else 0)
    opt = torch.optim.AdamW(gate.parameters(), lr=LR)
    incorrect = torch.tensor([1.0 - float(y[i]) for i in tr_idx], dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    gate.train()
    for _ in range(N_EPOCHS):
        perm = torch.randperm(len(tr_idx), generator=g).tolist()
        for b0 in range(0, len(tr_idx), BATCH):
            bpos = perm[b0:b0 + BATCH]
            if len(bpos) < 2:
                continue
            rows = [tr_idx[p] for p in bpos]
            h = H[rows] if H is not None and mode in ("hs", "hybrid") else None
            x = Xs[rows] if Xs is not None and mode in ("nllshape", "hybrid") else None
            q, _ = batch_scores(gate, [logl[i] for i in rows], h, x)
            target = _true_rank(incorrect[bpos])
            loss = ((_soft_rank(q) - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if not torch.isfinite(loss):
                raise SystemExit(f"NaN loss in {mode} seed {seed} — integrity failure, stopping")
    return gate


def score_gate(gate, mode, logl, H, Xs, idx):
    gate.eval()
    with torch.no_grad():
        h = H[idx] if H is not None and mode in ("hs", "hybrid") else None
        x = Xs[idx] if Xs is not None and mode in ("nllshape", "hybrid") else None
        q, beta = batch_scores(gate, [logl[i] for i in idx], h, x)
        g_h, g_n, _ = gate.g_parts(h, x)
    return (q.numpy(), beta.numpy(),
            g_h.numpy() if g_h is not None else None,
            g_n.numpy() if g_n is not None else None)


def flush(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="integrity only; no PRR printed/saved")
    ap.add_argument("--eval", default=None, help="restrict to one eval dataset")
    ap.add_argument("--seeds", default=None, help="comma list override (default 1,2,3)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    evals = [args.eval] if args.eval else LONG
    sha, carve = git_sha(), "legacy"
    out = ROOT / "results" / f"adaptive_lehmer__{cache._slug(MODEL)}.csv"
    diag_out = ROOT / "results" / f"adaptive_lehmer__{cache._slug(MODEL)}__diag.csv"

    print("=" * 100)
    print(f"W8 ADAPTIVE LEHMER  model={MODEL} layer={LAYER} beta_max={BETA_MAX} seeds={seeds} "
          f"commit={sha} carve={carve}{'  [SMOKE — no PRR is printed or saved]' if args.smoke else ''}")
    print("=" * 100)

    PT, LOGL, SHP, NLLV = {}, {}, {}, {}
    for d in sorted(set(LONG_SRC) | set(evals)):
        feats, split, y, records, orig = load_dataset(d)
        if args.smoke:                                    # tiny but split-diverse sample
            keep = np.arange(len(records))[:: max(1, len(records) // 120)]
            feats, split, y = feats[keep], split[keep], y[keep]
            records = [records[i] for i in keep]
        PT[d] = (feats, split, y, records, orig)
        LOGL[d], SHP[d], NLLV[d] = precompute(records)
        print(f"  {d}: {len(records)} rows (label={label_of(d)})", flush=True)

    header = ["model", "eval", "rung", "seed", "method", "prr", "n_train", "n_test", "layer",
              "beta_max", "train_sources", "mean_beta_test", "std_beta_test", "commit", "carve"]
    dheader = ["eval", "rung", "seed", "method", "mean_beta_train", "std_beta_train",
               "mean_beta_test", "std_beta_test", "beta_q10", "beta_q25", "beta_q50", "beta_q75",
               "beta_q90", "frac_beta_lt_0.5", "frac_beta_0.5_2", "frac_beta_gt_8",
               "mean_g_h", "std_g_h", "mean_g_n", "std_g_n", "gh_gn_sign_agree",
               "corr_beta_len", "corr_beta_max_z", "corr_beta_top5", "corr_beta_entropy",
               "corr_beta_quality"]
    out_rows, diag_rows = [], []

    for rung, X, spec in cells_long(set(PT), evals):
        if X not in PT:
            continue
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            allrows = train_rows + test_rows
            n_tr = len(train_rows)
            tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, len(allrows)))
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = y[te_idx]
            logl = [LOGL[d][i] for d, i in allrows]
            nllv = [NLLV[d][i] for d, i in allrows]
            Xs_raw = np.stack([SHP[d][i] for d, i in allrows])
            H_raw = np.stack([PT[d][0][i] for d, i in allrows])
            # standardise on TRAIN stats only
            ss_x = ShapeStandardiser(Xs_raw[tr_idx]); ss_h = ShapeStandardiser(H_raw[tr_idx])
            Xs = torch.from_numpy(ss_x(Xs_raw)).float()
            H = torch.from_numpy(ss_h(H_raw)).float()
            # B6.4 shuffled-hidden copy: permute TRAIN rows' h within each source dataset
            H_shuf = H.clone()
            for src in {d for d, _ in train_rows}:
                pos = [k for k, (d, _) in enumerate(train_rows) if d == src]
                rng = np.random.RandomState(100000 * sd + zlib.crc32(src.encode()) % 9973)
                H_shuf[pos] = H[np.array(pos)[rng.permutation(len(pos))]]

            srcs = "+".join(f"{d}:{sum(1 for dd, _ in train_rows if dd == d)}"
                            for d in dict.fromkeys(d for d, _c in spec))
            for m in METHODS:
                mode = MODE_OF[m]
                Huse = H_shuf if m.endswith("shuffled_h") else H
                gate = train_gate(mode, logl, Huse if mode in ("hs", "hybrid") else None,
                                  Xs if mode in ("nllshape", "hybrid") else None, y, tr_idx, sd)
                # test rows always use REAL hidden states (the control degrades training pairing)
                q, beta_te, g_h, g_n = score_gate(gate, mode, logl, H, Xs, te_idx)
                _, beta_tr, _, _ = score_gate(gate, mode, logl, Huse, Xs, tr_idx)
                if not np.isfinite(q).all():
                    raise SystemExit(f"non-finite score in {m} [{rung}/{X}] seed {sd}")
                prr = results.prr(yte, q)
                if not args.smoke:
                    out_rows.append({"model": MODEL, "eval": X, "rung": rung, "seed": sd,
                                     "method": m, "prr": round(float(prr), 4),
                                     "n_train": n_tr, "n_test": len(te_idx), "layer": LAYER,
                                     "beta_max": BETA_MAX, "train_sources": srcs,
                                     "mean_beta_test": round(float(beta_te.mean()), 4),
                                     "std_beta_test": round(float(beta_te.std()), 4),
                                     "commit": sha, "carve": carve})
                    lens = np.array([len(nllv[i]) for i in te_idx], float)
                    xs_te = Xs_raw[te_idx]
                    def _c(a, b):
                        return (float(np.corrcoef(a, b)[0, 1])
                                if np.std(a) > 0 and np.std(b) > 0 else np.nan)
                    diag_rows.append({
                        "eval": X, "rung": rung, "seed": sd, "method": m,
                        "mean_beta_train": round(float(beta_tr.mean()), 4),
                        "std_beta_train": round(float(beta_tr.std()), 4),
                        "mean_beta_test": round(float(beta_te.mean()), 4),
                        "std_beta_test": round(float(beta_te.std()), 4),
                        **{f"beta_q{qq}": round(float(np.percentile(beta_te, qq)), 4)
                           for qq in (10, 25, 50, 75, 90)},
                        "frac_beta_lt_0.5": round(float((beta_te < 0.5).mean()), 4),
                        "frac_beta_0.5_2": round(float(((beta_te >= 0.5) & (beta_te <= 2)).mean()), 4),
                        "frac_beta_gt_8": round(float((beta_te > 8).mean()), 4),
                        "mean_g_h": round(float(g_h.mean()), 4) if g_h is not None else "",
                        "std_g_h": round(float(g_h.std()), 4) if g_h is not None else "",
                        "mean_g_n": round(float(g_n.mean()), 4) if g_n is not None else "",
                        "std_g_n": round(float(g_n.std()), 4) if g_n is not None else "",
                        "gh_gn_sign_agree": (round(float((np.sign(g_h) == np.sign(g_n)).mean()), 4)
                                             if g_h is not None and g_n is not None else ""),
                        "corr_beta_len": _c(beta_te, lens),
                        "corr_beta_max_z": _c(beta_te, xs_te[:, 4]),
                        "corr_beta_top5": _c(beta_te, xs_te[:, 6]),
                        "corr_beta_entropy": _c(beta_te, xs_te[:, 8]),
                        "corr_beta_quality": _c(beta_te, yte),   # diagnostic only, never a score
                    })
            if args.smoke:
                print(f"  SMOKE [{rung}/{X}] seed {sd}: {len(METHODS)} gates trained+scored, "
                      f"all finite, schema OK — no PRR shown by design")
                return
            flush(out, out_rows, header); flush(diag_out, diag_rows, dheader)
            print(f"[{rung:14s}] eval={X} seed={sd} train={srcs} — {len(METHODS)} methods "
                  f"landed ({len(out_rows)} rows on disk)", flush=True)

    print(f"\nwrote {out} and {diag_out}")


if __name__ == "__main__":
    main()
