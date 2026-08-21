#!/usr/bin/env python
"""Export per-token CAWSA lambda=2 weights for the three fixed token-weight figure examples.

Population discipline: the three (dataset, record idx) pairs are COPIED VERBATIM from
scripts/tools/fig_wmsp_token_weights.py's PANELS list -- the deterministic A5 audit set
(scripts/checks/nll_token_audit.py) intersected with the pre-existing weight dump. No example is
re-selected here, and nothing is chosen on any score.

WHAT CHANGES vs results/analysis/fig_wmsp_token_weights_data.csv:
that export carried `wMSP_pairwise`, the UNREGULARISED weighting (reg_lambda = 0), read straight out
of cache/viz/*__wmsp_weights.npz. The npz dump's CURATED key list is
["wMSP-pairwise", "wMSP-shrink10", "wMSP-content", "wMSP-segment", "wMSP-smooth3", ...] -- it has
shrink10 but NOT shrink2, so there is no stored lambda=2 vector anywhere on disk and no checkpoint:
`train_weighted_msp` returns an in-memory model and nothing persists it. The lambda=2 weighter is
therefore RETRAINED here with the production recipe, byte-for-byte the same call the ladder uses
(scripts/checks/probedriftlong.py's "wmsp_shrink2" row) and the same call visualise_token_weights.py
makes for its `wm_shrink2` track:

    train_weighted_msp(states, records, y, tr, device, weight_mode="normalised",
                       length_normalise=True, seed=1, loss="pairwise",
                       reg=weighting.shrink_to_uniform, reg_lambda=2.0)

REPRODUCTION CONTROL (this is the point of the script, not a nicety): the SAME loop also retrains the
lambda=0 model and compares its per-token weights against the stored `wMSP_pairwise` vectors in the
npz. If retraining here reproduces the shipped lambda=0 numbers to ~1e-6, then the lambda=2 numbers
come off the same production path and are trustworthy. If it does NOT, the script says so loudly and
still writes the file with the mismatch recorded, so a silent drift can never be mistaken for a result.

Output: results/analysis/fig_hapes_lambda2_token_weights_data.csv (one row per generated token).
The output filename keeps its original spelling so the committed figure data stays in place.
Run on a COMPUTE NODE (it loads ~6.8 GB of per-token caches).
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from luq import cache                                                    # noqa: E402
from luq.config import Config                                            # noqa: E402
from luq import weighted_msp, weighting                                  # noqa: E402
from luq.weighted_msp import content_keep, per_token_nll                 # noqa: E402
from attn_pool import load_per_token, PROMPT_REGIME                      # noqa: E402
import xl_rungs                                                          # noqa: E402
import viz_common as V                                                   # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = cache._slug(MODEL)
LAYER = 15
LABEL_FIELD = "correctness"
LAMBDA = 2.0

# VERBATIM from fig_wmsp_token_weights.py PANELS. Do not edit, do not add, do not re-select.
PANELS = [("cnn_dailymail", 480), ("samsum", 642), ("pubmed_qa", 1264)]

OUT = ROOT / "results" / "analysis" / "fig_hapes_lambda2_token_weights_data.csv"


def train(states, records, y, tr, device, lam):
    """The production weighted-MSP training call. lam=0 -> the unregularised `wMSP-pairwise`
    (what the old figure export used); lam=2 -> CAWSA lambda=2 (`wmsp_shrink2` in the ladder)."""
    return weighted_msp.train_weighted_msp(
        states, records, y, tr, device, weight_mode="normalised", length_normalise=True,
        seed=1, loss="pairwise",
        reg=(weighting.shrink_to_uniform if lam > 0 else None), reg_lambda=lam)


def weights_for(model, state, rec, device):
    """Per-token weights exactly as the scoring code forms them: the MLP's raw logits, then the
    average-1 softmax with special tokens forced to -inf pre-softmax (`content_keep`). Mirrors
    visualise_token_weights.per_token_signals._wshow, which is how the figure's numbers were made."""
    g = len(rec["gen_token_ids"])
    with torch.no_grad():
        asx = torch.from_numpy(weighted_msp.answer_states(state)).to(device)
        keep = torch.from_numpy(content_keep(rec)).to(device)
        w = weighted_msp._weights_from_raw(model(asx), "normalised", keep=keep)
    return w.cpu().numpy()[:g]


def main():
    device = "cpu"                       # small MLP, and the states are already in RAM
    tok = V.load_tokenizer(MODEL)
    special_ids = set(int(i) for i in tok.all_special_ids)
    rows, control = [], []

    for dataset, idx in PANELS:
        cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID",
                     prompt_regime=PROMPT_REGIME.get(dataset, ""))
        records = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, dataset, "ID"))
        loaded = load_per_token(MODEL, dataset, LAYER, LABEL_FIELD)
        if loaded is None:
            raise SystemExit(f"no per-token cache for {dataset} at L{LAYER}")
        states, split_pt, y, layer, records_pt = loaded
        assert len(states) == len(records) == len(records_pt), f"{dataset}: length mismatch"
        for i in (0, len(records) // 2, len(records) - 1):
            assert list(records[i]["gen_token_ids"]) == list(records_pt[i]["gen_token_ids"]), \
                f"{dataset}: pertok cache disagrees with records at position {i} -- stale cache?"

        # The example is located by RECORD POSITION via the npz's record_pos, the same way the
        # figure did it -- record `idx` is not unique, so a bare idx lookup is not safe on its own.
        z = np.load(ROOT / "cache" / "viz" / f"{SLUG}__{dataset}__ID__wmsp_weights.npz",
                    allow_pickle=True)
        pos = z["record_pos"].tolist()
        hit = [p for p in pos if records[p].get("idx") == idx]
        if not hit:
            raise SystemExit(f"{dataset} #{idx} is not in the weight dump -- refusing to invent it")
        p = hit[0]
        rec = records[p]
        w_old_stored = np.asarray(z["wMSP_pairwise"][pos.index(p)], dtype=float)

        # Same deterministic train carve the ladders and the viz use.
        tr_idx, _ = xl_rungs.eval_split(np.asarray(split_pt))
        tr = [int(i) for i in tr_idx]
        print(f"[{dataset}] {len(states)} examples, {len(tr)} train rows; record pos {p}, idx {idx}")

        m0 = train(states, records, y, tr, device, 0.0)          # reproduction control
        m2 = train(states, records, y, tr, device, LAMBDA)       # the thing we actually want
        w0 = weights_for(m0, states[p], rec, device)
        w2 = weights_for(m2, states[p], rec, device)

        nll = per_token_nll(rec).astype(float)
        keep = content_keep(rec).astype(bool)
        ids = list(rec["gen_token_ids"])
        pieces = V.token_pieces(tok, ids)
        if not (len(w2) == len(nll) == len(keep) == len(ids) == len(pieces)):
            raise SystemExit(f"{dataset} #{idx}: length mismatch, refusing to export")

        d = float(np.max(np.abs(w0 - w_old_stored)))
        control.append((dataset, idx, d, len(ids)))
        print(f"  lambda=0 reproduction control: max |retrained - stored wMSP_pairwise| = {d:.3e}")

        q = rec.get("correctness", rec.get("factuality"))
        # Ranked positions. Ties would make a bare argmax arbitrary, so flag EVERY tied maximum.
        w2_masked = np.where(keep, w2, -np.inf)
        max_w, max_n = w2_masked.max(), nll.max()
        for t in range(len(ids)):
            rows.append({
                "dataset": dataset,
                "example_idx": idx,
                "quality": f"{float(q):.2f}",
                "token_position": t,
                "token_string": pieces[t],
                "token_nll": f"{nll[t]:.6f}",
                "cawsa_lambda2_weight": f"{w2[t]:.6f}",
                "is_content_token": int(bool(keep[t])),
                "is_special_token": int(int(ids[t]) in special_ids),
                "is_max_nll": int(nll[t] == max_n),
                "is_max_cawsa_weight": int(keep[t] and w2[t] == max_w),
            })
        jw, jn = int(np.argmax(w2_masked)), int(np.argmax(nll))
        print(f"  lambda={LAMBDA:g}: maxW={pieces[jw]!r} (pos {jw})  maxNLL={pieces[jn]!r} (pos {jn})")

    import csv
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwrote {OUT}  ({len(rows)} token rows)")
    worst = max(d for _ds, _i, d, _n in control)
    print("lambda=0 reproduction control, worst over the three responses: "
          f"max |retrained - stored| = {worst:.3e}")
    if worst > 1e-5:
        print("*** CONTROL FAILED: retraining does NOT reproduce the shipped lambda=0 weights. "
              "The lambda=2 numbers above are NOT on the same path as the published figure. ***")
    else:
        print("control PASSED: the retrained lambda=0 weights match the shipped figure vectors, so "
              "the lambda=2 weights come off the identical production path.")


if __name__ == "__main__":
    main()
