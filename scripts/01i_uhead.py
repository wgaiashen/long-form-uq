"""GPU step: the UHead ARCHITECTURE-ABLATION variant (paper Table 16, `hs_middle` features).

NOTE: this is NOT the Table-1 headline Uhead baseline. The paper's Table-1 Uhead uses attention +
token-probability features and is reproduced by running the actual `luh` code in
`scripts/01j_uhead_reference.py` (see src/luq/uhead_reference.py). THIS script feeds the same
`full_sequence_uhead` transformer head the `hs_middle` per-token hidden states, which is the
architecture-ablation setting (Table 16: same features as SAPLMA, different head). Keep it only as
that ablation / an apples-to-apples "head architecture" comparison, not as the Uhead baseline.

Extract the middle-layer per-token hidden states over the FULL [prompt + generation] sequence,
train the transformer-encoder uncertainty head on the train split, score the test split, and write
a `uhead` scores column that 04_eval picks up next to SAPLMA / Lookback / P(True).

    python scripts/01i_uhead.py --model meta-llama/Meta-Llama-3.1-8B --dataset sciq --variant v1

Extraction + training + scoring happen in ONE job because the per-token full-sequence states are
large; we hold them in RAM (fp16) rather than caching ~tens of GB to disk (Tier-3: compute on
demand, do not hoard). The frozen base means training on these cached states is identical to the
end-to-end Trainer -- see src/luq/uhead_fullseq.py for the faithfulness argument.

Defaults match the SAPLMA keystone forward (fp32 + eager, layer 15), so the hidden states are the
same ones our SAPLMA `hs_middle` features were built from -- exactly the `hs_middle` setting.

--max-seq bounds the sequence length for the longest XSum articles (some exceed 5000 tokens): it
keeps ALL generated tokens and the most recent context tokens, truncating the oldest context. This
mirrors features/lookback.py's identical bound and only affects the longest few percent.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, generate, results, uhead_fullseq  # noqa: E402
from luq.config import Config  # noqa: E402

_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _features_for_record(model, tok, record, layer, max_seq):
    """Return (X, output_mask) for one record: per-token L=layer states over [prompt+gen], with the
    last position dropped (the "predicts the next token" alignment), and output_mask marking the
    generated tokens (the output_mask[1:] convention -> generated iff index >= len(prompt)-1)."""
    prompt_ids = list(record["prompt_token_ids"])
    gen_ids = list(record["gen_token_ids"])
    n_gen = len(gen_ids)

    # Bound very long sequences: keep every generated token and the most recent context tokens.
    if len(prompt_ids) + n_gen > max_seq:
        keep = max(1, max_seq - n_gen)
        prompt_ids = prompt_ids[-keep:]
    P = len(prompt_ids)

    full_ids = prompt_ids + gen_ids
    states = generate.recompute_states(model, tok, full_ids, [layer])[0]  # (seq, hidden) fp32 cpu
    X = states[:-1].numpy().astype(np.float16)                            # (seq-1, hidden): drop last
    # output_mask over the shifted positions: index j (0..seq-2) is a generated token iff j >= P-1.
    idx = np.arange(X.shape[0])
    output_mask = (idx >= P - 1).astype(np.int64)
    assert output_mask.sum() == n_gen, (output_mask.sum(), n_gen)          # sanity: exactly len(gen)
    return X, output_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--variant", default="v1", choices=list(uhead_fullseq.VARIANTS),
                    help="the head config: v1 (dim768/1L/16H/6ep) or v2 (dim768/2L/4H/7ep)")
    ap.add_argument("--layer", type=int, default=15,
                    help="middle layer (ceil(N/2)-1); 15 for Llama-3.1-8B, matching hs_middle")
    ap.add_argument("--max-seq", type=int, default=2048,
                    help="cap on [prompt+gen] length; keeps all gen + most recent context")
    ap.add_argument("--name", default="uhead",
                    help="scores name (use 'uhead' for v1, 'uhead_v2' for v2 so both show in 04_eval)")
    ap.add_argument("--label-field", default="correctness",
                    help="correctness field to train against (default: the judge label)")
    # Match the SAPLMA keystone forward so the states equal the hs_middle features they were built from.
    ap.add_argument("--dtype", default="fp32", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--attn", default="eager", choices=["auto", "eager", "sdpa"])
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag (must match the one used by 01_extract).")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    L = args.layer
    dtype = None if args.dtype == "auto" else _DTYPE[args.dtype]
    attn = None if args.attn == "auto" else args.attn
    model, tok = generate.load_model(cfg.model_name, attn_implementation=attn, dtype=dtype)
    n_layers = model.config.num_hidden_layers + 1
    if not 0 <= L < n_layers:
        sys.exit(f"--layer {L} out of range 0..{n_layers - 1}")

    # One teacher-forced forward per record to build the per-token features (held in RAM, fp16).
    feats, masks = [], []
    for i, r in enumerate(records):
        X, m = _features_for_record(model, tok, r, L, args.max_seq)
        feats.append(X)
        masks.append(m)
        if (i + 1) % 200 == 0:
            print(f"  extracted {i + 1}/{len(records)}", flush=True)
    # Free the base model before training the head (head training is CPU/GPU-light).
    del model
    torch.cuda.empty_cache()

    lf = args.label_field
    y = np.array([r[lf] for r in records], dtype=float)
    split = np.array([r["split"] for r in records])
    tr, te = split == "train", split == "test"
    if np.ptp(y[tr]) < 1e-6:
        sys.exit("all train labels identical — nothing for the head to learn")

    idx_tr = np.where(tr)[0]
    idx_te = np.where(te)[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training uhead {args.variant} on {len(idx_tr)} train examples "
          f"(feature_dim {feats[0].shape[1]}, layer {L}) ...", flush=True)
    head = uhead_fullseq.build_head(feats[0].shape[1], args.variant)
    head = uhead_fullseq.train_head(
        head,
        [feats[i] for i in idx_tr], [masks[i] for i in idx_tr], y[idx_tr],
        variant=args.variant, device=device,
    )

    unc = uhead_fullseq.predict(head, [feats[i] for i in idx_te], [masks[i] for i in idx_te],
                                device=device)

    # Provenance: stamp the records mtime so 04_eval refuses these scores after a relabel. (There is
    # no feature file for uhead -- the head is retrained from states each run -- so no feat_method
    # stamp; 04_eval takes its benign "no feature provenance" warning branch for uhead.)
    rec_file = cache.records_path(cfg.cache_dir, key)
    path = cache.save_scores(unc, cfg.cache_dir, key, method=args.name, layer=L,
                             variant=args.variant, label_field=lf, max_seq=args.max_seq,
                             rec_mtime=rec_file.stat().st_mtime)
    prr = results.prr(y[idx_te], unc)
    print(f"saved {len(unc)} test uncertainties -> {path}")
    print(f"PRR  uhead {args.variant} (layer {L}) [{args.dataset} {args.ood}]: {prr:.3f}")


if __name__ == "__main__":
    main()
