"""GPU step: the Table-1 UHead baseline reproduced by running the reference `luh` code.

Reproduces the paper's headline Uhead (Shelmanov et al., transformer probe over attention maps +
token probabilities). Uses his real feature extractors + FullSeqHead (imported from
Temp_robust_UQ_probes with a TF stub), driven by our cached Llama records via a teacher-forced
forward. Trains on AlignScore (the paper's primary Table-1 correctness function), scores the test
split, and writes the `uhead` / `uhead_v2` column 04_eval reads. See src/luq/uhead_reference.py for the
faithfulness argument and the documented deviations.

    python scripts/01j_uhead_reference.py --model meta-llama/Meta-Llama-3.1-8B --dataset sciq --variant v1

Defaults: fp32 + eager base (matches the reference implementation, and eager returns the attentions the extractor needs) and
AlignScore as the training/eval signal. To compare against the paper's Table 1, run 04_eval with
--label-field correctness_alignscore too. --max-seq caps the longest XSum articles (all generated
tokens + most recent context kept); lower it or pass --dtype bf16 if the all-layer attention OOMs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, generate, results, uhead_reference  # noqa: E402
from luq.config import Config  # noqa: E402

_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--variant", default="v1", choices=list(uhead_reference.VARIANTS),
                    help="his architecture_ablations uhead config: v1 (1L/16H/6ep) or v2 (2L/4H/7ep)")
    ap.add_argument("--name", default="uhead",
                    help="scores name ('uhead' for v1, 'uhead_v2' for v2 so both show in 04_eval)")
    ap.add_argument("--label-field", default="correctness_alignscore",
                    help="correctness field to TRAIN the head on (default: AlignScore, the paper's "
                         "Table-1 signal). Use 'correctness' for the judge as a secondary column.")
    ap.add_argument("--max-seq", type=int, default=2048,
                    help="cap on [prompt+gen] length; keeps all gen + most recent context")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"],
                    help="head training precision: fp16 = the fp16=True (autocast+GradScaler)")
    # Base forward: fp32 + eager matches the reference implementation (it passes no torch_dtype, --attn eager) and eager is
    # REQUIRED (the attention extractor needs attention weights; SDPA returns none).
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag (must match the one used by 01_extract).")
    args = ap.parse_args()

    if args.attn != "eager":
        sys.exit("uhead needs attention weights -> --attn eager (SDPA returns none)")

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    lf = args.label_field
    bad = sum(not isinstance(r.get(lf), (int, float)) for r in records)
    if bad:
        sys.exit(f"ERROR: {bad} records lack a numeric '{lf}' label — label it first "
                 f"(AlignScore via 01f_alignscore.py, or the judge via 02_label.py).")

    model, tok = generate.load_model(cfg.model_name, attn_implementation=args.attn,
                                     dtype=_DTYPE[args.dtype])
    head = uhead_reference.build_uhead(model, args.variant)
    print(f"built his uhead {args.variant}: feature_dim {head.feature_extractor.feature_dim()} "
          f"(= 4 token-probs + 2*{model.config.num_hidden_layers}*"
          f"{model.config.num_attention_heads} attention)", flush=True)

    # One teacher-forced forward per record -> run HIS extractors -> cache (X, masks) in RAM.
    feats, oms, ams = [], [], []
    for i, r in enumerate(records):
        li, lo, _ = uhead_reference.build_llm_io(model, tok, r, max_seq=args.max_seq)
        X, om, am = uhead_reference.extract_features(head, li, lo)
        feats.append(X); oms.append(om); ams.append(am)
        if (i + 1) % 200 == 0:
            print(f"  extracted {i + 1}/{len(records)}", flush=True)

    y = np.array([r[lf] for r in records], dtype=float)
    split = np.array([r["split"] for r in records])
    tr, te = split == "train", split == "test"
    if np.ptp(y[tr]) < 1e-6:
        sys.exit("all train labels identical — nothing for the head to learn")
    idx_tr, idx_te = np.where(tr)[0], np.where(te)[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training his uhead {args.variant} on {len(idx_tr)} examples (signal '{lf}') ...",
          flush=True)
    head = uhead_reference.train_head(
        head,
        [feats[i] for i in idx_tr], [oms[i] for i in idx_tr], [ams[i] for i in idx_tr],
        y[idx_tr], variant=args.variant, device=device, precision=args.precision,
    )
    unc = uhead_reference.predict(head, [feats[i] for i in idx_te], [oms[i] for i in idx_te],
                            [ams[i] for i in idx_te], device=device)

    rec_file = cache.records_path(cfg.cache_dir, key)
    path = cache.save_scores(unc, cfg.cache_dir, key, method=args.name, layer=-1,
                             variant=args.variant, label_field=lf, max_seq=args.max_seq,
                             source="reference_luh", rec_mtime=rec_file.stat().st_mtime)
    prr = results.prr(y[idx_te], unc)
    print(f"saved {len(unc)} test uncertainties -> {path}")
    print(f"PRR  uhead {args.variant} (his luh, trained/eval on {lf}) "
          f"[{args.dataset} {args.ood}]: {prr:.3f}")


if __name__ == "__main__":
    main()
