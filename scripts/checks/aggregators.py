"""The aggregation set: compare ways of turning a response's token states into one score.

The SAPLMA baseline mean-pools every answer-token hidden state into one vector (uniform weight per
token), so a single load-bearing clause is diluted by the rest. This script runs that baseline next
to the alternative aggregators, all on the SAME per-token cache and the SAME SAPLMA window
([P-1 : P+G], the last-prompt token plus the answer tokens), so the only thing that varies is how
the per-token states are combined:

    mean          average all window states, then SAPLMA MLP        (the project baseline)
    last-token    SAPLMA MLP on the final token's state             (the other naive aggregator)
    per-sentence  split the answer into sentences, mean-pool each, score each sentence with a
                  SAPLMA MLP (every sentence carries its response's label), then combine the
                  per-sentence P(correct) into one score (mean / min / geomean -- best reported)
    attention     the softmax-attention pooling family (from attn_pool.py), linear head

Orgad et al.'s filter-then-aggregate is intentionally NOT here: their important-token method keys
on the exact-answer token and the long-form adaptation has to be read off their released code, not
paraphrased, so it waits for that check.

Everything is reported as PRR against the judge label, with the mean baseline alongside as the bar
to beat. The mean baseline must reproduce the cached SAPLMA PRR (it is that mean) -- a cache gate.

Runs on the per-token cache (cache/pertok, layer 15) plus the tokenizer (for sentence / filler
splitting). CPU is fine; the attention probe uses GPU if present.

    python scripts/checks/aggregators.py --datasets sciq,trivia_qa,pubmed_qa
    python scripts/checks/aggregators.py --datasets xsum            # once xsum cache + labels exist
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/vol/gpudata/gs925-msc_project/hf_cache")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from luq import cache, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402
from attn_pool import AttnPool, train_attn, attn_prr  # noqa: E402  reuse the learned aggregator

MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"


def window_token_ids(record):
    """The token ids over the SAPLMA window, aligned to the per-token cache rows: the last-prompt
    token followed by every answer token (length G+1, matching states row count)."""
    return [record["prompt_token_ids"][-1]] + list(record["gen_token_ids"])


def sentence_ids(pieces):
    """Assign each window row a sentence index. A sentence ends on a piece carrying . ! ? or newline;
    the next piece opens the next sentence. Row 0 (last-prompt token) joins the first sentence."""
    sent = np.zeros(len(pieces), dtype=int)
    cur = 0
    for i, p in enumerate(pieces):
        sent[i] = cur
        if re.search(r"[.!?\n]", p):
            cur += 1
    return sent


def load_per_token(model, dataset, layer):
    """Per-example token states (positional alignment, since record idx is not unique) + records."""
    path = ROOT / "cache" / "pertok" / f"{cache._slug(model)}__{dataset}__ID__L{layer}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID")
    records = cache.load_records(cfg.cache_dir, cache.run_key(model, dataset, "ID"))
    st = z["states"]
    if len(st) != len(records):
        raise SystemExit(f"{dataset}: per-token cache {len(st)} != records {len(records)}")
    states = [np.asarray(st[k], dtype=np.float32) for k in range(len(records))]
    return states, records, int(z["layer"])


def aggregate_conf(p, how):
    """Combine a response's per-sentence (or per-token) P(correct) into one confidence."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if how == "mean":
        return p.mean()
    if how == "min":
        return p.min()
    if how == "geomean":
        return float(np.exp(np.log(p).mean()))
    raise ValueError(how)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--datasets", default="sciq,trivia_qa,pubmed_qa")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--seeds", default="1,2,3", help="seeds for the attention probe (mean +/- std)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    tok = AutoTokenizer.from_pretrained(args.model)

    for dataset in args.datasets.split(","):
        loaded = load_per_token(args.model, dataset, args.layer)
        if loaded is None:
            print(f"\n==== {dataset}: no per-token cache, skip ====")
            continue
        states, records, layer = loaded
        y = np.array([r.get("correctness", np.nan) for r in records], dtype=float)
        if np.isnan(y).any():
            print(f"\n==== {dataset}: missing judge labels, skip ====")
            continue
        split = np.array([r["split"] for r in records])
        tr = np.where(split == "train")[0]
        te = np.where(split == "test")[0]
        ytr, yte = y[tr], [y[i] for i in te]
        print(f"\n==== {dataset} (layer {layer}, train {len(tr)}, test {len(te)}) ====")

        # Per-row pieces for every response (for sentence / content splitting). Guard that the
        # piece count matches the cached state-row count, else the window is misaligned.
        pieces = [tok.convert_ids_to_tokens(window_token_ids(r)) for r in records]
        for k in range(len(records)):
            if len(pieces[k]) != states[k].shape[0]:
                raise SystemExit(f"{dataset} row {k}: {len(pieces[k])} pieces != "
                                 f"{states[k].shape[0]} states -- window misaligned")

        # ---- mean baseline (== cached SAPLMA): gate then report ----
        Xmean = np.stack([s.mean(axis=0) for s in states])
        base = probe.train_probe_mlp(Xmean[tr], ytr)
        prr_mean = results.prr(yte, probe.uncertainty(base, Xmean[te]))
        feat = cache.load_features(Config(model_name=args.model, dataset=dataset,
                                          ood_setting="ID").cache_dir,
                                   cache.run_key(args.model, dataset, "ID"), "saplma")
        prr_feat = results.prr(yte, probe.uncertainty(
            probe.train_probe_mlp(feat[tr, layer, :], ytr), feat[te, layer, :]))
        gate = "OK" if abs(prr_mean - prr_feat) < 0.03 else "MISMATCH -- per-token cache suspect"
        print(f"  [cache gate] mean {prr_mean:.3f} vs cached SAPLMA L{layer} {prr_feat:.3f}  {gate}")

        # ---- last-token ----
        Xlast = np.stack([s[-1] for s in states])
        last = probe.train_probe_mlp(Xlast[tr], ytr)
        prr_last = results.prr(yte, probe.uncertainty(last, Xlast[te]))

        # ---- Orgad filter-then-aggregate: DEFERRED until the authors' repo is checked. ----
        # Orgad et al.'s important-token method keys on the exact-answer token, which has no clean
        # long-form analogue, so the right adaptation must be read off their code, not paraphrased.
        # Do not run a stand-in here (an earlier stopword filter was removed for this reason).

        # ---- per-sentence: mean-pool each sentence, SAPLMA on sentence vectors, aggregate ----
        def sentence_vecs(k):
            sid = sentence_ids(pieces[k])
            return np.stack([states[k][sid == s].mean(axis=0) for s in range(sid.max() + 1)])
        sent_tr = [sentence_vecs(k) for k in tr]
        Xsent_tr = np.concatenate(sent_tr, axis=0)
        ysent_tr = np.concatenate([np.full(len(v), ytr[j]) for j, v in enumerate(sent_tr)])
        sent_clf = probe.train_probe_mlp(Xsent_tr, ysent_tr)
        per_sent_conf = [sent_clf.p_correct(sentence_vecs(k)) for k in te]
        prr_sent = {how: results.prr(yte, [1 - aggregate_conf(p, how) for p in per_sent_conf])
                    for how in ("mean", "min", "geomean")}
        best_sent = max(prr_sent, key=prr_sent.get)

        # ---- attention-pool (learned aggregator), mean +/- std over seeds ----
        attn_prrs = []
        for sd in seeds:
            mdl = train_attn(states, y, list(tr), device, seed=sd)
            attn_prrs.append(attn_prr(mdl, states, y, list(te), device))
        attn_prrs = np.array(attn_prrs)

        def line(name, prr):
            return f"  {name:26s} {prr:+.3f}   (vs mean {prr - prr_mean:+.3f})"
        print(line("mean-pool (baseline)", prr_mean))
        print(line("last-token", prr_last))
        print(line(f"per-sentence ({best_sent})", prr_sent[best_sent])
              + f"    [mean {prr_sent['mean']:+.3f} min {prr_sent['min']:+.3f} "
                f"geomean {prr_sent['geomean']:+.3f}]")
        spread = f" +/- {attn_prrs.std():.3f}" if len(seeds) > 1 else ""
        print(line("attention-pool", attn_prrs.mean()) + spread)


if __name__ == "__main__":
    main()
