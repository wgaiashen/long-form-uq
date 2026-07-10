"""Per-dataset visualiser for the WEIGHTED-MSP family (where does each method put its per-token weight?).

WHAT THIS IS (in plain words)
-----------------------------
The sister of `visualise_attention.py`. That one shows the ATTENTION family (the learned pooler's
weights and the model's own self-attention). THIS one shows the WEIGHTED-MSP family: the methods that
decide *which token's MSP log-prob matters* when we score a whole generation. It writes ONE HTML page
per dataset, with the exact same toggle UI (click a button, every token recolours from that signal),
so the two families look and read identically.

The per-token signals it can show (a toggle each), in order:
  surprisal(MSP)   -log p(token)          the raw MSP signal every weighting re-weights (deeper = the
                                          model was less sure of this token)
  wMSP-pairwise    learned weight         softmax weight our weighted-MSP MLP puts on each token
                                          (trained with the pairwise ranking loss)
  wMSP-Blondel     learned weight         same, trained with the Blondel differentiable-rank loss
                                          (only shown if torchsort is installed)
  uniform          flat weight            the mean-pool baseline (frozen-query pooler = every token
                                          equal); the "no weighting" control
  orgad(0/1)       answer-span mask       the model's OWN exact-answer tokens (leak-free, from the
                                          LLM-extracted answer); shown only for datasets whose
                                          cache/orgad_llm/*.json has been built (short-form only)
  SAR(<gran>)      relevance R~           TokenSAR relevance: how much removing the token changes the
                                          answer's meaning; shown only where cache/sar/* exists

Every signal is min/max-normalised WITHIN each example for colour, and its raw value is on hover.
Optional signals simply do not appear as toggles for a dataset whose cache is not built yet -- exactly
how the attention viz degrades without its sidecar -- so this same script lights up the Orgad and SAR
toggles automatically as those caches complete.

NO GPU. Everything is trained on the per-token hidden states already cached at layer 15
(`cache/pertok/*__L15.npz`); there is no model forward pass. Runs on a CPU node.

USAGE
-----
    python scripts/tools/visualise_token_weights.py --dataset sciq --ood ID \
        --model meta-llama/Meta-Llama-3.1-8B

Writes results/viz/token_weights__<key>.html (override with --out). Open it in the browser / VS Code.

ALIGNMENT (the classic bug this guards against)
-----------------------------------------------
Three things must line up token-for-token: the cached record's `gen_token_ids`/`token_logprobs` (length
G), the per-token hidden-state window `states[pos]` (length G+1: row 0 is the last prompt token, dropped
by `answer_states`), and every weight vector. We assert G == len(logprobs), assert the pertok cache and
the canonical records describe the same generations, and clip every signal to G before rendering.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402

from luq import cache, weighted_msp  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import orgad_llm  # noqa: E402
from attn_pool import load_per_token, pad_batch, train_attn  # noqa: E402
from viz_common import (  # noqa: E402
    load_tokenizer, token_pieces, method_scores, minmax,
    interestingness, render_example, render_html,
)

LAYER = 15


# --------------------------------------------------------------------------------------
# Optional weight-source caches (each toggle appears only if its cache exists)
# --------------------------------------------------------------------------------------

def load_sar(model, dataset, gran):
    """TokenSAR relevance cache (cache/sar/<slug>__<ds>__ID__<gran>.npz). Indexed by full-record
    position (01s_sar_relevance.py enumerates cache.load_records). Returns the list of length-G R~
    arrays, or None if not built for this dataset/granularity yet."""
    p = ROOT / "cache" / "sar" / f"{cache._slug(model)}__{dataset}__ID__{gran}.npz"
    if not p.exists():
        return None
    return list(np.load(p, allow_pickle=True)["relevance"])


def load_orgad_json(model, dataset):
    """Orgad LLM-extracted model-own answer cache (cache/orgad_llm/<slug>__<ds>__ID.json).
    Keyed by f"{split}:{idx}" (idx alone collides across train/test: 2800 sciq records but 1800 unique
    idx, so bare idx mis-joins test rows -- see the audit fix). Returns {"split:idx": extracted} or None."""
    p = ROOT / "cache" / "orgad_llm" / f"{cache._slug(model)}__{dataset}__ID.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# --------------------------------------------------------------------------------------
# Per-token signals for the weighted-MSP family
# --------------------------------------------------------------------------------------

def per_token_signals(record, pos, ctx):
    """Return {signal_name: (raw_values, normalised_values)} for one example, mirroring the
    attention viz's `per_token_signals` so the shared renderer treats both families identically.

    `ctx` carries the trained models and caches so this stays a pure per-example function:
      states, wm_pair, wm_bl, unif, device, orgad_json, sar_rel, gran, tok
    """
    g = len(record["gen_token_ids"])

    # 1) the raw MSP signal every weighting re-weights.
    surprisal = [-lp for lp in record["token_logprobs"]][:g]
    signals = {"surprisal(MSP)": (surprisal, minmax(surprisal))}

    states = ctx["states"]
    device = ctx["device"]
    with torch.no_grad():
        # answer_states drops row 0 (the last-prompt anchor), so its G rows align 1:1 with the G
        # generated tokens. The weight is softmax(MLP(states)) over the sequence -- exactly the
        # `normalised` weight the weighted-MSP score uses.
        asx = torch.from_numpy(weighted_msp.answer_states(states[pos])).to(device)

        # 2) wMSP-pairwise
        w_pair = torch.softmax(ctx["wm_pair"](asx), dim=0).cpu().numpy()[:g]
        signals["wMSP-pairwise"] = (list(w_pair), minmax(w_pair))

        # 3) wMSP-Blondel (only if trained)
        if ctx["wm_bl"] is not None:
            w_bl = torch.softmax(ctx["wm_bl"](asx), dim=0).cpu().numpy()[:g]
            signals["wMSP-Blondel"] = (list(w_bl), minmax(w_bl))

        # 4) uniform (frozen-query pooler = mean-pool): the "no weighting" control.
        X, m, posf = pad_batch([states[pos]], device)
        _, a = ctx["unif"](X, m, posf)
        # a[0] spans the G+1 window; drop row 0 (anchor) to align with the G tokens.
        unif_w = a[0, : states[pos].shape[0]].cpu().numpy()[1: g + 1]
        signals["uniform"] = (list(unif_w), minmax(unif_w))

    # 5) orgad important-token mask (leak-free), keyed by split:idx. Only if located here.
    #    QA datasets store ONE answer string; summarisation (xsum/samsum) stores a LIST of important
    #    phrases. Normalise to a list of strings and union each located span into the mask.
    orgad_json = ctx["orgad_json"]
    if orgad_json is not None:
        extracted = orgad_json.get(f"{record['split']}:{record['idx']}")
        spans = [extracted] if isinstance(extracted, str) else (extracted or [])
        mask = [0.0] * g
        for s in spans:
            if not isinstance(s, str) or not s.strip():
                continue
            rows, found = orgad_llm.locate_extracted_rows(ctx["tok"], record["gen_token_ids"], s)
            if found:
                for r in rows:            # cache row r -> gen token index r-1
                    j = r - 1
                    if 0 <= j < g:
                        mask[j] = 1.0
        if any(mask):
            signals["orgad(0/1)"] = (mask, minmax(mask))

    # 6) SAR relevance, indexed by full-record position.
    sar_rel = ctx["sar_rel"]
    if sar_rel is not None and pos < len(sar_rel):
        rel = np.asarray(sar_rel[pos], dtype=float)[:g]
        if len(rel) == g:
            signals[f"SAR({ctx['gran']})"] = (list(rel), minmax(rel))

    return signals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--label-field", default="correctness",
                    help="which correctness field to show / sort by / train the weights on.")
    ap.add_argument("--primary-method", default="saplma",
                    help="method used to rank 'interesting' examples; falls back to any scored.")
    ap.add_argument("--max-examples", type=int, default=60,
                    help="cap the number of examples rendered (keeps the HTML light).")
    ap.add_argument("--split", default="test", choices=["test", "train", "all"],
                    help="which split to show. 'test' has method scores; 'train' does not.")
    ap.add_argument("--sort", default="interesting", choices=["interesting", "index"],
                    help="'interesting' surfaces confident-wrong / uncertain-right first.")
    ap.add_argument("--out", default="",
                    help="output HTML path (default: results/viz/token_weights__<key>.html).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    print(f"loading run: {key}  (device={device})")

    # Canonical records (full fields: prompt, idx, target, token_logprobs) for rendering + NLL.
    records = cache.load_records(cfg.cache_dir, key)
    for i, r in enumerate(records):
        assert len(r["gen_token_ids"]) == len(r["token_logprobs"]), (
            f"record {i}: {len(r['gen_token_ids'])} gen tokens vs "
            f"{len(r['token_logprobs'])} logprobs -- tokenisation/logprob mismatch")

    # Per-token hidden states (layer 15) -- what the weight models train on. No GPU forward.
    loaded = load_per_token(cfg.model_name, args.dataset, LAYER, args.label_field)
    if loaded is None:
        sys.exit(f"ERROR: no per-token cache for {args.dataset} at L{LAYER}. "
                 f"Run scripts/01h_pertoken.py --dataset {args.dataset} first.")
    states, split_pt, y, layer, records_pt = loaded

    # Alignment guard: the pertok cache and the canonical records must describe the SAME generations.
    assert len(states) == len(records) == len(records_pt), (
        f"length mismatch: {len(states)} states vs {len(records)} records vs {len(records_pt)} pertok")
    for i in (0, len(records) // 2, len(records) - 1):
        assert list(records[i]["gen_token_ids"]) == list(records_pt[i]["gen_token_ids"]), (
            f"pertok cache and records disagree on gen_token_ids at position {i} -- stale cache?")

    tr = [i for i in range(len(states)) if split_pt[i] == "train"]
    print(f"  {len(states)} examples, {len(tr)} train; training the weight models on CPU...")

    # Train the weighting models ONCE per dataset (cheap on CPU: small MLP / linear query).
    wm_pair = weighted_msp.train_weighted_msp(
        states, records, y, tr, device, weight_mode="normalised",
        length_normalise=True, seed=1, loss="pairwise")
    wm_bl = None
    if weighted_msp._HAVE_TORCHSORT:
        wm_bl = weighted_msp.train_weighted_msp(
            states, records, y, tr, device, weight_mode="normalised",
            length_normalise=True, seed=1, loss="blondel")
    else:
        print("  (torchsort not available -> skipping the wMSP-Blondel toggle)")
    unif = train_attn(states, y, tr, device, seed=1, freeze_query=True)

    # Optional weight-source caches (toggles appear only if built).
    orgad_json = load_orgad_json(cfg.model_name, args.dataset)
    gran = "token" if args.dataset in ("sciq", "trivia_qa") else "sentence"
    sar_rel = load_sar(cfg.model_name, args.dataset, gran)
    print(f"  orgad cache: {'yes' if orgad_json else 'no'} | "
          f"SAR({gran}) cache: {'yes' if sar_rel else 'no'}")

    tok = load_tokenizer(cfg.model_name)
    pieces_by_pos = {i: token_pieces(tok, r["gen_token_ids"]) for i, r in enumerate(records)}

    # Method scores + the same "interesting" sort as the attention viz.
    test_positions = [i for i, r in enumerate(records) if r["split"] == "test"]
    methods = method_scores(cfg.cache_dir, key, test_positions)
    print(f"  methods with scores: {list(methods.keys()) or '(none)'}")

    if args.split == "all":
        candidates = list(range(len(records)))
    else:
        candidates = [i for i, r in enumerate(records) if r["split"] == args.split]
    primary = args.primary_method if args.primary_method in methods else next(iter(methods), None)
    if args.sort == "interesting" and primary is not None:
        candidates.sort(
            key=lambda i: interestingness(i, records[i].get(args.label_field, 0.0), methods, primary),
            reverse=True)
        print(f"  sorted by 'interesting' using primary method: {primary}")
    shown = candidates[:args.max_examples]

    ctx = {"states": states, "wm_pair": wm_pair, "wm_bl": wm_bl, "unif": unif,
           "device": device, "orgad_json": orgad_json, "sar_rel": sar_rel,
           "gran": gran, "tok": tok}

    per_example = {i: per_token_signals(records[i], i, ctx) for i in shown}
    signal_names = []
    for i in shown:
        for name in per_example[i]:
            if name not in signal_names:
                signal_names.append(name)
    if not signal_names:
        signal_names = ["surprisal(MSP)"]

    ex_html = [render_example(records[i], i, methods, per_example[i],
                              args.label_field, pieces_by_pos[i]) for i in shown]

    subtitle = ("weighted-MSP family: surprisal is the raw MSP signal; wMSP-* are our LEARNED per-token "
                "weights; uniform is mean-pool (no weighting); orgad/SAR are unsupervised weight sources "
                "(shown where their cache exists).")
    out = Path(args.out) if args.out else (cfg.results_dir / "viz" / f"token_weights__{key}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(key, records, methods, args.label_field,
                               signal_names, "\n".join(ex_html), subtitle=subtitle))
    print(f"wrote {out}  ({len(shown)} of {len(candidates)} {args.split} examples; "
          f"toggles: {signal_names})")


if __name__ == "__main__":
    main()
