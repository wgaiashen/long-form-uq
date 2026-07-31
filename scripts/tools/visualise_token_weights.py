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
import functools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402

from luq import cache, weighted_msp, weighting, token_subsets  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import orgad_llm, sar  # noqa: E402
from attn_pool import load_per_token, pad_batch, train_attn, PROMPT_REGIME  # noqa: E402
from msp_ablations import build_masks  # noqa: E402  (P1.2 token-subset keep-masks)
from viz_common import (  # noqa: E402
    load_tokenizer, token_pieces, method_scores, minmax,
    interestingness, render_example, render_html,
)

LAYER = 15

# Short, technical "what is this + how is it computed" blurb per toggle, shown in the page's Method
# reference panel. Keys MUST match the signal names produced by per_token_signals(). Trusted HTML
# (we author it) -- a little <code> is fine. The point is to make the "token selection is heuristic"
# reading legible: you can see each weighting's mechanism next to where it actually lands.
METHOD_DOCS = {
    "surprisal":
        "The raw MSP signal every weighting re-weights: per-token surprisal <code>-log p(token)</code> "
        "read straight from the cached generation logprobs. Deeper = the model was less sure of that token. "
        "Plain MSP is the unweighted sum of these.",
    "wMSP-pairwise":
        "Our LEARNED weighted-MSP (primary), <b>with EOS/special tokens dropped</b> (the default keep-mask): a "
        "4-layer MLP (<code>4096&rarr;256&rarr;128&rarr;64&rarr;1</code>) reads each token's layer-15 hidden "
        "state and emits one logit; a <code>softmax</code> over the sequence (average-1 normalised, EOS forced "
        "to weight 0 pre-softmax) gives the weight, and the score is "
        "<code>&Sigma; w&#7511;&middot;(-log p&#7511;)</code>. Trained on correctness with a pairwise soft-rank "
        "loss. This is the 'learnt MSP without EOS' variant; the token-selection family below drops progressively "
        "more (special_punct = +punctuation, content = +stop-words).",
    "wMSP-Blondel":
        "Same MLP and score as wMSP-pairwise, but trained with the Blondel differentiable soft-rank loss "
        "(torchsort, O(n log n) exact) instead of the pairwise surrogate &mdash; it optimises the whole-batch "
        "ranking (PRR) directly.",
    "wMSP-shrink2":
        "wMSP-pairwise plus a shrink-to-uniform penalty during training "
        "(<code>+2&middot;&Sigma;(w-1)&sup2;</code>). This pulls the weight distribution back toward uniform "
        "(= plain MSP) to stop it spiking on lone tokens &mdash; a P1.1 moderation variant.",
    "wMSP-shrink10":
        "As wMSP-shrink2 but the HEAVY penalty (<code>+10&middot;&Sigma;(w-1)&sup2;</code>): weights are pulled "
        "hard toward uniform, so this is close to plain length-normalised MSP (perplexity). Use it to SEE where "
        "even a strongly-shrunk weighter still deviates from flat &mdash; the token-level view of the (now "
        "fair-floor-corrected) cnn story.",
    "wMSP-smooth3":
        "wMSP-pairwise but the per-token logits are neighbour-averaged over a 3-token window "
        "(<code>smooth_raw</code>, before the softmax) at both train and score time, encoding "
        "&lsquo;importance is a span property, not a lone token&rsquo; &mdash; a P1.1 smoothing variant.",
    "wMSP-entropy_hinge":
        "wMSP-pairwise plus a THRESHOLDED entropy penalty (<code>+2&middot;relu(0.7&minus;H/log&nbsp;n)&sup2;</code>): "
        "unlike shrink, it is 0 while the weights stay smooth and only pushes back once the distribution gets "
        "too peaked (normalised entropy &lt; 0.7). Joe&rsquo;s item-1 hinge &mdash; should look like wMSP-pairwise "
        "except where it clips a spike.",
    "wMSP-special_punct":
        "wMSP-pairwise with EOS/special tokens AND punctuation/whitespace-only pieces dropped (weight 0 "
        "pre-softmax) &mdash; the &lsquo;learnt MSP without punctuation&rsquo; step. Middle of the token-selection "
        "family (pairwise = EOS only &rarr; special_punct = +punctuation &rarr; content = +stop-words); the "
        "<code>keep=</code> special_punct mask.",
    "wMSP-content":
        "wMSP-pairwise but the learned softmax is RESTRICTED to content tokens only &mdash; special "
        "tokens, punctuation and stop-words (negation kept) get weight 0 pre-softmax, so the weighter "
        "cannot key on filler. The &lsquo;learnt MSP without stop-words&rsquo; variant &mdash; Joe&rsquo;s "
        "&lsquo;exclude stop words&rsquo; idea (the <code>keep=</code> content mask).",
    "wMSP-segment":
        "wMSP but the MLP emits ONE weight per SENTENCE, broadcast to that sentence&rsquo;s tokens "
        "(segment-mean of the raw logits, then softmax over sentences) &mdash; attacks the single-token "
        "spike degeneracy structurally. Joe&rsquo;s segment-weighting idea (<code>segment_ids=</code>).",
    "uniform":
        "The mean-pool baseline: a frozen-query attention pooler, so every token gets an equal weight. "
        "The &lsquo;no weighting&rsquo; control &mdash; if a learned weighting cannot beat this, its token "
        "selection is not buying anything.",
    "orgad":
        "Orgad important-token mask (0/1). The model&rsquo;s OWN exact-answer span, located by an LLM "
        "extraction of the answer (leak-free), mapped onto the generated tokens. QA stores one answer "
        "string; summarisation stores a list of key phrases, each located and unioned into the mask. "
        "Shown only where <code>cache/orgad_llm/</code> is built.",
    "orgad-broad":
        "REFINED Orgad mask (0/1) from the broadened long-form-QA prompt (2026-07-19): the SET of "
        "claim-bearing spans (yes/no verdict + findings + entities/numbers), not just the one exact answer. "
        "Fixes the degenerate short-form prompt (on pubmed the exact version collapsed to just &lsquo;Yes&rsquo;). "
        "Compare against <code>orgad</code> to SEE the broadening. Shown where <code>__broad.json</code> is built.",
    "SAR-token":
        "TokenSAR relevance <code>R&#771;</code> at the token level: remove each token, re-measure "
        "cross-encoder semantic similarity to the full answer, <code>R=1-sim</code>, normalised over the "
        "sequence. Higher = removing it changes the meaning more. Token-level is used for short-form.",
    "SAR-sentence":
        "TokenSAR relevance <code>R&#771;</code> at the sentence level (each token inherits its "
        "sentence&rsquo;s relevance): remove each sentence, re-measure cross-encoder similarity, "
        "<code>R=1-sim</code>, normalised. Used for long-form, where token-level removal is near-uniform.",
    "abl-minus_stop":
        "P1.2 MSP-ablation keep-mask (0/1): the tokens MSP would keep after dropping function/stop words. "
        "Built by grouping BPE pieces into words via the leading-space glyph and dropping stop words.",
    "abl-first_of_word":
        "P1.2 keep-mask: only the FIRST sub-word piece of each word is kept (word-initial tokens).",
    "abl-last_of_word":
        "P1.2 keep-mask: only the LAST sub-word piece of each word is kept.",
    "abl-content_word":
        "P1.2 keep-mask: content words only &mdash; non-stop words, first sub-word piece.",
    "abl-first_sentence":
        "P1.2 keep-mask: only the tokens of the first sentence (split on <code>.!?</code>) are kept.",
}


# --------------------------------------------------------------------------------------
# Optional weight-source caches (each toggle appears only if its cache exists)
# --------------------------------------------------------------------------------------

def load_sar(model, dataset):
    """TokenSAR relevance cache (cache/sar/<slug>__<ds>__ID__<gran>.npz). Auto-detect the granularity,
    preferring the SHARPER token-level answer-only (`token__noprepend`) cache the results actually use,
    then plain `token`, then `sentence` (near-uniform on long-form). Returns (list of length-G R~ arrays,
    display_gran) or (None, None). Mirrors weighted_msp_sar.load_sar so the viz shows the same signal the
    ladder scored, not the old near-uniform sentence SAR."""
    for gran, disp in (("token__noprepend", "token"), ("token", "token"), ("sentence", "sentence")):
        p = ROOT / "cache" / "sar" / f"{cache._slug(model)}__{dataset}__ID__{gran}.npz"
        if p.exists():
            return list(np.load(p, allow_pickle=True)["relevance"]), disp
    return None, None


def load_orgad_json(model, dataset, variant="exact"):
    """Orgad LLM-extracted answer cache. variant='broad' reads the refined long-form-QA span cache (__broad),
    so the viz can show the broadened important-token set beside the exact-answer one.
    Keyed by f"{split}:{idx}" (idx alone collides across train/test -- see the audit fix)."""
    suffix = "__broad" if variant == "broad" else ""
    p = ROOT / "cache" / "orgad_llm" / f"{cache._slug(model)}__{dataset}__ID{suffix}.json"
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
    # NOTE: signal names become HTML data-attribute keys (data-<name>), so they MUST be valid attribute
    # names -- no '(', ')', '/'. (An earlier version used "surprisal(MSP)"/"orgad(0/1)"/"SAR(token)",
    # which the browser could not parse, so those toggles silently coloured every token to 0.)
    surprisal = [-lp for lp in record["token_logprobs"]][:g]
    signals = {"surprisal": (surprisal, minmax(surprisal))}

    states = ctx["states"]
    device = ctx["device"]
    with torch.no_grad():
        # answer_states drops row 0 (the last-prompt anchor), so its G rows align 1:1 with the G
        # generated tokens. The weight is softmax(MLP(states)) over the sequence -- exactly the
        # `normalised` weight the weighted-MSP score uses.
        asx = torch.from_numpy(weighted_msp.answer_states(states[pos])).to(device)
        # The TRUE scored weight excludes special tokens (EOS) pre-softmax (the fix). Route every displayed
        # weight through _weights_from_raw(keep=) so the viz shows what the score actually uses -- EOS -> 0,
        # content tokens averaged to 1 -- instead of the old raw softmax that still lit up the EOS token.
        keep_t = torch.from_numpy(weighted_msp.content_keep(record)).to(device)

        def _wshow(raw, keep=keep_t):
            return weighted_msp._weights_from_raw(raw, "normalised", keep=keep).cpu().numpy()[:g]

        # 2) wMSP-pairwise
        w_pair = _wshow(ctx["wm_pair"](asx))
        signals["wMSP-pairwise"] = (list(w_pair), minmax(w_pair))

        # 3) wMSP-Blondel (only if trained)
        if ctx["wm_bl"] is not None:
            signals["wMSP-Blondel"] = (lambda w: (list(w), minmax(w)))(_wshow(ctx["wm_bl"](asx)))

        # 3b) P1.1 smoothing/moderation variants (shrink -> flatter; smooth -> gentler peaks).
        if ctx.get("wm_shrink2") is not None:
            signals["wMSP-shrink2"] = (lambda w: (list(w), minmax(w)))(_wshow(ctx["wm_shrink2"](asx)))
        if ctx.get("wm_shrink10") is not None:
            signals["wMSP-shrink10"] = (lambda w: (list(w), minmax(w)))(_wshow(ctx["wm_shrink10"](asx)))
        if ctx.get("wm_smooth3") is not None:
            raw_sm = weighting.smooth_raw(ctx["wm_smooth3"](asx), 3)      # smooth before the (keep-masked) softmax
            signals["wMSP-smooth3"] = (lambda w: (list(w), minmax(w)))(_wshow(raw_sm))
        if ctx.get("wm_hinge") is not None:
            signals["wMSP-entropy_hinge"] = (lambda w: (list(w), minmax(w)))(_wshow(ctx["wm_hinge"](asx)))

        # 3c) NEW constrained variants (Joe #4/#7): content-restricted + one-weight-per-sentence.
        if ctx.get("wm_special_punct") is not None:
            pieces_sp = ctx["tok"].convert_ids_to_tokens(record["gen_token_ids"])
            keep_sp = torch.from_numpy(token_subsets.keep_mask(record["gen_token_ids"], pieces_sp, "special_punct")).to(device)
            signals["wMSP-special_punct"] = (lambda w: (list(w), minmax(w)))(_wshow(ctx["wm_special_punct"](asx), keep_sp))
        if ctx.get("wm_content") is not None:
            pieces_c = ctx["tok"].convert_ids_to_tokens(record["gen_token_ids"])
            keep_c = torch.from_numpy(token_subsets.keep_mask(record["gen_token_ids"], pieces_c, "content")).to(device)
            signals["wMSP-content"] = (lambda w: (list(w), minmax(w)))(_wshow(ctx["wm_content"](asx), keep_c))
        if ctx.get("wm_segment") is not None:
            sid, _ = sar._token_sentence_ids(
                ctx["tok"], list(record["gen_token_ids"]),
                record.get("gen_text") or ctx["tok"].decode(record["gen_token_ids"], skip_special_tokens=True))
            seg_t = torch.from_numpy(np.asarray(sid, dtype=np.int64)).to(device)
            raw_seg = weighted_msp._segment_mean_raw(ctx["wm_segment"](asx), seg_t)
            signals["wMSP-segment"] = (lambda w: (list(w), minmax(w)))(_wshow(raw_seg))

        # 4) uniform (frozen-query pooler = mean-pool): the "no weighting" control.
        X, m, posf = pad_batch([states[pos]], device)
        _, a = ctx["unif"](X, m, posf)
        # a[0] spans the G+1 window; drop row 0 (anchor) to align with the G tokens.
        unif_w = a[0, : states[pos].shape[0]].cpu().numpy()[1: g + 1]
        signals["uniform"] = (list(unif_w), minmax(unif_w))

    # 5) orgad important-token mask (leak-free), keyed by split:idx. Only if located here.
    #    QA datasets store ONE answer string; summarisation (xsum/samsum) stores a LIST of important
    #    phrases. Normalise to a list of strings and union each located span into the mask.
    def _orgad_mask(oj):
        extracted = oj.get(f"{record['split']}:{record['idx']}")
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
        return mask
    # exact-answer Orgad (the original) and, where built, the refined BROAD span set (2026-07-19).
    if ctx["orgad_json"] is not None:
        m = _orgad_mask(ctx["orgad_json"])
        if any(m):                        # colour by RAW 0/1 (answer token fully shaded), not min/max
            signals["orgad"] = (m, m)
    if ctx.get("orgad_broad_json") is not None:
        mb = _orgad_mask(ctx["orgad_broad_json"])
        if any(mb):
            signals["orgad-broad"] = (mb, mb)

    # 6) SAR relevance, indexed by full-record position.
    sar_rel = ctx["sar_rel"]
    if sar_rel is not None and pos < len(sar_rel):
        rel = np.asarray(sar_rel[pos], dtype=float)[:g]
        if len(rel) == g:
            signals[f"SAR-{ctx['gran']}"] = (list(rel), minmax(rel))

    # 7) P1.2 MSP-ablation keep-masks (0/1): which tokens each ablation KEEPS. Built from the exact same
    #    tokenizer-piece grouping the ablation battery used (msp_ablations.build_masks over the HF pieces),
    #    so this is a direct visual check of that grouping. Coloured by raw 0/1 like orgad (an all-1 mask
    #    must not min/max-collapse to 0).
    if ctx.get("show_ablations"):
        pieces = ctx["tok"].convert_ids_to_tokens(record["gen_token_ids"])
        masks = build_masks(pieces)
        for name in ("minus_stop", "first_of_word", "last_of_word", "content_word", "first_sentence"):
            m = masks[name].astype(float).tolist()[:g]
            signals[f"abl-{name}"] = (m, m)

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
    ap.add_argument("--no-ablations", dest="show_ablations", action="store_false",
                    help="hide the P1.2 MSP-ablation keep-masks (shown by default).")
    ap.set_defaults(show_ablations=True)
    ap.add_argument("--dump-weights", default="",
                    help="CONSOLIDATION MODE: instead of rendering, compute the per-token weights for the WHOLE "
                         "test set and save a curated subset (+orgad/sar) to this npz, keyed by record position, "
                         "so visualise_allmethods.py can read them as tracks (no retrain). Skips the HTML.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # prompt-regime aware: a regime-namespaced XL set (ExpertQA -> cache/expertqa_rp12/) resolves its
    # records/scores from that root, exactly as load_per_token does -- so the viz covers it like any core set.
    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=PROMPT_REGIME.get(args.dataset, ""))
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

    # Training indices: a two-split set uses its baked train rows; a single-split XL set (ExpertQA all-"test",
    # med_quad/samsum all-"train") has a deterministic train carve, exactly as the ladders do (xl_rungs.eval_split).
    import xl_rungs  # noqa: E402  (shared organic-ProbeDriftXL split logic)
    tr_idx, _ = xl_rungs.eval_split(np.asarray(split_pt))
    tr = list(int(i) for i in tr_idx)
    print(f"  {len(states)} examples, {len(tr)} train ({'baked' if len(set(split_pt)) > 1 else 'carved'}); "
          f"training the weight models on CPU...")

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
    # P1.1 smoothing/moderation variants, trained exactly as the sweep did (cheap on CPU).
    wm_shrink2 = weighted_msp.train_weighted_msp(
        states, records, y, tr, device, weight_mode="normalised", length_normalise=True, seed=1,
        loss="pairwise", reg=weighting.shrink_to_uniform, reg_lambda=2.0)
    wm_smooth3 = weighted_msp.train_weighted_msp(
        states, records, y, tr, device, weight_mode="normalised", length_normalise=True, seed=1,
        loss="pairwise", smooth_n=3)
    # entropy_hinge: the thresholded entropy penalty (Joe #1) -- fires ONLY once the weights get too peaked
    # (normalised entropy < 0.7), so it should look like wMSP-pairwise while smooth and flatten only the spikes.
    wm_hinge = weighted_msp.train_weighted_msp(
        states, records, y, tr, device, weight_mode="normalised", length_normalise=True, seed=1,
        loss="pairwise", reg=functools.partial(weighting.entropy_hinge, threshold=0.7), reg_lambda=2.0)

    # Optional weight-source caches (toggles appear only if built).
    orgad_json = load_orgad_json(cfg.model_name, args.dataset)
    orgad_broad_json = load_orgad_json(cfg.model_name, args.dataset, variant="broad")
    sar_rel, gran = load_sar(cfg.model_name, args.dataset)
    gran = gran or "token"
    print(f"  orgad cache: {'yes' if orgad_json else 'no'} | "
          f"orgad-broad: {'yes' if orgad_broad_json else 'no'} | SAR({gran}) cache: {'yes' if sar_rel else 'no'}")

    tok = load_tokenizer(cfg.model_name)
    pieces_by_pos = {i: token_pieces(tok, r["gen_token_ids"]) for i, r in enumerate(records)}

    # NEW constrained-weighting variants (Joe #4/#7), trained like the keep-variants sweep. keep_mask needs
    # the BPE pieces (convert_ids_to_tokens, glyph-prefixed), NOT the decoded token_pieces.
    content_keep_list = [token_subsets.keep_mask(r["gen_token_ids"],
                                                 tok.convert_ids_to_tokens(r["gen_token_ids"]), "content")
                         for r in records]
    segids = [np.asarray(sar._token_sentence_ids(
                  tok, list(r["gen_token_ids"]),
                  r.get("gen_text") or tok.decode(r["gen_token_ids"], skip_special_tokens=True))[0], np.int64)
              for r in records]
    wm_content = weighted_msp.train_weighted_msp(
        states, records, y, tr, device, weight_mode="normalised", length_normalise=True, seed=1,
        loss="pairwise", keep=content_keep_list)
    wm_segment = weighted_msp.train_weighted_msp(
        states, records, y, tr, device, weight_mode="normalised", length_normalise=True, seed=1,
        loss="pairwise", segment_ids=segids)
    # shrink@10 -- the HEAVY moderation variant (was the cnn "headline"; kept for the token-level story of
    # WHERE a strongly-shrunk weighter still deviates from uniform). Same recipe as the all_variants sweep.
    wm_shrink10 = weighted_msp.train_weighted_msp(
        states, records, y, tr, device, weight_mode="normalised", length_normalise=True, seed=1,
        loss="pairwise", reg=weighting.shrink_to_uniform, reg_lambda=10.0)
    # special_punct keep-variant: the learnt weighting with EOS + PUNCTUATION dropped (the middle rung of the
    # token-selection family: pairwise = EOS only, special_punct = EOS+punct, content = EOS+punct+stop-words).
    special_punct_keep_list = [token_subsets.keep_mask(r["gen_token_ids"],
                                                       tok.convert_ids_to_tokens(r["gen_token_ids"]), "special_punct")
                               for r in records]
    wm_special_punct = weighted_msp.train_weighted_msp(
        states, records, y, tr, device, weight_mode="normalised", length_normalise=True, seed=1,
        loss="pairwise", keep=special_punct_keep_list)

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
           "wm_shrink2": wm_shrink2, "wm_shrink10": wm_shrink10, "wm_smooth3": wm_smooth3, "wm_hinge": wm_hinge,
           "wm_content": wm_content, "wm_special_punct": wm_special_punct,
           "wm_segment": wm_segment, "show_ablations": args.show_ablations,
           "device": device, "orgad_json": orgad_json, "orgad_broad_json": orgad_broad_json,
           "sar_rel": sar_rel, "gran": gran, "tok": tok}

    # CONSOLIDATION dump: compute per-token weights for the WHOLE test set and persist a curated subset so
    # visualise_allmethods.py reads them as tracks (no per-dataset retrain there). Reuses the SAME ctx +
    # per_token_signals as the render path, so the dumped vectors are byte-identical to what this tool shows.
    if args.dump_weights:
        CURATED = ["wMSP-pairwise", "wMSP-shrink10", "wMSP-content", "wMSP-segment", "wMSP-smooth3",
                   "orgad", "orgad-broad", "sar"]
        # key on the SAME test set the ladders/sidecars/all-methods use (xl_rungs.eval_split), NOT split=="test"
        # (which is empty for the all-'train' XL sets med_quad/samsum) -> guarantees overlap with all-methods.
        _, te_es = xl_rungs.eval_split(np.asarray(split_pt))
        dump_positions = [int(i) for i in te_es]
        cols = {k: [] for k in CURATED}
        rec_pos = []
        for i in dump_positions:
            sigs = per_token_signals(records[i], i, ctx)
            rec_pos.append(i)
            for k in CURATED:
                cols[k].append(np.asarray(sigs[k][0], dtype=np.float32) if k in sigs else None)
        save = {"record_pos": np.asarray(rec_pos, dtype=np.int64)}
        present = []
        for k in CURATED:                              # drop tracks absent for EVERY example (no cache for this set)
            if any(v is not None for v in cols[k]):
                save[k.replace("-", "_")] = np.array(cols[k], dtype=object)
                present.append(k)
        dp = Path(args.dump_weights); dp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dp, **save)
        print(f"[dump-weights] wrote {dp}  ({len(rec_pos)} test examples; tracks: {present})")
        return

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
                               signal_names, "\n".join(ex_html), subtitle=subtitle,
                               method_docs=METHOD_DOCS))
    print(f"wrote {out}  ({len(shown)} of {len(candidates)} {args.split} examples; "
          f"toggles: {signal_names})")


if __name__ == "__main__":
    main()
