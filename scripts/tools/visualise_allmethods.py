#!/usr/bin/env python
"""All-methods token overlay -- ONE browsable HTML per dataset with EVERY weighting method as a toggleable
per-token track, on the SAME generations, so you can see by eye what each method looks at and where they differ.

Builds on visualise_attention.py's data path; reuses viz_common (additively extended: token_meta hover + absent
tracks). HYPOTHESIS GENERATION ONLY -- STANDING RULE: any claim it suggests needs a full-population statistic
before it enters the project's working notes.

Tracks (each a per-token weight over the G generated tokens; every G+1 source has row 0 (prompt anchor) dropped):
  model-side : surprisal (per-token NLL), msp_min (ONE-HOT on the argmin-logprob token), perplexity/uniform
  probe-side : attnpool ID + attnpool <OOD rung> (the dissolution contrast), MultiMax (ONE-HOT on argmax W.x_j),
               content-mass prior, NLL prior, soft-Orgad prior (pubmed/med_quad/expertqa only)
  ABSENT (shown as greyed disabled toggles with the reason, NEVER a uniform fallback):
               wMSP variants (per-dataset MLP retrain -- see visualise_token_weights.py), self-attention meanq
               (GPU eager pass; only cached for sciq/trivia), multi-head heads 1-4 + armD (poolers not persisted).

ALIGNMENT ASSERT: pool weights are length G+1 (row 0 = prompt anchor); model-side weights are length G. Every
track is asserted == G after the row-0 drop, per example (the de-alignment family that caused the expertqa n=24
bug). A track that fails for an example is dropped for THAT example with a logged reason, never mis-coloured.

CPU/cache-bound: loads the dataset's per-token states once (cache/pertok). Run via PBS, not the login node.
"""
import argparse
import glob
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from luq import cache, msp                                                    # noqa: E402
import viz_common as vc                                                       # noqa: E402
import attn_pool as ap                                                        # noqa: E402
from pool_peak_tokens import classify_id                                      # noqa: E402
from xl_rungs import label_of                                                 # noqa: E402
from prior_builders import build_prior, OrgadCoverageError                    # noqa: E402
from luq.features import orgad_llm                                            # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15
VIZ = ROOT / "cache" / "viz"
PROBES = ROOT / "cache" / "probes"
OUTDIR = ROOT / "results" / "viz"

ORGAD_DATASETS = {"pubmed_qa", "med_quad", "expertqa"}   # only these have a cache/orgad_llm broad cache

METHOD_DOCS = {
    "surprisal": "Per-token NLL (−logprob). The raw model-side signal every floor is built from.",
    "msp_min(argmin)": "ONE-HOT on the single lowest-logprob token — the token msp_min (the pre-registered floor) keys on.",
    "perplexity/uniform": "Flat weight (perplexity = mean NLL; armB mean-pool = uniform attention). Shown flat as a reference.",
    "attnpool_ID": "The learned attention pooler's weights, trained IN-DISTRIBUTION (armA @ ID).",
    "MultiMax(pick)": "ONE-HOT on argmax_j (W·x_j) — the single token MultiMax's hard-max selects (Kramar et al. eq 9).",
    "content_mass": "Frozen prior: 1.0 on content tokens (non-punct/non-special), 0 elsewhere.",
    "NLL_prior": "Frozen prior: per-token NLL (≥0), length-normalised — the label-free 'importance = surprisal' prior.",
    "soft_orgad": "Soft-Orgad prior: the LLM-located claim-bearing spans (τ soft tiers). See the verbatim prompt in the header.",
}


def load_pooler(pk):
    if not torch.cuda.is_available():
        torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
    d = pickle.load(open(pk, "rb"))
    m = d["model"]
    return m, float(d.get("best_T", 1.0))


def softmax(z):
    z = np.asarray(z, float); z = z - z.max(); e = np.exp(z); return e / e.sum()


def onehot(idx, n):
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def sidecar_poolw(dataset, rung):
    key = cache.run_key(MODEL, dataset, "ID")
    suf = "" if rung == "ID" else f"__{rung}"
    p = VIZ / f"{key}__attn{suf}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return dict(zip(z["record_pos_all"].tolist(), z["pool_w"]))


# CONSOLIDATION: display names for the curated wMSP/orgad/sar tracks dumped by
# visualise_token_weights.py --dump-weights (npz key -> toggle label).
WMSP_DISPLAY = {"wMSP_pairwise": "wMSP-pairwise", "wMSP_shrink10": "wMSP-shrink10",
                "wMSP_content": "wMSP-content", "wMSP_segment": "wMSP-segment",
                "wMSP_smooth3": "wMSP-smooth3", "orgad": "orgad-hard",
                "orgad_broad": "orgad-broad", "sar": "SAR"}


def load_wmsp_weights(dataset):
    """Curated per-token wMSP/orgad/SAR weight tracks, keyed by record position. Same vectors this dataset's
    token_weights view renders (dumped from the SAME ctx). Returns {display_name: {record_pos: length-G vec}}
    or None if the dump has not been produced yet."""
    p = VIZ / f"{cache._slug(MODEL)}__{dataset}__ID__wmsp_weights.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    pos = z["record_pos"].tolist()
    return {WMSP_DISPLAY.get(k, k): dict(zip(pos, z[k])) for k in z.files if k != "record_pos"}


def load_mh_pooler(dataset, rung):
    """The persisted 4-head MultiMax-diagnostic pooler (mh_multimax_diag.py), or None. Only pubmed_qa/
    cnn_dailymail/xsum x {ID, DiffTask-long} exist (the diagnostic's scope)."""
    pk = PROBES / f"{cache._slug(MODEL)}__{dataset}__{rung}_mh4_s1__L15.pkl"
    if not pk.exists():
        return None
    m, _ = load_pooler(pk)
    return m


def mh_head_attention(model, x):
    """The K per-head attention vectors for one example, anchor row 0 dropped -> length G each."""
    X, mask, pos = ap.pad_batch([x], "cpu")
    model.eval()
    with torch.no_grad():
        _, a = model(X, mask, pos)              # (1, T, K)
    A = a[0].cpu().numpy()                       # (T, K)
    return [A[1:, h] for h in range(A.shape[1])]


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--dataset", required=True)
    ap_.add_argument("--ood-rung", default="DiffTask-long", help="the OOD pooler rung for the dissolution contrast")
    ap_.add_argument("--max-examples", type=int, default=60)
    ap_.add_argument("--per-bucket", type=int, default=15)
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--out", default=None)
    args = ap_.parse_args()
    D = args.dataset

    tok = vc.load_tokenizer(MODEL)
    special_ids = set(tok.all_special_ids)
    lf = label_of(D)   # 'correctness' for most; 'faithfulness' for expertqa etc. -- else y is all-NaN -> 0 examples
    loaded = ap.load_per_token(MODEL, D, LAYER, lf)
    if loaded is None:
        sys.exit(f"{D}: no per-token cache")
    states, split, y, _l, records = loaded
    n = len(records)

    # cached per-token weight sources
    pw_id = sidecar_poolw(D, "ID")
    pw_ood = sidecar_poolw(D, args.ood_rung)
    # CONSOLIDATED: curated wMSP/orgad/SAR tracks (from token_weights --dump-weights) + the 4-head diagnostic pooler
    wmsp = load_wmsp_weights(D)                       # {display: {pos: vec_G}} or None
    mh_id = load_mh_pooler(D, "ID")                   # 4-head pooler (pubmed/cnn/xsum only) or None
    # armA pooler for MultiMax (ID rung)
    armA_pk = PROBES / f"{cache._slug(MODEL)}__{D}__ID__attnpool_ID_s1__L15.pkl"
    W_armA = None
    if armA_pk.exists():
        mA, _ = load_pooler(armA_pk)
        W_armA = mA.head.weight.detach().numpy().ravel()

    # per-method cached uncertainty scores (SAPLMA/linear/ptrue) for the header + stratified sampling
    test_positions = list(range(n))
    methods = vc.method_scores(ROOT / "cache", cache.run_key(MODEL, D, "ID"), test_positions)
    primary = "saplma" if "saplma" in methods else (next(iter(methods), None))

    # ---- stratified example selection (fixed seed): 4 buckets correct/incorrect x confident/uncertain ----
    # CANDIDATE POOL = the examples the pooler was ACTUALLY scored on (the ID sidecar's record_pos = the eval
    # TEST rows). This is REQUIRED: pool_w only exists for those rows, so sampling from the full record set would
    # give train rows with no attnpool track (the dissolution contrast would be blank). Bucket by the floor
    # (msp_min) uncertainty percentile -- always available from logprobs, no dependency on cached probe scores.
    rng = np.random.RandomState(args.seed)
    if pw_id is not None:
        cand_pool = [int(i) for i in pw_id if int(i) < n and np.isfinite(y[int(i)])]
        selector_note = ("stratified 4-way (correct/incorrect × confident/uncertain) by the msp_min-floor "
                         "percentile, fixed seed, over the POOLER TEST SET (sidecar record_pos)")
    else:
        from xl_rungs import eval_split
        _, te = eval_split(np.asarray(split))
        cand_pool = [int(i) for i in te if np.isfinite(y[int(i)])]
        selector_note = ("stratified by msp_min-floor percentile over eval_split TEST rows (no ID sidecar; "
                         "attnpool tracks will be absent)")
    fu = {i: float(msp.msp_uncertainty(records[i]["token_logprobs"], "min")) for i in cand_pool}
    order = sorted(cand_pool, key=lambda i: fu[i])
    pct = {i: (r / max(1, len(order) - 1)) for r, i in enumerate(order)}
    buckets = {"correct+confident": [], "incorrect+confident": [], "correct+uncertain": [], "incorrect+uncertain": []}
    for i in cand_pool:
        correct = float(y[i]) >= 0.5
        uncertain = pct[i] >= 0.5
        buckets[("correct" if correct else "incorrect") + ("+uncertain" if uncertain else "+confident")].append(i)
    shown = []
    for b, idxs in buckets.items():
        rng.shuffle(idxs)
        shown += idxs[:args.per_bucket]
    shown = sorted(set(shown))[:args.max_examples]

    # ---- build per-example signals ----
    align_skips = []
    examples_html = []
    signal_union = []
    for i in shown:
        r = records[i]
        g = len(r["gen_token_ids"])
        lp = list(r["token_logprobs"])
        if len(lp) != g:
            align_skips.append(f"ex{i}: logprobs {len(lp)} != gen {g} (whole example skipped)")
            continue
        pieces = vc.token_pieces(tok, r["gen_token_ids"])
        x = states[i]                                  # (G+1, d)
        sig = {}
        meta = []

        def add(name, w_g):
            """add a length-G weight track; assert alignment, else skip THIS track for THIS example."""
            if len(w_g) != g:
                align_skips.append(f"ex{i}/{name}: len {len(w_g)} != G {g}")
                return
            sig[name] = (list(map(float, w_g)), vc.minmax(w_g))

        # model-side
        add("surprisal", [-v for v in lp])
        add("msp_min(argmin)", onehot(int(np.argmin(lp)), g))
        sig["perplexity/uniform"] = ([1.0 / g] * g, [0.45] * g)   # flat mid-colour (minmax of a constant = 0s)
        # probe-side (cached)
        if pw_id is not None and i in pw_id:
            w = list(pw_id[i])
            if len(w) == g + 1:
                add("attnpool_ID", w[1:])
        if pw_ood is not None and i in pw_ood:
            w = list(pw_ood[i])
            if len(w) == g + 1:
                add(f"attnpool_{args.ood_rung}", w[1:])
        # MultiMax pick
        if W_armA is not None:
            s = (x @ W_armA)                            # (G+1,)
            if len(s) == g + 1:
                add("MultiMax(pick)", onehot(int(np.argmax(s[1:])), g))
        # priors
        for pname, label in [("content_mass", "content_mass"), ("nll", "NLL_prior")]:
            try:
                pl, _ = build_prior(pname, [r], [x], tok, special_ids, datasets=[D])
                w = list(pl[0])
                if len(w) == g + 1:
                    add(label, w[1:])
            except Exception as e:
                align_skips.append(f"ex{i}/{label}: {type(e).__name__}")
        # soft-Orgad (only where the cache exists)
        if D in ORGAD_DATASETS:
            try:
                pl, _ = build_prior("orgad", [r], [x], tok, special_ids, datasets=[D])
                w = list(pl[0])
                if len(w) == g + 1:
                    add("soft_orgad", w[1:])
            except (OrgadCoverageError, Exception):
                pass
        # CONSOLIDATED wMSP family + orgad-hard + SAR (curated, already length-G, dumped from token_weights)
        if wmsp is not None:
            for name, byid in wmsp.items():
                if i in byid and byid[i] is not None:
                    add(name, list(byid[i]))
        # 4-head diagnostic pooler: the per-head attention (does each head attend to a different token?)
        if mh_id is not None:
            try:
                for h, vec in enumerate(mh_head_attention(mh_id, x)):
                    add(f"mh_head{h + 1}", vec)
            except Exception as e:
                align_skips.append(f"ex{i}/mh_head: {type(e).__name__}")

        # per-token hover metadata: logprob + token class
        for j in range(g):
            cls = classify_id(int(r["gen_token_ids"][j]), tok, special_ids)
            meta.append(f"logprob {lp[j]:+.2f} · {cls}")

        for name in sig:
            if name not in signal_union:
                signal_union.append(name)
        examples_html.append(vc.render_example(r, i, methods, sig, lf, pieces, token_meta=meta))

    # ---- absent tracks (explicit, with reason) ----
    absent = {"armD (annealed prior)": "the armD pooler was not persisted; would require a retrain"}
    if wmsp is None:
        absent["wMSP family / orgad-hard / SAR"] = ("weight dump not yet produced for this dataset — run "
                                                    "visualise_token_weights.py --dump-weights")
    else:
        absent["wMSP full 9-variant set (Blondel/shrink2/special_punct/entropy_hinge)"] = (
            "the curated 5 are shown here; the full nine live in the standalone visualise_token_weights.py view")
    if mh_id is None:
        absent["multi-head heads 1-4"] = ("the 4-head diagnostic pooler is only trained for "
                                          "pubmed_qa/cnn_dailymail/xsum (mh_multimax_diag.py scope)")
    if D not in ORGAD_DATASETS:
        absent["soft_orgad"] = f"no cache/orgad_llm broad cache for {D} (only pubmed_qa/med_quad/expertqa)"

    # ---- header / subtitle, incl. the verbatim Orgad prompt when the track is present ----
    sub = (f"ALL-METHODS token overlay. Selection: {selector_note} → {len(examples_html)} examples. "
           f"OOD pooler rung = {args.ood_rung}. Hypothesis-generation only (STANDING RULE: any claim needs a "
           f"full-population statistic first). Row-0 (prompt anchor) dropped from every G+1 track; {len(align_skips)} "
           f"per-track alignment skips (see console).")
    docs = dict(METHOD_DOCS)
    if D in ORGAD_DATASETS:
        prompt = orgad_llm.LONGFORM_QA_PROMPT if orgad_llm.is_longform_qa(D) else orgad_llm.SUMMARY_PROMPT
        docs["soft_orgad"] = (METHOD_DOCS["soft_orgad"] +
                              "<br><b>Verbatim prompt that produced the selection:</b>"
                              f"<pre style='white-space:pre-wrap;font-size:11px'>{prompt}</pre>")

    key = cache.run_key(MODEL, D, "ID")
    html = vc.render_html(key, records, methods, lf, signal_union, "\n".join(examples_html),
                          subtitle=sub, method_docs=docs, absent=absent)
    out = Path(args.out) if args.out else OUTDIR / f"ALLMETHODS_{D}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[allmethods] {D}: {len(examples_html)} examples, {len(signal_union)} tracks "
          f"({', '.join(signal_union)}), {len(absent)} absent, {len(align_skips)} align-skips")
    for s in align_skips[:20]:
        print("   align-skip:", s)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
