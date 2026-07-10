"""Per-example visualiser for a cached run (Phase 1 of the post-3-July plan).

WHAT THIS IS (in plain words)
-----------------------------
Lihu asked for an HTML viewer that shows, example by example, where a signal lands
on the generated tokens, so we can *see* why a probe succeeds or fails instead of
guessing. This script builds that viewer from data we already have on disk -- no GPU.

It reads one run's cached records (`cache/records/<key>.jsonl`) and the per-method
uncertainty scores (`cache/scores/<key>__<method>.npz`), then writes a single, self-
contained HTML file where each generated token is coloured by a chosen per-token
signal. For each example it also shows the question, the generation, the gold answer,
the graded correctness label, and every method's predicted uncertainty.

INCREMENT 1 (this file, CPU-only): the only per-token signal available is the token
**surprisal** (-logprob), which is exactly the MSP signal and lives in the record
already (`token_logprobs`). No model forward pass is needed.

INCREMENT 2 (next, needs one GPU job): add an **attention** signal (the learned
AttnPool weights and/or the model's own self-attention). Those are not cached, so a
small GPU step will dump a per-token sidecar that this renderer reads. The colour
toggle and the per-token data layout below are already built to accept extra signals,
so wiring attention in later is a small change, not a rewrite.

USAGE
-----
    python scripts/tools/visualise_attention.py --dataset sciq --ood ID \
        --model meta-llama/Meta-Llama-3.1-8B

Writes results/viz/<key>.html (override with --out). Open it in the browser / VS Code.

ALIGNMENT (the classic bug this guards against)
-----------------------------------------------
`gen_token_ids[k]` is the k-th generated token and `token_logprobs[k]` is its logprob,
so they are the same length G and line up one-to-one. We assert that. (The per-token
attention rows in Increment 2 are length G+1 -- an extra row 0 for the last prompt
token -- so that alignment gets its own assert when we add it.)
"""
import argparse
import html
import sys
from pathlib import Path

import numpy as np

# Make `import luq` work when this file is run directly from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402

# The generic page-building machinery (HTML, toggle JS, "interesting" sort, method scores,
# tokenizer) lives in viz_common so the attention family and the weighted-MSP family share
# exactly the same renderer. This script only adds the attention-specific signals below.
from viz_common import (  # noqa: E402
    load_tokenizer, token_pieces, method_scores, minmax,
    interestingness, render_example, render_html,
)


# --------------------------------------------------------------------------------------
# Per-token signal values (Increment 1: surprisal only)
# --------------------------------------------------------------------------------------

def load_viz_sidecar(cache_dir, key):
    """Load the attention sidecar written by dump_viz_attention.py, if it exists. Returns
    per-record-position lookups for each attention signal, or None if Increment 2 has not
    been run yet (in which case the renderer just shows the surprisal signal)."""
    path = Path(cache_dir) / "viz" / f"{key}__attn.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    pool = dict(zip(z["record_pos_all"].tolist(), z["pool_w"]))
    self_pos = z["record_pos_self"].tolist()
    return {
        "pool_w": pool,  # pos -> (G+1,) weights (row 0 = last-prompt token)
        "self_lastq": dict(zip(self_pos, z["self_lastq_block"])),
        "self_meanq": dict(zip(self_pos, z["self_meanq_block"])),
        "self_lastq_last": dict(zip(self_pos, z["self_lastq_last"])),
        "block": int(z["block"]), "layer": int(z["layer"]),
    }


def per_token_signals(record, record_pos, sidecar):
    """Return a dict {signal_name: (values, norm)} of the per-token signals AVAILABLE for
    this example. `values` is the raw per-token number (shown on hover); `norm` is that
    value scaled to [0, 1] *within this example* so the standout tokens show regardless of
    the example's overall scale. Deeper colour = higher `norm`.

    Always present: `surprisal` = -logprob (higher = the model found this token less
    likely = the MSP signal). When the Increment-2 sidecar exists, we add the attention
    signals for the examples it covers:
      attnpool         our learned pooler's weights (row 0 = last-prompt token is dropped
                       so the G values line up with the G generated tokens)
      selfattn_lastq   raw model self-attention from the final position onto each token
      selfattn_meanq   raw self-attention averaged over the response's own query rows
    """
    g = len(record["gen_token_ids"])
    surprisal = [-lp for lp in record["token_logprobs"]]
    signals = {"surprisal": (surprisal, minmax(surprisal))}
    if not sidecar:
        return signals

    if record_pos in sidecar["pool_w"]:
        w = list(sidecar["pool_w"][record_pos])
        # pool weights span the SAPLMA window (G+1): drop row 0 (last-prompt token) to
        # align with the G generated tokens. Guard the length so a mismatch is skipped,
        # not silently mis-coloured.
        if len(w) == g + 1:
            aligned = w[1:]
            signals["attnpool"] = (aligned, minmax(aligned))
    for name, store in (("selfattn_lastq", "self_lastq"),
                        ("selfattn_meanq", "self_meanq")):
        if record_pos in sidecar[store]:
            v = list(sidecar[store][record_pos])
            if len(v) == g:
                signals[name] = (v, minmax(v))
    return signals


# --------------------------------------------------------------------------------------
# The "interesting" sort, HTML rendering, method scores and tokenizer helpers all live in
# viz_common now (imported above), so the attention family and the weighted-MSP family
# share exactly the same renderer. This file only adds the attention signals above.
# --------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag (e.g. 'pdnew'); must match the run's cache.")
    ap.add_argument("--label-field", default="correctness",
                    help="which correctness field to show / sort by.")
    ap.add_argument("--primary-method", default="saplma",
                    help="method used to rank 'interesting' examples; falls back to any "
                         "available method, then to unsorted if none are scored.")
    ap.add_argument("--max-examples", type=int, default=60,
                    help="cap the number of examples rendered (keeps the HTML light).")
    ap.add_argument("--split", default="test", choices=["test", "train", "all"],
                    help="which split to show. 'test' has method scores; 'train' does not.")
    ap.add_argument("--sort", default="interesting", choices=["interesting", "index"],
                    help="'interesting' surfaces confident-wrong / uncertain-right first.")
    ap.add_argument("--out", default="",
                    help="output HTML path (default: results/viz/<key>.html).")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    print(f"loading run: {key}")
    records = cache.load_records(cfg.cache_dir, key)

    # --- the alignment guard the plan insists on: one logprob per generated token ---
    for i, r in enumerate(records):
        assert len(r["gen_token_ids"]) == len(r["token_logprobs"]), (
            f"record {i}: {len(r['gen_token_ids'])} gen tokens vs "
            f"{len(r['token_logprobs'])} logprobs -- tokenisation/logprob mismatch")

    test_positions = [i for i, r in enumerate(records) if r["split"] == "test"]
    methods = method_scores(cfg.cache_dir, key, test_positions)
    print(f"  methods with scores: {list(methods.keys()) or '(none)'}")

    sidecar = load_viz_sidecar(cfg.cache_dir, key)
    if sidecar:
        print(f"  attention sidecar found: pool_w for {len(sidecar['pool_w'])} test, "
              f"self-attn for {len(sidecar['self_lastq'])} (block {sidecar['block']})")
    else:
        print("  no attention sidecar (run dump_viz_attention.py for the attn signals); "
              "showing surprisal only")

    # Decode tokens once (a local cache) so rendering is cheap.
    tok = load_tokenizer(cfg.model_name)
    pieces_by_pos = {i: token_pieces(tok, r["gen_token_ids"]) for i, r in enumerate(records)}

    # Which records to show.
    if args.split == "all":
        candidates = list(range(len(records)))
    else:
        candidates = [i for i, r in enumerate(records) if r["split"] == args.split]

    # Pick the primary method for sorting: the requested one if scored, else any scored.
    primary = args.primary_method if args.primary_method in methods else (
        next(iter(methods), None))
    if args.sort == "interesting" and primary is not None:
        candidates.sort(
            key=lambda i: interestingness(i, records[i].get(args.label_field, 0.0),
                                          methods, primary),
            reverse=True)
        print(f"  sorted by 'interesting' using primary method: {primary}")
    else:
        print("  showing in record order")

    shown = candidates[:args.max_examples]

    # Compute each shown example's available signals, and the UNION of signal names across
    # them (the toggle buttons). Examples missing a signal simply keep their default colour
    # when that toggle is picked -- the JS treats an absent value as 0.
    per_example = {i: per_token_signals(records[i], i, sidecar) for i in shown}
    signal_names = []
    for i in shown:
        for name in per_example[i]:
            if name not in signal_names:
                signal_names.append(name)
    if not signal_names:
        signal_names = ["surprisal"]

    ex_html = []
    for i in shown:
        ex_html.append(render_example(records[i], i, methods, per_example[i],
                                      args.label_field, pieces_by_pos[i]))

    out = Path(args.out) if args.out else (cfg.results_dir / "viz" / f"{key}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(key, records, methods, args.label_field,
                               signal_names, "\n".join(ex_html)))
    print(f"wrote {out}  ({len(shown)} of {len(candidates)} {args.split} examples)")


if __name__ == "__main__":
    main()
