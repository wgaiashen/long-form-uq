"""ProbeDriftLong (W3 / H4): train EXCLUSIVELY on long-form datasets and evaluate on long-form, plus a
long->short transfer cell. the steer: the current ProbeDrift ID/LOO settings let probes lean on easy
short-form training data (long_form_loo §J showed SAPLMA's long-form OOD collapses to the floor once
short-form is removed). This driver removes that crutch by construction and asks: in the realistic
long-only setting, does weighted-MSP / shrink overtake the probes, measured against the HONEST fair floor
(max of msp_sum, perplexity, msp_min)?

LONG UNIVERSE: pubmed_qa, xsum, cnn_dailymail, med_quad, samsum, expertqa, asqa.
  UPDATE 2026-07-22 (author's decision): ExpertQA and ASQA are ORDINARY TRAINING SOURCES, so the long pool
  is MIXED-LABEL by default (ExpertQA = claim-precision, the rest = reference-agreement). Cells are tagged
  `different_label_projection`. `--label-homogeneous` drops ExpertQA from the sources to reproduce the
  pre-2026-07-22 baseline as the control.
  FINE families: correctness_qa = {pubmed_qa, med_quad, asqa};  factuality = {expertqa, factscore};
  summ = {xsum, cnn_dailymail, samsum}.  (factuality split from correctness_qa on 2026-07-27.)

RUNGS (long eval X): ID | SameTask-long (other long sets in X's family) | LOO-long (all other long sets)
  | DiffTask-long (opposite long family) | 1ds-Diff-long (one opposite-family long set).
  For a SHORT eval (sciq/trivia): the single "Long->Short" cell (train on the full long pool).

METHODS (the KEEP set): fair floor (msp_sum/perplexity/msp_min -> max), uniform + attention poolers,
  mean-pool SAPLMA (MLP), weighted-MSP {normalised, shrink@2, shrink@10, Blondel}. Paired bootstrap:
  best-wMSP vs fair-floor, best-wMSP vs best-pooler, attention vs fair-floor.

    python scripts/checks/probedriftlong.py --seeds 1,2,3
    python scripts/checks/probedriftlong.py --evals pubmed_qa,xsum --seeds 1   # smoke
"""
import argparse, csv as _csv, os, sys, pickle   # os: atomic replace in _flush_rows
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "checks"))
import torch  # noqa: E402
from luq import cache, msp, results, weighted_msp, probe  # noqa: E402
from luq.config import Config  # noqa: E402  (B.5: resolving a dataset's pooled-feature cache root)
from luq.features import sar  # noqa: E402  (shared sentence splitter)
from transformers import AutoTokenizer  # noqa: E402
from luq.weighting import shrink_to_uniform  # noqa: E402
from aggregation_table import load_per_token, attn_unc, paired_bootstrap, conf_meanpool, prr_from_conf  # noqa: E402
from attn_pool import train_attn, select_temperature, pad_batch, regime_tag, PROMPT_REGIME  # noqa: E402
from xl_rungs import build_rows, eval_split, label_of, different_label_projection  # noqa: E402

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B"
# MODEL is a module global (not a main() local) because the save helpers (_save_pooler, _save_perex)
# read it for cache keys and filenames. --model reassigns it at the top of main(), so every use site
# — run_key, probe/viz/perex filenames, the tokenizer, load_per_token, Config, the output CSV —
# follows the flag with no signature changes, and a no-arg run must stay byte-identical to before
# the port (the no-op control: re-run one Llama cell with no --model and diff against the published
# CSV). Every use is path/config construction; no logic branches on the model.
MODEL = DEFAULT_MODEL

# Re-export shim. The long-form grid is defined in `src/probe_drift_long/` and is re-exported
# here under its historical names, so the modules doing
# `import probedriftlong as pdl` and reaching for `pdl.LONG_SRC` / `pdl.cells_long` /
# `pdl.sampled_train_idx` keep working unchanged.
#
# Gate 1 (`scripts/checks/library_equivalence.py`) proves the move changed nothing: the taxonomy
# matches, the 42-cell grid is identical, and all 126 cell x seed row lists are BYTE-IDENTICAL.
# Re-run it after touching either side.
#
# Importing THIS module still costs a full torch + transformers import, because the method half
# below needs them. If all you want is the grid, `import probe_drift_long` instead — it is
# numpy-only. That is the whole reason the split exists.
from probe_drift_long import (LONG_DATASETS as LONG, LONG_SRC, SHORT_DATASETS as SHORT,   # noqa: E402
                              FINE_FAMILIES as FINE, XL_TOTAL, cells_long, rung_sources_long,
                              sampled_train_idx)
EVALS = LONG + SHORT
# wMSP KEEP variants: (col name, kwargs to weighted_msp_unc)  [all length_normalise=True]
WMSP = [("wmsp_norm", {"weight_mode": "normalised"}),
        # per-segment (sentence) wMSP -- FIRST run under ProbeDriftLong (design note 7; was standard-ladder
        # only). `_seg_ids` is a sentinel: the loop replaces it with this cell's sentence ids.
        ("wmsp_seg_flat", {"weight_mode": "normalised", "segment_ids": "_seg_ids"}),
        ("wmsp_seg_softmax", {"weight_mode": "normalised", "segment_ids": "_seg_ids",
                              "segment_mode": "softmax"}),
        # λ = 1.5, added 2026-08-19. SUPPLEMENTARY SENSITIVITY ONLY -- the project plan permits it
        # "only after the primary fixed-λ experiment completes, and it must not affect the primary
        # conclusion". The primary remains λ = 2 (`wmsp_shrink2`) against the λ = 0 control
        # (`wmsp_norm`). λ is NEVER selected on a replication population's test results -- reporting
        # whichever of 1.5 and 2 happens to score higher would be exactly the selection this panel
        # exists to avoid. Report the pair; do not pick a winner from it.
        ("wmsp_shrink1_5", {"weight_mode": "normalised", "reg": shrink_to_uniform, "reg_lambda": 1.5}),
        ("wmsp_shrink2", {"weight_mode": "normalised", "reg": shrink_to_uniform, "reg_lambda": 2.0}),
        ("wmsp_shrink10", {"weight_mode": "normalised", "reg": shrink_to_uniform, "reg_lambda": 10.0}),
        ("wmsp_blondel", {"weight_mode": "normalised", "loss": "blondel"}),
        # W2: Blondel loss on the KEEP shrink variants -> paired loss-only comparison vs wmsp_shrink2/10 above.
        ("wmsp_shrink2_blondel", {"weight_mode": "normalised", "reg": shrink_to_uniform,
                                  "reg_lambda": 2.0, "loss": "blondel"}),
        ("wmsp_shrink10_blondel", {"weight_mode": "normalised", "reg": shrink_to_uniform,
                                   "reg_lambda": 10.0, "loss": "blondel"})]
POOLERS = ["uniform", "attention"]
FLOORS = ["floor_sum", "floor_ppl", "floor_min"]
# B.5 — the baselines dropped for time and asked for again at the 31 July meeting. Recipe copied verbatim
# from ood_onegrid.py:48 (the only driver that ever ran them), so these stay faithful reproductions rather
# than re-derivations:  name -> (cached pooled feature, layer, standardize)
# uhead is deliberately ABSENT: implemented and verified against the authors, but never carried to the
# long-form grid, and dropped by decision on 2026-07-31 for time. Say so in the write-up.
# ptrue's layer is THE PROBED MIDDLE LAYER, not a constant: 15 is Llama's ceil(32/2)-1, and the
# feature cache stores only that plane (the rest are NaN), so pointing the Llama 15 at a Qwen cache
# aborts on the NaN plane. It is resolved from --layer in main() (same ceil(N/2)-1 rule -> 23 on
# Qwen); the default is 15, so every existing Llama invocation is untouched. lookback stays 0 by
# design (a single combined layer).
BASE_FEATS = {"ptrue": ("ptrue_accurate", 15, True), "lookback": ("lookback", 0, False)}
METHODS = FLOORS + ["fair_floor", "saplma"] + POOLERS + [w[0] for w in WMSP]


CSV_FIELDS = ["rung", "eval", "train", "method", "prr_mean", "prr_std", "n_seeds",
              "different_label_projection", "eval_med_len", "train_med_len", "train_max_len",
              "ci_lo", "ci_hi", "boot_p", "significant", "git_sha", "cluster", "env_hash",
              # WHICH TEST POPULATION produced this row (added 2026-08-08). "legacy" = drop
              # unlabelled rows then carve 30%; "all-rows" = carve 30% of ALL rows then score
              # whichever carry labels. They differ on expertqa + factscore ONLY, but a CSV that
              # does not say which rule it used is indistinguishable from one that used the other
              # -- and two agents are now writing results into this repo. Rows written before this
              # column existed are `legacy` by definition; they are NOT back-filled, because an
              # inferred stamp and a recorded one must not look the same.
              "carve"]


def _flush_rows(out, rows, prov):
    """Write every row accumulated SO FAR, atomically.

    WHY THIS IS CALLED PER CELL, NOT ONCE AT THE END (added 2026-08-03).
    This driver used to accumulate `out_rows` in memory for the whole run and write the CSV only after
    the final cell. A 16-hour run killed at hour 15 -- by the walltime, an OOM, or a node problem --
    lost EVERYTHING, with nothing on disk to show which cells had already succeeded. That is not
    hypothetical: a job was SIGTERM'd on cx3-14-9 earlier the same day after 36 seconds, and the
    ladders run for 8-16 hours.

    Written to a temp file then os.replace()d, so a crash DURING the write cannot leave a truncated CSV
    that later reads as a short-but-valid grid -- a half-written table is worse than none, because it
    looks complete.
    """
    for r in rows:                                     # provenance on EVERY row (method + VERDICT rows)
        r.update(prov)
    tmp = Path(str(out) + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, out)



def _provenance():
    """Per-row provenance: git_sha + cluster + env_hash. FAILS LOUD if the TRACKED working tree is dirty --
    a SHA stamped from a modified tree asserts a reproducibility that does not hold, so refuse to run rather
    than stamp a lie. Untracked scratch files are ignored (they do not change the committed code the SHA
    points at). Called at the top of main() so it aborts before the ~20-min pool load."""
    import subprocess, socket, hashlib

    def _git(*a):
        return subprocess.run(["git", *a], cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    sha = _git("rev-parse", "HEAD")
    if not sha:
        raise SystemExit("provenance: could not read git HEAD (not a repo?) -- refusing to run unversioned")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit("provenance: TRACKED working tree is DIRTY -- refusing to stamp a git_sha that does "
                         f"not reproduce. Commit or stash first, then resubmit.\n{dirty}")
    root = str(ROOT)
    cluster = "DoC" if root.startswith("/vol/gpudata") else ("RCS" if "/rds/" in root else socket.gethostname())
    try:
        from importlib.metadata import distributions
        pkgs = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in distributions()
                      if dist.metadata.get('Name'))
        env_hash = hashlib.sha256("\n".join(pkgs).encode()).hexdigest()[:12]
    except Exception as e:                              # never let env-hashing crash the run
        env_hash = f"unknown:{type(e).__name__}"
    # `carve` is read from xl_rungs (which resolves LUQ_CARVE) rather than re-read from the
    # environment here, so the CSV records what the run ACTUALLY used and cannot disagree with it.
    from xl_rungs import CARVE
    return {"git_sha": sha, "cluster": cluster, "env_hash": env_hash, "carve": CARVE}


def _save_pooler(pooler, best_T, rung, X, sd, layer, states, te_idx, test_rows, PT, device):
    """Persist the trained attention pooler (armA) + its test-set sidecar, byte-format-identical to
    scripts/tools/dump_ood_attention.py so RCS's post-hoc pass + G1 cross-check consume them unchanged.
    The pooler is trained identically here (same states/tr_idx/seed/best_T), so this adds only the cheap
    forward pass for pool_w -- no retrain. Sidecar filename carries NO seed (seed-1 convention); pkl carries
    s<seed>."""
    base_rung = rung.replace("-long", "")
    if any(c in rung for c in ">/\\"):                  # e.g. "Long->Short" -> unsafe filename; skip loudly
        print(f"  [save-pooler] skip {X} {rung!r}: unsafe rung name for a filename", flush=True)
        return
    ladder_family = "ID" if base_rung == "ID" else ("LONG" if rung.endswith("-long") else "STANDARD")
    probes = ROOT / "cache" / "probes"; probes.mkdir(parents=True, exist_ok=True)
    viz = ROOT / "cache" / "viz"; viz.mkdir(parents=True, exist_ok=True)
    pooler.eval()
    pool_w = []
    with torch.no_grad():
        for k in te_idx:
            Xk, mask, pos = pad_batch([states[k]], device)
            _, a = pooler(Xk, mask, pos)
            pool_w.append(a[0, : states[k].shape[0]].float().cpu().numpy())
    record_pos_all = np.array([int(PT[X][4][i]) for _d, i in test_rows])   # PT[.][4] = orig record positions
    key = cache.run_key(MODEL, X, "ID")
    suffix = "" if base_rung == "ID" else f"__{rung}"
    np.savez_compressed(viz / f"{key}__attn{suffix}.npz", record_pos_all=record_pos_all,
                        pool_w=np.array(pool_w, dtype=object), rung=rung, base_rung=base_rung,
                        ladder_family=ladder_family, seed=sd, layer=layer, best_T=float(best_T),
                        pool_config="post-TaskA-widened", n_train=len(states) - len(te_idx))
    pk = probes / f"{cache._slug(MODEL)}__{X}__ID__attnpool_{rung}_s{sd}__L{layer}.pkl"
    with open(pk, "wb") as f:
        pickle.dump({"model": pooler, "best_T": float(best_T), "rung": rung, "base_rung": base_rung,
                     "ladder_family": ladder_family, "seed": sd, "eval": X, "layer": layer,
                     "pool_config": "post-TaskA-widened"}, f)
    print(f"  [save-pooler] {X} {rung} s{sd} -> {pk.name}  (+sidecar {key}__attn{suffix}.npz)", flush=True)


PEREX_TOL = 1e-6


def _save_perex(perex_dir, X, rung, srcs, seeds_done, yte, unc_acc, stats, fair_name, layer, prov):
    """Dump this cell's PER-EXAMPLE uncertainty vectors so significance for ANY method pair becomes a
    post-hoc read instead of a re-run.

    Why this exists: `pdl_master` carries only a per-cell PRR, so every "is this difference significant?"
    question needed the whole ladder re-run. The vectors were always in memory here -- they were simply
    never written down.

    Layout, one .npz per cell:
      y            (n_te,)            test labels (asserted seed-invariant by the caller)
      seeds        (n_seeds,)         the seeds actually completed, in order
      unc__<m>     (n_seeds, n_te)    per-seed per-example uncertainty for method <m>
      prr__<m>     (n_seeds,)         per-seed PRR, recomputed here from the stored vectors
      prr_mean__<m> scalar            UNROUNDED mean over seeds (the CSV rounds to 4dp)
      + eval / rung / train / fair_floor_alias / layer / git_sha as metadata

    float64 throughout: the gate is 1e-6 and float32 storage would round the vectors enough to break it.

    SELF-GATE: PRR recomputed from the persisted vectors must reproduce the in-memory per-seed PRR to
    <1e-6, or this raises. A sidecar that silently disagrees with its own CSV is worse than no sidecar --
    it would look authoritative while quietly answering a different question.
    """
    methods = [m for m in unc_acc if unc_acc[m]]
    out = {"y": np.asarray(yte, np.float64), "seeds": np.asarray(seeds_done, np.int64)}
    meta = {"eval": X, "rung": rung, "train": srcs, "fair_floor_alias": fair_name,
            "layer": str(layer), "git_sha": prov.get("git_sha", "")}
    for m in methods:
        V = np.stack([np.asarray(v, np.float64) for v in unc_acc[m]])      # (n_seeds, n_te)
        if V.shape[1] != len(yte):
            raise SystemExit(f"FATAL [{rung}/{X}] sidecar: method {m} has {V.shape[1]} values for "
                             f"{len(yte)} test labels -- refusing to write a misaligned sidecar.")
        out[f"unc__{m}"] = V
        out[f"prr__{m}"] = np.array([results.prr(yte, V[i]) for i in range(V.shape[0])], np.float64)
        out[f"prr_mean__{m}"] = np.float64(np.mean(out[f"prr__{m}"]))
    # fair_floor is an ALIAS of whichever floor won; store it explicitly so a reader never has to re-derive
    # which floor was primary (re-deriving it is exactly how a floor mix-up would creep back in).
    if fair_name in methods:
        out["unc__fair_floor"] = out[f"unc__{fair_name}"]
        out["prr__fair_floor"] = out[f"prr__{fair_name}"]
        out["prr_mean__fair_floor"] = out[f"prr_mean__{fair_name}"]
    # THE GATE -- recomputed vs the in-memory stats that produced the CSV row
    worst, worst_m = 0.0, None
    for m in methods:
        d = abs(float(out[f"prr_mean__{m}"]) - float(stats[m][0]))
        if d > worst:
            worst, worst_m = d, m
    if worst >= PEREX_TOL:
        raise SystemExit(f"FATAL [{rung}/{X}] sidecar GATE FAILED: method {worst_m} PRR recomputed from the "
                         f"stored vectors differs from the in-memory mean by {worst:.3g} (tol {PEREX_TOL:g}). "
                         "The sidecar does not reproduce its own CSV row -- aborting rather than persisting it.")
    p = Path(perex_dir) / f"{X}__{rung}__{cache._slug(MODEL)}.npz"
    # NB the temp name must itself end in .npz -- np.savez_compressed APPENDS ".npz" to any path that does
    # not, which would write "<name>.npz.tmp.npz" and leave the rename below pointing at nothing.
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out, **{f"meta__{k}": np.array(v) for k, v in meta.items()})
    tmp.replace(p)                                     # atomic: never leave a half-written sidecar behind
    print(f"  [perex] {p.name}  {len(methods)} methods x {len(seeds_done)} seeds x {len(yte)} rows  "
          f"(gate max|Δ|={worst:.2g})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--evals", default=",".join(EVALS))
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model id whose caches to score (default: the Llama keystone, unchanged). "
                         "Every use is path/config construction (cache keys, probe/viz/perex filenames, "
                         "tokenizer, output CSV slug) — no logic branches on it. The per-token cache for "
                         "this model at --layer must already exist (01h_pertoken); pair Qwen/Qwen2.5-14B "
                         "with --layer 23 (the fixed ceil(N/2)-1 rule; do NOT rely on the layer default).")
    ap.add_argument("--include-expertqa", action="store_true",
                    help="DEPRECATED / no-op as of 2026-07-22: ExpertQA is now an ordinary training source by "
                         "default, so the mixed-label 'universal' pool IS the default. Kept so existing job "
                         "scripts keep working.")
    ap.add_argument("--label-homogeneous", action="store_true",
                    help="drop ExpertQA (faithfulness) from the TRAINING sources so every training label is "
                         "correctness. This reproduces the pre-2026-07-22 committed PART VII baseline and is "
                         "the control for 'does mixing label semantics help or hurt?'. ExpertQA remains an EVAL.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-wmsp", action="store_true",
                    help="skip the 10 wMSP KEEP variants (the training bottleneck). Poolers + floors + saplma "
                         "still run -- enough for §B.3 / P2a / Task D. Use when only the pooler numbers are "
                         "needed; wMSP (§C.4) is re-run separately.")
    ap.add_argument("--wmsp-only", default=None,
                    help="comma-separated wMSP variant names to KEEP (e.g. wmsp_norm,wmsp_shrink2,wmsp_shrink10). "
                         "Any WMSP variant not listed is EXCLUDED and its column is left ABSENT (never zero). "
                         "Each variant is a SEPARATE MLP fit, so this genuinely saves compute. Default None = all "
                         "committed variants (RCS default behaviour unchanged). Fails loud on an unknown name.")
    ap.add_argument("--wmsp-extra-lambdas", default=None,
                    help="comma-separated EXTRA shrink lambdas to APPEND as wmsp_lam<L> variants "
                         "(e.g. '1,1.5'). Additive and opt-in: the default None leaves the committed "
                         "WMSP list byte-identical, so no existing invocation changes. Combine with "
                         "--skip-wmsp to run ONLY the extra lambdas (the cheapest way to fill a "
                         "missing lambda cell). Added 2026-08-13 for PR4's short-form dev selection, "
                         "which needs lambda 1 and 1.5 on a --train-spec cell; every other lambda in "
                         "the registry already exists. It does NOT create a new registry variant -- "
                         "output lands in the --out file, never in a canonical master.")
    ap.add_argument("--sources", default=None,
                    help="comma-separated subset of LONG_SRC to use as the TRAINING-SOURCE POOL. Default "
                         "None = all 8. --evals restricts TARGETS ONLY; without this flag the rungs are "
                         "still built from the full 8-dataset pool, so a 'reduced panel' run would quietly "
                         "train on datasets it claims to have dropped. It also makes the restriction "
                         "EXPLICIT rather than an accident of which pertok caches happen to exist -- the "
                         "loader otherwise prints 'no pertok cache -> skip' and shrinks the pool silently. "
                         "Order is taken from LONG_SRC and never re-sorted, because 1ds-Diff-long reads "
                         "diff[:1]. Added 2026-08-15 for the W-Models six-dataset panel (the multi-model panel registration).")
    ap.add_argument("--rungs", default=None,
                    help="comma-separated BASE rung names to KEEP: ID,SameTask,DiffTask,LOO,1ds-Diff (they map "
                         "to the -long ladder names). Default None = every rung cells_long emits (RCS default "
                         "unchanged). Pass 'ID,LOO,DiffTask' for the 3 MASTER-GRID rungs only.")
    ap.add_argument("--save-pooler", action="store_true",
                    help="persist the seed-1 attention pooler (armA) trained in every cell: pkl "
                         "(cache/probes/...attnpool_<rung>_s1__L15.pkl) + sidecar (cache/viz/...__attn[__<rung>].npz "
                         "with pool_w + record_pos_all), byte-format-identical to dump_ood_attention so RCS's "
                         "post-hoc pass + G1 cross-check read them directly. The pooler is already trained, so "
                         "this is ~zero extra compute. Default off.")
    ap.add_argument("--baselines", default=None,
                    help="comma-separated supervised baselines to ADD as extra rows: ptrue,lookback (B.5). "
                         "Off by default so existing runs stay byte-identical and do not pay to read the "
                         "pooled feature caches. A dataset without the cached feature makes its cells "
                         "UNCOMPUTABLE, and those cells are skipped LOUDLY and left BLANK -- never zero. "
                         "As of 2026-08-01 the features exist for sciq/trivia/pubmed/xsum (v1) and "
                         "asqa/expertqa/factscore (DoC); med_quad, samsum and cnn_dailymail are pending "
                         "their regeneration, so most OOD cells stay blank until then.")
    ap.add_argument("--skip-poolers", action="store_true",
                    help="skip the uniform + attention poolers (select_temperature is 5 temperatures x 40 "
                         "epochs, then 2 more fits -- the second bottleneck after wMSP). Floors + saplma still "
                         "run. Use with --skip-wmsp for the cheap floors-and-SAPLMA pass that the significance "
                         "re-score needs. Skipped methods are left ABSENT from the CSV, never zero.")
    ap.add_argument("--perex-dir", default=None,
                    help="ALSO dump a per-example sidecar per cell to <dir>/{eval}__{rung}__{slug}.npz: every "
                         "method's per-seed per-example uncertainty vector + the labels + the unrounded "
                         "prr_mean. Additive -- the CSV schema is untouched. This is what makes significance "
                         "for ANY method pair a post-hoc read instead of a re-run. Self-gated: PRR recomputed "
                         "from the stored vectors must reproduce the in-memory prr_mean to <1e-6 or the run "
                         "ABORTS (a sidecar that disagrees with its own CSV is worse than no sidecar).")
    ap.add_argument("--train-spec", default=None,
                    help="EXPLICIT training pool, 'ds:n,ds:n' (e.g. 'sciq:900,trivia_qa:900'). Replaces the "
                         "rung grid from cells_long with ONE cell per --evals target built from exactly this "
                         "spec, for the cross-length transfer experiment (the long grid otherwise has no way "
                         "to say 'half long, half short'). The named datasets are added to the cache-load set, "
                         "so short-form sources (sciq/trivia_qa, which live in SHORT_DATASETS not LONG_SRC) "
                         "load correctly. A source contributing 0 rows is already a hard failure in "
                         "build_rows, so a mis-typed name cannot silently shrink the pool. REQUIRES "
                         "--rung-name and --out.")
    ap.add_argument("--rung-name", default=None,
                    help="the rung label written to the CSV when --train-spec is used. Required with it: an "
                         "unlabelled custom cell would be indistinguishable from a canonical rung in the "
                         "results file.")
    args = ap.parse_args()
    # Reassign the module global so the save helpers (and every path/key below) follow the flag.
    # Default leaves it untouched, so a no-arg invocation is exactly the pre-port driver.
    global MODEL
    MODEL = args.model
    # ptrue's feature plane follows the probed layer (see BASE_FEATS note). --layer's default is 15,
    # so a no-arg Llama run resolves to the identical (15) and stays byte-identical.
    BASE_FEATS["ptrue"] = ("ptrue_accurate", args.layer, True)
    # ---- explicit training pool (cross-length transfer) ------------------------------------------
    # `spec = [(source, cap)]` is the currency cells_long produces and build_rows consumes; build_rows
    # does not care where it came from, so an explicit pool needs no library change. Parsed here so a
    # malformed spec fails before the (slow) cache load.
    train_spec = None
    if args.train_spec:
        if not args.rung_name:
            raise SystemExit("--train-spec requires --rung-name (an unlabelled custom cell is "
                             "indistinguishable from a canonical rung in the CSV)")
        if not args.out:
            raise SystemExit("--train-spec requires --out (the default filename is the canonical "
                             "grid's, and a custom pool must never be written there)")
        train_spec = []
        for part in args.train_spec.split(","):
            part = part.strip()
            if not part:
                continue
            if part.count(":") != 1:
                raise SystemExit(f"--train-spec: bad entry {part!r}; expected 'dataset:count'")
            d, n = part.split(":")
            d = d.strip()
            if not n.strip().isdigit() or int(n) <= 0:
                raise SystemExit(f"--train-spec: bad count in {part!r}; expected a positive integer")
            train_spec.append((d, int(n)))
        if not train_spec:
            raise SystemExit("--train-spec is empty")
        dups = [d for d, _ in train_spec if [x for x, _ in train_spec].count(d) > 1]
        if dups:
            raise SystemExit(f"--train-spec: duplicate source(s) {sorted(set(dups))}")
        total = sum(n for _, n in train_spec)
        print(f"TRAIN-SPEC: rung={args.rung_name!r} pool="
              + "+".join(f"{d}:{n}" for d, n in train_spec)
              + f" (requested total {total}; cells_long grid BYPASSED)", flush=True)
        if total != XL_TOTAL:
            print(f"  requested total {total} != the matched budget {XL_TOTAL} — intentional? "
                  f"the realised pool is recorded per row in the `train` column either way", flush=True)

    # Provenance runs AFTER the cheap argument validation above: `git status` on this tree takes a
    # while, and a mis-typed --train-spec should fail instantly rather than behind it. It still
    # aborts long before the ~20-min pool load, which is the guarantee that matters.
    prov = _provenance()   # aborts here if the tracked tree is dirty
    print(f"PROVENANCE: git_sha={prov['git_sha'][:12]} cluster={prov['cluster']} env_hash={prov['env_hash']}"
          + ("  [+save-pooler seed-1]" if args.save_pooler else ""), flush=True)

    want_rungs = set(s.strip() for s in args.rungs.split(",") if s.strip()) if args.rungs else None
    if want_rungs is not None and train_spec is not None:
        raise SystemExit("--rungs filters the cells_long grid, which --train-spec replaces; pass one or "
                         "the other")
    if want_rungs is not None:
        _valid = {"ID", "SameTask", "DiffTask", "LOO", "1ds-Diff", "Long->Short"}
        _bad = want_rungs - _valid
        if _bad:
            raise SystemExit(f"--rungs: unknown rung(s) {sorted(_bad)}; valid = {sorted(_valid)}")
        print(f"RUNGS filter: keeping base rungs {sorted(want_rungs)} (others skipped)", flush=True)
    # active method set: wMSP is the per-cell training bottleneck (10 variants); drop it when only the poolers
    # and floors are needed. best_w / the wmsp verdicts are guarded below when wMSP is off.
    active_wmsp = [] if args.skip_wmsp else WMSP
    if args.wmsp_only and not args.skip_wmsp:
        keep_names = [s.strip() for s in args.wmsp_only.split(",") if s.strip()]
        known = {w[0] for w in WMSP}
        unknown = [n for n in keep_names if n not in known]
        if unknown:
            raise SystemExit(f"--wmsp-only: unknown variant(s) {unknown}; valid = {sorted(known)}")
        excluded = [w[0] for w in WMSP if w[0] not in keep_names]
        active_wmsp = [w for w in WMSP if w[0] in keep_names]
        print(f"WMSP-ONLY: keeping {[w[0] for w in active_wmsp]} ; EXCLUDED {excluded} "
              "(their columns are left ABSENT, not zero)", flush=True)
    # EXTRA lambdas (opt-in, additive). Appended AFTER --wmsp-only so the two flags compose, and
    # allowed even under --skip-wmsp so "only the missing lambda" is expressible. Default None means
    # this block never runs and the committed variant list is untouched.
    if args.wmsp_extra_lambdas:
        have = {w[0] for w in active_wmsp}
        for tok_ in args.wmsp_extra_lambdas.split(","):
            tok_ = tok_.strip()
            if not tok_:
                continue
            try:
                lam_ = float(tok_)
            except ValueError:
                raise SystemExit(f"--wmsp-extra-lambdas: {tok_!r} is not a number")
            if lam_ <= 0:
                raise SystemExit(f"--wmsp-extra-lambdas: lambda must be > 0, got {lam_}")
            name_ = f"wmsp_lam{tok_}"
            if name_ in have:
                raise SystemExit(f"--wmsp-extra-lambdas: {name_} is already active; refusing to fit it twice")
            active_wmsp = active_wmsp + [(name_, {"weight_mode": "normalised",
                                                  "reg": shrink_to_uniform, "reg_lambda": lam_})]
            have.add(name_)
        print(f"WMSP-EXTRA-LAMBDAS: appended {[w[0] for w in active_wmsp if w[0].startswith('wmsp_lam')]}",
              flush=True)
    active_poolers = [] if args.skip_poolers else POOLERS
    active_base = []
    if args.baselines:
        active_base = [b.strip() for b in args.baselines.split(",") if b.strip()]
        unknown = [b for b in active_base if b not in BASE_FEATS]
        if unknown:
            raise SystemExit(f"--baselines: unknown {unknown}; valid = {sorted(BASE_FEATS)}")
        print(f"BASELINES: adding {active_base} (uncomputable cells are skipped LOUDLY, left blank)", flush=True)
    active_methods = (FLOORS + ["fair_floor", "saplma"] + active_poolers
                      + [w[0] for w in active_wmsp] + active_base)
    if args.skip_wmsp:
        print("SKIP-WMSP: wMSP variants skipped (their columns are ABSENT, not zero)", flush=True)
    if args.skip_poolers:
        print("SKIP-POOLERS: uniform + attention skipped (their columns are ABSENT, not zero)", flush=True)
    if args.perex_dir:
        Path(args.perex_dir).mkdir(parents=True, exist_ok=True)
        print(f"PER-EXAMPLE SIDECARS -> {args.perex_dir} (gated: recomputed PRR must match to <1e-6)", flush=True)
    global LONG_SRC
    if args.sources:
        want = [s.strip() for s in args.sources.split(",") if s.strip()]
        unknown = [s for s in want if s not in LONG_SRC]
        if unknown:
            raise SystemExit(f"--sources names {unknown}, which are not in LONG_SRC {LONG_SRC}")
        # Preserve LONG_SRC ORDER: `1ds-Diff-long` takes diff[:1], so re-ordering the pool silently
        # changes that rung's composition on every eval (dataset_configs.py:8-10).
        LONG_SRC = [d for d in LONG_SRC if d in want]
        print(f"RESTRICTED SOURCE POOL ({len(LONG_SRC)}): {LONG_SRC}", flush=True)
        print("  rung composition is defined by THIS pool, not the 8-dataset one. SameTask/DiffTask/"
              "LOO differ from the full grid; ID and 1ds-Diff may not. Caption tables accordingly.",
              flush=True)
    if args.label_homogeneous:
        LONG_SRC = [d for d in LONG_SRC if d != "expertqa"]
        print("LABEL-HOMOGENEOUS mode: ExpertQA removed from training sources (correctness labels only)",
              flush=True)
    elif not args.sources:
        print("default MIXED-LABEL pool: ExpertQA (faithfulness) is an ordinary training source", flush=True)
    evals = args.evals.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | seeds {seeds} | evals {evals}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    # NON-NEGOTIABLE FOR A NON-LLAMA MODEL: register the real special-token ids, or
    # weighted_msp.content_keep silently falls back to the Llama-3 `id >= 128000` range test — on
    # Qwen2.5 (specials at 151,643+) that would zero the weight of a large band of ORDINARY content
    # tokens: no crash, just a quietly different method. Guarded to non-default models only so a
    # no-arg Llama run keeps the byte-identical legacy path (where the two tests provably agree on
    # every cached row). token_subsets gets the same registration for any keep-mode that routes
    # through it.
    if MODEL != DEFAULT_MODEL:
        from luq import token_subsets  # noqa: E402
        weighted_msp.set_special_ids(tok.all_special_ids)
        token_subsets.set_special_ids(tok.all_special_ids)
        print(f"special-token ids registered from the {MODEL} tokenizer "
              f"({len(tok.all_special_ids)} ids; Llama range-test fallback OFF)", flush=True)
    PT, SEG, POOLED = {}, {}, {}
    # A --train-spec source must join the load set: sciq/trivia_qa live in SHORT_DATASETS, not
    # LONG_SRC, so without this they are never loaded and build_rows hits its zero-rows guard.
    _load = set(LONG_SRC) | set(evals) | ({d for d, _ in train_spec} if train_spec else set())
    for d in sorted(_load):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok cache -> skip", flush=True); continue
        states, split, y, _, records = loaded
        n_orig = len(records)                          # ORIGINAL record count, BEFORE the unlabelled filter
        orig = np.arange(len(records))                 # filtered-row -> ORIGINAL record position (for sidecars)
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: fully unlabelled ({label_of(d)}) -> skip", flush=True); continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]; orig = keep
        segs = []
        for r, st in zip(records, states):
            sid, _ = sar._token_sentence_ids(tok, list(r['gen_token_ids']),
                                             r.get('gen_text') or tok.decode(r['gen_token_ids'], skip_special_tokens=True))
            sid = np.asarray(sid, dtype=np.int64)
            g = int(np.asarray(st).shape[0])   # per-token window is G+1; wMSP uses answer_states (G) -> ids length G
            if len(sid) == g:                  # already G (state has the +1 anchor); trim to answer tokens
                sid = sid
            segs.append(sid)
        PT[d] = (states, split, y, records, orig); SEG[d] = segs
        # B.5: pooled baseline features, one vector per example. Loaded per dataset and immediately sliced
        # to the method's layer -- the full array is (n, 33, 4096) and only one layer is ever used, so
        # holding all of them would cost gigabytes for nothing. A dataset missing the cache is recorded as
        # absent so its cells can be skipped loudly rather than silently imputed.
        if active_base:
            cfg_d = Config(model_name=MODEL, dataset=d, ood_setting="ID",
                           prompt_regime=PROMPT_REGIME.get(d, ""))
            key_d = cache.run_key(MODEL, d, "ID")
            POOLED[d] = {}
            for bm in active_base:
                fm, layer_b, _std = BASE_FEATS[bm]
                try:
                    arr = cache.load_features(cfg_d.cache_dir, key_d, fm)
                except Exception as e:                 # genuinely absent cache -> blank cells, loudly
                    POOLED[d][bm] = None
                    print(f"    {d}: baseline '{bm}' UNAVAILABLE ({type(e).__name__}: {e}) -> cells using "
                          f"{d} will be left BLANK for this method", flush=True)
                    continue
                v_b = np.ascontiguousarray(arr[:, layer_b, :]); del arr
                # THE INDEX BASE IS THE TRAP. The feature cache is indexed by ORIGINAL record position,
                # but this loop has already DROPPED unlabelled rows, so every downstream index is a
                # FILTERED position. expertqa drops 292 of 2016 rows and factscore 45 of 500, so indexing
                # the features with a filtered index shifts those datasets by up to 292 places -- and still
                # returns a perfectly plausible PRR, which is the worst possible failure mode. Reindex ONCE
                # here, so every later `POOLED[d][bm][i]` uses the same filtered basis as states/records/y.
                # These are hard assertions, not comments: a mismatch here is a bug, not a missing input,
                # and must NOT be swallowed into the "unavailable" path that legitimately skips a cell.
                if len(v_b) != n_orig:
                    raise SystemExit(
                        f"FATAL {d}/{bm}: feature cache has {len(v_b)} rows but the record file has "
                        f"{n_orig}. These must be the same array length -- refusing to guess an alignment.")
                v_b = v_b[orig]                        # ORIGINAL basis -> FILTERED basis
                if len(v_b) != len(states):
                    raise SystemExit(
                        f"FATAL {d}/{bm}: after reindexing, {len(v_b)} feature rows vs {len(states)} "
                        f"states. The filtered bases disagree -- refusing to emit numbers built on that.")
                POOLED[d][bm] = v_b
                if n_orig != len(states):
                    print(f"    {d}: baseline '{bm}' reindexed ORIGINAL->FILTERED "
                          f"({n_orig} -> {len(states)} rows, {n_orig - len(states)} unlabelled dropped)",
                          flush=True)
        print(f"  {d}: {len(states)} rows (label={label_of(d)})", flush=True)
    sources = set(PT)

    if train_spec is not None:
        # Validate AFTER the load, against what actually has a cache -- a named source with no
        # pertok cache must fail loudly here rather than quietly shrinking the pool.
        missing = [d for d, _ in train_spec if d not in PT]
        if missing:
            raise SystemExit(f"--train-spec: no usable cache for {missing} (loaded: {sorted(PT)}). "
                             f"Refusing to run a pool that is smaller than the one requested.")
        # A source equal to the eval target is NOT leakage: build_rows routes `d == X` to X's own
        # TRAIN split via eval_split (splits.py:171-173), which is disjoint from the test rows by
        # construction even for the split-less sets. That is exactly how the canonical ID cell is
        # built, and it is what the ID-short arm of the cross-length experiment needs. Announced
        # rather than blocked, so it is visible in the log that this cell is an ID construction.
        selfsrc = [X for X in evals if X in {d for d, _ in train_spec}]
        if selfsrc:
            print(f"  note: {selfsrc} appear in their own pool -> ID construction (own TRAIN split, "
                  f"disjoint from the eval test rows)", flush=True)
        cells = [(args.rung_name, X, list(train_spec)) for X in evals]
    else:
        cells = cells_long(sources, evals)

    out_rows = []
    for rung, X, spec in cells:
        if want_rungs is not None and rung.replace("-long", "") not in want_rungs:
            continue
        if X not in PT:
            continue
        _, X_te = eval_split(PT[X][1])
        if len(X_te) == 0:
            continue
        xlbl = different_label_projection(X)
        per = {m: [] for m in active_methods}; unc_acc = {m: [] for m in active_methods}; yte_ref = None
        yte_per_seed = []; seeds_done = []
        for sd in seeds:
            train_rows, test_rows = build_rows(X, spec, PT, sd, sampled_train_idx)
            if not train_rows or not test_rows:
                continue
            n_tr = len(train_rows); tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], float)
            yte = np.array([y[i] for i in te_idx], float); yte_ref = yte
            states = [PT[d][0][i] for d, i in allrows]; records = [PT[d][3][i] for d, i in allrows]
            v = {}
            # unsupervised floors (per-example vectors)
            v["floor_sum"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "sum") for i in te_idx])
            v["floor_ppl"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity") for i in te_idx])
            v["floor_min"] = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "min") for i in te_idx])
            # SAPLMA mean-pool + MLP
            Xmean = np.stack([s.mean(axis=0) for s in states])
            v["saplma"] = 1.0 - conf_meanpool(Xmean, tr_idx, te_idx, y, sd)
            # poolers (skipped under --skip-poolers -- select_temperature is the second bottleneck)
            if active_poolers:
                best_T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
                v["uniform"] = np.asarray(attn_unc(train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True),
                                                   states, te_idx, device), float)
                attn_pooler = train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T)
                v["attention"] = np.asarray(attn_unc(attn_pooler, states, te_idx, device), float)
                if args.save_pooler and sd == seeds[0]:    # seed-1 pooler = the RCS post-hoc convention
                    _save_pooler(attn_pooler, best_T, rung, X, sd, args.layer, states, te_idx, test_rows, PT, device)
            # B.5 supervised baselines on the SAME sampled rows, so they are paired with everything else.
            # A cell is computable only if EVERY dataset it touches has the feature; otherwise it is skipped
            # loudly and left absent. Imputing here would be the banned no-op default -- a zero or a mean
            # would read as "measured and bad" rather than "not measured".
            for bm in active_base:
                fm, layer_b, std_b = BASE_FEATS[bm]
                missing = sorted({d for d, _ in allrows if POOLED.get(d, {}).get(bm) is None})
                if missing:
                    if sd == seeds[0]:
                        print(f"    [{rung}/{X}] baseline '{bm}' SKIPPED -- no cached feature for "
                              f"{missing}; cell left BLANK", flush=True)
                    continue
                # Join-site guard: `i` here is a FILTERED index into PT[d], and POOLED was reindexed to the
                # same basis at load. Assert the two agree per dataset rather than trusting it -- an
                # off-by-292 join would return a plausible number rather than an error.
                for _d in {d for d, _ in allrows}:
                    if len(POOLED[_d][bm]) != len(PT[_d][0]):
                        raise SystemExit(f"FATAL {_d}/{bm}: {len(POOLED[_d][bm])} feature rows vs "
                                         f"{len(PT[_d][0])} states -- index bases disagree at the join.")
                Xtr_b = np.vstack([POOLED[d][bm][i] for d, i in train_rows])
                Xte_b = np.vstack([POOLED[d][bm][i] for d, i in test_rows])
                clf_b = probe.train_probe(Xtr_b, y[tr_idx], standardize=std_b, seed=sd)
                v[bm] = np.asarray(list(probe.uncertainty(clf_b, Xte_b)), float)
            # wMSP KEEP variants (skipped under --skip-wmsp -- the training bottleneck)
            seg_cell = [SEG[d][i] for d, i in allrows]
            for name, kw in active_wmsp:
                kw2 = dict(kw)
                if kw2.get('segment_ids') == '_seg_ids':
                    kw2['segment_ids'] = seg_cell
                v[name] = np.asarray(weighted_msp.weighted_msp_unc(states, records, y, tr_idx, te_idx, device,
                                     length_normalise=True, seed=sd, **kw2), float)
            for m in v:
                per[m].append(results.prr(yte, v[m])); unc_acc[m].append(v[m])
            yte_per_seed.append(yte); seeds_done.append(sd)
        if yte_ref is None:
            continue
        # The existing `avg`/paired_bootstrap path averages uncertainty vectors ACROSS seeds and scores them
        # against a single yte_ref -- which is only valid if the test labels are seed-invariant. That has always
        # been assumed (eval_split is fixed-seed); assert it rather than trusting it, because a silent per-seed
        # test-set change would misalign every vector without changing any shape.
        for _s, _yt in zip(seeds_done, yte_per_seed):
            if _yt.shape != yte_ref.shape or not np.array_equal(_yt, yte_ref):
                raise SystemExit(f"FATAL [{rung}/{X}]: test labels differ across seeds (seed {_s} vs "
                                 f"{seeds_done[-1]}). Cross-seed vector averaging and the per-example sidecar "
                                 "both assume a fixed test set -- refusing to emit numbers built on that.")
        stats = {m: (float(np.mean(per[m])), float(np.std(per[m]))) for m in per if per[m]}
        # PRE-REGISTERED primary bar = floor_min (msp_min), FIXED across datasets (2026-07-24 meeting);
        # replaces the rejected max-of-three ("three shots for the baseline"). All three floors are still
        # persisted (METHODS) for the 3-variant table. `strongest` is tracked only for the DUAL-REPORT note
        # on datasets where a different variant is the strongest free score (cnn/samsum -> floor_ppl).
        _FLOOR_KEY = {"sum": "floor_sum", "perplexity": "floor_ppl", "min": "floor_min"}
        primary_name = _FLOOR_KEY[msp.PRIMARY_FLOOR_AGG]
        fair_name = primary_name if primary_name in stats else max(FLOORS, key=lambda f: stats[f][0])
        strongest = max(FLOORS, key=lambda f: stats[f][0])
        stats["fair_floor"] = stats[fair_name]
        avg = {m: np.mean(np.stack(unc_acc[m]), 0) for m in unc_acc if unc_acc[m]}
        avg["fair_floor"] = avg[fair_name]
        best_w = max((w[0] for w in active_wmsp), key=lambda m: stats[m][0]) if active_wmsp else None
        best_p = max(active_poolers, key=lambda m: stats[m][0]) if active_poolers else None
        # HONEST LABEL (Round-3 Task A): `train` from REALISED per-source counts (last seed; seed-stable), not
        # requested caps -- so the label can never overstate the pool (the V-A0 defect). realised==labelled.
        _real = {}
        for _d, _i in train_rows:
            _real[_d] = _real.get(_d, 0) + 1
        srcs = "+".join(f"{d}:{_real.get(d, 0)}" for d in dict.fromkeys(d for d, _c in spec))
        xf = "  [CROSS-LABEL]" if (xlbl and rung != "ID") else ""
        dual = "" if strongest == fair_name else f"  (strongest free = {strongest} {stats[strongest][0]:+.3f}; DUAL-REPORT)"
        # LENGTH COLUMNS (Round-3 Task C): domain-shift vs length-shift made visible, over the REALISED pool.
        _tl = [len(PT[d][3][i]["token_logprobs"]) for d, i in train_rows]
        _el = [len(PT[X][3][i]["token_logprobs"]) for _d, i in test_rows]
        _lens = {"eval_med_len": round(float(np.median(_el)), 1) if _el else "",
                 "train_med_len": round(float(np.median(_tl)), 1) if _tl else "",
                 "train_max_len": int(max(_tl)) if _tl else ""}
        if args.perex_dir:
            _save_perex(args.perex_dir, X, rung, srcs, seeds_done, yte_ref, unc_acc, stats, fair_name,
                        args.layer, prov)
        print(f"\n[{rung:14s}] eval={X} ({label_of(X)}) train={srcs}{xf}  primary_floor={fair_name} {stats['fair_floor'][0]:+.3f}{dual}", flush=True)
        for m in active_methods:
            if m in stats:
                print(f"    {m:14s} {stats[m][0]:+.3f} +/- {stats[m][1]:.3f}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": m,
                                 "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                                 "n_seeds": len(per[m]), "different_label_projection": bool(xlbl and rung != "ID"),
                                 **_lens})
        for vk, a, b in [("bestw_vs_fairfloor", best_w, "fair_floor"),
                         ("bestw_vs_bestpooler", best_w, best_p),
                         ("attention_vs_fairfloor", "attention", "fair_floor")]:
            if a in avg and b in avg:
                mg, lo, hi, p, sig = paired_bootstrap(yte_ref, avg[a], avg[b])
                print(f"    [verdict] {vk:24s} ({a} vs {b}) margin {mg:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} {'SIG' if sig else 'ns'}", flush=True)
                out_rows.append({"rung": rung, "eval": X, "train": srcs, "method": f"VERDICT:{vk}",
                                 "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                 "boot_p": round(p, 4), "significant": bool(sig), "n_seeds": len(seeds)})

        # CRASH SAFETY: land this cell on disk before starting the next one. Each cell costs minutes
        # to hours, so an interruption should cost the CURRENT cell, never the whole run.
        _cell_out = Path(args.out) if args.out else (
            ROOT / "results" / f"probedriftlong{regime_tag()}__{cache._slug(MODEL)}.csv")
        _flush_rows(_cell_out, out_rows, prov)
        print(f"    [saved] {len(out_rows)} rows -> {_cell_out.name}", flush=True)

    # The regime tag goes in the DEFAULT filename so a v2 run cannot silently overwrite the v1 CSV in
    # place. The drivers write to a fixed path, so without this a single `LUQ_REGIME=...` run would land
    # on top of the v1 results and the only clue would be an mtime. v1_frozen/ makes that recoverable
    # rather than fatal, but recoverable is not a reason to allow it. Tag is '' when no override is set,
    # so the v1 default filename is unchanged.
    out = Path(args.out) if args.out else (
        ROOT / "results" / f"probedriftlong{regime_tag()}__{cache._slug(MODEL)}.csv")
    _flush_rows(out, out_rows, prov)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
