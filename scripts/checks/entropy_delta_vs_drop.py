"""A1 / prereg 0.1 — does the pooler's attention FLATTENING predict its PERFORMANCE DROP?

Joe's explicit request (31 July): *"Is there a correlation between how much it flattens -- that delta --
and how much performance drops? If so, that's a really useful signal."* It is the candidate LABEL-FREE
switching signal for the proposed system (weighted-MSP backing off to an attention probe). If it is weak,
the back-off gates on OUTPUT LENGTH instead and the system is not rebuilt around entropy.

Pre-registered in `prereg/0.1_entropy_delta_vs_prr_drop.md` BEFORE this ran. Registered predictions:
  P1  dataset-level (n=8) correlation is WEAK, possibly wrong-signed (cnn is an explicit counterexample:
      its attention barely moves, -0.005, yet it drops as hard as pubmed, which flattens by +0.354).
  P2  per-cell (n=32) may carry more signal, because rung severity varies WITHIN a dataset.
  P3  even a real per-cell correlation may be driven by WITHIN-dataset rung ordering rather than ACROSS
      datasets -- which would make it useless as a switch, since the switch fires across datasets.
      So the within/across decomposition is REQUIRED output, not a nicety.

SIGN DISCIPLINE (registered): the hypothesis predicts a NEGATIVE correlation -- more flattening (larger
positive d_entropy) goes with a larger drop (more negative d_PRR). A POSITIVE correlation is not "weak
support", it is evidence AGAINST.

TWO QUANTITIES, and both conventions are fixed by the prereg:
  d_entropy = mean per-example NORMALISED attention entropy (H / log T) at the OOD rung, minus the same
              at ID. NORMALISED is non-negotiable: raw H scales with sequence length and our datasets
              span ~2 tokens (sciq) to ~384 (expertqa), so raw H would find a length artefact and call it
              flattening. Positive = flatter OOD = the learned query dissolved toward uniform.
  d_PRR     = the attention pooler's own PRR at the OOD rung minus its PRR at ID. Negative = worse.

THE CONFOUND CHECK IS RUN AND REPORTED WHETHER OR NOT THE RESULT IS FAVOURABLE. The prereg fixes it in
advance: raw H correlates with length, and length correlates with the drop, so a raw-H result would be a
confound rather than a signal. We therefore report the raw-H correlation and the length correlation
ALONGSIDE the normalised one, every time.

READ-ONLY. Reads `cache/viz/*__attn[__<rung>].npz` (the `pool_w` per-example softmax weight vectors) and
the per-eval ladder CSVs `results/pdl_fam_<eval>__<slug>.csv` (method `attention`). Writes one CSV + one
PNG under results/. Never touches a cache.

    python scripts/checks/entropy_delta_vs_drop.py
    python scripts/checks/entropy_delta_vs_drop.py --verify-convention   # assert vs pool_attention_ood_diag.panel
"""
import argparse
import csv
import os
import sys
from pathlib import Path

# read-only diagnostic on CACHED data -- never needs the network; force offline so any tokenizer load does
# not hang checking the hub (this is what stalled pool_attention_ood_diag on the login node).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache                                    # noqa: E402
from luq.config import Config                            # noqa: E402
from attn_pool import PROMPT_REGIME                      # noqa: E402  regime-namespaced cache roots

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
OOD_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
METHOD = "attention"          # armA, the learned-query attention pooler -- the thing that flattens


# ---------------------------------------------------------------------------------------------------
# entropy side
# ---------------------------------------------------------------------------------------------------
def entropy_of_sidecar(path, records):
    """Per-example NORMALISED attention entropy H/log(T) for one sidecar.

    Convention copied VERBATIM from `pool_attention_ood_diag.panel` (weights renormalised, the G+1
    alignment guard, H in nats, divided by log T). `--verify-convention` asserts the two agree, so this
    is a cheap re-implementation of a shared convention rather than a second, drifting definition -- the
    reason it is re-implemented at all is that `panel` also does a per-token content/punct classification
    that costs a tokenizer pass over every token of every example, which we do not need here.
    """
    z = np.load(path, allow_pickle=True)
    pool = dict(zip(z["record_pos_all"].tolist(), z["pool_w"]))
    ne, raw, lens = [], [], []
    skipped_zero = skipped_align = 0
    for i, w in pool.items():
        w = np.asarray(w, float)
        if w.sum() <= 0:
            skipped_zero += 1
            continue
        w = w / w.sum()
        G = len(records[i]["gen_token_ids"])
        if len(w) != G + 1:                    # alignment guard: never pad/trim silently
            skipped_align += 1
            continue
        T = len(w)
        H = float(-(w * np.log(w + 1e-12)).sum())
        ne.append(H / np.log(T))
        raw.append(H)
        lens.append(T)
    return (np.array(ne), np.array(raw), np.array(lens),
            {"zero_mass": skipped_zero, "misaligned": skipped_align, "kept": len(ne)})


def sidecar_path(dataset, rung, cache_dir):
    """Resolve a viz sidecar, BASE `cache/viz/` first then the regime namespace.

    ⚠️ The regime-namespaced datasets (asqa/expertqa/factscore) have their records under
    `cache/<regime>/records/` but their sidecars under the BASE `cache/viz/` -- `dump_viz_attention`
    writes to `cfg.cache_dir/viz` and was evidently run without `--prompt-regime` for those three, and
    no `cache/*/viz` directory exists at all. Building the sidecar path from the namespaced cache_dir
    (the first version of this script) silently found nothing for exactly those three datasets and
    dropped 12 of 32 cells -- which read as 'not measured' rather than 'looked in the wrong place'.
    Both locations are tried, and the caller fails loud if neither has it."""
    suffix = "" if rung == "ID" else f"__{rung}"
    name = f"{SLUG}__{dataset}__ID__attn{suffix}.npz"
    base = ROOT / "cache" / "viz" / name
    if base.exists():
        return base
    return cache_dir / "viz" / name


def load_entropy_table(datasets):
    """{(dataset, rung): (mean_norm_entropy, mean_raw_entropy, mean_len, n_kept)}; missing cells ABSENT.

    Absent means NOT MEASURED and must render blank downstream -- never 0.0, which would read as
    'measured and flat'."""
    out, notes = {}, []
    for d in datasets:
        cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID",
                     prompt_regime=PROMPT_REGIME.get(d, ""))
        try:
            records = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, d, "ID"))
        except Exception as e:                                   # noqa: BLE001
            notes.append(f"{d}: records unreadable ({type(e).__name__}) -> whole dataset skipped")
            continue
        for rung in ["ID"] + OOD_RUNGS:
            p = sidecar_path(d, rung, cfg.cache_dir)
            if not p.exists():
                notes.append(f"{d}/{rung}: no sidecar at {p.name}")
                continue
            ne, raw, lens, info = entropy_of_sidecar(p, records)
            if len(ne) == 0:
                notes.append(f"{d}/{rung}: sidecar present but 0 usable rows ({info})")
                continue
            out[(d, rung)] = (float(ne.mean()), float(raw.mean()), float(lens.mean()), info["kept"])
            # ALIGNMENT VISIBILITY. The G+1 guard silently skips a row whose sidecar weight vector does
            # not match its record's token count. A few is fine; a large fraction means the sidecar and
            # the records describe DIFFERENT generations (e.g. a sidecar dumped against another prompt
            # regime), and then the surviving mean is a number computed over an unknown subpopulation.
            # Report the rate always, and shout past 5%, rather than averaging over whatever survived.
            tot = info["kept"] + info["misaligned"] + info["zero_mass"]
            if info["misaligned"]:
                frac = info["misaligned"] / max(tot, 1)
                flag = "  ⚠️ >5% — sidecar/records may describe different generations" if frac > 0.05 else ""
                notes.append(f"{d}/{rung}: {info['misaligned']}/{tot} rows dropped on the G+1 "
                             f"alignment guard ({frac:.1%}){flag}")
    return out, notes


# ---------------------------------------------------------------------------------------------------
# PRR side
# ---------------------------------------------------------------------------------------------------
def load_prr_table(datasets, resdir):
    """{(dataset, rung): (prr_mean, n_seeds)} for the attention pooler, from the PER-EVAL ladder CSVs.

    Deliberately reads `pdl_fam_<eval>__<slug>.csv` rather than the assembled `pdl_master`: the master is
    regenerated by a separate assembler and goes stale silently (its own header says so), and the ladders
    are mid-flight as this runs. Reading the per-eval files means the population is exactly the one the
    ladder wrote, and the file mtimes are stamped into the output so a later re-read is comparable."""
    out, stamps, notes, src = {}, {}, [], {}
    for d in datasets:
        # A rung that was killed on walltime is re-run into its OWN file (e.g. the xsum 1ds-Diff recovery
        # job, commit 865c9b1 -> `pdl_fam_xsum_1ds-Diff__<slug>.csv`). Reading only `pdl_fam_<d>__` would
        # silently lose that rung and report it as 'not measured'. `_base` files hold the unsupervised
        # baselines and carry no `attention` rows, so they are excluded by name rather than by luck.
        # Enumerated explicitly rather than globbed: the main file, plus one optional per-rung recovery
        # file for each rung. A glob on `pdl_fam_{d}*` would also swallow `_base` and, for a dataset whose
        # name prefixes another, the wrong eval entirely.
        candidates = [Path(resdir) / f"pdl_fam_{d}__{SLUG}.csv"]
        candidates += [Path(resdir) / f"pdl_fam_{d}_{r.replace('-long', '')}__{SLUG}.csv"
                       for r in OOD_RUNGS]
        paths = [p for p in candidates if p.exists()]
        if not paths:
            notes.append(f"{d}: no ladder CSV -> all rungs blank")
            continue
        for p in paths:
            stamps[p.name] = p.stat().st_mtime
            with p.open() as f:
                for r in csv.DictReader(f):
                    if r.get("method") != METHOD:
                        continue
                    key = (d, r["rung"])
                    try:
                        val = (float(r["prr_mean"]), int(r.get("n_seeds") or 0))
                    except (TypeError, ValueError):
                        notes.append(f"{d}/{r['rung']}: unparseable prr_mean {r.get('prr_mean')!r}")
                        continue
                    # CONFLICT = FAIL LOUD. If two files both claim the same cell with different values,
                    # one of them is a different population/run and letting the later read win silently is
                    # exactly how a cross-population number gets published.
                    if key in out and abs(out[key][0] - val[0]) > 1e-9:
                        raise SystemExit(
                            f"{d}/{r['rung']}: conflicting PRR from {src[key]} ({out[key][0]:+.4f}) and "
                            f"{p.name} ({val[0]:+.4f}). Refusing to guess which run is canonical.")
                    out[key] = val
                    src[key] = p.name
    return out, stamps, notes


# ---------------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------------
def corrs(x, y, n_boot=10000, seed=0):
    """Pearson + Spearman with a percentile bootstrap CI over the PAIRS. Returns None-filled dict when
    n < 3 rather than emitting a number that cannot mean anything."""
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return {"n": n, "pearson": None, "spearman": None, "p_pearson": None,
                "ci_lo": None, "ci_hi": None}
    pr, pp = stats.pearsonr(x, y)
    sr, _ = stats.spearmanr(x, y)
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_boot):
        k = rng.randint(0, n, n)
        if np.std(x[k]) == 0 or np.std(y[k]) == 0:
            continue
        boot.append(stats.pearsonr(x[k], y[k])[0])
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
    return {"n": n, "pearson": float(pr), "spearman": float(sr), "p_pearson": float(pp),
            "ci_lo": float(lo), "ci_hi": float(hi)}


def within_across(cells):
    """P3 decomposition. `cells` = [(dataset, d_entropy, d_prr)].

    ACROSS = correlate the per-dataset MEANS (this is the component a per-dataset switch could actually
    use -- the switch fires across datasets, per Lihu's constraint that selection is per dataset).
    WITHIN = correlate after subtracting each dataset's own mean from BOTH variables (rung ordering
    inside a dataset). A correlation that lives only in WITHIN is useless as a switching signal, which is
    precisely what P3 warns about."""
    by = {}
    for d, de, dp in cells:
        by.setdefault(d, []).append((de, dp))
    ax, ay, wx, wy = [], [], [], []
    for d, pts in by.items():
        e = np.array([p[0] for p in pts], float)
        r = np.array([p[1] for p in pts], float)
        ax.append(e.mean()); ay.append(r.mean())
        wx.extend(e - e.mean()); wy.extend(r - r.mean())
    return (ax, ay), (wx, wy)


# ---------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(LONG))
    ap.add_argument("--resdir", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / f"entropy_delta_vs_drop__{SLUG}.csv"))
    ap.add_argument("--plot", default=str(ROOT / "results" / f"entropy_delta_vs_drop__{SLUG}.png"))
    ap.add_argument("--verify-convention", action="store_true",
                    help="assert our normalised entropy equals pool_attention_ood_diag.panel's on one cell")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    print("A1 / prereg 0.1 — attention flattening vs performance drop")
    print(f"  population: ProbeDriftLong cells_long, method='{METHOD}' (armA), model {SLUG}")
    print("  REGISTERED SIGN: the hypothesis predicts a NEGATIVE correlation.\n")

    ent, ent_notes = load_entropy_table(datasets)
    prr, stamps, prr_notes = load_prr_table(datasets, args.resdir)
    for n in ent_notes + prr_notes:
        print(f"  [note] {n}")

    if args.verify_convention:
        verify_convention(datasets, ent)

    # ---- build the per-cell table; a cell needs BOTH sides at ID and at the rung ----
    rows, cells = [], []
    for d in datasets:
        for rung in OOD_RUNGS:
            have = all(k in ent for k in [(d, "ID"), (d, rung)]) and \
                   all(k in prr for k in [(d, "ID"), (d, rung)])
            if not have:
                rows.append({"dataset": d, "rung": rung, "d_entropy": "", "d_entropy_raw": "",
                             "d_PRR": "", "ne_ID": "", "ne_OOD": "", "prr_ID": "", "prr_OOD": "",
                             "mean_len": "", "n_seeds": "", "status": "not measured"})
                continue
            ne_i, raw_i, len_i, _ = ent[(d, "ID")]
            ne_o, raw_o, len_o, _ = ent[(d, rung)]
            p_i, ns_i = prr[(d, "ID")]
            p_o, ns_o = prr[(d, rung)]
            de, dr = ne_o - ne_i, p_o - p_i
            cells.append({"ds": d, "rung": rung, "de": de, "dr": dr,
                          "de_raw": raw_o - raw_i, "len": len_o})
            rows.append({"dataset": d, "rung": rung, "d_entropy": f"{de:.4f}",
                         "d_entropy_raw": f"{raw_o - raw_i:.4f}", "d_PRR": f"{dr:.4f}",
                         "ne_ID": f"{ne_i:.4f}", "ne_OOD": f"{ne_o:.4f}",
                         "prr_ID": f"{p_i:.4f}", "prr_OOD": f"{p_o:.4f}",
                         "mean_len": f"{len_o:.1f}", "n_seeds": min(ns_i, ns_o), "status": "ok"})

    n_target = len(datasets) * len(OOD_RUNGS)
    print(f"\n  COVERAGE: {len(cells)}/{n_target} OOD cells measured.")
    missing = [f"{r['dataset']}/{r['rung']}" for r in rows if r["status"] != "ok"]
    if missing:
        print(f"  MISSING (blank, not zero): {', '.join(missing)}")

    # ---- per-cell table ----
    print(f"\n  PER-CELL (n={len(cells)})   [population: cells_long; blank = not measured]")
    print(f"  {'dataset':16s}{'rung':16s}{'d_entropy':>11s}{'d_PRR':>9s}{'ne_ID':>9s}{'ne_OOD':>9s}")
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['dataset']:16s}{r['rung']:16s}{'':>11s}{'':>9s}{'':>9s}{'':>9s}   (not measured)")
        else:
            print(f"  {r['dataset']:16s}{r['rung']:16s}{float(r['d_entropy']):+11.3f}"
                  f"{float(r['d_PRR']):+9.3f}{float(r['ne_ID']):9.3f}{float(r['ne_OOD']):9.3f}")

    if len(cells) < 3:
        print("\n  Too few measured cells to correlate. Stopping rather than emitting a number.")
        write_csv(args.out, rows, stamps)
        return

    de = [c["de"] for c in cells]
    dr = [c["dr"] for c in cells]
    de_raw = [c["de_raw"] for c in cells]
    lens = [c["len"] for c in cells]

    percell = corrs(de, dr)
    (ax, ay), (wx, wy) = within_across([(c["ds"], c["de"], c["dr"]) for c in cells])
    across = corrs(ax, ay)
    within = corrs(wx, wy)
    raw_c = corrs(de_raw, dr)
    len_c = corrs(lens, dr)
    len_e = corrs(lens, de)

    print("\n  ================ CORRELATIONS (registered sign: NEGATIVE supports the hypothesis) ========")

    def line(tag, c, note=""):
        if c["pearson"] is None:
            print(f"  {tag:34s} n={c['n']:<3d}  (n<3, not computed)")
            return
        print(f"  {tag:34s} n={c['n']:<3d}  Pearson {c['pearson']:+.3f} "
              f"[{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}]  Spearman {c['spearman']:+.3f}"
              f"  p={c['p_pearson']:.3f}  {note}")

    line("PER-CELL  d_entropy vs d_PRR", percell, "<- P2, the version that was unrun")
    line("ACROSS-dataset (means)", across, "<- P1 / the only one a switch could use")
    line("WITHIN-dataset (rung ordering)", within, "<- P3")
    print("\n  ---- CONFOUND CHECKS (reported regardless of the result, per the prereg) ----")
    line("RAW-H d_entropy vs d_PRR", raw_c, "<- if this beats normalised, it is a length artefact")
    line("mean length vs d_PRR", len_c, "<- length as a competing explanation")
    line("mean length vs d_entropy", len_e)

    breakdown(cells)

    # ---- the n=4 subset the prereg says must reproduce ----
    byd = {}
    for c in cells:
        if c["ds"] in ("pubmed_qa", "xsum", "expertqa", "cnn_dailymail"):
            byd.setdefault(c["ds"], []).append((c["de"], c["dr"]))
    if byd:
        print("\n  ---- n=4 SUBSET CHECK (prereg tabled these from the E1 dump; they must reproduce) ----")
        print(f"  {'dataset':16s}{'d_entropy (mean over rungs)':>30s}{'d_PRR (mean)':>16s}")
        for d in ("pubmed_qa", "xsum", "expertqa", "cnn_dailymail"):
            if d not in byd:
                print(f"  {d:16s}{'not measured':>30s}")
                continue
            e = np.mean([p[0] for p in byd[d]])
            r = np.mean([p[1] for p in byd[d]])
            print(f"  {d:16s}{e:+30.3f}{r:+16.3f}")
        print("  (prereg reference: pubmed +0.354/-0.538, xsum +0.080/-0.337, "
              "expertqa +0.011/-0.431, cnn -0.005/-0.506)")

    # ---- registered decision rule ----
    print("\n  ================ DECISION (rule fixed in advance) ================")
    strong = (percell["pearson"] is not None and percell["pearson"] < -0.5 and percell["ci_hi"] < 0)
    across_ok = (across["pearson"] is not None and across["pearson"] < -0.5)
    if strong and across_ok:
        print("  STRONG and negative at BOTH levels -> entropy is a CANDIDATE gate. Per the prereg it must")
        print("  now be compared head-to-head against the LENGTH gate leave-one-dataset-out before adoption.")
        print("  ⚠️ And it contradicts the four points already on record -> treat as a SUSPECT and re-run")
        print("     the four checks the prereg lists before believing it.")
    elif strong and not across_ok:
        print("  Negative per-cell but NOT across datasets -> P3 confirmed: the correlation lives in the")
        print("  WITHIN-dataset rung ordering. USELESS as a switching signal, because the switch fires")
        print("  across datasets. Gate on LENGTH.")
    else:
        print("  WEAK or wrong-signed -> entropy is NOT the switching signal. Phase 3 gates on LENGTH.")
        print("  This is a clean negative answering Joe's question, and it is a result, not a failure.")

    write_csv(args.out, rows, stamps)
    make_plot(args.plot, cells, percell, across)


def breakdown(cells, n_perm=20000, seed=0):
    """Is there a correlation in SOME slice, even if not in general? Split the 32 cells two ways.

    BY RUNG (4 groups of 8 datasets): "within one severity of shift, do the datasets that flatten most
    drop most?" This is the SWITCH-RELEVANT question -- a per-dataset switch fires within a rung, ranking
    datasets against each other, so a per-rung rank correlation is what a router could actually exploit.

    BY DATASET (8 groups of 4 rungs): "within one dataset, does the rung that flattens it most hurt it
    most?" A correlation here CANNOT drive a switch (the switch chooses between datasets, not rungs) but
    it would still be a real mechanism finding: flattening would track damage once dataset identity is
    held fixed.

    ⚠️ THE CONTROL IS THE POINT, not an afterthought. Twelve subgroup tests on 32 points will throw up a
    strong-looking one by chance -- and the groups are tiny (n=8 and n=4; with n=4 there are only 4!=24
    orderings, so |rho|=1.0 carries one-sided p=1/24=0.042 and is the WEAKEST possible "significant"
    result). So we report the null distribution of the BEST-LOOKING subgroup: permute d_PRR across all 32
    cells, recompute all twelve, take max|rho|, repeat. A GLOBAL permutation is used rather than a
    within-group one because each cell belongs to one rung group AND one dataset group, so the twelve
    tests are dependent and the null has to respect that. Observed max|rho| is read against that
    distribution, not against 0.05.
    """
    from scipy import stats

    def rho(pts):
        if len(pts) < 3:
            return None
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        if np.std(x) == 0 or np.std(y) == 0:
            return None
        return float(stats.spearmanr(x, y)[0])

    groups = {}                                    # name -> [(d_entropy, d_PRR)]
    for r in OOD_RUNGS:
        pts = [(c["de"], c["dr"]) for c in cells if c["rung"] == r]
        if pts:
            groups[f"rung:{r}"] = pts
    for d in sorted({c["ds"] for c in cells}):
        pts = [(c["de"], c["dr"]) for c in cells if c["ds"] == d]
        if pts:
            groups[f"data:{d}"] = pts

    print("\n  ================ BREAKDOWN — is there a correlation in SOME slice? ================")
    print("  Registered sign is still NEGATIVE. Spearman (rank) is the switch-relevant statistic.\n")
    print(f"  {'slice':28s}{'n':>4s}{'Spearman':>11s}{'Pearson':>10s}   interpretation")
    obs = {}
    for name, pts in groups.items():
        s = rho(pts)
        obs[name] = s
        if s is None:
            print(f"  {name:28s}{len(pts):>4d}{'--':>11s}{'--':>10s}   n<3, not computed")
            continue
        x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        pr = float(stats.pearsonr(x, y)[0])
        tag = "supports (negative)" if s < -0.5 else ("WRONG SIGN" if s > 0.5 else "flat")
        print(f"  {name:28s}{len(pts):>4d}{s:+11.3f}{pr:+10.3f}   {tag}")

    vals = {k: v for k, v in obs.items() if v is not None}
    if not vals:
        return
    best = max(vals, key=lambda k: abs(vals[k]))
    best_abs = abs(vals[best])

    # ---- null distribution of the BEST subgroup, by global permutation of d_PRR ----
    rng = np.random.RandomState(seed)
    de_all = np.array([c["de"] for c in cells], float)
    dr_all = np.array([c["dr"] for c in cells], float)
    idx = {name: [i for i, c in enumerate(cells)
                  if (c["rung"] == name.split(":", 1)[1] if name.startswith("rung:")
                      else c["ds"] == name.split(":", 1)[1])]
           for name in groups}
    null_max = []
    for _ in range(n_perm):
        perm = rng.permutation(dr_all)
        m = 0.0
        for name, ii in idx.items():
            if len(ii) < 3:
                continue
            x, y = de_all[ii], perm[ii]
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            m = max(m, abs(float(stats.spearmanr(x, y)[0])))
        null_max.append(m)
    null_max = np.array(null_max)
    p_best = float((null_max >= best_abs).mean())
    q95 = float(np.percentile(null_max, 95))

    print(f"\n  STRONGEST SLICE: {best}  Spearman {vals[best]:+.3f}  (|rho| = {best_abs:.3f})")
    print(f"  NULL for the strongest of {len(vals)} slices (global permutation of d_PRR, {n_perm} draws):")
    print(f"    95th percentile of max|rho| under the null = {q95:.3f}")
    print(f"    P(max|rho| >= {best_abs:.3f} by chance)      = {p_best:.3f}")
    if p_best < 0.05 and vals[best] < 0:
        print("  -> the strongest slice SURVIVES the multiple-comparisons null AND has the registered sign.")
        print("     Worth following up; still one slice out of twelve, so treat as a lead, not a finding.")
    elif p_best < 0.05:
        print("  -> survives the null but has the WRONG SIGN, so it is not support for the hypothesis.")
    else:
        print("  -> does NOT survive. A slice this strong is what twelve tests on 32 points produce anyway,")
        print("     so there is no subgroup in which flattening predicts the drop.")

    n_neg = sum(1 for v in vals.values() if v < 0)
    print(f"\n  SIGN COUNT across the {len(vals)} slices: {n_neg} negative (hypothesis direction), "
          f"{len(vals) - n_neg} positive. Under the null this is a coin flip; "
          f"a real effect should push most slices negative.")


def verify_convention(datasets, ent):
    """CONTROL: our entropy must equal pool_attention_ood_diag.panel's on a real cell, or the two
    definitions have drifted and every number here is on a different footing from the existing diagnostic."""
    import pool_attention_ood_diag as pood
    from transformers import AutoTokenizer
    d = next((x for x in datasets if (x, "ID") in ent), None)
    if d is None:
        print("  [verify] no measured ID cell to check against")
        return
    cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID", prompt_regime=PROMPT_REGIME.get(d, ""))
    records = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, d, "ID"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    special = set(tok.all_special_ids)
    P = pood.panel(sidecar_path(d, "ID", cfg.cache_dir), records, special, tok)
    theirs = float(np.mean(P["norm_entropy"]))
    ours = ent[(d, "ID")][0]
    ok = abs(theirs - ours) < 1e-9
    print(f"  [verify] {d}/ID normalised entropy — ours {ours:.9f} vs panel {theirs:.9f} "
          f"-> {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("entropy convention has drifted from pool_attention_ood_diag.panel — refusing to "
                         "report numbers on a second, silently different definition.")


def write_csv(path, rows, stamps):
    cols = ["dataset", "rung", "d_entropy", "d_entropy_raw", "d_PRR", "ne_ID", "ne_OOD",
            "prr_ID", "prr_OOD", "mean_len", "n_seeds", "status"]
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    os.replace(tmp, path)                      # atomic: a crash mid-write cannot leave a short-but-valid CSV
    print(f"\n  wrote {path}")
    if stamps:
        import time
        print("  SOURCE LADDER CSVs (mtime stamped — the ladders were mid-flight when this ran):")
        for name, t in sorted(stamps.items()):
            print(f"    {name}: {time.strftime('%Y-%m-%d %H:%M', time.localtime(t))}")


def make_plot(path, cells, percell, across):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = sorted({c["ds"] for c in cells})
    cmap = plt.get_cmap("tab10")
    colour = {d: cmap(i % 10) for i, d in enumerate(ds)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    a = axes[0]
    for d in ds:
        pts = [(c["de"], c["dr"]) for c in cells if c["ds"] == d]
        a.scatter([p[0] for p in pts], [p[1] for p in pts], s=55, color=colour[d], label=d, alpha=0.85)
    a.axhline(0, lw=0.7, color="0.6")
    a.axvline(0, lw=0.7, color="0.6")
    r = percell["pearson"]
    a.set_title(f"Per-cell (n={percell['n']})   Pearson {r:+.3f}"
                f" [{percell['ci_lo']:+.3f},{percell['ci_hi']:+.3f}]" if r is not None
                else f"Per-cell (n={percell['n']})")
    a.set_xlabel("Δ normalised attention entropy (OOD − ID)   →  flatter")
    a.set_ylabel("Δ PRR (OOD − ID)   ↓  worse")
    a.legend(fontsize=7, ncol=2)

    b = axes[1]
    for d in ds:
        pts = [(c["de"], c["dr"]) for c in cells if c["ds"] == d]
        mx, my = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
        b.scatter(mx, my, s=110, color=colour[d], label=d)
        b.annotate(d, (mx, my), fontsize=7, xytext=(4, 4), textcoords="offset points")
    b.axhline(0, lw=0.7, color="0.6")
    b.axvline(0, lw=0.7, color="0.6")
    ra = across["pearson"]
    b.set_title(f"Per-dataset means (n={across['n']})   Pearson {ra:+.3f}" if ra is not None
                else f"Per-dataset means (n={across['n']})")
    b.set_xlabel("Δ normalised attention entropy (mean over OOD rungs)")
    b.set_ylabel("Δ PRR (mean over OOD rungs)")

    fig.suptitle("A1 / prereg 0.1 — does attention flattening predict the performance drop?   "
                 "population: ProbeDriftLong cells_long, method=attention (armA)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
