"""S9 — is the "selection law" really about ANSWER LENGTH? (prereg/response_length_vs_selection_law.md)

THE LAW: probe advantage over the untrained `msp_min` baseline is strongly NEGATIVELY correlated with how
strong `msp_min` already is (PDL r=-0.846 n=8; XL r=-0.895 n=10). Short-form QA sits at the extreme
high-baseline end AND has by far the shortest answers, so the two are confounded across datasets.

THE MECHANICAL WORRY, TESTED NOT ASSUMED: `msp_min` is a MIN over tokens, so more tokens = more draws =
a lower minimum, regardless of the model being any less certain. `perplexity` is length-normalised and
should NOT share that dependence. **That contrast is the control.**

LENGTH DEFINITION. `mean_len` in `id_entropy_vs_drop__*.csv` is **T = G+1** -- generated tokens PLUS
one prompt token, because the pooling window is `[last_prompt_token] + gen_tokens`
(`entropy_delta_vs_drop.py:89-97`). It INCLUDES a prompt token. This driver therefore recomputes
**G = len(record["gen_token_ids"])**, generation only, identically for all 10 datasets, and prints
mean(T)-1 beside it as a cross-check (they must agree, since T = G+1 per row).

CPU only. No GPU, no judge calls, no money.
"""
import csv
import glob
import itertools
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/rds/general/user/gs925/home/gs925-msc_project/msc-project-gs925")
SLUG = "meta-llama_Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
SHORT = ["sciq", "trivia_qa"]
OOD_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
TERCILE_SETS = ["expertqa", "cnn_dailymail", "pubmed_qa"]   # widest length spread


def rd():
    local = ROOT / "results"
    return local if list(local.glob("pdl_fam_*.csv")) else BASE / "results"


# expertqa/asqa/factscore live in a PROMPT-REGIME namespace, exactly as attn_pool.PROMPT_REGIME says.
# Looking only in cache/records/ finds nothing for them and would drop 3 of the 8 long sets.
PROMPT_REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}


def cd(ds=None):
    root = ROOT if (ROOT / "cache" / "records").exists() else BASE
    sub = PROMPT_REGIME.get(ds, "")
    return (root / "cache" / sub / "records") if sub else (root / "cache" / "records")


# ---------------------------------------------------------------- stats (no scipy dependency)
def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for pos, i in enumerate(order):
        r[i] = pos + 1
    return r


def spearman(a, b):
    return pearson(rank(a), rank(b))


def perm_p(a, b, use_rank=True):
    """EXACT permutation p for n<=10 (<=3.6M orderings at n=10 is too many; sample when n>8)."""
    f = (lambda x, y: pearson(rank(x), rank(y))) if use_rank else pearson
    obs = abs(f(a, b))
    n = len(a)
    if n <= 8:
        tot = hit = 0
        for perm in itertools.permutations(b):
            tot += 1
            if abs(f(a, list(perm))) >= obs - 1e-12:
                hit += 1
        return hit / tot, tot, "exact"
    # n>8: deterministic subsample of orderings, seeded by construction (no RNG allowed in this repo's
    # analysis scripts to stay reproducible) -- use every k-th ordering of a fixed generator.
    tot = hit = 0
    for k, perm in enumerate(itertools.permutations(b)):
        if k % 97:                      # fixed stride, deterministic, ~37k of 3.6M at n=10
            continue
        tot += 1
        if abs(f(a, list(perm))) >= obs - 1e-12:
            hit += 1
    return hit / tot, tot, "stride-97"


def boot_ci(a, b, fn, iters=2000):
    """Deterministic leave-one-out jackknife CI -- no RNG, reproducible, honest about n."""
    vals = []
    for i in range(len(a)):
        aa = [a[j] for j in range(len(a)) if j != i]
        bb = [b[j] for j in range(len(b)) if j != i]
        v = fn(aa, bb)
        if v == v:
            vals.append(v)
    if len(vals) < 3:
        return float("nan"), float("nan")
    return min(vals), max(vals)


def partial(a, b, c):
    """Partial correlation of a,b controlling for c (Pearson on the raw values)."""
    rab, rac, rbc = pearson(a, b), pearson(a, c), pearson(b, c)
    den = ((1 - rac ** 2) * (1 - rbc ** 2)) ** 0.5
    return (rab - rac * rbc) / den if den else float("nan")


# ---------------------------------------------------------------- data
def load_records(ds):
    hits = [f for f in glob.glob(str(cd(ds) / f"*__{ds}__ID.jsonl"))
            if "Meta-Llama-3.1-8B" in f and not f.endswith(".bak")
            and ".bak_" not in f]
    if len(hits) != 1:
        raise SystemExit(f"{ds}: expected exactly 1 Llama record file, found {len(hits)} -- refusing to guess")
    return [json.loads(l) for l in open(hits[0]) if l.strip()]


def gen_lengths(ds):
    """G = number of GENERATED tokens. One definition, all 10 datasets, no prompt tokens."""
    recs = load_records(ds)
    return [len(r["gen_token_ids"]) for r in recs], recs


def reused_mean_len():
    """mean_len from the entropy table = T = G+1 (INCLUDES one prompt token). Cross-check only."""
    p = rd() / f"id_entropy_vs_drop__{SLUG}.csv"
    if not p.exists():
        return {}
    return {r["dataset"]: float(r["mean_len"]) for r in csv.DictReader(open(p)) if r.get("mean_len")}


def prr_from_fam(method_names):
    """{(dataset, rung): prr} from the per-eval ladder files -- the same source the master assembles."""
    out = {}
    for f in sorted(glob.glob(str(rd() / f"pdl_fam_*__{SLUG}.csv"))):
        for r in csv.DictReader(open(f)):
            if r["method"] in method_names and r.get("prr_mean"):
                try:
                    out[(r["eval"], r["rung"], r["method"])] = float(r["prr_mean"])
                except ValueError:
                    pass
    return out


def ood_value(prr, ds, method):
    """Long sets: mean over the 4 long OOD rungs. Short sets: the single Long->Short cell.
    These are DIFFERENT rung constructions -- flagged in the output, not silently merged."""
    if ds in SHORT:
        v = prr.get((ds, "Long->Short", method))
        return (v, 1) if v is not None else (None, 0)
    vals = [prr[(ds, g, method)] for g in OOD_RUNGS if (ds, g, method) in prr]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def main():
    prr = prr_from_fam({"floor_min", "floor_ppl", "saplma"})
    reuse = reused_mean_len()

    print("=" * 100)
    print("LENGTH DEFINITION, PRINTED IN FULL")
    print("=" * 100)
    print("  USED HERE      G = len(record['gen_token_ids'])  -- GENERATED TOKENS ONLY, no prompt tokens.")
    print("                 Tokenizer: the Llama-3.1-8B tokenizer that produced the generation; the ids")
    print("                 are stored at generation time, so no re-tokenisation is involved.")
    print("                 Source: cache/records/<slug>__<dataset>__ID.jsonl (Tier-1 records).")
    print("  NOT USED       `mean_len` in id_entropy_vs_drop__*.csv is T = G+1: generated tokens PLUS the")
    print("                 last PROMPT token, because the pooling window is [last_prompt_token]+gen.")
    print("                 It INCLUDES a prompt token, so it is reported below only as a cross-check.")

    rows = []
    for ds in LONG + SHORT:
        G, recs = gen_lengths(ds)
        mn, md, ne = st.mean(G), st.median(G), len(G)
        f_min, n_min = ood_value(prr, ds, "floor_min")
        f_ppl, _ = ood_value(prr, ds, "floor_ppl")
        sap, _ = ood_value(prr, ds, "saplma")
        adv = (sap - f_min) if (sap is not None and f_min is not None) else None
        rows.append(dict(dataset=ds, mean_len=mn, median_len=md, n=ne, msp_min=f_min,
                         ppl=f_ppl, saplma=sap, adv=adv, n_rungs=n_min))

    print("\n" + "=" * 100)
    print("THE TABLE — one row per dataset")
    print("population: long sets = mean over 4 long OOD rungs; short sets = the single Long->Short cell")
    print("=" * 100)
    hdr = (f"{'dataset':14s}{'mean_len':>10s}{'median_len':>12s}{'n_examples':>12s}"
           f"{'msp_min_OOD':>13s}{'perplexity':>12s}{'SAPLMA_OOD':>12s}{'probe_adv':>11s}{'rungs':>7s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        def f(x, w=13):
            return f"{x:+{w}.4f}" if x is not None else f"{'—':>{w}s}"
        flag = "  n<200" if r["n"] < 200 else ""
        print(f"{r['dataset']:14s}{r['mean_len']:10.1f}{r['median_len']:12.1f}{r['n']:12d}"
              f"{f(r['msp_min'])}{f(r['ppl'],12)}{f(r['saplma'],12)}{f(r['adv'],11)}{r['n_rungs']:>7d}{flag}")

    print("\n  CROSS-CHECK of the reused column (mean_len from the entropy table should be G+1):")
    for r in rows:
        t = reuse.get(r["dataset"])
        if t is None:
            print(f"    {r['dataset']:14s} not in that file (expected for the short sets)")
        else:
            print(f"    {r['dataset']:14s} T={t:7.1f}   T-1={t-1:7.1f}   mean(G)={r['mean_len']:7.1f}"
                  f"   diff={t-1-r['mean_len']:+7.2f}")

    print("\n  VERIFICATION against the master tables (must match to 1e-6):")
    targets = {"pubmed_qa": 0.371, "factscore": 0.428, "sciq": 0.7555, "trivia_qa": 0.7465}
    ok = True
    for ds, want in targets.items():
        got = next(r["msp_min"] for r in rows if r["dataset"] == ds)
        d = abs(got - want)
        # the master prints 3-4 dp, so compare at the printed precision
        good = d < 5e-4
        ok &= good
        print(f"    {ds:14s} computed {got:+.4f}  target {want:+.4f}  diff {d:.2e}  {'OK' if good else '*** MISMATCH ***'}")
    if not ok:
        raise SystemExit("STOP: msp_min does not match the master tables -- the populations differ.")

    # ------------------------------------------------------------ correlations
    def block(sub, label):
        print("\n" + "=" * 100)
        print(f"CORRELATIONS — {label}  (n = {len(sub)})")
        print("=" * 100)
        L = [r["mean_len"] for r in sub]
        M = [r["msp_min"] for r in sub]
        P = [r["ppl"] for r in sub]
        A = [r["adv"] for r in sub]
        pairs = [("length vs msp_min_OOD", L, M),
                 ("length vs perplexity_OOD   <- CONTROL", L, P),
                 ("length vs probe_advantage", L, A),
                 ("msp_min_OOD vs probe_advantage", M, A)]
        print(f"  {'pair':40s}{'pearson':>10s}{'spearman':>10s}{'perm p':>10s}{'jackknife CI (spearman)':>28s}")
        print("  " + "-" * 96)
        for name, a, b in pairs:
            pe, sp = pearson(a, b), spearman(a, b)
            p, tot, kind = perm_p(a, b)
            lo, hi = boot_ci(a, b, spearman)
            print(f"  {name:40s}{pe:+10.4f}{sp:+10.4f}{p:10.4f}   [{lo:+.4f},{hi:+.4f}] ({kind})")
        print("\n  PARTIAL CORRELATIONS (Pearson):")
        print(f"    msp_min vs advantage, controlling for length : {partial(M, A, L):+.4f}")
        print(f"    length  vs advantage, controlling for msp_min: {partial(L, A, M):+.4f}")
        if len(sub) <= 8:
            print("    n<=8: point estimates only. NOT reported as significant (5 residual df).")
        print("\n  DECISION RULE (prereg/S9): length explains the law iff BOTH")
        rho_la = abs(spearman(L, A))
        pma = abs(partial(M, A, L))
        print(f"    (a) |rho(length, advantage)| >= 0.70   ->  {rho_la:.4f}  {'PASS' if rho_la >= 0.70 else 'FAIL'}")
        print(f"    (b) partial(msp_min, advantage | length) < 0.40  ->  {pma:.4f}  {'PASS' if pma < 0.40 else 'FAIL'}")
        print(f"    VERDICT: {'LENGTH EXPLAINS THE LAW' if (rho_la >= 0.70 and pma < 0.40) else 'length does NOT explain the law'}")

    complete = [r for r in rows if None not in (r["msp_min"], r["ppl"], r["saplma"])]
    block([r for r in complete if r["dataset"] in LONG], "8 LONG-FORM SETS (clean: one rung construction)")
    block(complete, "ALL 10 mixes two rung constructions (long OOD mean vs Long->Short)")

    # ------------------------------------------------------------ within-dataset terciles
    print("\n" + "=" * 100)
    print("WITHIN-DATASET LENGTH TERCILES — the decisive check")
    print("Model and task held fixed; only length varies. msp_min is a MIN over tokens; perplexity is")
    print("length-normalised. If msp_min falls across terciles and perplexity does not, the mechanical")
    print("effect is real regardless of the between-dataset correlations.")
    print("=" * 100)
    print(f"\n  {'dataset':14s}{'tercile':10s}{'n':>7s}{'mean_len':>10s}{'msp_min':>11s}{'perplexity':>12s}")
    print("  " + "-" * 64)
    for ds in TERCILE_SETS:
        recs = load_records(ds)
        items = []
        for r in recs:
            tl = r.get("token_logprobs")
            if not tl:
                continue
            g = len(r["gen_token_ids"])
            mn = min(tl)                                  # msp_min: the least-probable token
            ppl = -sum(tl) / len(tl)                      # length-normalised mean NLL
            items.append((g, mn, ppl))
        items.sort(key=lambda t: t[0])
        n = len(items)
        for k, lab in enumerate(["short", "mid", "long"]):
            seg = items[k * n // 3:(k + 1) * n // 3]
            if not seg:
                continue
            print(f"  {ds if k == 0 else '':14s}{lab:10s}{len(seg):7d}"
                  f"{st.mean(x[0] for x in seg):10.1f}"
                  f"{st.mean(x[1] for x in seg):+11.4f}{st.mean(x[2] for x in seg):+12.4f}")
        print("  " + "-" * 64)
    print("\n  NOTE: these two columns are the RAW per-example quantities (min token logprob; mean NLL),")
    print("  NOT PRR. PRR needs labels and a train/test split; this asks only whether the underlying")
    print("  statistic drifts with length, which is what the mechanical worry is about.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
