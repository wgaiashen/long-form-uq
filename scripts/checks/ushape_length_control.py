"""Is the entropy U-shape a LENGTH effect in disguise?

The observation: distance from the middle of the entropy range predicts the ID->OOD performance drop.
The worry: ID attention entropy correlates with output length at rho ~ +0.69, so a length effect could
produce the same pattern without entropy meaning anything.

The control is `|length - centre|` vs the drop, on the same 8 datasets and the same harness. If it
matches or beats the entropy version, the observation is length wearing an entropy costume.

POST-HOC, and labelled as such. Both the functional form (distance from a centre) and the centre
itself are chosen after seeing the data, on n = 8. `prereg/0.2` already recorded this analysis as NOT
BLIND. Nothing here can be promoted to a finding on its own.

THE RESULT IS FAVOURABLE TO THE OBSERVATION, so it is treated as a suspect. The extra checks below are
the ones that would have been run had the control gone the other way:
  - CENTRE SENSITIVITY. The centre is estimated from the same 8 points. If the pattern only survives at
    the median it is an artifact of that one choice.
  - PARTIAL CORRELATION. Does |entropy - centre| still predict the drop once length is removed, and does
    |length - centre| predict anything once entropy is removed?

Exact permutation p over all 8! = 40,320 orderings; n = 8 makes asymptotic p-values meaningless.

Reads one saved CSV. No compute node.
"""
import csv
import itertools
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/rds/general/user/gs925/home/gs925-msc_project/msc-project-gs925")
SLUG = "meta-llama_Meta-Llama-3.1-8B"
NAME = f"id_entropy_vs_drop__{SLUG}.csv"
PREREG_ALPHA = 0.025          # prereg/0.2's Bonferroni threshold, fixed before any of this was run


def src():
    """`results/` is gitignored and lives in the main checkout only, not in a git worktree."""
    local = ROOT / "results" / NAME
    return local if local.exists() else BASE / "results" / NAME


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0] * len(v)
    for pos, i in enumerate(order):
        rk[i] = pos + 1
    return rk


def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def spearman(a, b):
    return pearson(rank(a), rank(b))


def perm_p(a, b):
    """Exact two-sided permutation p on the Spearman statistic: enumerate every ordering of b."""
    obs = abs(spearman(a, b))
    ra = rank(a)
    hits = total = 0
    for perm in itertools.permutations(rank(b)):
        total += 1
        if abs(pearson(ra, list(perm))) >= obs - 1e-12:
            hits += 1
    return hits / total, total


def partial(a, b, c):
    """Partial Spearman of a,b controlling for c — correlation of the rank residuals."""
    ra, rb, rc = rank(a), rank(b), rank(c)
    r_ac, r_bc, r_ab = pearson(ra, rc), pearson(rb, rc), pearson(ra, rb)
    den = ((1 - r_ac ** 2) * (1 - r_bc ** 2)) ** 0.5
    return (r_ab - r_ac * r_bc) / den if den else float("nan")


def main():
    rows = list(csv.DictReader(open(src())))
    ds = [r["dataset"] for r in rows]
    ne = [float(r["ne_ID"]) for r in rows]
    ln = [float(r["mean_len"]) for r in rows]
    dr = [float(r["drop_mean"]) for r in rows]
    n = len(ds)

    print("=" * 78)
    print(f"U-SHAPE vs LENGTH — n = {n} datasets, ID attention entropy and mean output length")
    print(f"population: the 8 long-form evals, drop = mean over that dataset's 4 OOD rungs")
    print("=" * 78)

    print("\n1. HARNESS CHECK — a quantity already on record must come back identical")
    r, (p, tot) = spearman(ne, dr), perm_p(ne, dr)
    ok = abs(r - (-0.1429)) < 5e-4 and abs(p - 0.7520) < 5e-4
    print(f"   raw entropy vs drop   rho={r:+.4f}  p={p:.4f}   "
          f"{'MATCHES the recorded -0.1429 / 0.7520' if ok else 'DIFFERS — harness suspect, stop'}")
    if not ok:
        raise SystemExit("harness check failed: this does not reproduce id_entropy_vs_drop's own number")

    print(f"\n2. THE CONTROL  (exact permutation over {tot:,} orderings)")
    med_ne, med_ln = st.median(ne), st.median(ln)
    u_ne = [abs(x - med_ne) for x in ne]
    u_ln = [abs(x - med_ln) for x in ln]
    print(f"   {'predictor of the drop':38s} {'rho':>8s} {'exact p':>9s}")
    print("   " + "-" * 58)
    for label, v in [("|entropy - median|   (the observation)", u_ne),
                     ("|length  - median|   (THE CONTROL)", u_ln),
                     ("raw length", ln),
                     ("raw entropy", ne)]:
        rr, (pp, _) = spearman(v, dr), perm_p(v, dr)
        flag = ""
        if label.startswith("|entropy"):
            flag = "  <- vs prereg/0.2 alpha=%.3f: %s" % (
                PREREG_ALPHA, "PASSES" if pp < PREREG_ALPHA else "**STILL FAILS**")
        print(f"   {label:38s} {rr:+8.4f} {pp:9.4f}{flag}")
    print(f"\n   entropy<->length coupling: rho={spearman(ne, ln):+.4f}  (the reason the control was needed)")

    print("\n3. CENTRE SENSITIVITY — the centre is estimated from the SAME 8 points")
    centres = {"median": (st.median(ne), st.median(ln)),
               "mean": (st.mean(ne), st.mean(ln)),
               "midrange": ((min(ne) + max(ne)) / 2, (min(ln) + max(ln)) / 2)}
    print(f"   {'centre':10s} {'|entropy-c| rho':>17s} {'p':>8s} {'|length-c| rho':>16s} {'p':>8s}")
    print("   " + "-" * 62)
    survived = 0
    for name, (cn, cl) in centres.items():
        en = [abs(x - cn) for x in ne]
        le = [abs(x - cl) for x in ln]
        re_, (pe, _) = spearman(en, dr), perm_p(en, dr)
        rl_, (pl, _) = spearman(le, dr), perm_p(le, dr)
        survived += int(pe < 0.05)
        print(f"   {name:10s} {re_:+17.4f} {pe:8.4f} {rl_:+16.4f} {pl:8.4f}")
    print(f"   -> the entropy U-shape reaches p<0.05 under {survived}/3 centre choices")
    if survived < 3:
        print("      NOT robust to the centre. Since the centre is a free parameter fitted to the same")
        print("      8 points, a pattern that depends on which one is chosen is not evidence.")

    print("\n4. PARTIAL CORRELATION — does each survive once the other is removed?")
    print(f"   |entropy-median| vs drop, controlling for length : {partial(u_ne, dr, ln):+.4f}")
    print(f"   |length -median| vs drop, controlling for entropy : {partial(u_ln, dr, ne):+.4f}")

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    p_ne = perm_p(u_ne, dr)[0]
    p_ln = perm_p(u_ln, dr)[0]
    print(f"  The control does NOT explain the observation: |length-median| reaches only p={p_ln:.4f}")
    print(f"  against the entropy version's p={p_ne:.4f}. So it is not length in disguise.")
    print(f"  BUT the observation STILL misses its own pre-registered threshold of {PREREG_ALPHA}")
    print(f"  (p={p_ne:.4f}). Surviving a confound check does not promote it.")
    print("  n=8, a post-hoc functional form, and a centre fitted to the same points. This stays an")
    print("  OBSERVATION to raise in review -- not a finding, and not a foundation for a router.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
