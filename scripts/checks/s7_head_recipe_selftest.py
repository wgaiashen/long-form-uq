"""S7 self-test — pooling heads with DIFFERENT FIXED recipes (design notes 3+4).

Runs on synthetic tensors in seconds, no cache and no GPU, so it can gate the real run cheaply.

The assertions, and why each one exists:

  A. The single-head path is BYTE-IDENTICAL to before. The S7 change lives in the multi-head branch, but
     a leak into arms A/B/C/D would silently move every existing number. This is the regression that
     matters most.
  B. A frozen head's attention EQUALS its renormalised recipe. If it does not, the "fixed recipe" head is
     not fixed and the collapse-proof property is gone.
  C. Frozen heads receive NO gradient on their query row. Same property, checked at the parameter rather
     than the output.
  D. Different recipes produce DIFFERENT attention. This is the whole premise -- B.2 died because heads
     sharing a condition collapsed to correlation 1.0000.
  E. IDENTICAL recipes produce correlation 1.0. The same-recipe control must behave as a control; if it
     did not, a low correlation in the real arm would be uninterpretable.
  F. A zero-mass recipe RAISES rather than degrading to a near-zero attention row. A clamp there would
     hand a recipe that selected nothing a plausible-looking pooled vector -- the banned failure.
  G. A free head next to frozen heads still LEARNS (its query moves off its init).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attn_pool import AttnPool, pad_head_priors, train_attn        # noqa: E402

# Single-threaded on purpose. The problem is tiny (40 synthetic examples), and torch's default spawned 64
# threads, which on a SHARED LOGIN NODE is exactly the load RCS kill processes for. One thread also makes
# the determinism assertions in A independent of the thread count of whatever machine runs it.
torch.set_num_threads(1)

TOL = 1e-6
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def synth(n=40, T=12, d=16, seed=0):
    """n examples of T tokens each, all the same length so the mask is full (padding is exercised by the
    real driver; here we want the invariants, not the padding)."""
    rng = np.random.RandomState(seed)
    states = [rng.randn(T, d).astype(np.float32) for _ in range(n)]
    y = (rng.rand(n) > 0.5).astype(float)
    return states, y, T, d


def recipe(n, T, kind, seed=0):
    """A per-example weight vector per recipe kind, mimicking prior_builders' UNNORMALISED convention."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        if kind == "flat":
            w = np.ones(T, dtype=np.float32)
        elif kind == "front":                      # mass on the first third
            w = np.zeros(T, dtype=np.float32); w[: max(1, T // 3)] = 1.0
        elif kind == "back":                       # mass on the last third
            w = np.zeros(T, dtype=np.float32); w[-max(1, T // 3):] = 1.0
        elif kind == "rand":
            w = rng.rand(T).astype(np.float32) + 1e-3
        elif kind == "dead":                       # selects nothing -> must raise
            w = np.zeros(T, dtype=np.float32)
        else:
            raise ValueError(kind)
        out.append(w)
    return out


def main():
    torch.manual_seed(0)
    dev = "cpu"
    states, y, T, d = synth()
    n = len(states)
    tr = list(range(n))

    print("A. single-head path unchanged")
    m1 = train_attn(states, y, tr, dev, seed=1, epochs=3)
    m2 = train_attn(states, y, tr, dev, seed=1, epochs=3)
    same = torch.allclose(m1.q, m2.q, atol=0) and torch.allclose(m1.head.weight, m2.head.weight, atol=0)
    check("single-head deterministic and untouched by S7", same)
    X = torch.tensor(np.stack(states), dtype=torch.float32)
    mask = torch.ones(n, T)
    pos = torch.zeros(n, T, 2)
    lg, a = m1(X, mask, pos)
    check("single-head attention is (B,T) and sums to 1", a.dim() == 2 and abs(float(a.sum(1).mean()) - 1) < TOL)

    print("\nB/C/D. frozen recipe heads")
    Q = 3
    hp = [recipe(n, T, "front"), recipe(n, T, "back"), None]        # head 2 free
    model = AttnPool(d, n_query=Q, n_head=Q, frozen_heads=(0, 1))
    prior = pad_head_priors(hp, list(range(n)), T, dev)
    lg, a = model(X, mask, pos, prior=prior)
    check("multi-head attention is (B,T,Q)", tuple(a.shape) == (n, T, Q), str(tuple(a.shape)))
    for h, kind in [(0, "front"), (1, "back")]:
        w = torch.tensor(np.stack(recipe(n, T, kind)), dtype=torch.float32)
        want = w / w.sum(1, keepdim=True)
        check(f"head {h} attention == renormalised '{kind}' recipe",
              torch.allclose(a[:, :, h], want, atol=TOL),
              f"max|d|={float((a[:, :, h] - want).abs().max()):.2e}")
    corr01 = float(np.corrcoef(a[:, :, 0].detach().numpy().ravel(),
                               a[:, :, 1].detach().numpy().ravel())[0, 1])
    check("different recipes give DIFFERENT attention (|corr| < 0.9)", abs(corr01) < 0.9, f"corr={corr01:+.4f}")

    # C — the frozen heads' query rows must get no gradient.
    loss = lg.sum()
    loss.backward()
    gr = model.q_rest.grad
    check("frozen head query row has zero/absent gradient",
          gr is None or float(gr[0].abs().max()) == 0.0,
          "q_rest row 0 backs head 1 (frozen)")

    print("\nE. identical recipes = the same-recipe control")
    hp_same = [recipe(n, T, "front"), recipe(n, T, "front")]
    m_same = AttnPool(d, n_query=2, n_head=2, frozen_heads=(0, 1))
    a_s = m_same(X, mask, pos, prior=pad_head_priors(hp_same, list(range(n)), T, dev))[1]
    corr_same = float(np.corrcoef(a_s[:, :, 0].detach().numpy().ravel(),
                                  a_s[:, :, 1].detach().numpy().ravel())[0, 1])
    check("identical recipes give correlation 1.0", abs(corr_same - 1.0) < 1e-9, f"corr={corr_same:.9f}")

    print("\nF. a recipe that selected nothing must RAISE, not degrade")
    raised = False
    try:
        m_dead = AttnPool(d, n_query=2, n_head=2, frozen_heads=(0, 1))
        m_dead(X, mask, pos, prior=pad_head_priors([recipe(n, T, "dead"), recipe(n, T, "front")],
                                                   list(range(n)), T, dev))
    except SystemExit:
        raised = True
    check("zero-mass recipe raises", raised)

    print("\nG. a free head beside frozen heads still learns")
    hp2 = [recipe(n, T, "front"), recipe(n, T, "back"), None]
    trained = train_attn(states, y, tr, dev, seed=1, epochs=8, n_query=3, n_head=3,
                         head_priors=hp2, frozen_heads=(0, 1))
    moved = float(trained.q_rest[1].abs().max())                    # q_rest row 1 == query head 2 (free)
    check("free head's query moved off its init", moved > 0, f"max|q|={moved:.3e}")

    print("\nH. misuse is rejected loudly")
    for name, fn in [
        ("head_priors and prior_list together",
         lambda: train_attn(states, y, tr, dev, epochs=1, n_query=2, n_head=2,
                            head_priors=[recipe(n, T, "front"), None], frozen_heads=(0,),
                            prior_list=recipe(n, T, "flat"))),
        ("frozen head with a None recipe",
         lambda: train_attn(states, y, tr, dev, epochs=1, n_query=2, n_head=2,
                            head_priors=[None, None], frozen_heads=(0,))),
        ("frozen_heads index out of range",
         lambda: AttnPool(d, n_query=2, n_head=2, frozen_heads=(5,))),
    ]:
        got = False
        try:
            fn()
        except (SystemExit, ValueError):
            got = True
        check(name + " raises", got)

    print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
