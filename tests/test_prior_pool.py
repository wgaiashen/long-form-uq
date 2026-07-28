"""Grounding tests for S3 prior-init pooling (the `prior=` channel on AttnPool).

Each check is a NUMERIC limit that would catch a silent bug where an arm stops being what it claims:

  arm C (frozen prior), flat prior  -> attention is uniform  -> reduces EXACTLY to the uniform/mean pooler.
  arm C (frozen prior), known prior -> attention is the renormalised prior (a weighted mean, verifiable).
  arm D (annealed prior), beta=0    -> the log-prior term vanishes -> identical to the plain learned pooler.
  frozen-prior TRAINING              -> the query stays bit-identical to its init (only the head trains):
                                        without this, arm C could secretly be arm A (a learned query).
  batching                           -> a frozen-prior example's weights depend only on THAT example
                                        (the "prior is example-local" invariant, the basis of the method).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from attn_pool import AttnPool, pad_batch, pad_prior, train_attn  # noqa: E402

D, T, B = 8, 6, 4
TOL = 1e-6


def _states(seed=0):
    rng = np.random.RandomState(seed)
    return [rng.randn(T, D).astype(np.float32) for _ in range(B)]


def _weights(model, states, prior_list=None):
    """Return the attention weights `a` (B, T) for one forward pass over all states."""
    X, mask, pos = pad_batch(states, "cpu")
    prior = pad_prior(prior_list, X.shape[1], "cpu") if prior_list is not None else None
    with torch.no_grad():
        _, a = model(X, mask, pos, prior=prior)
    return a.numpy()


def test_flat_prior_frozen_equals_uniform():
    """arm C with an all-ones prior == the uniform (frozen-query) pooler: both put 1/T on every token."""
    states = _states()
    frozen = AttnPool(D, frozen_prior=True)
    uni = AttnPool(D, freeze_query=True)
    a_frozen = _weights(frozen, states, prior_list=[np.ones(T) for _ in range(B)])
    a_uni = _weights(uni, states)
    assert np.allclose(a_frozen, a_uni, atol=TOL)
    assert np.allclose(a_frozen, 1.0 / T, atol=TOL)          # genuinely uniform


def test_frozen_prior_is_renormalised_prior():
    """arm C attention == the prior renormalised over real tokens (a pure weighted mean, no query)."""
    states = _states()
    priors = [np.abs(np.random.RandomState(i).randn(T)) + 0.1 for i in range(B)]
    model = AttnPool(D, frozen_prior=True)
    a = _weights(model, states, prior_list=priors)
    for i, p in enumerate(priors):
        assert np.allclose(a[i], p / p.sum(), atol=TOL)


def test_annealed_beta_zero_is_noop():
    """arm D with beta=0: the log-prior term drops out -> identical weights to the plain learned pooler."""
    states = _states()
    priors = [np.abs(np.random.RandomState(i + 9).randn(T)) + 0.1 for i in range(B)]
    model = AttnPool(D, frozen_prior=False, beta=0.0)
    with torch.no_grad():                                    # give the query a non-trivial value
        model.q.copy_(torch.randn(D))
    a_prior = _weights(model, states, prior_list=priors)
    a_noprior = _weights(model, states, prior_list=None)
    assert np.allclose(a_prior, a_noprior, atol=TOL)


def test_frozen_training_leaves_query_untouched():
    """Training a frozen-prior pooler must NOT move the query (else arm C is secretly a learned query)."""
    states = _states(1)
    y = np.array([0.0, 1.0, 0.0, 1.0])
    priors = [np.abs(np.random.RandomState(i + 3).randn(T)) + 0.1 for i in range(B)]
    q0 = torch.zeros(D)
    model = train_attn(states, y, [0, 1, 2, 3], "cpu", seed=0, epochs=5, bs=2,
                       prior_list=priors, frozen_prior=True)
    assert torch.equal(model.q.detach(), q0)                 # query bit-identical to its init
    assert not model.q.requires_grad                         # and frozen


def test_prior_is_example_local():
    """A frozen-prior example's weights must not depend on the OTHER examples batched with it."""
    states = _states(2)
    priors = [np.abs(np.random.RandomState(i + 5).randn(T)) + 0.1 for i in range(B)]
    model = AttnPool(D, frozen_prior=True)
    a_full = _weights(model, states, prior_list=priors)
    a_solo = _weights(model, [states[0]], prior_list=[priors[0]])
    assert np.allclose(a_full[0], a_solo[0], atol=TOL)
