"""The probe: a small supervised classifier on the cached feature vectors.

Response-level SAPLMA is exactly this on the pooled hidden-state vector. At the
response level a linear probe performs about as well as a deep one, so start linear.
The output is an UNCERTAINTY score = 1 - P(correct), higher = more uncertain.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_probe(X: np.ndarray, y: np.ndarray):
    """X: (n, hidden) features. y: correctness.

    TODO:
      - The graded judge label is 0..1; LogisticRegression needs binary targets, so
        threshold it (the >= 0.5 below) or switch to a regressor. Decide and note it.
      - Standardising features first usually helps a linear probe.
    """
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(X, (np.asarray(y) >= 0.5).astype(int))
    return clf


def uncertainty(clf, X: np.ndarray) -> np.ndarray:
    """Return 1 - P(correct): higher = more uncertain."""
    p_correct = clf.predict_proba(X)[:, 1]
    return 1.0 - p_correct
