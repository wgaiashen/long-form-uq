"""The probe: a small supervised probe on the cached feature vectors.

Response-level SAPLMA is exactly this on the pooled hidden-state vector. At the
response level a linear probe performs about as well as a deep one, so it stays linear.

The judge's correctness label is GRADED (0..1), not binary, and PRR is scored on that
graded label. So we train on the SOFT label directly instead of thresholding it at 0.5:
a single linear logit -> sigmoid -> binary cross-entropy against the graded target.
sigmoid(logit) is pushed toward the judge's score, so the gradation is kept (0.6 and 1.0
are no longer both "correct"). This is the standard soft-label BCE and is Joe's setup.
The output is an UNCERTAINTY score = 1 - P(correct), higher = more uncertain.
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class SoftProbe:
    """A linear probe trained with soft-label BCE: sigmoid(w·x + b) ~ graded label.

    Stores the fitted feature scaler and the learned linear weights as plain numpy, so
    prediction is a one-line sigmoid and the object pickles without dragging in torch.
    """
    def __init__(self, scaler, w, b):
        self.scaler = scaler
        self.w = w          # (hidden,) learned weight vector
        self.b = b          # scalar bias

    def p_correct(self, X: np.ndarray) -> np.ndarray:
        """P(correct) = sigmoid(w·x + b) on standardised features."""
        z = self.scaler.transform(X) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))


def train_probe(X: np.ndarray, y: np.ndarray,
                epochs: int = 300, lr: float = 0.05,
                weight_decay: float = 1e-3, seed: int = 1) -> SoftProbe:
    """X: (n, hidden) features. y: GRADED correctness in [0, 1] (NOT thresholded).

    Trains a linear logit with BCE against the soft target: sigmoid(logit) is pushed
    toward the judge's graded score, keeping the gradation. Features are standardised
    first, which a linear probe usually needs. weight_decay is light L2, the analogue
    of LogisticRegression's built-in regulariser; it, lr and epochs are the knobs if a
    layer's PRR looks off. Full-batch on CPU over a couple of thousand examples is
    sub-second, so this stays the cheap, GPU-free step it was.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Standardise features (fit on train only); a linear probe usually needs it.
    scaler = StandardScaler().fit(X)

    torch.manual_seed(seed)  # reproducible weight init (project keeps seed=1)
    Xt = torch.tensor(scaler.transform(X), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)   # (n, 1) SOFT targets

    linear = torch.nn.Linear(Xt.shape[1], 1)
    # BCEWithLogitsLoss = sigmoid + BCE in one numerically stable step, and it accepts
    # continuous targets in [0, 1] — exactly the soft-label objective we want.
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(linear.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(linear(Xt), yt)          # BCE(sigmoid(logit), soft label)
        loss.backward()
        opt.step()

    # Keep the learned line as plain numpy so SoftProbe needs no torch to predict.
    w = linear.weight.detach().numpy().ravel()  # (hidden,)
    b = float(linear.bias.detach().numpy().ravel()[0])
    return SoftProbe(scaler, w, b)


def uncertainty(clf: SoftProbe, X: np.ndarray) -> np.ndarray:
    """Return 1 - P(correct): higher = more uncertain."""
    return 1.0 - clf.p_correct(np.asarray(X, dtype=np.float64))
