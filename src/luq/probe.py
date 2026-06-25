"""The probe: a small supervised probe on the cached feature vectors.

Response-level SAPLMA is exactly this on the pooled hidden-state vector.

The judge's correctness label is GRADED (0..1), not binary, and PRR is scored on that
graded label. So we train on the SOFT label directly instead of thresholding it at 0.5:
sigmoid(logit) -> binary cross-entropy against the graded target, so the gradation is
kept (0.6 and 1.0 are no longer both "correct"). This is the standard soft-label BCE and
is Joe's setup. The output is an UNCERTAINTY score = 1 - P(correct), higher = more uncertain.

Two probe ARCHITECTURES share that soft-label objective:

  * `train_probe_mlp` / `MLPProbe` — the 4-layer MLP (256 -> 128 -> 64 -> 1, ReLU, sigmoid).
    This is the `saplma` method: a FAITHFUL reproduction of Azaria & Mitchell, run as published
    (5 epochs, batch 32, no weight decay, raw features). It is a BASELINE, so it is NOT tuned —
    see `train_probe_mlp` for why a faithful baseline that overfits is still reported as-is.
  * `train_probe` / `SoftProbe` — a single LINEAR logit (logistic regression). Used by our own
    `linear` probe and `ptrue` (validation-tuned, standardised features) AND by the `lookback`
    baseline (Chuang's logistic regression: `standardize=False`, default-strength L2, NOT tuned).

Which method uses what is wired in `scripts/03_probe.py` METHOD_SPEC (per-method hyperparameters).
A `standardize` toggle lets a method skip feature scaling to stay faithful to its source paper
(SAPLMA and lookback use raw features; our linear/ptrue standardise). Both probe objects expose
`.p_correct(X)` (and handle `scaler=None`), so `uncertainty(clf, X)` works on either.
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
        self.scaler = scaler  # fitted StandardScaler, or None when trained on raw features
        self.w = w          # (hidden,) learned weight vector
        self.b = b          # scalar bias

    def p_correct(self, X: np.ndarray) -> np.ndarray:
        """P(correct) = sigmoid(w·x + b). Uses raw features when scaler is None."""
        X = np.asarray(X, dtype=np.float64)
        Xs = X if self.scaler is None else self.scaler.transform(X)
        z = Xs @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))


def train_probe(X: np.ndarray, y: np.ndarray,
                epochs: int = 300, lr: float = 0.05,
                weight_decay: float = 1e-3, seed: int = 1,
                standardize: bool = True) -> SoftProbe:
    """X: (n, hidden) features. y: GRADED correctness in [0, 1] (NOT thresholded).

    Trains a linear logit (= logistic regression) with BCE against the soft target:
    sigmoid(logit) is pushed toward the judge's graded score, keeping the gradation.
    `weight_decay` is L2, the analogue of LogisticRegression's built-in regulariser; it,
    `lr` and `epochs` are the knobs (set per-method in 03_probe's METHOD_SPEC). Full-batch
    on CPU over a couple of thousand examples is sub-second.

    `standardize=True` fits a StandardScaler on the train features (what our own `linear` and
    `ptrue` methods use). `standardize=False` trains on RAW features — used by the `lookback`
    baseline to stay faithful to Chuang et al., whose logistic regression takes no scaler.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Optional standardisation (fit on train only). None => raw features (paper-faithful).
    scaler = StandardScaler().fit(X) if standardize else None
    Xin = scaler.transform(X) if scaler is not None else X

    torch.manual_seed(seed)  # reproducible weight init (project keeps seed=1)
    Xt = torch.tensor(Xin, dtype=torch.float32)
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


class MLPProbe:
    """The 4-layer SAPLMA MLP (Azaria & Mitchell), trained with soft-label BCE.

    Like SoftProbe, it keeps the fitted scaler and the learned weights as plain numpy and
    runs its forward pass in numpy, so it pickles and predicts without dragging in torch.
    `weights[i]` has shape (in, out) and `biases[i]` has shape (out,), layer by layer; the
    last layer outputs a single logit, the earlier ones use a ReLU.
    """
    def __init__(self, scaler, weights, biases):
        self.scaler = scaler     # fitted StandardScaler, or None when trained on raw features
        self.weights = weights   # list of (in, out) numpy arrays, input layer first
        self.biases = biases     # list of (out,) numpy arrays

    def p_correct(self, X: np.ndarray) -> np.ndarray:
        """P(correct) = sigmoid(MLP(x)). Uses raw features when scaler is None."""
        X = np.asarray(X, dtype=np.float64)
        a = X if self.scaler is None else self.scaler.transform(X)
        # Hidden layers: linear then ReLU. The final layer is a bare logit (no ReLU).
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            a = np.maximum(0.0, a @ W + b)
        z = a @ self.weights[-1] + self.biases[-1]
        return (1.0 / (1.0 + np.exp(-z))).ravel()


def train_probe_mlp(X: np.ndarray, y: np.ndarray,
                    hidden_sizes: tuple = (256, 128, 64),
                    epochs: int = 5, lr: float = 1e-3, weight_decay: float = 0.0,
                    batch_size: int = 32, seed: int = 1,
                    standardize: bool = False) -> MLPProbe:
    """SAPLMA: a FAITHFUL reproduction of Azaria & Mitchell, run as published — this is a
    BASELINE and is NOT tuned by us.

    Recipe from the released code (`saplma-ref/classify_sentences.py:133-141`, the
    sisinflab/HidingInTheHiddenStates repo that reproduces A&M):
      - architecture: Dense 256/128/64 (ReLU) -> 1 (sigmoid);
      - epochs=5, batch_size=32  (both EXPLICIT in the code);
      - optimizer Adam with lr = the Keras framework DEFAULT (A&M did not set it explicitly;
        that default is ~1e-3, which we pass here);
      - NO weight decay / L2 / dropout; RAW features (no scaler).
    A&M control overfitting purely via few epochs + minibatch-SGD noise (no explicit
    regulariser). We keep exactly that.

    Two deliberate, project-wide deviations (both exactly as Joe did in Hidden Failures, so
    every method is comparable under one yardstick):
      (1) train on the GRADED 0..1 label via BCEWithLogitsLoss, not a binary 0/1 label (PRR
          is graded);
      (2) one seeded run instead of A&M's 20-rerun metric average (we need one deployable probe;
          seeding the init AND the minibatch shuffle keeps reruns byte-identical, which
          `scripts/reproduce.py` relies on).

    This is a baseline: we do NOT tune `epochs`/`weight_decay` to improve PRR or reduce
    overfitting. With 3584-dim hidden states and 1800 train examples (p >> n) it may still
    overfit; if so, that is a reportable property of SAPLMA on our setup, not something to tune
    away (tuning would make it our method, not A&M's). Report the train/test gap honestly.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # A&M use raw features (standardize defaults False). None => no scaling at predict time.
    scaler = StandardScaler().fit(X) if standardize else None
    Xin = scaler.transform(X) if scaler is not None else X

    torch.manual_seed(seed)  # seeds the Linear weight init (reproducible)
    Xt = torch.tensor(Xin, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)   # (n, 1) SOFT targets

    # Build Linear -> ReLU -> ... -> Linear(., 1): the 256/128/64 -> 1 SAPLMA stack.
    layers = []
    prev = Xt.shape[1]
    for h in hidden_sizes:
        layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
        prev = h
    layers += [torch.nn.Linear(prev, 1)]     # final bare logit
    net = torch.nn.Sequential(*layers)

    # BCEWithLogitsLoss = sigmoid + BCE in one stable step; accepts continuous targets.
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # A&M's minibatch SGD (batch 32), the implicit regulariser. A separate seeded generator
    # drives the per-epoch shuffle so the whole run stays deterministic / byte-identical.
    n = Xt.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx]), yt[idx])   # BCE(sigmoid(mlp), soft label)
            loss.backward()
            opt.step()

    # Pull each Linear's weights out as numpy (transpose to (in, out) for `x @ W`), so the
    # MLPProbe predicts without torch — same torch-free prediction as SoftProbe.
    weights, biases = [], []
    for layer in net:
        if isinstance(layer, torch.nn.Linear):
            weights.append(layer.weight.detach().numpy().T.copy())   # (in, out)
            biases.append(layer.bias.detach().numpy().copy())        # (out,)
    return MLPProbe(scaler, weights, biases)


def uncertainty(clf, X: np.ndarray) -> np.ndarray:
    """Return 1 - P(correct): higher = more uncertain. Works on SoftProbe or MLPProbe."""
    return 1.0 - clf.p_correct(np.asarray(X, dtype=np.float64))
