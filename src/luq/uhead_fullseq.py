"""UHead baseline, faithful to Hidden Failures (Joe Stacey's `full_sequence_uhead`).

This is the "train a small uncertainty head FROM SCRATCH" baseline, NOT the orphaned
pretrained `luh` checkpoint (that checkpoint will not load into any public or the `luh`;
`features/uhead.py` is the old pretrained-head attempt and stays only for reference). the
Hidden Failures repo (`Temp_robust_UQ_probes`) never loads a pretrained head -- it trains a
transformer-encoder head on the base model's own per-token hidden states. This module
re-implements that head and its training loop in our own thin spine.

WHAT THE HEAD IS (mirrors `luh/heads/full_seq_head.py::FullSeqHead`):
  per-token feature X_t (a hidden state at the middle layer, one per sequence position)
    -> proj MLP  (Linear -> LN -> GELU -> Dropout -> Linear -> LN -> GELU)
    -> add a learned "is this token in the generated output?" entity embedding
    -> a 1-2 layer TransformerEncoder over the WHOLE sequence (context + generated)
    -> masked MEAN over the GENERATED tokens only
    -> classifier MLP -> one logit
  sigmoid(logit) is the response-level UNCERTAINTY (the reference implementation trains it to predict 1 - correctness).

WHICH FEATURE (mirrors the `--probe_feature_extractor_setting hs_middle`):
  a single hidden layer, the "middle" = ceil(num_hidden_layers/2) - 1 (Llama-3.1-8B -> layer 15),
  over the full [prompt + generation] sequence. The transformer needs the context tokens as
  memory even though only the generated tokens are pooled, so the features span the whole
  sequence, not just the answer span.

FEATURE / MASK ALIGNMENT (verified against `feature_supervision.py` + `heads/utils.py`):
  the reference implementation reads the generation-time hidden states, which have length seq-1 (they are the states that
  PREDICT tokens 1..seq-1), and an output_mask that is also shifted by one (`output_mask[1:]`).
  We teacher-force the cached [prompt+gen] token ids through the frozen base and take all
  per-position states (length seq), then DROP THE LAST position -> length seq-1, the exact same
  "state that predicts the next token" set. A feature at index j is a generated token iff
  j >= len(prompt) - 1, which gives exactly the len(gen) generated positions the reference implementation pools.

WHY TRAIN ON CACHED STATES (a deviation in mechanism, not in maths): the reference implementation trains end-to-end with a
HuggingFace Trainer that re-runs the frozen base every step to recompute the states. The base is
frozen, so no gradient ever flows into it -- precomputing the states once and training the tiny
head on them is identical in expectation and vastly cheaper (it is exactly the project's
"cache features, retrain the cheap head for free" design). The one numerical deviation we make
explicit: the Trainer uses fp16 autocast for the head; we compute the head in fp32 (the head's
own `proj` already upcasts features to fp32), which is more stable for a reported baseline and is
a precision choice, not a tuning knob.
"""
import numpy as np
import torch
import torch.nn as nn


# The two head configurations the reference implementation sweeps in `scripts/architecture_ablations.sh`. head_dim, the
# transformer depth/width and dropout differ; the optimiser settings below are shared. These are
# the numbers verbatim -- a faithful baseline, not tuned by us.
VARIANTS = {
    "v1": dict(head_dim=768, n_layers=1, n_heads=16, dropout=0.05, num_train_epochs=6),
    "v2": dict(head_dim=768, n_layers=2, n_heads=4, dropout=0.20, num_train_epochs=7),
}
# Shared training hyperparameters (the `feature_supervision.py` TrainingArguments + the
# architecture_ablations flags): AdamW, linear schedule with warmup, gradient accumulation so the
# effective batch is train_batch_size * grad_accum = 1 * 4 = 4.
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
TRAIN_BATCH_SIZE = 1
GRAD_ACCUM = 4
SEED = 1  # project convention (ProbeDrift seed=1)


class FullSeqHead(nn.Module):
    """Faithful re-implementation of the `FullSeqHead` for one instance at a time.

    feature_dim = the base model's hidden size (we feed one hidden layer, so no layer concat).
    Processes a single (T, feature_dim) sequence -> one uncertainty logit. Batching is handled by
    the training loop as gradient accumulation over single instances, which matches the
    per-instance head loop (his `_compute_tensors` iterates the batch and pools each item alone).
    """

    def __init__(self, feature_dim: int, head_dim: int, n_layers: int, n_heads: int,
                 dropout: float):
        super().__init__()
        # proj: Linear -> LN -> GELU -> Dropout -> Linear -> LN -> GELU (the exact stack).
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, head_dim * 2),
            nn.LayerNorm(head_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim * 2, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
        )
        # 0 = context token, 1 = generated token. Added to every position's projected feature.
        self.entity_embedding = nn.Embedding(2, head_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=head_dim, nhead=n_heads, dropout=dropout, activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(self, X: torch.Tensor, output_mask: torch.Tensor) -> torch.Tensor:
        """X: (T, feature_dim) per-token states. output_mask: (T,) long, 1 = generated token.

        Returns a scalar tensor: the uncertainty logit for this instance.
        """
        features = self.proj(X.to(torch.float32))               # (T, head_dim); upcast like the reference implementation
        ent = self.entity_embedding(output_mask)                # (T, head_dim)
        out = (features + ent).unsqueeze(0)                     # (1, T, head_dim)
        # Single unpadded sequence, so there is no src_key_padding_mask (the mask is all-valid
        # here). Disable the cuDNN SDPA kernel to match his workaround for the same shapes; it only
        # changes which kernel computes the identical attention.
        with torch.backends.cuda.sdp_kernel(enable_cudnn=False):
            out = self.transformer_encoder(out)                # (1, T, head_dim)
        out = out.squeeze(0)                                    # (T, head_dim)
        m = output_mask.unsqueeze(-1).to(out.dtype)             # (T, 1)
        pooled = (out * m).sum(dim=0) / m.sum().clamp(min=1)    # masked mean over generated tokens
        return self.classifier(pooled).squeeze(-1)             # scalar logit


def _reinitialize_weights(module):
    """the `CausalLMWithUncertaintyLayer.reinitialize_weights`, applied after construction.

    Xavier-uniform for 2-D weights (Linear), uniform for 1-D weights (LayerNorm / embedding rows),
    zeros for biases. Skips positional-encoding weights (none here). Replicated for faithfulness.
    """
    name = getattr(module, "name", "")
    if hasattr(module, "weight") and module.weight is not None and "positional_encoding" not in name:
        if module.weight.ndim >= 2:
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.uniform_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)


def build_head(feature_dim: int, variant: str) -> FullSeqHead:
    """Construct a head for the named variant ('v1' or 'v2') and apply the weight re-init."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown uhead variant {variant!r}; choose from {sorted(VARIANTS)}")
    cfg = VARIANTS[variant]
    head = FullSeqHead(feature_dim, cfg["head_dim"], cfg["n_layers"], cfg["n_heads"],
                       cfg["dropout"])
    head.apply(_reinitialize_weights)
    return head


def train_head(head: FullSeqHead, feats_train, masks_train, y_train, variant: str,
               device: str = "cuda", log_every: int = 500) -> FullSeqHead:
    """Train the head to predict 1 - correctness with BCE, matching the optimiser settings.

    feats_train: list of (T_i, feature_dim) float arrays (per-token states, one per train example).
    masks_train: list of (T_i,) int arrays, 1 = generated token.
    y_train:     (n,) correctness in [0, 1]. The head's TARGET is 1 - correctness (uncertainty).
    """
    torch.manual_seed(SEED)
    cfg = VARIANTS[variant]
    epochs = cfg["num_train_epochs"]
    head = head.to(device).train()

    n = len(feats_train)
    targets = 1.0 - np.asarray(y_train, dtype=np.float32)      # uncertainty target (reference: 1 - metric)

    opt = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Effective batch = TRAIN_BATCH_SIZE * GRAD_ACCUM; one optimiser step per effective batch.
    steps_per_epoch = int(np.ceil(n / (TRAIN_BATCH_SIZE * GRAD_ACCUM)))
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(WARMUP_RATIO * total_steps)

    def lr_lambda(step):  # linear warmup then linear decay to 0 (HF "linear" scheduler)
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.BCEWithLogitsLoss()

    gen = torch.Generator().manual_seed(SEED)
    for epoch in range(epochs):
        order = torch.randperm(n, generator=gen).tolist()
        opt.zero_grad()
        running = 0.0
        for k, idx in enumerate(order):
            X = torch.from_numpy(np.asarray(feats_train[idx], dtype=np.float32)).to(device)
            mask = torch.from_numpy(np.asarray(masks_train[idx], dtype=np.int64)).to(device)
            target = torch.tensor(targets[idx], device=device)
            logit = head(X, mask)
            loss = loss_fn(logit, target) / GRAD_ACCUM
            loss.backward()
            running += loss.item() * GRAD_ACCUM
            if (k + 1) % GRAD_ACCUM == 0 or (k + 1) == n:
                torch.nn.utils.clip_grad_norm_(head.parameters(), MAX_GRAD_NORM)
                opt.step()
                sched.step()
                opt.zero_grad()
            if log_every and (k + 1) % log_every == 0:
                print(f"  epoch {epoch + 1}/{epochs}  {k + 1}/{n}  "
                      f"loss {running / (k + 1):.4f}  lr {sched.get_last_lr()[0]:.2e}", flush=True)
    return head.eval()


@torch.no_grad()
def predict(head: FullSeqHead, feats, masks, device: str = "cuda") -> np.ndarray:
    """Return sigmoid(logit) = uncertainty in [0, 1] for each example (higher = more uncertain)."""
    head = head.to(device).eval()
    out = []
    for X_arr, m_arr in zip(feats, masks):
        X = torch.from_numpy(np.asarray(X_arr, dtype=np.float32)).to(device)
        mask = torch.from_numpy(np.asarray(m_arr, dtype=np.int64)).to(device)
        out.append(torch.sigmoid(head(X, mask)).item())
    return np.array(out, dtype=np.float32)
