"""UHead reproduction that runs the reference `luh` code (Hidden Failures, Shelmanov et al.).

Decision (2026-07-01, with the supervisor): reproduce the paper's headline Uhead baseline
(Table 1), NOT a reimplementation. The paper's Uhead (Appendix A) is a transformer probe over
**attention maps + token probabilities** (`_FEATURE_EXTRACTORS['uhead']` = token_probabilities
top-4 + basic_attention all-layers, attn_history_sz 2), fed into the `full_sequence_uhead`
transformer head (`luh/heads/full_seq_head.py::FullSeqHead`), with the head hyper-parameters from
`scripts/architecture_ablations.sh` (v1: dim768/1L/16H/drop0.05/6ep; v2: dim768/2L/4H/drop0.2/7ep).

To be faithful we import and run HIS OWN extractor + head classes rather than re-implementing them
(the project rule: verify against the authors' code, and here we use it directly). His `luh`
package eagerly imports TensorFlow (an unrelated Keras SAPLMA head at package import), so we stub
TF the same way `scripts/checks/alignscore_vs_authors.py` does. His full generation/lm_polygraph
engine will NOT run under our newer transformers, but the extractors and head are pure torch and
only need an `llm_outputs` dict; we build that dict from a teacher-forced forward over our cached
token ids (greedy generation is deterministic, so teacher-forcing reproduces the exact attentions
and logits his generate() produced).

WHAT WE MATCH vs HIS RECIPE:
  * extractor + head: his exact classes (imported).
  * head hparams: his architecture_ablations values (per variant).
  * training: his `feature_supervision.py` TrainingArguments -- AdamW, lr 2e-4, wd 0.1, linear
    warmup_ratio 0.1, max_grad_norm 1.0, gradient_accumulation_steps 4, per-device batch 1
    (-> effective batch 4), fp16 autocast + GradScaler (his fp16=True), and his xavier/uniform
    weight re-init. Target = 1 - correctness, BCEWithLogits. Base model in EVAL mode (his code
    asserts base.training == False), fp32 + eager (he passes no torch_dtype; --attn eager), so the
    attentions/logits equal his.
  * training signal: AlignScore (the paper's primary correctness function for Table 1). Pass the
    AlignScore label field; the judge is a secondary column.

DEVIATIONS (documented, honest):
  * we train the head on features cached from a teacher-forced forward instead of his end-to-end
    Trainer that re-runs the frozen base each step. The base is frozen and in eval, so this is
    identical in expectation (no gradient into the base) -- our standard cache-then-retrain design.
  * `--max-seq` caps the sequence for the longest XSum articles (his generate has no such cap); we
    keep all generated tokens and the most recent context, matching features/lookback.py's bound.
"""
import sys
import types
import importlib.machinery
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .uhead_fullseq import _reinitialize_weights  # his reinit rule (replicated once, reused here)

REFERENCE_REPO = Path("/vol/gpudata/gs925-msc_project/Temp_robust_UQ_probes")

# His architecture_ablations.sh uhead configs (the head hyper-parameters).
VARIANTS = {
    "v1": dict(head_dim=768, n_layers=1, n_heads=16, dropout=0.05, num_train_epochs=6),
    "v2": dict(head_dim=768, n_layers=2, n_heads=4, dropout=0.20, num_train_epochs=7),
}
# The paper's `uhead` feature setting (run_polygraph.py::_FEATURE_EXTRACTORS['uhead']).
UHEAD_FEATURE = [
    {"name": "luh.feature_extractors.token_probabilities", "top_n": 4},
    {"name": "luh.feature_extractors.basic_attention", "layer_nums": "all",
     "attn_history_sz": 2, "pool": False, "offset": 0},
]
# Shared training hyperparameters (feature_supervision.py TrainingArguments + architecture_ablations).
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
GRAD_ACCUM = 4
TRAIN_BATCH_SIZE = 1
SEED = 1


class _AttrDict(dict):
    """dict that also allows attribute access, so it works both as `cfg.name` (how his loaders read
    it) and as `**cfg` (how his extractor constructors read it) -- what OmegaConf provides in his
    pipeline, without adding the omegaconf dependency."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def _stub_tensorflow():
    """Register a no-op tensorflow so `import luh` works without TF (his package imports a Keras
    SAPLMA head at load). Identical to scripts/checks/alignscore_vs_authors.py. No effect on the
    torch extractors/head we use."""
    if "tensorflow" in sys.modules:
        return
    tf = types.ModuleType("tensorflow")
    tf.config = types.SimpleNamespace(
        optimizer=types.SimpleNamespace(set_jit=lambda *a, **k: None),
        set_visible_devices=lambda *a, **k: None,
    )
    tf.__version__ = "0.0.0-stub"
    keras = types.ModuleType("tensorflow.keras")
    models = types.ModuleType("tensorflow.keras.models")
    layers = types.ModuleType("tensorflow.keras.layers")
    models.Sequential = object
    layers.Dense = object
    keras.models, keras.layers = models, layers
    tf.keras = keras
    for name, mod in [("tensorflow", tf), ("tensorflow.keras", keras),
                      ("tensorflow.keras.models", models), ("tensorflow.keras.layers", layers)]:
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = mod


def build_uhead(base_model, variant: str):
    """Construct his FullSeqHead + combined uhead feature extractor for the given variant, using
    his AutoUncertaintyHead.from_config. Applies his xavier/uniform weight re-init."""
    if str(REFERENCE_REPO) not in sys.path:
        sys.path.insert(0, str(REFERENCE_REPO))
    _stub_tensorflow()
    from luh import AutoUncertaintyHead  # noqa: E402  (import here so the TF stub is in place)

    cfg = VARIANTS[variant]
    config = _AttrDict(
        head_type="full_sequence_uhead",
        feature_extractor=[_AttrDict(**fe) for fe in UHEAD_FEATURE],
        uncertainty_head=_AttrDict(head_dim=cfg["head_dim"], n_layers=cfg["n_layers"],
                                   n_heads=cfg["n_heads"], dropout=cfg["dropout"]),
    )
    head = AutoUncertaintyHead.from_config(config, base_model=base_model)
    head.apply(_reinitialize_weights)
    return head


@torch.no_grad()
def build_llm_io(model, tok, record: dict, max_seq: int = 2048):
    """Build the (llm_inputs, llm_outputs) dicts his extractors expect, from ONE teacher-forced
    forward over [prompt + gen] token ids. Reproduces the structure HF `generate` hands his code:
      llm_outputs['attentions'] : length G tuple; [0] = prefill per-layer (1,H,P,P); [k] for
                                  k=1..G-1 = per-layer (1,H,1,P+k) for query position P+k-1.
      llm_outputs['scores']     : length G list; scores[k] = logits predicting gen token k
                                  (teacher-forced logits at position P-1+k), each (1, vocab).
      llm_outputs['sequences']  : (1, S) full ids ; ['context_lengths'] = tensor([P]) ;
      llm_outputs['full_attention_mask'] : ones(1, S).
      llm_inputs['attention_mask'] : ones(1, P) (context mask) ;
      llm_inputs['output_mask']    : (1, S-1) long, his output_mask[1:] (1 iff index >= P-1).
    Returns (llm_inputs, llm_outputs, n_gen).
    """
    prompt_ids = list(record["prompt_token_ids"])
    gen_ids = list(record["gen_token_ids"])
    G = len(gen_ids)
    if len(prompt_ids) + G > max_seq:                 # bound long sequences (see module docstring)
        keep = max(1, max_seq - G)
        prompt_ids = prompt_ids[-keep:]
    P = len(prompt_ids)
    full_ids = prompt_ids + gen_ids
    S = P + G
    ids = torch.tensor([full_ids], device=model.device)
    out = model(ids, output_attentions=True, output_hidden_states=False)
    attn = out.attentions                              # tuple over layers, each (1, H, S, S)
    logits = out.logits                                # (1, S, vocab)
    L = len(attn)

    # Attentions in generate-step layout (length G).
    attentions = [tuple(attn[l][:, :, :P, :P] for l in range(L))]          # prefill
    for k in range(1, G):
        q = P + k - 1
        attentions.append(tuple(attn[l][:, :, q:q + 1, :q + 1] for l in range(L)))
    # Scores: one per generated token; scores[k] predicts gen token k (position P-1+k).
    scores = [logits[:, P - 1 + k, :] for k in range(G)]

    output_mask = torch.zeros(S, device=model.device)
    output_mask[P:] = 1
    output_mask = output_mask[1:].to(torch.long).unsqueeze(0)              # (1, S-1)
    assert int(output_mask.sum()) == G, (int(output_mask.sum()), G)

    llm_outputs = {
        "attentions": attentions,
        "scores": scores,
        "sequences": ids,
        "context_lengths": torch.tensor([P], device=model.device),
        "full_attention_mask": torch.ones(1, S, device=model.device, dtype=torch.long),
    }
    llm_inputs = {
        "attention_mask": torch.ones(1, P, device=model.device, dtype=torch.long),
        "output_mask": output_mask,
    }
    return llm_inputs, llm_outputs, G


@torch.no_grad()
def extract_features(head, llm_inputs, llm_outputs):
    """Run HIS combined uhead feature extractor -> (T, F) fp16 features (cpu), the output_mask (T,)
    and the head's attention mask (T,), all detached for caching. T = S-1, F = 4 + 2*L*H."""
    feats = head.feature_extractor(llm_inputs, llm_outputs)   # (1, T, F)
    x_attn_mask = head._get_attn_mask(llm_inputs, llm_outputs)  # full_attention_mask[:,1:] -> (1,T)
    X = feats[0].to(torch.float16).cpu().numpy()
    om = llm_inputs["output_mask"][0].to(torch.long).cpu().numpy()
    am = x_attn_mask[0].to(torch.long).cpu().numpy()
    return X, om, am


def _head_logit(head, X, om, am, device):
    """One forward of his head on precomputed features -> scalar uncertainty logit.
    Calls his `_compute_tensors(llm_inputs, X, X_attn_mask)` directly, bypassing re-extraction."""
    Xg = torch.from_numpy(np.asarray(X, np.float32)).to(device).unsqueeze(0)   # (1, T, F)
    om_g = torch.from_numpy(np.asarray(om, np.int64)).to(device).unsqueeze(0)  # (1, T)
    am_g = torch.from_numpy(np.asarray(am, np.int64)).to(device).unsqueeze(0)  # (1, T)
    out = head._compute_tensors({"output_mask": om_g}, Xg, am_g)               # (1, 1, 1)
    return out.reshape(())


def train_head(head, feats, oms, ams, y_train, variant: str, device: str = "cuda",
               precision: str = "fp16", log_every: int = 500):
    """Train his head to predict 1 - correctness (BCE), matching his TrainingArguments.

    feats/oms/ams: lists of per-example (X, output_mask, attn_mask) arrays (train split).
    y_train: (n,) correctness in [0, 1]. precision fp16 = his fp16=True (autocast + GradScaler).
    """
    torch.manual_seed(SEED)
    epochs = VARIANTS[variant]["num_train_epochs"]
    head = head.to(device).train()
    n = len(feats)
    targets = 1.0 - np.asarray(y_train, dtype=np.float32)

    opt = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = int(np.ceil(n / (TRAIN_BATCH_SIZE * GRAD_ACCUM)))
    total_steps = steps_per_epoch * epochs
    warmup = int(WARMUP_RATIO * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = (precision == "fp16" and device == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)          # torch >= 2.4 API
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)             # older-torch fallback

    gen = torch.Generator().manual_seed(SEED)
    for epoch in range(epochs):
        order = torch.randperm(n, generator=gen).tolist()
        opt.zero_grad()
        running = 0.0
        for k, idx in enumerate(order):
            target = torch.tensor(targets[idx], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logit = _head_logit(head, feats[idx], oms[idx], ams[idx], device)
                loss = loss_fn(logit.to(torch.float32), target) / GRAD_ACCUM
            scaler.scale(loss).backward()
            running += loss.item() * GRAD_ACCUM
            if (k + 1) % GRAD_ACCUM == 0 or (k + 1) == n:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(head.parameters(), MAX_GRAD_NORM)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
            if log_every and (k + 1) % log_every == 0:
                print(f"  epoch {epoch + 1}/{epochs}  {k + 1}/{n}  loss {running / (k + 1):.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
    return head.eval()


@torch.no_grad()
def predict(head, feats, oms, ams, device: str = "cuda") -> np.ndarray:
    """sigmoid(logit) uncertainty in [0, 1] per example (higher = more uncertain)."""
    head = head.to(device).eval()
    return np.array([torch.sigmoid(_head_logit(head, X, om, am, device)).item()
                     for X, om, am in zip(feats, oms, ams)], dtype=np.float32)
