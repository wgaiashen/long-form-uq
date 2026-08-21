"""W4 — HIERARCHICAL (two-level) attention pooling over a long-form generation.

The idea (the "move from per-token to per-segment"): a long answer is not a flat bag of tokens, it is a
sequence of SENTENCES, each of which carries a claim. So instead of one softmax over all T tokens, do it in
two levels:

    level 1 (WITHIN a segment)  : attend over the tokens of each sentence  -> one vector per sentence
    level 2 (ACROSS segments)   : attend over the sentence vectors         -> one vector per response
                                  -> linear head -> P(correct)

Why this might beat the flat pooler: the flat pooler must spend one softmax budget across every token in the
response, so a long response dilutes the few load-bearing tokens, and a single spiky token can dominate the
whole score. Splitting the budget means "which token matters IN THIS SENTENCE" is decided separately from
"which SENTENCE matters", which is closer to how a long-form answer actually fails (one bad claim among
several fine ones).

GROUNDING (the reduction properties this module must satisfy — these are unit-tested in
tests/test_hier_pool.py, following the project convention that every aggregation variant carries a numeric
correctness check, not just a deliverable):

  * ONE segment containing every token   -> level 2 is a softmax over a single item (= 1.0), so the module
                                            collapses to the FLAT attention pooler.
  * EVERY token its own segment          -> level 1 is a softmax over single items (= 1.0), so each segment
                                            vector IS its token, and again the module collapses to the FLAT
                                            attention pooler.

Both limits must reproduce the flat pooler EXACTLY (same q, same head). If either fails, the two-level
wiring is wrong. This mirrors `constant == plain MSP` for weighted-MSP.

Segment ids come from the SAME helper the weighted-MSP segment variant uses
(`luq.features.sar._token_sentence_ids`), so "sentence" means the same thing across both tracks.

CPU-friendly: operates on the cached L15 per-token states, no GPU needed.
"""

import numpy as np
import torch
import torch.nn as nn


def segment_softmax(scores, mask, seg, n_seg):
    """Softmax over tokens WITHIN each segment (a scatter-based, differentiable segment-wise softmax).

    scores, mask, seg : (B, T).  `seg` holds the segment index of each token (pad tokens may hold any
    value -- they are removed by `mask`).  Returns (a, flat_seg) where `a` is (B, T) attention weights that
    sum to 1 WITHIN each (batch, segment) group, and `flat_seg` is the flattened group index used again by
    the caller to build the segment vectors.

    Numerics: we subtract a per-segment max before exponentiating (the standard softmax stabilisation), and
    guard the denominator, so an empty/padded segment cannot produce NaN.
    """
    B, T = scores.shape
    dev = scores.device
    # flatten (batch, segment) into ONE group index so we can use scatter ops over a 1-D buffer
    flat_seg = (torch.arange(B, device=dev).unsqueeze(1) * n_seg + seg).reshape(-1)
    s = scores.masked_fill(mask == 0, float("-inf")).reshape(-1)

    # per-group max, for numerical stability
    seg_max = torch.full((B * n_seg,), float("-inf"), device=dev, dtype=s.dtype)
    seg_max = seg_max.scatter_reduce(0, flat_seg, s, reduce="amax", include_self=True)
    seg_max = torch.where(torch.isinf(seg_max), torch.zeros_like(seg_max), seg_max)

    e = torch.exp(s - seg_max[flat_seg]) * mask.reshape(-1)
    denom = torch.zeros(B * n_seg, device=dev, dtype=s.dtype).index_add_(0, flat_seg, e)
    a = e / torch.clamp(denom[flat_seg], min=1e-9)
    return a.reshape(B, T), flat_seg


class HierPool(nn.Module):
    """Two-level attention pooling: tokens -> sentences -> response -> logit.

    `q_tok` decides where to attend WITHIN a sentence; `q_seg` decides which SENTENCE matters. Both are
    initialised to zero, so training starts from uniform-within-sentence + uniform-across-sentences, i.e.
    exactly mean-pooling -- the same "start at the unsupervised prior" convention as AttnPool.

    freeze_tok / freeze_seg pin the corresponding level to uniform, which is what lets us ablate the two
    levels separately (is the win coming from picking tokens, or from picking sentences?).
    """

    def __init__(self, d, temperature=1.0, freeze_tok=False, freeze_seg=False):
        super().__init__()
        self.q_tok = nn.Parameter(torch.zeros(d))
        self.q_seg = nn.Parameter(torch.zeros(d))
        if freeze_tok:
            self.q_tok.requires_grad_(False)
        if freeze_seg:
            self.q_seg.requires_grad_(False)
        self.head = nn.Linear(d, 1)
        self.scale = d ** 0.5
        self.temperature = temperature

    def forward(self, X, mask, seg, n_seg):
        """X (B,T,d), mask (B,T), seg (B,T) long, n_seg = max segments in this batch.
        Returns (logit, a_tok, a_seg) so both levels of weights can be visualised."""
        B, T, d = X.shape

        # ---- level 1: attend over tokens within each segment -> segment vectors ----
        tok_scores = (X @ self.q_tok) / (self.scale * self.temperature)          # (B,T)
        a_tok, flat_seg = segment_softmax(tok_scores, mask, seg, n_seg)
        weighted = (a_tok.unsqueeze(-1) * X).reshape(-1, d)                      # (B*T, d)
        seg_vec = torch.zeros(B * n_seg, d, device=X.device, dtype=X.dtype)
        seg_vec = seg_vec.index_add_(0, flat_seg, weighted).reshape(B, n_seg, d)  # (B,S,d)

        # a segment is real iff it contains at least one real token
        seg_mask = torch.zeros(B * n_seg, device=X.device, dtype=X.dtype)
        seg_mask = seg_mask.index_add_(0, flat_seg, mask.reshape(-1)).reshape(B, n_seg)
        seg_mask = (seg_mask > 0).to(X.dtype)                                    # (B,S)

        # ---- level 2: attend over segment vectors -> one response vector ----
        seg_scores = (seg_vec @ self.q_seg) / (self.scale * self.temperature)    # (B,S)
        seg_scores = seg_scores.masked_fill(seg_mask == 0, float("-inf"))
        a_seg = torch.softmax(seg_scores, dim=1)                                 # (B,S)
        pooled = (a_seg.unsqueeze(-1) * seg_vec).sum(dim=1)                      # (B,d)
        return self.head(pooled).squeeze(-1), a_tok, a_seg


def pad_batch_seg(states_list, seg_list, device):
    """Pad states AND segment ids to (B, Tmax). Returns X, mask, seg, n_seg.

    Segment ids are re-based per example (0..k-1, contiguous) so `n_seg` stays as small as possible --
    the scatter buffers are B*n_seg wide, so a stray large id would waste memory.
    """
    d = states_list[0].shape[1]
    tmax = max(s.shape[0] for s in states_list)
    X = torch.zeros(len(states_list), tmax, d, dtype=torch.float32)
    mask = torch.zeros(len(states_list), tmax, dtype=torch.float32)
    seg = torch.zeros(len(states_list), tmax, dtype=torch.long)
    n_seg = 1
    for i, (s, sd) in enumerate(zip(states_list, seg_list)):
        t = s.shape[0]
        X[i, :t] = torch.from_numpy(np.ascontiguousarray(s))
        mask[i, :t] = 1.0
        ids = np.asarray(sd[:t], dtype=np.int64)
        _, contig = np.unique(ids, return_inverse=True)      # re-base to 0..k-1
        seg[i, :t] = torch.from_numpy(contig)
        n_seg = max(n_seg, int(contig.max()) + 1 if len(contig) else 1)
    return X.to(device), mask.to(device), seg.to(device), n_seg


def train_hier(states, seg_ids, y, tr_idx, device, seed=1, temperature=1.0, freeze_tok=False,
               freeze_seg=False, epochs=60, bs=32, lr=1e-3, weight_decay=1e-2):
    """Train HierPool with BCE on soft correctness labels -- the SAME loss/optimiser/epochs/batch as
    attn_pool.train_attn, so a HierPool-vs-AttnPool comparison is a pooling-only difference (the
    head-confound guard: never let the head or optimiser differ across compared rows)."""
    torch.manual_seed(seed)
    d = states[0].shape[1]
    model = HierPool(d, temperature=temperature, freeze_tok=freeze_tok, freeze_seg=freeze_seg).to(device)
    # queries get NO weight decay (they are the pooling distribution, not a capacity knob); head does.
    groups = [{"params": [model.q_tok, model.q_seg], "weight_decay": 0.0},
              {"params": list(model.head.parameters()), "weight_decay": weight_decay}]
    opt = torch.optim.Adam(groups, lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    idx_all = np.array(tr_idx)
    rng = np.random.RandomState(seed)
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(len(idx_all))
        for b in range(0, len(perm), bs):
            idx = idx_all[perm[b: b + bs]]
            X, mask, seg, n_seg = pad_batch_seg([states[i] for i in idx], [seg_ids[i] for i in idx], device)
            logit, _, _ = model(X, mask, seg, n_seg)
            loss = lossf(logit, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def hier_unc(model, states, seg_ids, te_idx, device, bs=64):
    """Per-example uncertainty (1 - sigmoid(logit)) -- same convention as attn_unc."""
    model.eval()
    preds = np.zeros(len(te_idx))
    with torch.no_grad():
        for b in range(0, len(te_idx), bs):
            idx = te_idx[b: b + bs]
            X, mask, seg, n_seg = pad_batch_seg([states[i] for i in idx], [seg_ids[i] for i in idx], device)
            logit, _, _ = model(X, mask, seg, n_seg)
            preds[b: b + len(idx)] = torch.sigmoid(logit).cpu().numpy()
    return 1.0 - preds
