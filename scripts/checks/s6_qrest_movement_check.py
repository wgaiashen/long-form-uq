"""Did q_rest MOVE during training? Reproduce the deterministic init (same seed train_attn uses),
re-train MH on one real cell, compare ||q_rest_final - q_rest_init|| per head. Decides finding vs artifact."""
import sys, numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/checks")
from attn_pool import AttnPool, train_attn
from aggregation_table import load_per_token
from xl_rungs import label_of

MODEL = "meta-llama/Meta-Llama-3.1-8B"; LAYER = 15; seed = 1; K = 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"
states, split, y, _, records = load_per_token(MODEL, "pubmed_qa", LAYER, label_of("pubmed_qa"))
tr = [i for i in range(len(states)) if split[i] == "train" and np.isfinite(y[i])][:256]   # subsample for speed
d = states[0].shape[1]
print(f"cell=pubmed_qa (subsample {len(tr)} train), K={K}, seed={seed}, epochs=60")

# reproduce train_attn's internal init EXACTLY (same seed, same constructor args, same order)
torch.manual_seed(seed)
m0 = AttnPool(d, temperature=1.0, n_query=K, n_head=K)
q_rest_init = m0.q_rest.detach().clone()
q_init = m0.q.detach().clone()
print("q_rest.requires_grad:", m0.q_rest.requires_grad, "| q_rest in a leaf param:", m0.q_rest.is_leaf)

# train (train_attn re-seeds to the same seed -> same init internally)
model = train_attn(states, y, tr, DEV, seed=seed, temperature=1.0, n_query=K, n_head=K)
q_rest_fin = model.q_rest.detach().cpu()
q_fin = model.q.detach().cpu()

print("\n=== q_rest MOVEMENT per extra head ===")
for h in range(K - 1):
    ni, nf = q_rest_init[h].norm().item(), q_rest_fin[h].norm().item()
    dn = (q_rest_fin[h] - q_rest_init[h]).norm().item()
    print(f"  head {h+1}: ||init||={ni:.4f}  ||final||={nf:.4f}  ||Δ||={dn:.4f}  (Δ/init = {dn/ni:.2f}x)")
print(f"primary q:  ||init||={q_init.norm().item():.4f} (should be 0)  ||final||={q_fin.norm().item():.4f}")
tot = (q_rest_fin - q_rest_init).norm().item()
print(f"\nq_rest total ||Δ|| = {tot:.4f}")
print("VERDICT:", "MOVED SUBSTANTIALLY -> collapse is a real finding" if tot > 0.1 else
      "~ZERO -> heads never trained -> RETRACT S6 finding")
