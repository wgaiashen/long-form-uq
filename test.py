# test.py — does the pipeline work? data -> generate -> reach a hidden state

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from probe_drift import get_datasets

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="cuda")

train_ds, eval_ds = get_datasets(eval_dataset="sciq", ood_setting="ID", batch_size=1)
print("train size:", len(train_ds.x))
print("first prompt:\n", train_ds.x[0][:300])

for i in range(10):
    inputs = tok(train_ds.x[i], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=20,
                         output_hidden_states=True, return_dict_in_generate=True)
    gen = tok.decode(out.sequences[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"[{i}] gold={train_ds.y[i]!r}  gen={gen!r}")

