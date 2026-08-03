# Canonical v1 cache manifest — sha256

**Written 2026-08-03.** Tonight the med_quad v2 rsync landed and the instruction was to verify the
canonical v1 record was not overwritten. **No recorded sha256 existed**, so "unchanged" could only
rest on an mtime. That is the gap this file closes: presence and mtime lie, a hash does not.

Regenerate/compare with:
```
cd msc-project-gs925 && sha256sum -c prereg/CANONICAL_v1_MANIFEST.sha256
```

| bytes | sha256 | file |
|---|---|---|
| 42524934 | `f711cacf7536829ff041119252924c5b35bbccba08d94e62ffd1dc86478ccca2` | `cache/records/meta-llama_Meta-Llama-3.1-8B__cnn_dailymail__ID.jsonl` |
| 32861188 | `5cd5aee16f0e7c107dea88cc68f21d868fb75cf6366d96fa2eedeabef72eb68c` | `cache/records/meta-llama_Meta-Llama-3.1-8B__med_quad__ID.jsonl` |
| 20684215 | `986b964ddc3646323dc8354fb4f2e85e6a0c674a72c8af009008bd0868e3526b` | `cache/records/meta-llama_Meta-Llama-3.1-8B__pubmed_qa__ID.jsonl` |
| 5730348 | `df984792c251f4c35bc21bf4c95cc59e91abf178f1250e92ccb662a84ec984d5` | `cache/records/meta-llama_Meta-Llama-3.1-8B__samsum__ID.jsonl` |
| 5884420 | `f6f0f8f0354a2ba3bcd94a6bf869f135ebf4fd382c7d741aec0b80ef7adef8f7` | `cache/records/meta-llama_Meta-Llama-3.1-8B__sciq__ID.jsonl` |
| 9974371 | `8e28a331e09add817555e5603fb3266c5de2a0ca68b15e9cdc86f8d6c3d9d476` | `cache/records/meta-llama_Meta-Llama-3.1-8B__trivia_qa__ID.jsonl` |
| 26028834 | `6aa50902e9c3a6703a31d24a51b7d883d979d5f7d0e85799500f5f8d332c88e6` | `cache/records/meta-llama_Meta-Llama-3.1-8B__xsum__ID.jsonl` |
| 4219884 | `75565879561c68e18c1b6e72630f569af2f55f250a53d84c1509d6009ceabeb7` | `cache/asqa_rp12/records/meta-llama_Meta-Llama-3.1-8B__asqa__ID.jsonl` |
| 17671790 | `3dcca0e6c670a723034e696cbdc4dc9e86d64be5d8ea0e5e1fb4b5870893264d` | `cache/expertqa_rp12/records/meta-llama_Meta-Llama-3.1-8B__expertqa__ID.jsonl` |
| 1775876 | `743f083c55de0d04a0813f9e796a6e5e2cf2df72c0e7fe04327d992d04b51a16` | `cache/factscore_rp12/records/meta-llama_Meta-Llama-3.1-8B__factscore__ID.jsonl` |
