# Canonical v1 cache manifest — sha256

**Written 2026-08-03.** Tonight the med_quad v2 rsync landed and the instruction was to verify the
canonical v1 record was not overwritten. **No recorded sha256 existed**, so "unchanged" could only
rest on an mtime. That is the gap this file closes: presence and mtime lie, a hash does not.

Regenerate/compare with:
```
sha256sum -c scripts/checks/CANONICAL_v1_MANIFEST.sha256
```

| bytes | sha256 | file |
|---|---|---|
| 43013351 | `0660de832c36c6ecfef71fda6533455be55bfbd9d590b8788e7c11637a2b0ad5` | `cache/records/meta-llama_Meta-Llama-3.1-8B__cnn_dailymail__ID.jsonl` |
| 33092735 | `3bc377452be78f13704bdc026a84b74fc7a231e8519fde22acf9c074e65638df` | `cache/records/meta-llama_Meta-Llama-3.1-8B__med_quad__ID.jsonl` |
| 20684215 | `986b964ddc3646323dc8354fb4f2e85e6a0c674a72c8af009008bd0868e3526b` | `cache/records/meta-llama_Meta-Llama-3.1-8B__pubmed_qa__ID.jsonl` |
| 5961443 | `596fa42ec8ac79aba21a56a307650e7872fb0c3783fc3b99d9c421ecab16b987` | `cache/records/meta-llama_Meta-Llama-3.1-8B__samsum__ID.jsonl` |
| 5884420 | `f6f0f8f0354a2ba3bcd94a6bf869f135ebf4fd382c7d741aec0b80ef7adef8f7` | `cache/records/meta-llama_Meta-Llama-3.1-8B__sciq__ID.jsonl` |
| 9974371 | `8e28a331e09add817555e5603fb3266c5de2a0ca68b15e9cdc86f8d6c3d9d476` | `cache/records/meta-llama_Meta-Llama-3.1-8B__trivia_qa__ID.jsonl` |
| 26028834 | `6aa50902e9c3a6703a31d24a51b7d883d979d5f7d0e85799500f5f8d332c88e6` | `cache/records/meta-llama_Meta-Llama-3.1-8B__xsum__ID.jsonl` |
| 4341760 | `a0167579f4954fd666261eb0a3dba0b2e56b35b919792814bde7d6e3256d07f7` | `cache/asqa_rp12/records/meta-llama_Meta-Llama-3.1-8B__asqa__ID.jsonl` |
| 17931184 | `fa49d064c7c8f4e40ff19250d35e012dbf480efcbaa9e3e4c5851b64f73c24e8` | `cache/expertqa_rp12/records/meta-llama_Meta-Llama-3.1-8B__expertqa__ID.jsonl` |
| 1840040 | `af676ac70aaa9f202683a03d110e7c034561fcb50e3eabbf459da2dd634ed8fb` | `cache/factscore_rp12/records/meta-llama_Meta-Llama-3.1-8B__factscore__ID.jsonl` |

---

## AMENDMENT — 2026-08-04: hashes refreshed after a deliberate change
**Reason:** merged the unsupervised-P(True) sidecars from DoC (additive only: ptrue_unsup, ptrue_unsup_mass, ptrue_unsup_model; judge labels verified bit-identical against the backup)
The table above now holds the CURRENT hashes. The files below changed; their previous hashes are recorded here so a `sha256sum -c` failure against an older copy can still be resolved.
**Pre-change copies:** `BACKUP_records_pre_ptrueunsup_2026-08-04` (verified against the OLD hashes below before the change was made).

| file | old sha256 | new sha256 |
|---|---|---|
| `cache/records/meta-llama_Meta-Llama-3.1-8B__cnn_dailymail__ID.jsonl` | `f711cacf7536829ff041119252924c5b35bbccba08d94e62ffd1dc86478ccca2` | `0660de832c36c6ecfef71fda6533455be55bfbd9d590b8788e7c11637a2b0ad5` |
| `cache/records/meta-llama_Meta-Llama-3.1-8B__med_quad__ID.jsonl` | `5cd5aee16f0e7c107dea88cc68f21d868fb75cf6366d96fa2eedeabef72eb68c` | `3bc377452be78f13704bdc026a84b74fc7a231e8519fde22acf9c074e65638df` |
| `cache/records/meta-llama_Meta-Llama-3.1-8B__samsum__ID.jsonl` | `df984792c251f4c35bc21bf4c95cc59e91abf178f1250e92ccb662a84ec984d5` | `596fa42ec8ac79aba21a56a307650e7872fb0c3783fc3b99d9c421ecab16b987` |
| `cache/asqa_rp12/records/meta-llama_Meta-Llama-3.1-8B__asqa__ID.jsonl` | `75565879561c68e18c1b6e72630f569af2f55f250a53d84c1509d6009ceabeb7` | `a0167579f4954fd666261eb0a3dba0b2e56b35b919792814bde7d6e3256d07f7` |
| `cache/expertqa_rp12/records/meta-llama_Meta-Llama-3.1-8B__expertqa__ID.jsonl` | `3dcca0e6c670a723034e696cbdc4dc9e86d64be5d8ea0e5e1fb4b5870893264d` | `fa49d064c7c8f4e40ff19250d35e012dbf480efcbaa9e3e4c5851b64f73c24e8` |
| `cache/factscore_rp12/records/meta-llama_Meta-Llama-3.1-8B__factscore__ID.jsonl` | `743f083c55de0d04a0813f9e796a6e5e2cf2df72c0e7fe04327d992d04b51a16` | `af676ac70aaa9f202683a03d110e7c034561fcb50e3eabbf459da2dd634ed8fb` |
