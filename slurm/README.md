# `slurm/`: Slurm job scripts

117 job scripts, plus two job-level Python helpers. They are a record of what was submitted to the
Slurm cluster, not a curated interface. As with the PBSPro directory, most name partitions and card
types specific to that cluster and will not run elsewhere.

This directory and `pbs/` are **not** a one-for-one mirror. Work was routed to whichever cluster
suited it: jobs needing an 80 GB card ran here, and cache-bound work ran where the caches were. Most
jobs therefore exist on one side only.

## The files worth opening

| file | why |
|---|---|
| `qwen_generate_full.sbatch` | the generation specification for the second base model: per-dataset budgets and decoding controls, and the explicit `--dtype fp32 --attn eager` that keeps a cache comparable rather than letting the default resolve to half precision. Several later scripts transcribe their settings from this one. |
| `wmodels_extract.sbatch` | the same for the additional model families. |
| `finish_partial_join.py`, `rebuild_and_assert_baseline.py` | job-level helpers, not application code: they make a long job survive a walltime kill and join partial output correctly. |

## Reading the rest

Start from the driver in `scripts/checks/` that produced the result you care about, then grep this
directory for its name to find the job that ran it. The scheduling is here; the method is there.
