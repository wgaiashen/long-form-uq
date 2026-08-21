# Running this project on the RCS HPC (CX3 Phase 2)

The RCS cluster is a **second, independent** place to run jobs, added alongside the
Department of Computing (DoC) Slurm cluster. The Python pipeline is identical on both;
only the submission wrapper and a few paths differ. The DoC/Slurm workflow is unchanged
— `slurm/*.sbatch` still works exactly as before.

**Scheduler:** PBSPro (`qsub` / `qstat -u $USER` / `qdel <id>`), not Slurm.
**Use RCS for:** the many small independent OOD jobs (train-on-A / test-on-B across
sciq × pubmed_qa × xsum) where a second queue and no 3-GPU cap help. **Keep the heaviest
job** (Gemma-2-9B + the eager fp32 Lookback recompute) **on DoC's A100-80GB** — RCS's
A100s are only 40GB and scarce; default to the **L40S (48GB)** on RCS.

> RCS docs: https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/

---

## One-time setup

1. **Log in** (password only, no key auth):
   ```bash
   ssh <user>@login.cx3.hpc.imperial.ac.uk
   ```

2. **Where data lives.** The RCS **home directory has a large allocation (~930GB)**, so the
   repo, the HF cache, and the venv all just live in `$HOME` — nothing to configure. The
   one place NOT to use is **`$EPHEMERAL`**: it is wiped after 30 days, so weights cached
   there would vanish monthly and re-download constantly. `/vol/gpudata` and `/vol/bitbucket`
   do **not** exist on RCS. The scripts default to `$HOME/hf_cache` and `$HOME/venv`.

3. **Get the code into your home directory:**
   ```bash
   cd ~
   git clone <repo-url> msc-project-gs925   # or rsync the repo over; code only, not caches
   cd msc-project-gs925
   ```

4. **Load Python and create the venv (in `$HOME`):**
   ```bash
   module load tools/prod
   module load Python/3.11.5-GCCcore-13.2.0   # matches the project's 3.11 env
   python -m venv ~/venv          # matches the default LUQ_VENV
   source ~/venv/bin/activate
   pip install --upgrade pip
   ```
   This repo's own `luq` package is **not** pip-installed — the scripts (and
   `luq_activate`, via `PYTHONPATH`) import it straight from `src/`, so there is no
   `pip install -e .` to run. You only install the **dependencies**:
   ```bash
   # core deps (versions match the working DoC env; let pip pick the right torch CUDA wheel)
   pip install torch transformers==4.57.6 datasets==5.0.0 numpy==1.26.4 \
       scikit-learn==1.9.0 scipy==1.12.0 openai==2.41.0
   # one dep that lives on GitHub (same as DoC):
   pip install "git+https://github.com/IINemo/lm-polygraph.git@dev"
   ```
   - **ProbeDrift is not distributed with this repository.** It supplies the formatted prompts
     and gold targets for each dataset and rung, and it is installed editable from a local
     checkout:
     ```bash
     pip install -e <path-to>/ProbeDrift
     ```
     Without it the extraction and ladder stages cannot build their populations.
   - Prefer a **conda env** instead? Create it, then submit with
     `LUQ_CONDA_ENV_RCS=<env>` and `LUQ_CONDA_SH_RCS=<path>/etc/profile.d/conda.sh` set;
     `pbs/_env.sh` will use conda instead of the venv. See the RCS conda guide.
   - Do heavy `pip` builds (e.g. torch) in an interactive job, not the login node, if the
     login node is tight on memory.

5. **Pre-download model weights** on the login node (compute nodes may lack internet):
   ```bash
   export HF_HOME=~/hf_cache       # matches the default; do NOT use $EPHEMERAL
   hf download google/gemma-2-9b-it
   ```

---

## End-to-end check (do this before any real GPU job)

```bash
qsub -I -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=L40S -l walltime=1:00:00
# ...once the interactive job starts:
cd <repo>; source pbs/_env.sh; luq_activate
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Or submit the batch check job and read its log:
```bash
qsub pbs/end_to_end.pbs        # writes luq_e2e.o<jobid>
```

---

## Running jobs

Use the wrapper (it picks `qsub` on RCS, `sbatch` on DoC):
```bash
export LUQ_CLUSTER=rcs               # or rely on hostname auto-detect
./scripts/submit.sh extract pubmed_qa ID      # parametrised extraction
./scripts/submit.sh gemma_gpu                  # all three datasets, all GPU features
```
Or `qsub` directly:
```bash
qsub -v LUQ_DATASET=pubmed_qa,LUQ_OOD=ID pbs/extract.pbs
qsub pbs/gemma_gpu.pbs
```

Monitor / cancel:
```bash
qstat -u $USER
qdel <jobid>
```

The `pbs/*.pbs` scripts mirror their `slurm/*.sbatch` twins one-for-one (same Python
entrypoints, same resumability/sentinels). The one-off P(True) wording sweeps
(`slurm/ptrue_*`) were **not** ported — they were completed experiments, not part of the
ongoing pipeline.

## Slurm ↔ PBS quick map

| Action | DoC (Slurm) | RCS (PBSPro) |
|---|---|---|
| submit | `sbatch slurm/x.sbatch` | `qsub pbs/x.pbs` |
| queue | `squeue -u $USER` | `qstat -u $USER` |
| cancel | `scancel <id>` | `qdel <id>` |
| interactive | `salloc --gres=gpu:1 ...` | `qsub -I -l select=1:ngpus=1:...` |
| resources | `#SBATCH --partition=a100 --gres=gpu:1` | `#PBS -l select=1:ngpus=1:gpu_type=A100` |
| GPU | A100 **80GB** | **L40S 48GB** (default), A100 **40GB** (scarce) |
| env | `conda activate luq` | `module load` + venv (PBS skips `~/.bashrc`) |
