# `pbs/`: PBSPro job scripts

299 job scripts, plus a shared environment file, two shared job bodies, a watchdog and a setup note. They are a record of what was
submitted to the PBSPro cluster, not a curated interface, and most of them will not be useful to
anyone outside that cluster: they name queues, walltimes and card types specific to it, and many
carry absolute paths from the machine they ran on. They are kept because they are the evidence of
how each reported result was actually produced.

Almost all of them have the same shape: a scheduler header, then a call to one driver in
`scripts/checks/` with its flags. The method is in the driver. The job script adds the scheduling.

## The files worth opening

Most of the 299 are not worth reading. These are.

| file | why |
|---|---|
| `_env.sh` | resolves the repository root, the model cache and the Python environment for whichever cluster it runs on. Sourced by nearly every other script here, and by anyone running the pipeline by hand. Its Python twin is `src/luq/cluster.py`. |
| `SETUP_RCS.md` | first-time setup on this cluster. |
| `wmodels_extract.pbs` | the per-dataset generation specification: token budget and decoding controls for each of the eight evaluation sets, one line each, in the `case` block. This is the configuration the reproducibility table in the write-up states, in the form it was actually run. |
| `extract_neighbour.pbs` | the same for the datasets brought in as training sources, and the reasoning for using an n-gram ban rather than a repetition penalty on one of them. |
| `_md_hybrids_rmd_body.sh`, `_pertoken_multilayer_body.sh` | shared bodies sourced by families of wrappers that differ only in which datasets they cover and which cards they request. |

## Reading the rest

If you want to know how a particular result was produced, start from the driver named in
`scripts/checks/`, not from here. To find the job that ran it, grep this directory for the driver
name. Where a job script carries a comment explaining a flag, that comment is usually the only place
the reasoning is written down, which is the other reason these files are kept.
