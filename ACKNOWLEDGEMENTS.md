# Acknowledgement of AI assistance

This project used Claude Code (Anthropic) as a coding assistant. This document records what that
assistance covered, what it did not, and the checks applied to it. Short comments in the files
concerned point back to this document.

## What the assistant was used for

**Cluster job scripts.** The scheduler boilerplate across `pbs/` and `slurm/` was largely drafted
with assistance: PBS and Slurm resource directives, environment activation, log redirection, array
indexing and the failure handling repeated across several hundred near-identical scripts. What each
job runs, and the parameters it runs with, were specified by the author.

**Visualisation and figure code.** The per-token HTML visualisers in `scripts/tools/`
(`viz_common.py` and the `visualise_*.py` family) and the figure scripts (`fig_*.py`) were drafted
with assistance: HTML and CSS generation, colour mapping, page layout and matplotlib plotting code.
The quantities displayed, the figure designs and the interpretation of what they show are the
author's own.

**Documentation and utilities.** README text, docstrings, and small file-handling and data-
processing utilities in `scripts/tools/`. The assistant also maintained a dated log of what was run
and why, which is not part of this repository but facilitated the author's tracking of progress and writing of the final report.

**Debugging, and running checks.** The assistant was used to diagnose failures and to run the
repository's verification scripts and report the outcome.

## What the assistant was not used for

The research questions, the benchmark design, the uncertainty estimation method, the experimental
design, the pre-registered hypotheses with their thresholds and decision rules, the validity
criteria, the analysis and the conclusions are the author's own work. So are the unit tests in
`tests/` and the verification and gate scripts in `scripts/checks/`, which encode acceptance
criteria the author set. No assistant output was treated as authoritative, and none was used in
place of a cited source.

## Limitations observed, and what was done about them

Assistance was most reliable on mechanical and repetitive code and least reliable on questions about
what the code or the cached data actually contained. Confident explanations of pipeline behaviour,
cached artifact contents and numerical precision were repeatedly wrong when checked against the
artifacts themselves or against the original authors' released code. In one case an assistant-written
audit of this repository stated that a set of results was absent from the published tables when the
rows existed in a different file, which was found only by reading the files rather than the audit.

The response was to make verification structural rather than optional, which is visible throughout
this repository:

- every result row carries the commit, cluster and environment it was produced under, and the
  pipeline refuses to stamp a commit that does not reproduce the working tree;
- ported baseline methods are checked numerically against the original authors' released code
  rather than against a description of it;
- equivalence gates compare per-example score vectors rather than summary values, because a summary
  can agree while the underlying scores differ;
- coverage travels with every row, so a method measured on part of the grid cannot be averaged as
  though it were complete;
- a missing input raises or leaves a field blank rather than substituting a neutral value, so an
  absent measurement can never be read as a measured one;
- a result that moved in the expected direction was treated as a suspected error until the check
  that would have been run had it gone the other way had been run.

Code produced with assistance was reviewed and tested before use, and numerical results that entered
the report were recomputed or re-run independently.

## Where the comments are

Comments naming this assistance appear for example in `pbs/_env.sh`, which every job script sources, in the
visualisation and figure scripts and utilities in `scripts/tools/`, and in `README.md` and
`published_results/README.md`.
