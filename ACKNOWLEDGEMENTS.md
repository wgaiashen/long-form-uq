# Acknowledgement of AI assistance

This project used Claude Code (Anthropic) as a coding assistant. This document records what that assistance covered, what it did not, the limitations observed, and the checks applied to it. I have also added acknowledgements as comments in the relevant parts of the codebase, with those comments pointing back to this document. ChatGPT use for report drafting and review is declared separately in the MSc report.

## What the assistant was used for

**Cluster job scripts.** The scheduler boilerplate across `pbs/` and `slurm/` was largely drafted with assistance, including PBS and Slurm resource directives, environment activation, log redirection, array indexing and failure handling repeated across several hundred near-identical scripts. What each job runs, and the parameters it runs with, were specified by the author.

**Visualisation and figure code.** The per-token HTML visualisers in `scripts/tools/` (`viz_common.py` and the `visualise_*.py` family) and the figure scripts (`fig_*.py`) were drafted with assistance, including HTML and CSS generation, colour mapping, page layout and matplotlib plotting code. The quantities displayed, the figure designs and the interpretation of what they show are the author's own.

**Documentation and utilities.** Assistance was used for README text, docstrings, and small file-handling and data-processing utilities in `scripts/tools/`. The assistant also maintained a dated log of what was run and why. That log is not part of this repository but supported the author's tracking of experimental progress and preparation of the final report.

**Debugging and running checks.** The assistant was used to diagnose implementation failures and to run the repository's verification scripts and report their outputs. It did not define the acceptance criteria encoded by those checks.

## What the assistant was not used for

The research questions, ProbeDriftLong benchmark design, CAWSA uncertainty estimation method, experimental design, pre-registered hypotheses with their thresholds and decision rules, validity criteria, analysis and conclusions are the author's own work. So are the unit tests in `tests/` and the verification and gate scripts in `scripts/checks/`, which encode acceptance criteria set by the author. No assistant output was treated as authoritative or used in place of a cited source.

## Limitations observed and how they were managed

Assistance was most reliable on mechanical and repetitive code and least reliable on questions about what the code or cached data actually contained. Confident explanations of pipeline behaviour, cached artifact contents and numerical precision were repeatedly wrong when checked against the artifacts themselves or against the original authors' released code. In one case an assistant-written audit stated that a set of results was absent from the published tables when the rows existed in a different file, which was found only by reading the files rather than relying on the audit.

The response was to make verification structural rather than optional. This is visible throughout the repository:

- every report-facing result row carries the commit, cluster and environment under which it was produced, and the pipeline refuses to stamp a commit that does not reproduce the working tree;
- ported baseline methods are checked numerically against the original authors' released code rather than against a prose description;
- equivalence gates compare per-example score vectors rather than only summary values, because a summary can agree while the underlying scores differ;
- coverage travels with every row, so a method measured on only part of the benchmark cannot be averaged as though it were complete;
- a missing input raises or leaves a field blank rather than substituting a neutral value, so absence cannot be mistaken for a measurement; and
- a result moving in the expected direction was still treated as a suspected error until the same checks that would have been run for an unexpected result had been completed.

Code produced with assistance was reviewed and tested before use. Numerical results that entered the report were recomputed or re-run independently, and report-facing interpretations were checked against repository outputs and source papers rather than accepted from assistant explanations.

## Where the comments are

Comments acknowledging this assistance appear, for example, in `pbs/_env.sh`, which every job script sources, in the visualisation and figure scripts and utilities in `scripts/tools/`, and in `README.md` and `published_results/README.md`. These comments point back to this document so that the scope of assistance and the verification policy can be inspected in one place.
