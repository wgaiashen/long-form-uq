"""Overnight finalizer for the ExpertQA labelling run: WAIT for the detached labeller to finish
(or detect it stalled), then write a human-readable summary to results/expertqa/label_summary.txt
so the results are there in the morning regardless of whether the shell that started it survives.

Deliberately does NOT relaunch the labeller (a second writer could corrupt the paid labels). If the
labeller dies before 2016, it writes a PARTIAL/STALLED summary and stops; resume is a manual one-liner.

Run detached:
    source /vol/gpudata/gs925-msc_project/.openai_key   # not needed here, but harmless
    nohup python -u scripts/tools/finalize_expertqa_label.py >> logs/finalize_label.out 2>&1 &
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache                    # noqa: E402
from luq.config import Config            # noqa: E402

KEY = cache.run_key("meta-llama/Meta-Llama-3.1-8B", "expertqa", "ID")
CDIR = Config(prompt_regime="expertqa_rp12").cache_dir
OUT = Path("results/expertqa/label_summary.txt")


def labeller_running():
    r = subprocess.run(["pgrep", "-f", "02_label_expertqa"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def counts():
    recs = cache.load_records(CDIR, KEY)
    done = [r for r in recs if r.get("factuality_model")]
    return recs, done


def summarise(recs, done, status):
    import numpy as np
    lines = [f"ExpertQA label summary  —  status: {status}",
             f"records: {len(recs)} | labelled: {len(done)}", ""]
    fdef = [r["factuality"] for r in done if r.get("factuality") is not None]
    n_alluncov = sum(1 for r in done if r.get("factuality") is None and not r.get("factuality_parse_fail"))
    n_quar = sum(1 for r in done if r.get("factuality_quarantined"))
    n_incoh = sum(1 for r in done if r.get("coherent") is False and not r.get("factuality_quarantined"))
    n_pf = sum(1 for r in done if r.get("factuality_parse_fail"))
    unc = [r["uncovered"] for r in done if r.get("uncovered") is not None]
    if fdef:
        fa = np.array(fdef)
        lines += [f"FAITHFULNESS (the label, over covered claims): n={len(fdef)}",
                  f"  mean {fa.mean():.3f}  median {np.median(fa):.3f}  "
                  f"| ==0: {np.mean(fa==0):.0%}  ==1: {np.mean(fa==1):.0%}  strictly-mid: {np.mean((fa>0)&(fa<1)):.0%}"]
    if unc:
        u = np.array(unc)
        lines += ["", f"UNCOVERED / blind spot: mean {u.mean():.3f}  median {np.median(u):.3f}  "
                      f">0.5: {np.mean(u>0.5):.0%}"]
    n = len(done) or 1
    lines += ["",
              f"ALL-uncovered (no factuality signal, excluded from probe): {n_alluncov} ({n_alluncov/n:.0%})",
              f"distrust-quarantined SEVERE (factuality=0): {n_quar} ({n_quar/n:.0%})",
              f"coherent=false marginal (factuality=0): {n_incoh} ({n_incoh/n:.0%})",
              f"judge parse failures: {n_pf} ({n_pf/n:.0%})",
              "", "judge = gpt-5-mini | label field = `factuality` (+ uncovered, coherent, factuality_quarantined)",
              "caveats to report: ~12% base-model derailment floor + this ~58%-scale blind spot; ExpertQA -> Role-C."]
    if status != "COMPLETE":
        lines += ["", "RESUME (single writer only):",
                  "  source /vol/gpudata/gs925-msc_project/.openai_key && \\",
                  "  PYTHONPATH=src python scripts/02_label_expertqa.py --prompt-regime expertqa_rp12 --judge gpt-5-mini"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    print(f"finalizer watching label {KEY} ...", flush=True)
    while True:
        recs, done = counts()
        if len(done) >= len(recs):
            summarise(recs, done, "COMPLETE")
            return
        if not labeller_running():
            summarise(recs, done, f"STALLED at {len(done)}/{len(recs)} (labeller not running)")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
