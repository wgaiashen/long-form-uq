#!/usr/bin/env python
"""Numerical equivalence of our Mahalanobis port against the reference implementation itself.

The project rule is that a ported method is checked against the authors' released code, not against
the paper text and not against memory. This imports the reference modules directly out of the
Hidden Failures checkout in the parent directory and runs both implementations on identical random
inputs. Numbers decide, not prose.

Four things are compared:
  1. compute_inv_covariance      -- the jittered, float64-inverted covariance
  2. the single-centroid distance -- our einsum against the reference's general einsum
  3. total_uncertainty_linear_step -- the hybrid uncertainty combination used by HUQ
  4. grid_search_hp              -- the hyperparameter search over that combination

The hybrid back-off is checked separately in md_hybrids.py's unit test because its reference lives in
a runnable script rather than an importable module.

    python scripts/checks/md_port_equivalence.py
"""
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT.parent / "RobustUQProbes"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REF))

# The reference's HUQ module imports a sibling that calls nltk.download at import time, which would
# reach the network from a login node. A stub keeps the import local; nothing downstream uses nltk.
def _load_by_path(mod_name, rel_path, prereg=None):
    """Load one reference source file directly, bypassing its package __init__.

    Importing `lm_polygraph_lite` as a package pulls in an uncertainty-head class that imports
    tensorflow, which is not installed here and is irrelevant to the Mahalanobis code. Loading the
    leaf files by path executes the authors' literal source without dragging in that chain, so the
    comparison is still against their code and not a paraphrase of it.
    """
    import importlib.util
    for name, mod in (prereg or {}).items():
        sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location(mod_name, REF / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# A synthetic parent package so the reference files' relative imports resolve to the leaves we load.
_pkg = types.ModuleType("refpkg")
_pkg.__path__ = []
sys.modules["refpkg"] = _pkg

_mahal = _load_by_path("refpkg.mahalanobis_distance",
                       "lm_polygraph_lite/estimators/mahalanobis_distance.py")
ref_inv_cov = _mahal.compute_inv_covariance
ref_md = _mahal.mahalanobis_distance_with_known_centroids_sigma_inv

_uem = _load_by_path("refpkg.ue_metric", "lm_polygraph_lite/ue_metrics/ue_metric.py")
_pra_src = (REF / "lm_polygraph_lite/ue_metrics/pred_rej_area.py").read_text()
_pra_ns = {"np": np, "List": list, "UEMetric": _uem.UEMetric, "normalize": _uem.normalize}
exec(compile(_pra_src.replace("from .ue_metric import UEMetric, normalize", ""),
             "pred_rej_area.py", "exec"), _pra_ns)
RefPRR = _pra_ns["PredictionRejectionArea"]

# total_uncertainty_linear_step and grid_search_hp are self-contained top-level functions in the
# reference's HUQ module, whose module-level imports would drag in sklearn and an nltk download. The
# two function definitions are lifted out of the file's own source text and executed verbatim, so
# what runs below is the authors' code character for character.
import ast as _ast

_huq_src = (REF / "satmd_baseline/huq_msp_lrtmd.py").read_text()
_tree = _ast.parse(_huq_src)
_wanted = {"total_uncertainty_linear_step", "grid_search_hp"}
_ns = {"np": np, "rankdata": __import__("scipy.stats", fromlist=["rankdata"]).rankdata,
       "PredictionRejectionArea": RefPRR}
for _node in _tree.body:
    if isinstance(_node, _ast.FunctionDef) and _node.name in _wanted:
        exec(compile(_ast.Module(body=[_node], type_ignores=[]), "huq_msp_lrtmd.py", "exec"), _ns)
missing = _wanted - set(_ns)
if missing:
    sys.exit(f"could not lift {missing} out of the reference HUQ module")
ref_step = _ns["total_uncertainty_linear_step"]
ref_grid = _ns["grid_search_hp"]

from luq import mahalanobis as ours  # noqa: E402

FAILS = []


def report(name, val, tol):
    ok = bool(val <= tol)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} max|d| = {val:.3e}  (tol {tol:.0e})")
    if not ok:
        FAILS.append(name)


def main():
    rng = np.random.RandomState(0)
    D, N, M = 64, 900, 120
    feats = rng.randn(N, D).astype(np.float32) * 1.7 + 0.4
    centroid = feats.mean(axis=0)

    print("1. compute_inv_covariance")
    ours_inv, ours_jit = ours.compute_inv_covariance(centroid, feats)
    ref_inv, ref_jit = ref_inv_cov(torch.from_numpy(centroid).unsqueeze(0), torch.from_numpy(feats))
    ref_inv = ref_inv.cpu().numpy()
    report("inverse covariance", float(np.max(np.abs(ours_inv - ref_inv))), 1e-9)
    print(f"        jitter ours {ours_jit:g} vs reference {ref_jit:g} "
          f"{'(same)' if ours_jit == ref_jit else '(DIFFERENT)'}")
    if ours_jit != ref_jit:
        FAILS.append("jitter choice")

    print("2. single-centroid distance")
    ev = rng.randn(M, D).astype(np.float32)
    ref_d = ref_md(torch.from_numpy(centroid), None, torch.from_numpy(ref_inv),
                   torch.from_numpy(ev))[:, 0].cpu().numpy()
    stats = ours.MDStats(centroid=centroid.astype(np.float64), sigma_inv=ours_inv,
                         n_tokens=N, n_rows=1, jitter=ours_jit, key="")
    ours_d = ours.md_tokens([ev], stats)[0]
    report("per-token distance", float(np.max(np.abs(ours_d - ref_d))), 1e-4)

    print("3. total_uncertainty_linear_step")
    worst = 0.0
    for trial in range(6):
        r = np.random.RandomState(trial)
        epi, ale = r.randn(200), r.randn(200)
        for tmin, tmax, alpha in [(0.0, 1.0, 0.0), (0.1, 0.9, 0.1), (0.25, 0.85, 0.5), (0.3, 0.8, 1.0)]:
            a = ours.total_uncertainty_linear_step(epi, ale, tmin, tmax, alpha)
            b = ref_step(epi, ale, tmin, tmax, alpha)
            worst = max(worst, float(np.max(np.abs(a - b))))
    report("combined ranks, 24 parameter settings", worst, 0.0)

    print("4. grid_search_hp")
    ref_prr = RefPRR()
    r = np.random.RandomState(7)
    epi, ale = r.randn(150), r.randn(150)
    metrics = np.clip(0.5 + 0.3 * r.randn(150), 0, 1)
    # The reference's target metric is called as metric(uncertainty, target); ours takes
    # (target, uncertainty), so the adapter here is the ONLY difference between the two calls.
    ours_best = ours.grid_search_hp(epi, ale, metrics, prr_fn=lambda y, u: ref_prr(u, y))
    ref_best = ref_grid(epi, ale, metrics, target_metric=ref_prr)
    same = all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(ours_best, ref_best))
    print(f"  {'PASS' if same else 'FAIL'}  selected (prr, t_min, t_max, alpha)")
    print(f"        ours      {tuple(round(float(x), 6) for x in ours_best)}")
    print(f"        reference {tuple(round(float(x), 6) for x in ref_best)}")
    if not same:
        FAILS.append("grid_search_hp")

    print()
    if FAILS:
        sys.exit(f"PORT EQUIVALENCE FAILED: {FAILS}. The port is NOT the published method and must "
                 f"not be used as a baseline until this passes.")
    print("PORT EQUIVALENCE: all checks pass against the reference implementation.")


if __name__ == "__main__":
    main()
