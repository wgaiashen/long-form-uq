"""Numerical check: our AlignScore label matches the AlignScore wrapper.

Our scorer (src/luq/labels/alignscore.py) vendors the AlignScorer, so the underlying model code
is the same. What this check confirms is the GLUE around it: the claims/contexts direction, the
max over multiple references (trivia aliases), and the empty-output handling, end to end on real
cached records. We score a handful of (output, target) pairs with both our `score()` and the
`AlignScore.__call__`, then assert they agree to < 1e-3.

For the side, multiple references are reduced exactly as his AggregatedMetric does for trivia:
score each alias on its own, then take the max.

Needs a GPU: the literal scorer calls torch.cuda.synchronize() unconditionally (the very call
our vendored copy guards), so it cannot instantiate on a CPU-only node. Run inside a GPU session:

    python scripts/checks/alignscore_vs_authors.py
    python scripts/checks/alignscore_vs_authors.py --n 30
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_REPO = ROOT.parent / "Temp_robust_UQ_probes"
sys.path.insert(0, str(ROOT / "src"))

# This round is Llama-3.1-8B; Config.model_name still defaults to the old Qwen dev model.
MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.labels.alignscore import score as our_score  # noqa: E402

TOL = 1e-3


def _stub_tensorflow():
    """Register a no-op tensorflow so the package imports without TF installed.

    Importing AlignScore pulls in lm_polygraph_lite, whose __init__ eagerly loads an unrelated
    Keras SAPLMA head that does `import tensorflow` + `tf.config...` +
    `from tensorflow.keras... import Sequential, Dense` at module load. AlignScore itself uses
    torch, not TF, so a stub lets us run the authors' real AlignScore code (yuh-zha/AlignScore).
    No effect on scoring.
    """
    import types
    import importlib.machinery
    if "tensorflow" in sys.modules:
        return
    tf = types.ModuleType("tensorflow")
    tf.config = types.SimpleNamespace(
        optimizer=types.SimpleNamespace(set_jit=lambda *a, **k: None),
        set_visible_devices=lambda *a, **k: None,
    )
    tf.__version__ = "0.0.0-stub"
    keras = types.ModuleType("tensorflow.keras")
    models = types.ModuleType("tensorflow.keras.models")
    layers = types.ModuleType("tensorflow.keras.layers")
    models.Sequential = object
    layers.Dense = object
    keras.models, keras.layers = models, layers
    tf.keras = keras
    for name, mod in [("tensorflow", tf), ("tensorflow.keras", keras),
                      ("tensorflow.keras.models", models), ("tensorflow.keras.layers", layers)]:
        # A real ModuleSpec so importlib.util.find_spec("tensorflow") (transformers probes for
        # it) returns cleanly instead of raising on __spec__ == None.
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = mod


def reference_scorer():
    """the AlignScore wrapper, imported from his repo. Kept import-local so the rest of the
    file can be read without his package installed."""
    if str(REFERENCE_REPO) not in sys.path:
        sys.path.insert(0, str(REFERENCE_REPO))
    _stub_tensorflow()
    from utils.alignscore import AlignScore  # noqa: E402
    return AlignScore(batch_size=1)


def reference_value(ref, record):
    """the AlignScore for one record. A list target (trivia aliases) is reduced by max, the
    same as his AggregatedMetric."""
    out = record["gen_text"]
    tgt = record["target"]
    golds = tgt if isinstance(tgt, list) else [tgt]
    vals = [float(ref({"greedy_texts": [out]}, [g])[0]) for g in golds]
    return max(vals)


def sample_records(cache_dir, model, datasets, n):
    """A spread of records across datasets, taken from the front of each cache. Returns
    (dataset, record) pairs."""
    per = max(1, n // len(datasets))
    picked = []
    for d in datasets:
        key = cache.run_key(model, d, "ID")
        try:
            recs = cache.load_records(cache_dir, key)
        except FileNotFoundError:
            print(f"  [skip] no cached records for {d}")
            continue
        picked += [(d, r) for r in recs[:per]]
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--datasets", default="sciq,trivia_qa,pubmed_qa,xsum")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset="sciq", ood_setting="ID")
    pairs = sample_records(cfg.cache_dir, args.model, args.datasets.split(","), args.n)
    if not pairs:
        sys.exit("no records to check")

    ref = reference_scorer()

    print(f"{'dataset':12s}{'ours':>9}{'ref':>9}{'|diff|':>10}")
    print("-" * 40)
    worst = 0.0
    for d, r in pairs:
        ours = our_score(r, d)
        theirs = reference_value(ref, r)
        if ours is None:
            print(f"{d:12s}{'None':>9}{theirs:>9.4f}   our scorer returned None")
            continue
        diff = abs(ours - theirs)
        worst = max(worst, diff)
        flag = "" if diff < TOL else "  <-- OVER TOL"
        print(f"{d:12s}{ours:>9.4f}{theirs:>9.4f}{diff:>10.2e}{flag}")

    print("-" * 40)
    print(f"worst |diff| = {worst:.2e}  (tolerance {TOL:.0e})")
    if worst < TOL:
        print("PASS: our AlignScore matches the wrapper.")
    else:
        sys.exit("FAIL: our AlignScore disagrees with the beyond tolerance.")


if __name__ == "__main__":
    main()
