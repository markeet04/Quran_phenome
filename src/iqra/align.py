"""Thin wrapper over the official scorer's own alignment code.

Imports metric.py and align_data.py from the pinned mdd_eval folder unchanged, so
candidates and per-clip counts use exactly the scorer's Align and op labels.
Importing them only defines functions (align_data.py guards main()).
"""
import importlib.util
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from .config import SCORER_COMMIT, SCORER_REPO

EPS = "<eps>"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def mdd_eval_dir() -> Path:
    """Locate the pinned mdd_eval folder.

    Order: $IQRA_MDD_EVAL, the repo's third_party submodule, else a pinned clone
    in ~/.cache/iqra (pip installs on Kaggle don't carry the submodule).
    """
    env = os.environ.get("IQRA_MDD_EVAL")
    if env:
        return _checked(Path(env))
    sub = _REPO_ROOT / "third_party" / "interspeech_IqraEval" / "mdd_eval"
    if (sub / "metric.py").exists():
        return _checked(sub)
    cache = Path.home() / ".cache" / "iqra" / "interspeech_IqraEval"
    if not (cache / "mdd_eval" / "metric.py").exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", SCORER_REPO, str(cache)], check=True)
        subprocess.run(["git", "-C", str(cache), "checkout", "-q", SCORER_COMMIT], check=True)
    return _checked(cache / "mdd_eval")


def _checked(path: Path) -> Path:
    for name in ("metric.py", "align_data.py", "ins_del_cor_sub_analysis.py"):
        if not (path / name).exists():
            raise FileNotFoundError(f"official scorer file missing: {path / name}")
    repo = path.parent
    if (repo / ".git").exists():
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        if head and head != SCORER_COMMIT:
            raise RuntimeError(f"scorer at {repo} is at {head}, expected pinned {SCORER_COMMIT}")
    return path


@lru_cache(maxsize=None)
def _official(module_name: str):
    folder = mdd_eval_dir()
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))      # align_data.py does `from metric import ...`
    spec = importlib.util.spec_from_file_location(f"_iqra_official_{module_name}",
                                                  folder / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    # never write __pycache__ into the pinned scorer folder
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def scorer_env():
    """Environment for running the official scripts without writing bytecode beside them."""
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


def Align(seq1, seq2):
    """Official Needleman-Wunsch alignment: returns (aligned1, aligned2) with '<eps>' gaps."""
    return _official("metric").Align(seq1, seq2)


def get_op(aligned1, aligned2):
    """Official op labels per aligned position: 'C', 'S', 'D' (gap in 2), 'I' (gap in 1)."""
    return _official("align_data").get_op(aligned1, aligned2)


def align_ops(seq1, seq2):
    """(aligned1, aligned2, ops) exactly as align_data.get_align writes them."""
    a1, a2 = Align(seq1, seq2)
    return a1, a2, get_op(a1, a2)


def accuracy_counts(trans, pred):
    """Official Accuracy(): (edit count, len(trans)) used for the scorer's PER."""
    return _official("metric").Accuracy(trans, pred)
