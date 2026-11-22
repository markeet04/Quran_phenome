"""Deletion rule and base-rate projection (SPEC decision 37)."""
import pandas as pd
import pytest

from iqra import transfer as X
from iqra.candidates import alignment


def test_deletion_rule_reverts_only_deletions():
    expected = "b a t u l".split()
    heard = "b i t q u".split()           # S a>i, insertion q, D of final l
    ops = [op for op, *_ in alignment(expected, heard)]
    assert "D" in ops and "S" in ops
    out = X.deletion_rule(expected, heard)
    assert out[-1] == "l"                  # deletion reverted
    assert "i" in out                      # substitution kept
    assert len(out) == len(heard) + ops.count("D")   # nothing dropped, insertions kept
    assert X.deletion_rule(expected, expected) == expected


def test_projection_identity_and_examples():
    base = X.projected(1.0, 1.0)
    assert base["precision"] == pytest.approx(0.634) and base["recall"] == pytest.approx(0.703)
    assert base["f1"] == pytest.approx(2 * 0.634 * 0.703 / (0.634 + 0.703))
    row4 = X.projected(129 / 192, 135 / 704)
    assert row4["f1"] == pytest.approx(0.609, abs=0.002)
    assert X.projected(1.0, 0.0)["precision"] == 1.0


def test_keep_fractions():
    a = pd.DataFrame({"TR": [3, 1], "FR": [4, 4]})
    b = pd.DataFrame({"TR": [2, 1], "FR": [1, 1]})
    assert X.keep_fractions(a, b) == {"keep_tr": 0.75, "keep_fr": 0.25}
