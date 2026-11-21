"""T4 weights: PER, agreement filter, confusion table, TTS weights."""
import math

import pytest

from conftest import PHONEMES
from iqra import weights

PH = [p for p in PHONEMES]


def test_clip_per_and_agreement():
    assert weights.clip_per("a b c d".split(), "a b c d".split()) == 0.0
    assert weights.clip_per("a b c d".split(), "a x c".split()) == pytest.approx(0.5)
    assert weights.agreement_weight(0.31) == 0.2 and weights.agreement_weight(0.30) == 1.0
    # empty label -> PER undefined -> broken label, down-weighted (NaN > 0.30 is False)
    assert weights.clip_per([], "a b".split()) != weights.clip_per([], "a b".split())   # NaN
    assert weights.agreement_weight(float("nan")) == 0.2
    import numpy as np
    assert weights.agreement_weight(np.float64("nan")) == 0.2


def _table():
    t = weights.ConfusionTable(PH)
    for _ in range(20):
        t.add("q a l b".split(), "k a l b".split())      # q -> k is the common real error
    t.add("s a b".split(), "s a".split())                 # one deletion of b
    t.add("m a".split(), "m a n".split())                 # one insertion of n
    return t


def test_confusion_counts_and_smoothing():
    t = _table()
    assert t.sub["q"]["k"] == 20 and t.sub["b"][weights.DELETE] == 1 and t.ins["n"] == 1
    assert t.top_pairs(1) == [(("q", "k"), 20)]
    K = t.n_outcomes
    assert t.p_error("q", "k") == pytest.approx(21 / (20 + K))
    assert t.p_error("z", "k") == pytest.approx(1 / K)    # unseen expected phoneme -> uniform
    total = sum(t.p_error("q", o) for o in [p for p in PH if p != "q"] + [weights.DELETE])
    assert total == pytest.approx(1.0)


def test_tts_weights_favour_realistic_errors():
    t = _table()
    K = t.n_outcomes
    realistic = weights.tts_weight("q a l".split(), "k a l".split(), t)     # q -> k, the real habit
    unusual = weights.tts_weight("q a l".split(), "S a l".split(), t)       # q -> S, never seen
    unseen = weights.tts_weight("q a l".split(), "q a S".split(), t)        # l has no recorded errors
    assert realistic > 1.0 > unusual
    assert realistic == pytest.approx(min(3.0, 21 / (20 + K) * K))
    assert unusual == pytest.approx(max(0.5, K / (20 + K)))
    assert unseen == pytest.approx(1.0)                                       # uniform prior: neutral
    assert weights.tts_weight("a b".split(), "a b".split(), t) == 1.0
    assert 0.5 <= weights.tts_weight("q a b".split(), "k a".split(), t) <= 3.0
