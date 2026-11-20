"""Forced alignment vs brute-force enumeration of every CTC path."""
import itertools

import numpy as np
import pytest

from iqra.forced_align import ctc_forced_align, token_spans


def _collapse(path, blank=0):
    out, prev = [], None
    for p in path:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return out


def _brute_force(lp, targets, blank=0):
    T, V = lp.shape
    best, best_path = -np.inf, None
    symbols = sorted(set(targets) | {blank})
    for path in itertools.product(symbols, repeat=T):
        if _collapse(path, blank) == list(targets):
            score = sum(lp[t, p] for t, p in enumerate(path))
            if score > best:
                best, best_path = score, list(path)
    return best, best_path


@pytest.mark.parametrize("seed", range(40))
def test_viterbi_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    V = 5
    T = int(rng.integers(3, 8))
    L = int(rng.integers(1, min(4, T) + 1))
    targets = [int(x) for x in rng.integers(1, V, L)]
    if seed % 5 == 0 and L >= 2:
        targets[1] = targets[0]                       # a repeated token needs a blank between
    lp = np.log(rng.dirichlet(np.ones(V), size=T))
    repeats = sum(1 for i in range(1, L) if targets[i] == targets[i - 1])
    if T < L + repeats:
        with pytest.raises(ValueError):
            ctc_forced_align(lp, targets)
        return
    path, score, _ = ctc_forced_align(lp, targets)
    best, _ = _brute_force(lp, targets)
    assert _collapse(path) == targets
    assert score == pytest.approx(best)
    assert score == pytest.approx(sum(lp[t, p] for t, p in enumerate(path)))


def test_token_spans_cover_their_frames():
    V, T = 6, 30
    targets = [1, 2, 2, 3]
    truth = [0] * 3 + [1] * 5 + [0] * 2 + [2] * 4 + [0] * 3 + [2] * 3 + [0] * 2 + [3] * 6 + [0] * 2
    lp = np.full((T, V), np.log(0.02))
    lp[np.arange(T), truth] = np.log(0.9)
    spans, failed = token_spans(lp, targets)
    assert not failed
    assert spans == [(3, 7), (10, 13), (17, 19), (22, 27)]


def test_token_spans_fallback_when_too_short():
    lp = np.log(np.full((3, 4), 0.25))
    spans, failed = token_spans(lp, [1, 2, 3, 1, 2])
    assert failed and len(spans) == 5
    assert all(0 <= a <= b < 3 for a, b in spans)
