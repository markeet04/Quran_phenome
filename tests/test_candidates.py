"""Revert rule, candidate keys and labels (SPEC T0, section 5; insertions decision)."""
import random

from conftest import make_synthetic
from iqra.candidates import (alignment, extract_candidates, label_candidates,
                             remove_insertions, revert)


def _pairs():
    truth, pred = make_synthetic(50, seed=3)
    for (_, t), (_, p) in zip(truth.iterrows(), pred.iterrows()):
        yield t["Reference_phn"].split(), p["Prediction"].split()


def _expected_with_insertions(expected, heard):
    return [d if op == "I" else e for op, _, _, e, d in alignment(expected, heard)]


def test_candidates_are_only_substitutions_and_deletions():
    n_ins = 0
    for expected, heard in _pairs():
        assert {c.op for c in extract_candidates(expected, heard)} <= {"S", "D"}
        n_ins += sum(1 for r in alignment(expected, heard) if r[0] == "I")
    assert n_ins > 0          # the test data does contain insertions


def test_revert_reject_all_gives_expected_plus_insertions():
    for expected, heard in _pairs():
        keep = {c.pos: False for c in extract_candidates(expected, heard)}
        out = revert(expected, heard, keep)
        assert out == _expected_with_insertions(expected, heard)
        if not any(r[0] == "I" for r in alignment(expected, heard)):
            assert out == expected


def test_revert_keep_all_leaves_prediction():
    for expected, heard in _pairs():
        keep = {c.pos: True for c in extract_candidates(expected, heard)}
        assert revert(expected, heard, keep) == heard
        assert revert(expected, heard, {}) == heard


def test_revert_never_removes_insertions():
    rng = random.Random(0)
    for expected, heard in _pairs():
        cands = extract_candidates(expected, heard)
        keep = {c.pos: rng.random() < 0.5 for c in cands}
        out = revert(expected, heard, keep)
        n_rej_D = sum(1 for c in cands if c.op == "D" and not keep[c.pos])
        assert len(out) == len(heard) + n_rej_D


def test_remove_insertions():
    for expected, heard in _pairs():
        n_ins = sum(1 for r in alignment(expected, heard) if r[0] == "I")
        assert len(remove_insertions(expected, heard)) == len(heard) - n_ins


def test_candidate_indices():
    expected = "b i s m i".split()
    heard = "b u s i x".split()
    for c in extract_candidates(expected, heard):
        assert expected[c.j] == c.expected
        if c.op == "S":
            assert heard[c.h] == c.heard


def test_labels_follow_the_human():
    expected = "q a l b i".split()
    said = "k a l b i".split()          # human substituted q -> k
    heard = "k a l b u".split()         # model: q -> k (real), i -> u (false alarm)
    cands = extract_candidates(expected, heard)
    labels = dict(zip([(c.op, c.j) for c in cands], label_candidates(expected, said, cands)))
    assert labels == {("S", 0): 1, ("S", 4): 0}
