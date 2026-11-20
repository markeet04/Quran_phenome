"""Candidate flags, their labels, and the revert rule (SPEC section 5).

The verifier scores only S and D positions in the official alignment of expected vs heard:
  S: expected index j was heard as something else (heard index h)
  D: expected index j was not heard
Insertions (heard phonemes with no expected counterpart) are never candidates and are
never removed. In the official scorer a model insertion is never a false rejection; at a
gap where the reciter also inserted it is a true rejection (ins_ins / ins_ins1). So
removing one can only lose TR. remove_insertions() exists only for the dev check that
confirms this on Extra-dev.
Features are added in T3 (verifier); this module only fixes keys, labels and edits.
"""
from dataclasses import dataclass

from .align import align_ops


@dataclass(frozen=True)
class Candidate:
    op: str          # "S" or "D"
    pos: int         # position in the alignment
    j: int           # expected index
    h: int           # heard index (S); -1 for D
    expected: str    # expected phoneme
    heard: str       # heard phoneme ("" for D)


def alignment(expected, heard):
    """Official alignment of expected vs heard with expected/heard indices per position.

    Returns a list of (op, j, h, expected_tok, heard_tok). For D, h = -1; for I, j is the
    gap index (the insertion sits before expected index j).
    """
    a1, a2, ops = align_ops(list(expected), list(heard))
    rows, j, h = [], 0, 0
    for e, d, op in zip(a1, a2, ops):
        if op in ("C", "S"):
            rows.append((op, j, h, e, d)); j += 1; h += 1
        elif op == "D":
            rows.append((op, j, -1, e, "")); j += 1
        else:  # "I"
            rows.append((op, j, h, "", d)); h += 1
    assert j == len(expected) and h == len(heard)
    return rows


def extract_candidates(expected, heard):
    """S and D positions only."""
    return [Candidate(op, pos, j, h, e, d)
            for pos, (op, j, h, e, d) in enumerate(alignment(expected, heard))
            if op in ("S", "D")]


def label_candidates(expected, said, candidates):
    """1 if the human also deviates at expected index j (substituted or deleted it), else 0.

    `said` is what was actually spoken (phoneme_mis / Annotation_phn). A kept flag where
    the human deviated is a true rejection; where the human was correct, a false rejection.
    """
    deviated = {j for op, j, _, _, _ in alignment(expected, said) if op in ("S", "D")}
    return [int(c.j in deviated) for c in candidates]


def revert(expected, heard, keep):
    """Rebuild the prediction after the verifier: reject = revert, never drop.

    `keep` maps alignment position -> bool for S/D candidates (missing = keep).
      C: heard phoneme
      S: keep -> heard phoneme,  reject -> expected phoneme
      D: keep -> nothing,        reject -> expected phoneme
      I: always the heard phoneme (insertions are never removed)
    """
    out = []
    for pos, (op, _, _, e, d) in enumerate(alignment(expected, heard)):
        kept = keep.get(pos, True)
        if op in ("C", "I"):
            out.append(d)
        elif op == "S":
            out.append(d if kept else e)
        elif not kept:  # "D"
            out.append(e)
    return out


def remove_insertions(expected, heard):
    """The prediction with every insertion dropped. Dev alignment-shift check only."""
    return [d for op, _, _, _, d in alignment(expected, heard) if op in ("C", "S")]
