"""Training-data weights for stage 2 (SPEC T4, section 5).

- Agreement filter: PER between the stage-1 decode and a clip's label; weight 0.2 above
  PER 0.30, else 1 (single model, decision 5c). Caveat: stage 1 trained on these same
  clips, so it agrees with its own training labels more than an independent model would.
- Confusion table: expected -> said substitutions, deletions and insertions counted over
  Extra folds A + B (official alignment), add-one smoothed.
- TTS weight: mean log-ratio of a clip's injected errors' real-confusion probability
  against a uniform error model, exponentiated and clipped to [0.5, 3]; error-free 1.
"""
import math
from collections import Counter, defaultdict

import numpy as np

from .align import accuracy_counts, align_ops

DELETE = "<del>"


def clip_per(label, decoded):
    """Official-style PER of one clip: (S + D + I) / len(label)."""
    if not label:
        return float("nan")
    edits, n = accuracy_counts(list(label), list(decoded))
    return edits / n


def agreement_weight(per, cutoff=0.30, low=0.2):
    return low if per > cutoff else 1.0


class ConfusionTable:
    """Counts of what real reciters said instead of each expected phoneme."""

    def __init__(self, phonemes):
        self.phonemes = sorted(phonemes)
        self.sub = defaultdict(Counter)      # expected -> Counter(outcome or <del>)
        self.ins = Counter()                 # inserted phoneme
        self.correct = Counter()

    def add(self, expected, said):
        a1, a2, ops = align_ops(list(expected), list(said))
        for e, s, op in zip(a1, a2, ops):
            if op == "C":
                self.correct[e] += 1
            elif op == "S":
                self.sub[e][s] += 1
            elif op == "D":
                self.sub[e][DELETE] += 1
            else:
                self.ins[s] += 1
        return self

    @property
    def n_outcomes(self):
        """Error outcomes for one expected phoneme: any other phoneme, or deletion."""
        return len(self.phonemes)

    def p_error(self, expected, outcome):
        """Add-one smoothed P(outcome | expected phoneme was mispronounced)."""
        c = self.sub.get(expected, Counter())
        return (c[outcome] + 1) / (sum(c.values()) + self.n_outcomes)

    def p_insert(self, phoneme):
        return (self.ins[phoneme] + 1) / (sum(self.ins.values()) + len(self.phonemes))

    def top_pairs(self, k=10):
        pairs = Counter({(e, o): n for e, c in self.sub.items() for o, n in c.items()})
        return pairs.most_common(k)

    def to_json(self):
        return {"substitutions": {e: dict(c) for e, c in self.sub.items()},
                "insertions": dict(self.ins), "correct": dict(self.correct),
                "phonemes": self.phonemes}


def tts_weight(phoneme_ref, phoneme_mis, table: ConfusionTable, lo=0.5, hi=3.0):
    """Weight of one TTS clip by how realistic its injected errors are (1.0 if none)."""
    a1, a2, ops = align_ops(list(phoneme_ref), list(phoneme_mis))
    logs = []
    for e, s, op in zip(a1, a2, ops):
        if op == "S":
            logs.append(math.log(table.p_error(e, s) * table.n_outcomes))
        elif op == "D":
            logs.append(math.log(table.p_error(e, DELETE) * table.n_outcomes))
        elif op == "I":
            logs.append(math.log(table.p_insert(s) * len(table.phonemes)))
    if not logs:
        return 1.0
    return float(np.clip(math.exp(float(np.mean(logs))), lo, hi))
