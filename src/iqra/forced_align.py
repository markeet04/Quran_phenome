"""CTC forced alignment of the expected phonemes over frame log-posteriors (SPEC section 5).

Viterbi over the standard CTC state graph (blank, t1, blank, t2, ..., tL, blank): the
single most likely frame-level path that spells exactly the expected token sequence.
Returns, for every expected token, the frames it occupies. Pure NumPy; tested against
brute-force enumeration of all alignments. (torchaudio's forced_align is not used:
its availability alongside Kaggle's torch is uncertain; decision 24's fallback.)
"""
import numpy as np

NEG = -1e30


def ctc_forced_align(log_probs, targets, blank=0):
    """Most likely CTC path for `targets` through `log_probs` [T, V].

    Returns (path, score): path is a length-T list of token ids (blank or a target id),
    score the path log-probability. Raises ValueError if T is too short.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    T = lp.shape[0]
    L = len(targets)
    repeats = sum(1 for i in range(1, L) if targets[i] == targets[i - 1])
    if T < L + repeats:
        raise ValueError(f"{T} frames cannot align {L} tokens ({repeats} repeats)")
    S = 2 * L + 1
    labels = np.full(S, blank, dtype=np.int64)
    labels[1::2] = targets
    # skip allowed into odd state s from s-2 when the token differs from the previous token
    can_skip = np.zeros(S, dtype=bool)
    for s in range(3, S, 2):
        can_skip[s] = labels[s] != labels[s - 2]

    alpha = np.full(S, NEG)
    alpha[0] = lp[0, labels[0]]
    if S > 1:
        alpha[1] = lp[0, labels[1]]
    back = np.zeros((T, S), dtype=np.int8)          # 0 stay, 1 from s-1, 2 from s-2
    for t in range(1, T):
        stay = alpha
        prev1 = np.concatenate(([NEG], alpha[:-1]))
        prev2 = np.where(can_skip, np.concatenate(([NEG, NEG], alpha[:-2])), NEG)
        stacked = np.stack([stay, prev1, prev2])
        choice = stacked.argmax(axis=0)
        alpha = stacked[choice, np.arange(S)] + lp[t, labels]
        back[t] = choice
    # end in the last token or the final blank
    ends = [S - 1, S - 2] if S > 1 else [0]
    s = max(ends, key=lambda e: alpha[e])
    score = float(alpha[s])
    if score <= NEG / 2:
        raise ValueError("no valid alignment")
    states = [s]
    for t in range(T - 1, 0, -1):
        s -= int(back[t, s])
        states.append(s)
    states.reverse()
    return [int(labels[s]) for s in states], score, states


def token_spans(log_probs, targets, blank=0):
    """Frames occupied by each expected token: list of (first, last) per target index.

    Returns (spans, failed). On failure (clip too short), spans are evenly spaced over
    the clip and failed=True, so callers can set the fa_failed feature.
    """
    T = np.shape(log_probs)[0]
    L = len(targets)
    try:
        _, _, states = ctc_forced_align(log_probs, targets, blank)
    except ValueError:
        edges = np.linspace(0, T, L + 1)
        return [(int(edges[i]), max(int(edges[i]), int(np.ceil(edges[i + 1])) - 1)) for i in range(L)], True
    spans = [None] * L
    for t, s in enumerate(states):
        if s % 2 == 1:
            j = (s - 1) // 2
            spans[j] = (spans[j][0], t) if spans[j] else (t, t)
    return spans, False
