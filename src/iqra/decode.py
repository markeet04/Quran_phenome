"""Greedy CTC decoding (NB5 convention) with emission spans, and ensemble averaging."""
import numpy as np

from .config import DROP_TOKENS


def greedy_decode(scores, id2tok, blank_id, drop=DROP_TOKENS):
    """Argmax per frame, collapse repeats, drop blank, then drop `drop` tokens.

    Identical output to NB5's recognise_phonemes. `scores` is [T, V] logits or
    log-posteriors (argmax is the same). Returns (tokens, spans) where spans[k] is the
    (first_frame, last_frame) run that emitted tokens[k].
    """
    ids = np.asarray(scores).argmax(-1).tolist()
    tokens, spans = [], []
    prev, start = None, 0
    for t, i in enumerate(ids):
        if i != prev:
            start = t
        if i != prev and i != blank_id:
            sym = id2tok[i] if 0 <= i < len(id2tok) else "<unk>"
            if sym not in drop:
                tokens.append(sym)
                spans.append([t, t])
            else:
                spans.append(None)          # placeholder so the run can be extended
                tokens.append(None)
        elif i == prev and i != blank_id and spans and spans[-1] is not None:
            spans[-1][1] = t
        prev = i
    keep = [k for k, tok in enumerate(tokens) if tok is not None]
    return [tokens[k] for k in keep], [tuple(spans[k]) for k in keep]


def log_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


def ensemble_log_posteriors(members):
    """Average member probabilities frame by frame: logsumexp over members minus log K.

    `members` is a list of [T, V] log-posterior arrays for one clip; all T must match.
    """
    if not members:
        raise ValueError("no ensemble members")
    shapes = {np.shape(m) for m in members}
    if len(shapes) != 1:
        raise AssertionError(f"ensemble members disagree on [T, V]: {sorted(shapes)}")
    stack = np.stack([np.asarray(m, dtype=np.float64) for m in members])
    top = stack.max(axis=0)
    return top + np.log(np.exp(stack - top).sum(axis=0)) - np.log(len(members))
