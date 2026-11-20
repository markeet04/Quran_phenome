"""Decode parity with NB5, emission spans, ensemble averaging (SPEC T0)."""
import numpy as np
import pytest

from conftest import VOCAB_TOKENS
from iqra.decode import ensemble_log_posteriors, greedy_decode, log_softmax


def nb5_recognise_from_logits(logits, id2tok, blank_id):
    """NB5's recognise_phonemes body (nb5_official_scorer.ipynb), from logits onward."""
    ids = logits.argmax(-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank_id:
            sym = id2tok.get(int(i), "<unk>")
            if sym not in ("<sil>", "sil", "<pad>", "<unk>"):
                out.append(sym)
        prev = i
    return " ".join(out)


UNK, SIL = VOCAB_TOKENS.index("<unk>"), VOCAB_TOKENS.index("<sil>")


def _frames(rng, V, T):
    """Frame ids in runs, heavy on blank (0), with <unk> and <sil> mixed in."""
    ids = []
    while len(ids) < T:
        r = rng.random()
        tok = 0 if r < 0.45 else (UNK if r < 0.5 else (SIL if r < 0.55 else rng.integers(0, V)))
        ids += [int(tok)] * int(rng.integers(1, 5))
    return np.array(ids[:T])


def test_decode_parity_with_nb5():
    rng = np.random.default_rng(0)
    id2tok_list = VOCAB_TOKENS
    id2tok_dict = {i: t for i, t in enumerate(VOCAB_TOKENS)}
    V = len(VOCAB_TOKENS)
    for trial in range(500):
        T = int(rng.integers(1, 120))
        if trial % 2:
            logits = rng.normal(size=(T, V))                   # unstructured random logits
        else:
            logits = rng.normal(scale=0.5, size=(T, V))
            logits[np.arange(T), _frames(rng, V, T)] += 6.0    # realistic runs and blanks
        ref = nb5_recognise_from_logits(logits, id2tok_dict, 0)
        toks, spans = greedy_decode(logits, id2tok_list, 0)
        assert " ".join(toks) == ref, trial
        assert len(spans) == len(toks)
        # log-posteriors decode the same as logits
        assert greedy_decode(log_softmax(logits), id2tok_list, 0)[0] == toks


def test_emission_spans():
    V = len(VOCAB_TOKENS)
    b, s = VOCAB_TOKENS.index("b"), VOCAB_TOKENS.index("s")
    sil = VOCAB_TOKENS.index("<sil>")
    ids = [0, b, b, b, 0, s, sil, sil, s, s, 0]
    logits = np.full((len(ids), V), -5.0)
    logits[np.arange(len(ids)), ids] = 5.0
    toks, spans = greedy_decode(logits, VOCAB_TOKENS, 0)
    assert toks == ["b", "s", "s"]
    assert spans == [(1, 3), (5, 5), (8, 9)]


def test_ensemble_average():
    rng = np.random.default_rng(1)
    a = log_softmax(rng.normal(size=(20, 7)))
    assert np.allclose(ensemble_log_posteriors([a, a, a]), a)
    b = log_softmax(rng.normal(size=(20, 7)))
    avg = ensemble_log_posteriors([a, b])
    assert np.allclose(np.exp(avg), (np.exp(a) + np.exp(b)) / 2)
    with pytest.raises(AssertionError):
        ensemble_log_posteriors([a, a[:-1]])
