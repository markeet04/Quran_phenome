import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The 71 tokens of the trained model's vocab.json (NB1's phoneme_vocab.json), in id
# order: <pad>=0 (CTC blank), <unk>=1, then sorted symbols, so <sil> is id 8.
VOCAB_TOKENS = ["<pad>", "<unk>", "$", "$$", "*", "**", "<", "<<", "<sil>", "A", "AA", "D",
                "DD", "E", "EE", "H", "HH", "I", "II", "S", "SS", "T", "TT", "U", "UU", "Z",
                "ZZ", "^", "^^", "a", "aa", "b", "bb", "d", "dd", "f", "ff", "g", "gg", "h",
                "hh", "i", "ii", "j", "jj", "k", "kk", "l", "ll", "m", "mm", "n", "nn", "q",
                "qq", "r", "rr", "s", "ss", "t", "tt", "u", "uu", "w", "ww", "x", "xx", "y",
                "yy", "z", "zz"]
PHONEMES = [t for t in VOCAB_TOKENS if not t.startswith("<") or t in ("<", "<<")]


def _mutate(seq, rng, p_sub, p_del, p_ins):
    out = []
    for tok in seq:
        r = rng.random()
        if r < p_del:
            continue
        if r < p_del + p_sub:
            out.append(rng.choice([p for p in PHONEMES if p != tok]))
        else:
            out.append(tok)
        if rng.random() < p_ins:
            out.append(rng.choice(PHONEMES))
    return out or [rng.choice(PHONEMES)]


def make_synthetic(n_clips=50, seed=0, mixed=False, min_len=4, max_len=30):
    """Canonical, reciter annotation (with errors) and a noisy prediction per clip.

    mixed=True bases half the predictions on the canonical text instead of the
    annotation, so the model also misses real errors (exercises every FA branch).
    """
    rng = random.Random(seed)
    rows, preds = [], []
    for k in range(n_clips):
        canon = [rng.choice(PHONEMES) for _ in range(rng.randint(min_len, max_len))]
        trans = _mutate(canon, rng, p_sub=0.10, p_del=0.06, p_ins=0.06)
        base = canon if (mixed and k % 2) else trans
        pred = _mutate(base, rng, p_sub=0.10, p_del=0.05, p_ins=0.05)
        cid = f"clip{k:03d}"
        rows.append({"ID": cid, "Reference_phn": " ".join(canon), "Annotation_phn": " ".join(trans)})
        preds.append({"ID": cid, "Prediction": " ".join(pred)})
    return pd.DataFrame(rows), pd.DataFrame(preds)


@pytest.fixture
def synthetic():
    return make_synthetic(50, seed=0)
