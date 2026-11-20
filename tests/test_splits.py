"""T1 split logic: group disjointness, sizes, duration slice, safe IDs, Quran matching."""
import numpy as np
import pytest

from iqra import data


def _groups(n=1333, n_groups=180, seed=0):
    rng = np.random.default_rng(seed)
    return [f"g{int(x)}" for x in rng.integers(0, n_groups, n)]


def test_group_split_is_disjoint_and_sized():
    groups = _groups()
    split = data.group_split(groups, dev_frac=0.20, seed=13)
    data.assert_group_disjoint(split, groups)
    n = len(groups)
    assert 0.20 * n <= len(split["extra_dev"]) <= 0.20 * n + 60    # stops at the 20% mark
    a, b = len(split["fold_a"]), len(split["fold_b"])
    assert abs(a - b) <= 60                                        # balanced folds


def test_group_split_is_deterministic():
    groups = _groups()
    assert data.group_split(groups, seed=13) == data.group_split(groups, seed=13)
    assert data.group_split(groups, seed=13) != data.group_split(groups, seed=14)


def test_disjointness_assert_catches_leaks():
    groups = ["x", "x", "y", "z"]
    with pytest.raises(AssertionError):
        data.assert_group_disjoint({"extra_dev": [0], "fold_a": [1, 2], "fold_b": [3]}, groups)


def test_duration_slice():
    durations = np.full(2588, 5.0)                     # 5 s clips
    rows, hours = data.duration_slice(durations, 1.0, seed=13)
    assert len(rows) == 720 and hours == pytest.approx(1.0)
    assert len(set(rows)) == len(rows)


def test_safe_ids():
    ids, mapping = data.safe_ids(["a1", "a2"], "extra")
    assert ids == ["a1", "a2"] and mapping is None
    ids, mapping = data.safe_ids(["prefix_1", "x y", "x y"], "extra")
    assert ids == ["extra00000", "extra00001", "extra00002"]
    assert mapping["extra00001"] == "x y"              # reversible
    ids, mapping = data.safe_ids([None, None], "extra")
    assert mapping == {"extra00000": None, "extra00001": None}


def test_quranic_flags():
    quran = ["بسم الله الرحمن الرحيم", "الحمد لله رب العالمين"]
    sentences = ["بِسْمِ ٱللَّهِ الرَّحْمَٰنِ",          # with diacritics and alif wasla
                 "ذهب الولد إلى المدرسة",             # not Quranic
                 "الله"]                               # too short to count
    assert data.quranic_flags(sentences, quran) == [True, False, False]


def test_quranic_window_flags_handle_error_markers():
    quran = ["إذ يغشيكم النعاس أمنة منه وينزل عليكم من السماء ماء",
             "وليستعفف الذين لا يجدون نكاحا حتى يغنيهم الله من فضله",
             "الله الذي خلقكم من ضعف ثم جعل من بعد ضعف قوة",
             "وإذ ابتلى إبراهيم ربه بكلمات فأتمهن"]
    real = ["يُغ@خْشِيكُمُ النُّعَاسَ أَمَنَةً مِّنْهُ",           # inserted letter after @
            "وَلْيَسْتَعْف@  ِ الَّذِينَ لَا يَجِدُونَ نِكَاحًا",      # marker splits a word
            "اللَّهُ الَّذِي خَلَق@كُكُمْ مِنْ ضُعْفٍ ",
            "وَإِذِ ٱبۡتَلَىٰٓ إِبۡرَٰهِيمَ@ُ رَبُّهُ"]                 # @ before a vowel mark
    msa = ["ذهب الولد إلى المدرسة في الصباح الباكر", "قال الرجل إن الطقس جميل اليوم"]
    assert data.quranic_window_flags(real + msa, quran) == [True] * 4 + [False] * 2


def test_quranic_flags_uthmani_against_simple_spelling():
    # Real Iqra_Extra_IS26 sentence (Uthmani, with the '@' error marker) vs Tanzil 2:124
    quran = ["وإذ ابتلى إبراهيم ربه بكلمات فأتمهن"]
    uthmani = "وَإِذِ ٱبۡتَلَىٰٓ إِبۡرَٰهِيمَ@ُ رَبُّهُ"
    assert data.quranic_flags([uthmani], quran) == [True]


def test_audio_fingerprint():
    a = np.zeros(16000, dtype=np.float32)
    b = a.copy(); b[5] = 1e-3
    assert data.audio_fingerprint(a) == data.audio_fingerprint(a.copy())
    assert data.audio_fingerprint(a) != data.audio_fingerprint(b)
