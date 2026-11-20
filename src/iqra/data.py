"""Data access: training-split loaders, splits (T1), and the guarded test loader.

Only NB16 may touch the test set, and only with FINAL_RUN=1 in the environment
(SPEC rule 1, decision 19). The pipeline gets ID + Reference_phn (the canonical input
the task supplies); the full truth file, with Annotation_phn, goes only to scorer.py.
"""
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

from .config import Columns, Datasets

SPEAKER_COLUMNS = ("speaker", "speaker_id", "spk", "spk_id", "client_id")
ID_COLUMNS = ("ID", "id", "utt_id", "file_id")
TEXT_COLUMNS = ("sentence", "tashkeel_sentence", "text", "transcript", "sentence_ref")


def find_column(columns, candidates):
    """First of `candidates` present in `columns`, else None."""
    cols = list(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


# --------------------------------------------------------------------------- ids
_UNSAFE_ID = re.compile(r"\s|ref|hyp")


def safe_ids(raw_ids, prefix):
    """IDs the official line parser can handle (decision 23).

    Keeps the originals when every one is unique and safe; otherwise replaces all of
    them with `{prefix}{n:05d}`. Returns (ids, mapping safe_id -> original or None).
    """
    raw = [None if r is None else str(r) for r in raw_ids]
    ok = (all(r for r in raw) and len(set(raw)) == len(raw)
          and not any(_UNSAFE_ID.search(r) for r in raw))
    if ok:
        return raw, None
    ids = [f"{prefix}{n:05d}" for n in range(len(raw))]
    return ids, dict(zip(ids, raw))


# --------------------------------------------------------------------------- splits
def group_split(groups, dev_frac=0.20, seed=13):
    """Split rows into extra_dev, fold_a, fold_b with no group shared between them.

    `groups` is one key per row (a speaker, or the canonical sentence). Groups are
    shuffled with `seed`; extra_dev takes groups until it holds >= dev_frac of the rows;
    the remaining groups go one by one to whichever fold has fewer rows (ties to A).
    Returns {"extra_dev": [...], "fold_a": [...], "fold_b": [...]} of row indices.
    """
    groups = [str(g) for g in groups]
    members = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    order = sorted(members)
    np.random.default_rng(seed).shuffle(order)
    target = dev_frac * len(groups)
    out = {"extra_dev": [], "fold_a": [], "fold_b": []}
    k = 0
    while k < len(order) and len(out["extra_dev"]) < target:
        out["extra_dev"] += members[order[k]]
        k += 1
    for g in order[k:]:
        side = "fold_a" if len(out["fold_a"]) <= len(out["fold_b"]) else "fold_b"
        out[side] += members[g]
    return {name: sorted(rows) for name, rows in out.items()}


def assert_group_disjoint(split, groups):
    seen = {}
    for name, rows in split.items():
        for g in {str(groups[i]) for i in rows}:
            if g in seen:
                raise AssertionError(f"group {g!r} is in both {seen[g]} and {name}")
            seen[g] = name
    all_rows = [i for rows in split.values() for i in rows]
    if sorted(all_rows) != list(range(len(groups))):
        raise AssertionError("splits do not partition the rows exactly")


def duration_slice(durations, hours, seed=13):
    """Row indices of a random slice totalling about `hours` (stops once reached)."""
    order = np.random.default_rng(seed).permutation(len(durations))
    picked, total = [], 0.0
    for i in order:
        if total >= hours * 3600:
            break
        picked.append(int(i))
        total += float(durations[i])
    return sorted(picked), total / 3600


def decode_audio(entry):
    """Decode one stored audio cell ({"bytes", "path"} or {"array", "sampling_rate"}).

    Returns (float32 mono array, sampling rate) at the stored rate. Independent of the
    installed `datasets` version (4.x decodes through torchcodec; we never let it).
    """
    import io
    if entry.get("array") is not None:
        return np.asarray(entry["array"], dtype=np.float32), int(entry["sampling_rate"])
    import soundfile as sf
    src = io.BytesIO(entry["bytes"]) if entry.get("bytes") else entry.get("path")
    if src is None:
        raise ValueError("audio cell has neither bytes nor path")
    arr, sr = sf.read(src, dtype="float32", always_2d=True)
    return arr.mean(axis=1), int(sr)


def iter_audio(ds, column="audio", batch=256):
    """Yield (row index, array, sampling rate) from a datasets.Dataset without using its
    audio decoding: reads the raw Arrow column and decodes with soundfile."""
    table = ds.data
    n = len(ds)
    indices = getattr(ds, "_indices", None)
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        if indices is not None:
            rows = [int(indices.column(0)[i].as_py()) for i in range(start, stop)]
            cells = [table.slice(r, 1).column(column).to_pylist()[0] for r in rows]
        else:
            cells = table.slice(start, stop - start).column(column).to_pylist()
        for k, cell in enumerate(cells):
            arr, sr = decode_audio(cell)
            yield start + k, arr, sr


def audio_fingerprint(array) -> str:
    """Hash of the decoded waveform, to catch identical audio under different IDs."""
    a = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return hashlib.md5(a.tobytes()).hexdigest()


# --------------------------------------------------------------------------- quran
_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_LETTER_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي",
                             "ة": "ه", "ؤ": "و", "ئ": "ي"})


def normalise_arabic(text) -> str:
    """Strip diacritics and Quranic marks, unify letter variants, keep Arabic letters."""
    t = _DIACRITICS.sub("", str(text)).translate(_LETTER_MAP)
    t = re.sub(r"[^ء-ي\s]", " ", t)
    return " ".join(t.split())


def quranic_flags(sentences, quran_lines, min_words=2):
    """True where a sentence (normalised, >= min_words words) occurs verbatim in the Quran.

    `quran_lines` are verse texts (e.g. Tanzil simple-clean); matching runs over the
    verses joined by spaces, on word boundaries.
    """
    corpus = " " + " ".join(normalise_arabic(v) for v in quran_lines) + " "
    flags = []
    for s in sentences:
        n = normalise_arabic(s)
        flags.append(len(n.split()) >= min_words and f" {n} " in corpus)
    return flags


def load_tanzil_simple_clean(url=("https://tanzil.net/pub/download/index.php?"
                                  "quranType=simple-clean&outType=txt-2&agree=true")):
    """Quran text from Tanzil (simple-clean, 'sura|aya|text' lines). Needs internet."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read().decode("utf-8")
    lines = [ln.split("|", 2)[2] for ln in raw.splitlines() if ln.count("|") >= 2]
    if len(lines) != 6236:
        raise ValueError(f"expected 6,236 verses from Tanzil, got {len(lines)}")
    return lines


# --------------------------------------------------------------------------- truth
def truth_frame(ids, canonical, said) -> pd.DataFrame:
    """Scorer-format truth: ID, Reference_phn (canonical), Annotation_phn (what was said)."""
    return pd.DataFrame({"ID": list(ids), "Reference_phn": list(canonical),
                         "Annotation_phn": list(said)})


class TestSetLocked(PermissionError):
    pass


def _require_final_run():
    if os.environ.get("FINAL_RUN") != "1":
        raise TestSetLocked("the test set is locked: set FINAL_RUN=1, and only in NB16")


def _default_truth_loader():
    from datasets import load_dataset
    ds = load_dataset(Datasets().test_truth, split="test")
    return ds.to_pandas()


def load_test_canonical(loader=None) -> pd.DataFrame:
    """ID + Reference_phn for the pipeline. Never returns Annotation_phn."""
    _require_final_run()
    df = (loader or _default_truth_loader)()
    c = Columns()
    return df[["ID", c.test_canonical]].reset_index(drop=True).copy()


def load_test_truth_for_scorer(loader=None) -> pd.DataFrame:
    """Full truth (ID, Reference_phn, Annotation_phn). Pass it only to scorer.py,
    after every prediction CSV has been written."""
    _require_final_run()
    df = (loader or _default_truth_loader)()
    c = Columns()
    return df[["ID", c.test_canonical, c.test_annotation]].reset_index(drop=True).copy()


def load_test_audio():
    """QuranMB.v2 audio (ID, audio). NB16 only."""
    _require_final_run()
    from datasets import Audio, load_dataset
    from .config import SAMPLING_RATE
    ds = load_dataset(Datasets().test_audio, split="test")
    return ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))


def save_splits(splits: dict, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=1, sort_keys=True)


def load_splits(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
