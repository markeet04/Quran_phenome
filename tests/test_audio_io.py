"""iter_audio decodes stored WAV bytes with soundfile, whatever the datasets version."""
import io

import numpy as np
import pytest

datasets = pytest.importorskip("datasets")
sf = pytest.importorskip("soundfile")

from iqra import data  # noqa: E402


def _wav(seconds, sr, seed):
    x = np.random.default_rng(seed).normal(0, 0.05, int(seconds * sr)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue(), x


def _dataset(n=7):
    cells, arrays = [], []
    for i in range(n):
        b, x = _wav(0.5 + 0.1 * i, 16000 if i % 2 else 22050, i)
        cells.append({"bytes": b, "path": None}); arrays.append(x)
    # Build the column as a plain bytes/path struct, then cast it to Audio: the cast goes
    # through Arrow storage, so no audio encoder (torchcodec in datasets 4.x) is needed.
    feats = datasets.Features({"audio": {"bytes": datasets.Value("binary"),
                                         "path": datasets.Value("string")},
                               "k": datasets.Value("int32")})
    ds = datasets.Dataset.from_dict({"audio": cells, "k": list(range(n))}, features=feats)
    ds = ds.cast_column("audio", datasets.Audio(decode=False))
    assert isinstance(ds.features["audio"], datasets.Audio)
    return ds, arrays


def test_iter_audio_reads_every_row_exactly():
    ds, arrays = _dataset()
    out = list(data.iter_audio(ds, batch=3))
    assert [i for i, _, _ in out] == list(range(len(arrays)))
    for (i, arr, sr), x in zip(out, arrays):
        assert sr == (16000 if i % 2 else 22050)
        assert np.allclose(arr, x)


def test_iter_audio_respects_selected_rows():
    ds, arrays = _dataset()
    sub = ds.select([5, 1, 3])                       # creates an indices mapping
    got = [arr for _, arr, _ in data.iter_audio(sub, batch=2)]
    for arr, src in zip(got, [5, 1, 3]):
        assert np.allclose(arr, arrays[src])
