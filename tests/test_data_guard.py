"""The test loader is locked without FINAL_RUN=1 and never leaks Annotation_phn."""
import pandas as pd
import pytest

from iqra import data


def _fake_truth():
    return pd.DataFrame({"ID": ["a1", "a2"], "Reference_phn": ["b i", "s a"],
                         "Annotation_phn": ["b u", "s a"], "Extra": [1, 2]})


def test_locked_without_final_run(monkeypatch):
    monkeypatch.delenv("FINAL_RUN", raising=False)
    for fn in (data.load_test_canonical, data.load_test_truth_for_scorer):
        with pytest.raises(data.TestSetLocked):
            fn(loader=_fake_truth)
    with pytest.raises(data.TestSetLocked):
        data.load_test_audio()


def test_pipeline_gets_only_id_and_reference(monkeypatch):
    monkeypatch.setenv("FINAL_RUN", "1")
    df = data.load_test_canonical(loader=_fake_truth)
    assert list(df.columns) == ["ID", "Reference_phn"]
    full = data.load_test_truth_for_scorer(loader=_fake_truth)
    assert list(full.columns) == ["ID", "Reference_phn", "Annotation_phn"]
