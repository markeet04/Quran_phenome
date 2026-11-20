"""Version-parity fixture: deterministic inputs and local self-consistency (decision 27)."""
from pathlib import Path

import numpy as np
import pytest

from iqra import parity

MODEL_DIR = Path(__file__).resolve().parents[1] / "quran_demo_app" / "model"


def test_synthetic_audio_is_deterministic():
    for seed, secs in parity.SYNTHETIC:
        a, b = parity.synthetic_audio(seed, secs), parity.synthetic_audio(seed, secs)
        assert a.dtype == np.float32 and len(a) == int(16000 * secs)
        assert np.array_equal(a, b)
        assert 0.49 < np.max(np.abs(a)) <= 0.5


def test_fixture_is_present_and_complete():
    meta, arrays = parity.load_fixture()
    assert meta["transformers"] == "4.44.2"
    for seed, _ in parity.SYNTHETIC:
        for part in ("features", "mask", "logits"):
            assert f"synth{seed}_{part}" in arrays
    assert {"real0_logits", "real0_features_input"} <= set(arrays)
    assert all(k.endswith(("_logits", "_features", "_mask", "_features_input")) for k in arrays)


def test_compare_flags_differences():
    _, ref = parity.load_fixture()
    ok, _ = parity.compare(ref, dict(ref))
    assert ok
    bad = dict(ref)
    bad["synth1_logits"] = ref["synth1_logits"] + 0.01
    ok, rows = parity.compare(ref, bad)
    assert not ok


@pytest.mark.skipif(not (MODEL_DIR / "model.safetensors").exists(), reason="NB8 model not on this machine")
def test_local_rerun_reproduces_fixture():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.44.2":
        pytest.skip("reference run needs transformers 4.44.2")
    from transformers import AutoFeatureExtractor
    meta, ref = parity.load_fixture()
    assert parity.sha256_file(MODEL_DIR / "model.safetensors") == meta["weights_sha256"]
    model, info = parity.load_with_report(MODEL_DIR)
    assert not any(info.values())
    fe = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
    real = {k[: -len("_features_input")]: v for k, v in ref.items() if k.endswith("_features_input")}
    ok, rows = parity.compare(ref, parity.run(model, fe, "cpu", real))
    assert ok, rows
