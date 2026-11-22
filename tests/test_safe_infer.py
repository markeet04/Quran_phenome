"""Crash-isolated inference: a child that dies on a clip is restarted without it and the clip
is run on CPU; results equal plain single-clip inference."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from iqra import safe_infer as SI  # noqa: E402
from iqra.posteriors import clip_logits  # noqa: E402
from test_model import _tiny  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("si")
    m = _tiny()
    m.save_pretrained(d / "ckpt")
    rng = np.random.default_rng(0)
    audio = {f"c{i}": rng.normal(0, .1, int(16000 * s)).astype(np.float32) for i, s in enumerate((0.6, 1.4, 0.9, 2.0))}
    SI.save_audio(audio, d / "audio.npz")
    fe = transformers.AutoFeatureExtractor.from_pretrained(SI.FE_NAME)
    ref = {c: clip_logits(m, fe, a, "cpu") for c, a in audio.items()}
    return d, audio, ref


def test_audio_round_trip(setup):
    d, audio, _ = setup
    back = SI.load_audio(d / "audio.npz")
    assert list(back) == list(audio) and all(np.array_equal(back[c], audio[c]) for c in audio)


def test_child_crash_is_isolated_and_recovered(setup, monkeypatch):
    d, audio, ref = setup
    monkeypatch.setenv("IQRA_TEST_CRASH_ON", "c1,c3")
    logs = []
    out, rep = SI.run_model("m", str(d / "ckpt"), str(d / "audio.npz"), str(d / "work"), gpu_index=0,
                            child_device="cpu", log=logs.append)
    assert sorted(rep["cpu_clips"]) == ["c1", "c3"] and rep["gpu_restarts"] == 2
    assert sorted(out) == sorted(audio)
    for c in audio:
        np.testing.assert_allclose(out[c], ref[c], atol=1e-5)


def test_cpu_only(setup, monkeypatch):
    d, audio, ref = setup
    monkeypatch.delenv("IQRA_TEST_CRASH_ON", raising=False)
    out, rep = SI.run_model("m2", str(d / "ckpt"), str(d / "audio.npz"), str(d / "work2"), gpu_index=None)
    assert rep["cpu_clips"] == "all" and all(np.allclose(out[c], ref[c], atol=1e-5) for c in audio)


def test_crash_storm_sends_long_clips_to_cpu(setup, monkeypatch):
    d, audio, ref = setup
    monkeypatch.setenv("IQRA_TEST_CRASH_ON", "c2,c1,c3")          # lengths 0.9, 1.4, 2.0 s: all but c0
    out, rep = SI.run_model("m3", str(d / "ckpt"), str(d / "audio.npz"), str(d / "work3"), gpu_index=0,
                            child_device="cpu", storm_after=2, log=lambda *a: None)
    # first crash on c2 (shortest crashing clip) -> excluded; second crash on c1 -> c1 and every
    # longer clip (c3) go to CPU in one step
    assert sorted(rep["cpu_clips"]) == ["c1", "c2", "c3"] and rep["gpu_restarts"] == 2
    assert all(np.allclose(out[c], ref[c], atol=1e-5) for c in audio)


def test_eager_attention_matches_default(setup, tmp_path):
    d, audio, ref = setup
    from iqra.model import load_model
    m, info = load_model(d / "ckpt", "cpu", attn_implementation="eager")
    assert not any(info.values())
    fe = transformers.AutoFeatureExtractor.from_pretrained(SI.FE_NAME)
    for c, a in audio.items():
        np.testing.assert_allclose(clip_logits(m, fe, a, "cpu"), ref[c], atol=1e-4)
