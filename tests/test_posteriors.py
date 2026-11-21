"""Posterior cache: frame rate, round trip, cached decode == live decode (SPEC T2)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from conftest import VOCAB_TOKENS  # noqa: E402
from iqra import posteriors  # noqa: E402
from iqra.decode import greedy_decode  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    cfg = transformers.Wav2Vec2BertConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        output_hidden_size=32, vocab_size=len(VOCAB_TOKENS), pad_token_id=0, add_adapter=True,
        num_adapter_layers=1, adapter_stride=2, conv_depthwise_kernel_size=3)
    model = transformers.Wav2Vec2BertForCTC(cfg).eval()
    fe = transformers.SeamlessM4TFeatureExtractor()
    return model, fe


def _audio(seconds, seed):
    return np.random.default_rng(seed).normal(0, 0.1, int(16000 * seconds)).astype(np.float32)


def test_frame_rate_matches_config(tiny_model):
    model, fe = tiny_model
    rate = posteriors.model_frame_rate(model.config)
    assert rate == 25.0                                  # 50 frames/s, one adapter of stride 2
    for secs in (2.0, 5.0):
        T = posteriors.clip_logits(model, fe, _audio(secs, 1), "cpu").shape[0]
        assert abs(T / secs - rate) < 1.0


def test_cache_round_trip_and_decode_parity(tiny_model, tmp_path):
    model, fe = tiny_model
    entries, live = {}, {}
    for k in range(6):
        logits = posteriors.clip_logits(model, fe, _audio(1 + k * 0.5, k), "cpu")
        live[f"c{k}"] = greedy_decode(logits, VOCAB_TOKENS, 0)[0]
        entries[f"c{k}"] = posteriors.make_entry(logits, VOCAB_TOKENS, 0)
    path = tmp_path / "post_tiny_dev.pt"
    posteriors.write_cache(path, entries, checkpoint="tiny", split="dev", vocab_hash="abc",
                           frame_rate=25.0)
    meta, clips = posteriors.read_cache(path)            # weights_only=True inside
    assert meta["n_clips"] == 6 and meta["vocab_hash"] == "abc"
    for cid, entry in clips.items():
        toks, spans = posteriors.decode_entry(entry, VOCAB_TOKENS, 0)
        assert toks == live[cid] == entry["tokens"]
        assert [list(s) for s in spans] == entry["spans"]


def test_cached_logits_decode_exactly_even_with_near_ties(tiny_model, tmp_path):
    """The untrained tiny model's flat outputs are full of near-ties: float16 log-posteriors
    flip some decodes, the stored float32 logits never do (decision 26)."""
    model, fe = tiny_model
    flips = 0
    entries, live = {}, {}
    for k in range(20):
        logits = posteriors.clip_logits(model, fe, _audio(2.0, 100 + k), "cpu")
        live[k] = greedy_decode(logits, VOCAB_TOKENS, 0)[0]
        entries[f"c{k}"] = posteriors.make_entry(logits, VOCAB_TOKENS, 0)
        flips += posteriors.float16_would_flip(entries[f"c{k}"], VOCAB_TOKENS, 0)
    path = tmp_path / "post.pt"
    posteriors.write_cache(path, entries, checkpoint="t", split="s", vocab_hash="h", frame_rate=25.0)
    _, clips = posteriors.read_cache(path)
    for k in range(20):
        assert posteriors.decode_entry(clips[f"c{k}"], VOCAB_TOKENS, 0)[0] == live[k]
    assert flips > 0          # shows why float16 storage was dropped


def test_log_posteriors_normalised(tiny_model):
    model, fe = tiny_model
    entry = posteriors.make_entry(posteriors.clip_logits(model, fe, _audio(1.0, 7), "cpu"),
                                  VOCAB_TOKENS, 0)
    lp = posteriors.log_posteriors(entry)
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0)


def test_output_length_matches_single_clip_runs(tiny_model):
    model, fe = tiny_model
    for secs in (0.5, 1.03, 2.0, 3.37, 5.0):
        audio = _audio(secs, 11)
        feats = fe(audio, sampling_rate=16000, return_tensors="pt")
        T = posteriors.clip_logits(model, fe, audio, "cpu").shape[0]
        n_in = feats["input_features"].shape[1]          # frames the model actually saw
        assert posteriors.output_length(n_in, model.config) == T


def test_batch_logits_trimmed_and_close_to_single(tiny_model):
    model, fe = tiny_model
    audios = [_audio(s, 20 + i) for i, s in enumerate((0.7, 2.3, 1.1, 3.9))]
    batched = posteriors.batch_logits(model, fe, audios, "cpu")
    for a, b in zip(audios, batched):
        single = posteriors.clip_logits(model, fe, a, "cpu")
        assert b.shape == single.shape
        assert np.max(np.abs(b - single)) < 1e-3


def test_check_compatible():
    a = {"vocab_hash": "v1", "frame_rate": 25.0}
    posteriors.check_compatible([a, dict(a)])
    with pytest.raises(AssertionError):
        posteriors.check_compatible([a, {**a, "vocab_hash": "v2"}])
    with pytest.raises(AssertionError):
        posteriors.check_compatible([a, {**a, "frame_rate": 50.0}])
