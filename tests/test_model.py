"""Layer fusion (SPEC T5): init equivalence, loading, gradients, layerdrop, checkpointing."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from conftest import VOCAB_TOKENS  # noqa: E402
from iqra import model as M  # noqa: E402

REAL = Path(__file__).resolve().parents[1] / "quran_demo_app" / "model"


def _tiny(layerdrop=0.0, seed=0):
    torch.manual_seed(seed)
    cfg = transformers.Wav2Vec2BertConfig(
        hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=64,
        output_hidden_size=32, vocab_size=len(VOCAB_TOKENS), pad_token_id=0, add_adapter=True,
        num_adapter_layers=1, adapter_stride=2, conv_depthwise_kernel_size=3, layerdrop=layerdrop,
        ctc_loss_reduction="mean", ctc_zero_infinity=True)
    return transformers.Wav2Vec2BertForCTC(cfg).eval()


def _inputs(seconds=2.0, seed=1):
    fe = transformers.SeamlessM4TFeatureExtractor()
    x = np.random.default_rng(seed).normal(0, 0.1, int(16000 * seconds)).astype(np.float32)
    return fe(x, sampling_rate=16000, return_tensors="pt")


def test_last_hidden_state_is_last_layer_output():
    info = M.encoder_last_equals_last_layer(_tiny(), **_inputs())
    assert info["hidden_states[-1]==last_layer"], info


def test_fused_matches_plain_at_init():
    plain = _tiny()
    fused = M.add_layer_fusion(_tiny())
    assert M.fusion_weights(fused)[-1] >= 0.99
    feats = _inputs()
    with torch.no_grad():
        a, b = plain(**feats).logits, fused(**feats).logits
    assert torch.max(torch.abs(a - b)) < 1e-2


@pytest.mark.skipif(not (REAL / "model.safetensors").exists(), reason="NB8 model not on this machine")
def test_fused_matches_plain_at_init_on_the_real_model():
    feats = transformers.AutoFeatureExtractor.from_pretrained(REAL)(
        np.random.default_rng(3).normal(0, 0.1, 16000 * 3).astype(np.float32),
        sampling_rate=16000, return_tensors="pt")
    plain, _ = M.load_model(REAL)
    info = M.encoder_last_equals_last_layer(plain, **feats)
    assert info["hidden_states[-1]==last_layer"] and info["n_hidden_states"] == 25, info
    with torch.no_grad():
        a = plain(**feats).logits
        M.add_layer_fusion(plain)
        b = plain(**feats).logits
    assert torch.max(torch.abs(a - b)) < 1e-2
    assert torch.equal(a.argmax(-1), b.argmax(-1))


def test_plain_checkpoint_loads_into_fused_and_fused_round_trips(tmp_path):
    base = _tiny()
    base.save_pretrained(tmp_path / "plain")
    m, info = M.load_model(tmp_path / "plain")            # plain checkpoint
    assert not any(info.values()), info
    M.add_layer_fusion(m)
    with torch.no_grad():
        m.wav2vec2_bert.encoder.layer_fusion.logits.copy_(torch.linspace(-1, 2, 4))
    m.save_pretrained(tmp_path / "fused")
    m2, info2 = M.load_model(tmp_path / "fused")          # fused checkpoint
    assert not any(info2.values()), info2
    assert torch.equal(m2.wav2vec2_bert.encoder.layer_fusion.logits, torch.linspace(-1, 2, 4))
    feats = _inputs()
    with torch.no_grad():
        assert torch.allclose(m.eval()(**feats).logits, m2(**feats).logits)


def test_fusion_actually_changes_the_output():
    """With uniform weights the output must differ from the plain model (the hooks fire)."""
    plain = _tiny()
    fused = M.add_layer_fusion(_tiny())
    with torch.no_grad():
        fused.wav2vec2_bert.encoder.layer_fusion.logits.zero_()
        feats = _inputs()
        assert torch.max(torch.abs(plain(**feats).logits - fused(**feats).logits)) > 1e-2


def test_gradients_reach_the_fusion_weights():
    m = M.add_layer_fusion(_tiny()).train()
    feats = _inputs()
    labels = torch.tensor([[5, 9, 12, 7]])
    m(**feats, labels=labels).loss.backward()
    g = m.wav2vec2_bert.encoder.layer_fusion.logits.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


@pytest.mark.parametrize("layerdrop", [0.5, 1.0])
def test_fusion_with_layerdrop(layerdrop):
    m = M.add_layer_fusion(_tiny(layerdrop=layerdrop)).train()
    torch.manual_seed(0)
    out = m(**_inputs(), labels=torch.tensor([[5, 9, 12]]))
    assert torch.isfinite(out.loss)
    out.loss.backward()


def test_fusion_with_gradient_checkpointing():
    m = M.add_layer_fusion(_tiny()).train()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    out = m(**_inputs(), labels=torch.tensor([[5, 9, 12]]))
    out.loss.backward()
    assert m.wav2vec2_bert.encoder.layer_fusion.logits.grad.abs().sum() > 0
