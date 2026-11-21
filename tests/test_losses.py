"""Consistency loss: zero for identical outputs, symmetric, masked to real frames."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from iqra import losses  # noqa: E402
from test_model import _tiny  # noqa: E402


def test_symmetric_kl_basics():
    torch.manual_seed(0)
    a, b = torch.randn(2, 6, 9), torch.randn(2, 6, 9)
    mask = torch.ones(2, 6, dtype=torch.bool)
    assert losses.symmetric_kl(a, a, mask).item() == pytest.approx(0.0, abs=1e-6)
    assert losses.symmetric_kl(a, b, mask).item() == pytest.approx(losses.symmetric_kl(b, a, mask).item())
    assert losses.symmetric_kl(a, b, mask).item() > 0


def test_padding_frames_do_not_count():
    torch.manual_seed(1)
    a = torch.randn(1, 8, 5)
    b = a.clone()
    b[0, 6:] += 10.0                                  # differences only in padded frames
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, :6] = True
    assert losses.symmetric_kl(a, b, mask).item() == pytest.approx(0.0, abs=1e-6)


def test_real_frame_mask_matches_single_clip_lengths():
    fe = transformers.SeamlessM4TFeatureExtractor()
    m = _tiny()
    clips = [np.random.default_rng(i).normal(0, 0.1, int(16000 * s)).astype(np.float32)
             for i, s in enumerate((0.9, 2.1, 1.4))]
    batch = fe(clips, sampling_rate=16000, padding=True, return_tensors="pt")
    with torch.no_grad():
        T = m(**batch).logits.shape[1]
        singles = [m(**fe(c, sampling_rate=16000, return_tensors="pt")).logits.shape[1] for c in clips]
    mask = losses.real_frame_mask(batch["attention_mask"], m.config, T)
    assert mask.sum(dim=1).tolist() == singles


def test_ctc_with_consistency_in_train_mode():
    fe = transformers.SeamlessM4TFeatureExtractor()
    m = _tiny().train()
    m.config.hidden_dropout = 0.1
    clips = [np.random.default_rng(i).normal(0, 0.1, 16000 * 2).astype(np.float32) for i in range(2)]
    batch = dict(fe(clips, sampling_rate=16000, padding=True, return_tensors="pt"))
    batch["labels"] = torch.tensor([[5, 9, 12, -100], [7, 8, 9, 10]])
    total, ctc, kl, skipped = losses.ctc_with_consistency(m, batch, lam=1.0)
    assert torch.isfinite(total) and kl.item() >= 0
    total.backward()
    plain, ctc0, kl0, skipped0 = losses.ctc_with_consistency(m, batch, lam=0.0)
    assert kl0.item() == 0.0 and skipped0 is False


def test_kl_skipped_when_adapter_layerdrop_changes_the_frame_rate():
    """layerdrop also skips the adapter: passes at different lengths skip the KL term."""
    fe = transformers.SeamlessM4TFeatureExtractor()
    m = _tiny(layerdrop=0.5).train()
    clips = [np.random.default_rng(i).normal(0, 0.1, 16000 * 2).astype(np.float32) for i in range(2)]
    batch = dict(fe(clips, sampling_rate=16000, padding=True, return_tensors="pt"))
    batch["labels"] = torch.tensor([[5, 9, 12, -100], [7, 8, 9, 10]])
    torch.manual_seed(0)
    flags = [losses.ctc_with_consistency(m, batch, lam=1.0)[3] for _ in range(12)]
    assert any(flags) and not all(flags)
