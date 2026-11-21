"""Consistency loss (SPEC T5, section 5).

Two forward passes per batch with dropout (and SpecAugment, when on) active; CTC loss on
both; plus lambda x the symmetric KL between the two frame log-softmax outputs, averaged
over real frames only (output lengths from the attention mask), computed in fp32.
"""
import torch
import torch.nn.functional as F

from .posteriors import output_length


def real_frame_mask(attention_mask, config, T):
    """[B, T] bool mask of real output frames. Each clip's output length follows its own
    feature length rounded up to even (as a single-clip run), clamped to the batch's T."""
    n = attention_mask.sum(dim=1).tolist()
    lengths = [min(T, output_length(k + (k % 2), config)) for k in n]
    idx = torch.arange(T, device=attention_mask.device)[None, :]
    return idx < torch.tensor(lengths, device=attention_mask.device)[:, None]


def symmetric_kl(logits_a, logits_b, mask):
    """0.5 * (KL(p_a || p_b) + KL(p_b || p_a)) per frame, summed over the vocabulary and
    averaged over real frames; fp32."""
    la = F.log_softmax(logits_a.float(), dim=-1)
    lb = F.log_softmax(logits_b.float(), dim=-1)
    kl_ab = (la.exp() * (la - lb)).sum(-1)
    kl_ba = (lb.exp() * (lb - la)).sum(-1)
    per_frame = 0.5 * (kl_ab + kl_ba)
    m = mask.to(per_frame.dtype)
    return (per_frame * m).sum() / m.sum().clamp(min=1.0)


def ctc_with_consistency(model, batch, lam):
    """Returns (total loss, ctc mean of both passes, kl, kl_skipped). lam = 0 -> one plain pass.

    The library applies layerdrop to the adapter too (skipped with probability
    config.layerdrop in training, 0.1 here, as in NB2/NB8), so the two passes can come out
    at different frame rates. Those frames cannot be compared one to one: the KL term is
    skipped for that batch (CTC still trains) and kl_skipped=True is reported.
    """
    out1 = model(**batch)
    zero = torch.zeros((), device=out1.loss.device)
    if lam == 0:
        return out1.loss, out1.loss.detach(), zero, False
    out2 = model(**batch)
    ctc = 0.5 * (out1.loss + out2.loss)
    if out1.logits.shape != out2.logits.shape:
        return ctc, ctc.detach(), zero, True
    mask = real_frame_mask(batch["attention_mask"], model.config, out1.logits.shape[1])
    kl = symmetric_kl(out1.logits, out2.logits, mask)
    return ctc + lam * kl, ctc.detach(), kl.detach(), False
