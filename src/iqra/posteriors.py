"""Posterior caches: run a checkpoint once per split, keep its frame outputs.

One file per checkpoint and split (post_{ckpt}_{split}.pt), holding for every clip ID:
the raw float32 logits [T x V], the greedy tokens and each token's emission span.
Log-posteriors are computed on load (log_posteriors()). Storing float16 log-posteriors,
as SPEC section 5 first said, does not reproduce the live greedy decode: near-tied
frames flip their argmax after rounding (decision 26). Raw float32 logits decode
exactly as the live model does.
The file also stores the vocab hash and the frame rate, which ensembling checks.
Files are written with torch.save using only tensors, lists, dicts and strings, so
they load with torch.load(weights_only=True).
"""
import datetime

import numpy as np

from .decode import greedy_decode, log_softmax

CACHE_FORMAT = 1


def model_frame_rate(config, input_rate=50.0):
    """Output frames per second implied by the config: the SeamlessM4T feature extractor
    stacks 10 ms frames in pairs (50 per second); each adapter layer divides by its stride."""
    rate = input_rate
    if getattr(config, "add_adapter", False):
        rate /= config.adapter_stride ** config.num_adapter_layers
    return rate


def clip_logits(model, feature_extractor, audio_16k, device):
    """Logits [T, V] (float32 numpy) for one clip, exactly as NB5 runs the model."""
    import torch
    feats = feature_extractor(np.asarray(audio_16k, dtype=np.float32),
                              sampling_rate=16_000, return_tensors="pt")
    feats = {k: v.to(device) for k, v in feats.items()}
    with torch.no_grad():
        return model(**feats).logits[0].float().cpu().numpy()


def output_length(n_feature_frames, config):
    """Model output frames for an input of `n_feature_frames` feature frames: each adapter
    layer is a stride-`adapter_stride` convolution with kernel `adapter_kernel_size`,
    padding kernel // 2."""
    T = int(n_feature_frames)
    if getattr(config, "add_adapter", False):
        k, s = config.adapter_kernel_size, config.adapter_stride
        for _ in range(config.num_adapter_layers):
            T = (T + 2 * (k // 2) - k) // s + 1
    return T


def batch_logits(model, feature_extractor, audios_16k, device, fp16=False):
    """Logits for several clips in one padded forward pass, each trimmed to its own length.

    For bulk filtering only (the agreement filter, T4): padding and optional fp16 change
    logits slightly, so reported numbers and caches always use clip_logits.
    """
    import torch
    feats = feature_extractor([np.asarray(a, dtype=np.float32) for a in audios_16k],
                              sampling_rate=16_000, padding=True, return_tensors="pt")
    # A clip run on its own is padded by the feature extractor to an even frame count
    # (SeamlessM4TFeatureExtractor's default pad_to_multiple_of=2), and its output length
    # follows that padded count; trim each clip of the batch to the same length.
    n_frames = [n + (n % 2) for n in feats["attention_mask"].sum(dim=1).tolist()]
    feats = {k: v.to(device) for k, v in feats.items()}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                         enabled=fp16 and str(device).startswith("cuda")):
        out = model(**feats).logits.float().cpu().numpy()
    return [out[i, :output_length(n, model.config)] for i, n in enumerate(n_frames)]


def make_entry(logits, id2tok, blank_id):
    """Cache entry for one clip: raw float32 logits, greedy tokens and spans."""
    import torch
    logits = np.ascontiguousarray(np.asarray(logits, dtype=np.float32))
    tokens, spans = greedy_decode(logits, id2tok, blank_id)
    return {"logits": torch.from_numpy(logits.copy()), "tokens": tokens,
            "spans": [list(s) for s in spans]}


def logits_of(entry):
    return entry["logits"].numpy()


def log_posteriors(entry):
    """Frame log-posteriors [T, V] (float64) computed from the stored logits."""
    return log_softmax(logits_of(entry))


def decode_entry(entry, id2tok, blank_id):
    """Greedy decode from the stored logits: identical to the live model's decode."""
    return greedy_decode(logits_of(entry), id2tok, blank_id)


def float16_would_flip(entry, id2tok, blank_id):
    """Diagnostic: would float16 log-posterior storage have changed this clip's decode?"""
    import torch
    lp16 = torch.from_numpy(log_softmax(logits_of(entry)).astype(np.float32)).half().float().numpy()
    return greedy_decode(lp16, id2tok, blank_id)[0] != entry["tokens"]


def write_cache(path, entries: dict, *, checkpoint, split, vocab_hash, frame_rate, extra_meta=None):
    import torch
    meta = {"format": CACHE_FORMAT, "checkpoint": str(checkpoint), "split": str(split),
            "vocab_hash": str(vocab_hash), "frame_rate": float(frame_rate),
            "n_clips": len(entries),
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if extra_meta:
        meta.update({k: str(v) for k, v in extra_meta.items()})
    torch.save({"meta": meta, "clips": entries}, path)
    return meta


def read_cache(path):
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob["meta"].get("format") != CACHE_FORMAT:
        raise ValueError(f"{path}: unknown cache format {blob['meta'].get('format')}")
    return blob["meta"], blob["clips"]


def check_compatible(metas):
    """Caches can be ensembled only with one vocab and one frame rate (SPEC section 5)."""
    vocabs = {m["vocab_hash"] for m in metas}
    rates = {m["frame_rate"] for m in metas}
    if len(vocabs) != 1:
        raise AssertionError(f"caches use different vocabularies: {sorted(vocabs)}")
    if len(rates) != 1:
        raise AssertionError(f"caches use different frame rates: {sorted(rates)}")
