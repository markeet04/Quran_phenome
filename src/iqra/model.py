"""Layer fusion for Wav2Vec2BertForCTC (SPEC T5, section 5).

The CTC head reads a softmax-weighted sum of the 24 encoder layer outputs instead of the
last one; everything after the encoder (intermediate FFN, adapter, dropout, lm_head) runs
exactly as the library does. Implemented with forward hooks, not by re-writing the
library's forward: forward hooks on each encoder layer capture its output, and a hook on
the encoder replaces its last_hidden_state with the fused sum. That keeps working across
transformers versions (4.44.2 locally, 5.x on Kaggle) and renames no weights, so existing
checkpoints load with only the fusion logits new (`wav2vec2_bert.encoder.layer_fusion.logits`).

Initialised so the last layer gets >= 0.99 of the weight: at step 0 the fused model
computes (almost exactly) what the plain model computes.
"""
import json
from pathlib import Path

import torch
from torch import nn

FUSION_FLAG = "iqra_layer_fusion"          # stored in config.json of fused checkpoints
INIT_LAST_LOGIT = 12.0                     # softmax weight of the last layer ~0.99986 at init


class LayerFusion(nn.Module):
    def __init__(self, n_layers, init_last_logit=INIT_LAST_LOGIT):
        super().__init__()
        logits = torch.zeros(n_layers)
        logits[-1] = init_last_logit
        self.logits = nn.Parameter(logits)

    def weights(self):
        return torch.softmax(self.logits.float(), dim=0)


def _layer_output(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def add_layer_fusion(model, init_last_logit=INIT_LAST_LOGIT):
    """Attach fusion to a Wav2Vec2BertForCTC in place; returns the model."""
    encoder = model.wav2vec2_bert.encoder
    if hasattr(encoder, "layer_fusion"):
        return model
    layers = encoder.layers
    # create the fusion logits where the encoder already lives (the model may be on the GPU)
    encoder.layer_fusion = LayerFusion(len(layers), init_last_logit).to(next(encoder.parameters()).device)
    encoder._fusion_outputs = {}

    def capture(idx):
        def hook(module, args, out):
            encoder._fusion_outputs[idx] = _layer_output(out)
        return hook

    def fuse(module, args, out):
        outs = module._fusion_outputs
        # layerdrop (0.1 in the stage-1 config, kept as in NB8) skips random layers in
        # training; a skipped layer passes its input through unchanged, so its slot takes
        # the previous layer's output (or the encoder input for layer 0)
        filled, prev = [], outs.get(-1)
        for i in range(len(layers)):
            cur = outs.get(i, prev)
            if cur is None:
                raise RuntimeError("layer fusion found neither layer output nor encoder input")
            filled.append(cur)
            prev = cur
        w = module.layer_fusion.weights()
        stacked = torch.stack(filled, dim=0)
        fused = (w.to(stacked.dtype).view(-1, *([1] * (stacked.dim() - 1))) * stacked).sum(0)
        module._fusion_outputs = {}
        if isinstance(out, (tuple, list)):
            return (fused, *out[1:])
        # attribute assignment updates the stored field (and so out[0]); item assignment
        # with an int key would add a literal key 0 and make out[0] recurse forever
        out.last_hidden_state = fused
        return out

    def start(module, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        module._fusion_outputs = {-1: hidden}

    model._fusion_handles = [layer.register_forward_hook(capture(i)) for i, layer in enumerate(layers)]
    model._fusion_handles.append(encoder.register_forward_hook(fuse))
    model._fusion_handles.append(encoder.register_forward_pre_hook(start, with_kwargs=True))
    setattr(model.config, FUSION_FLAG, True)
    return model


def encoder_last_equals_last_layer(model, input_features, attention_mask):
    """Spec check: the encoder's last_hidden_state is the last layer's output (no extra
    final norm after the layers) in the installed transformers version."""
    enc = model.wav2vec2_bert.encoder
    captured = {}
    h = enc.layers[-1].register_forward_hook(lambda m, a, o: captured.__setitem__("last", _layer_output(o)))
    try:
        with torch.no_grad():
            out = model.wav2vec2_bert(input_features=input_features, attention_mask=attention_mask,
                                      output_hidden_states=True)
    finally:
        h.remove()
    hs = out.hidden_states
    last_layer = captured["last"]
    return {"n_hidden_states": len(hs),
            "hidden_states[-1]==last_layer": bool(torch.allclose(hs[-1], last_layer, atol=1e-5)),
            "uses_adapter": bool(getattr(model.config, "add_adapter", False))}


def load_model(model_dir, device="cpu", attn_implementation=None):
    """Load a plain or fused checkpoint. Returns (model, loading_info). For a fused
    checkpoint the fusion logits are restored after attaching the hooks.
    attn_implementation="eager" forces the plain attention computation (as transformers
    4.44.2 did); None keeps the library default (SDPA in 5.x)."""
    from transformers import Wav2Vec2BertForCTC
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    kw = {"attn_implementation": attn_implementation} if attn_implementation else {}
    model, info = Wav2Vec2BertForCTC.from_pretrained(model_dir, output_loading_info=True, **kw)
    info = {k: list(v) for k, v in info.items()}
    if cfg.get(FUSION_FLAG):
        add_layer_fusion(model)
        from safetensors.torch import load_file
        state = {}
        for f in sorted(model_dir.glob("*.safetensors")):
            state.update(load_file(str(f)))
        key = "wav2vec2_bert.encoder.layer_fusion.logits"
        model.wav2vec2_bert.encoder.layer_fusion.logits.data.copy_(state[key])
        info["unexpected_keys"] = [k for k in info.get("unexpected_keys", []) if k != key]
    # BASE (SpecAugment off): the model built its SpecAugment mask vector when checkpoint-5000
    # was loaded (mask_time_prob 0.05 there), then configure() set the mask probabilities to 0
    # and the checkpoint was saved with the vector but a config that no longer creates it. The
    # vector only replaces masked frames in training; inference never reads it. Accept exactly
    # that key, and only when this config has masking off (found by the T6 dry run).
    spec_key = "wav2vec2_bert.masked_spec_embed"
    if (not hasattr(model.wav2vec2_bert, "masked_spec_embed")
            and cfg.get("mask_time_prob", 0) == 0 and cfg.get("mask_feature_prob", 0) == 0):
        info["unexpected_keys"] = [k for k in info.get("unexpected_keys", []) if k != spec_key]
    return model.to(device).eval(), info


def fusion_weights(model):
    enc = model.wav2vec2_bert.encoder
    return enc.layer_fusion.weights().detach().cpu().numpy() if hasattr(enc, "layer_fusion") else None
