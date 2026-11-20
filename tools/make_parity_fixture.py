"""Build src/iqra/fixtures/parity_nb8_final.npz under transformers 4.44.2 (SPEC decision 27).

Run locally, where transformers 4.44.2 (the version NB2/NB8 trained with) is installed:
    python tools/make_parity_fixture.py <nb8_final_model_dir> <real_clip.mp3> [<real_clip.mp3> ...]
The real clips are stored only as feature-extractor output, never as audio.
"""
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iqra import parity  # noqa: E402


def load_16k(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != parity.SR:
        g = gcd(sr, parity.SR)
        x = resample_poly(x, parity.SR // g, sr // g).astype(np.float32)
    return x


def main(model_dir, clips):
    import torch
    import transformers
    from transformers import AutoFeatureExtractor
    assert transformers.__version__ == "4.44.2", f"needs transformers 4.44.2, have {transformers.__version__}"
    model_dir = Path(model_dir)
    model, info = parity.load_with_report(model_dir)
    assert not any(info.values()), f"checkpoint did not load cleanly: {info}"
    fe = AutoFeatureExtractor.from_pretrained(model_dir)

    real = {}
    for k, clip in enumerate(clips):
        feats = fe(load_16k(clip), sampling_rate=parity.SR, return_tensors="np")
        real[f"real{k}"] = feats["input_features"][0].astype(np.float32)
    out = parity.run(model, fe, "cpu", real)
    for name, feat in real.items():
        out[f"{name}_features_input"] = feat

    meta = {"transformers": transformers.__version__, "torch": torch.__version__,
            "numpy": np.__version__, "python": sys.version.split()[0],
            "model_dir_name": model_dir.name,
            "weights_sha256": parity.sha256_file(model_dir / "model.safetensors"),
            "config": json.loads((model_dir / "config.json").read_text()),
            "synthetic": parity.SYNTHETIC, "real_clips": [Path(c).name for c in clips],
            "argmax_phoneme_frames": {k: int((v.argmax(-1) != 0).sum())
                                      for k, v in out.items() if k.endswith("_logits")}}
    parity.FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(parity.FIXTURE, meta=np.array(json.dumps(meta)), **out)
    print("wrote", parity.FIXTURE, f"({parity.FIXTURE.stat().st_size / 1e3:.0f} kB)")
    print(json.dumps({k: meta[k] for k in ("transformers", "torch", "weights_sha256",
                                           "argmax_phoneme_frames")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
