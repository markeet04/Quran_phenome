"""Library-version parity for Wav2Vec2BertForCTC (SPEC decision 27).

NB2/NB8 trained with transformers 4.44.2, which cannot be installed on Kaggle's
Python 3.13 image. Before using Kaggle's newer transformers, NB14 checks that it
computes the same thing: the trained NB8 model is run on fixed inputs and compared
with reference outputs produced locally under transformers 4.44.2
(tools/make_parity_fixture.py).

Inputs:
  - synthetic clips, regenerated from code on both machines (feature extractor + model);
  - real recitations stored as feature-extractor output (model only; no audio in the repo).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).with_name("fixtures") / "parity_nb8_final.npz"
SR = 16_000
SYNTHETIC = [(0, 3.0), (1, 5.0), (2, 1.2), (3, 8.0)]     # (seed, seconds)


def synthetic_audio(seed, seconds):
    """Deterministic speech-band test signal: harmonic tones with an amplitude envelope,
    a chirp, a silent gap and low noise. Pure numpy, identical on every machine."""
    n = int(SR * seconds)
    t = np.arange(n, dtype=np.float64) / SR
    f0 = 110.0 + 25.0 * seed
    voiced = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    envelope = 0.5 * (1 + np.sin(2 * np.pi * (2.0 + seed) * t))
    chirp = np.sin(2 * np.pi * (200 * t + 900 * t ** 2 / max(seconds, 1e-9)))
    x = 0.3 * voiced * envelope + 0.1 * chirp
    x[int(0.4 * n):int(0.5 * n)] = 0.0
    x += 0.01 * np.random.default_rng(seed).standard_normal(n)
    return (x / np.max(np.abs(x)) * 0.5).astype(np.float32)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_reference_model(weights_sha256, search_root="/kaggle/input"):
    """Directory of the attached model.safetensors whose SHA-256 equals the reference's.

    Independent of folder names (the same NB8 model may be mounted as a dataset or as a
    notebook's output). Returns (dir, candidates checked as [(path, sha256)]); raises,
    listing every candidate, when none matches.
    """
    import glob
    import os
    checked = []
    for p in sorted(glob.glob(os.path.join(search_root, "**", "model.safetensors"), recursive=True)):
        h = sha256_file(p)
        checked.append((p, h))
        if h == weights_sha256:
            return os.path.dirname(p), checked
    listing = "\n".join(f"  {h[:12]}  {p}" for p, h in checked) or "  (no model.safetensors found)"
    raise FileNotFoundError(f"no attached model.safetensors matches the reference "
                            f"{weights_sha256[:12]}; attach the NB8 model output "
                            f"(notebook17aed3d165 / quran_model_final). Checked:\n{listing}")


def load_with_report(model_dir, device="cpu"):
    """Load a checkpoint and return (model, loading info); missing/unexpected keys must be empty."""
    from transformers import Wav2Vec2BertForCTC
    model, info = Wav2Vec2BertForCTC.from_pretrained(model_dir, output_loading_info=True)
    return model.to(device).eval(), {k: list(v) for k, v in info.items()}


def run(model, fe, device="cpu", real_features=None):
    """Features and logits for the synthetic clips; logits for given real feature arrays."""
    import torch
    out = {}
    for seed, secs in SYNTHETIC:
        feats = fe(synthetic_audio(seed, secs), sampling_rate=SR, return_tensors="pt")
        out[f"synth{seed}_features"] = feats["input_features"][0].numpy()
        out[f"synth{seed}_mask"] = feats["attention_mask"][0].numpy()
        with torch.no_grad():
            out[f"synth{seed}_logits"] = model(**{k: v.to(device) for k, v in feats.items()}
                                               ).logits[0].float().cpu().numpy()
    for name, feat in (real_features or {}).items():
        x = torch.from_numpy(np.asarray(feat, dtype=np.float32))[None].to(device)
        mask = torch.ones(x.shape[:2], dtype=torch.long, device=device)
        with torch.no_grad():
            out[f"{name}_logits"] = model(input_features=x, attention_mask=mask
                                          ).logits[0].float().cpu().numpy()
    return out


def load_fixture(path=FIXTURE):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    arrays = {k: z[k] for k in z.files if k != "meta"}
    return meta, arrays


def verify_environment(device="cpu", search_root="/kaggle/input"):
    """Decision 27 check, as run in NB14: the NB8 model (found by weights hash) on the
    reference inputs, on CPU, against the transformers 4.44.2 outputs. Raises on failure.
    Returns a report dict (rows, GPU-vs-CPU rows when a GPU is available, versions)."""
    import torch
    import transformers
    from transformers import AutoFeatureExtractor
    meta, ref = load_fixture()
    model_dir, checked = find_reference_model(meta["weights_sha256"], search_root)
    model, info = load_with_report(model_dir, "cpu")
    if any(info.values()):
        raise RuntimeError(f"NB8 model did not load cleanly: {info}")
    fe = AutoFeatureExtractor.from_pretrained(model_dir)
    real = {k[: -len("_features_input")]: v for k, v in ref.items() if k.endswith("_features_input")}
    cur = run(model, fe, "cpu", real)
    ok, rows = compare(ref, cur)
    gpu_rows = None
    if str(device).startswith("cuda"):
        cur_gpu = run(model.to(device), fe, device, real)
        gpu_rows = [{"input": r["input"],
                     "gpu_vs_cpu_max_abs_logit_diff": float(np.max(np.abs(cur_gpu[k] - cur[k]))),
                     "gpu_vs_cpu_argmax_mismatch_frames": int(np.sum(cur_gpu[k].argmax(-1) != cur[k].argmax(-1)))}
                    for r in rows for k in [r["input"] + "_logits"]]
    del model
    report = {"pass": ok, "rows": rows, "gpu_vs_cpu": gpu_rows, "model_dir": model_dir,
              "here": {"transformers": transformers.__version__, "torch": torch.__version__},
              "reference": {k: meta[k] for k in ("transformers", "torch")}}
    if not ok:
        raise RuntimeError(f"transformers on this image does not reproduce 4.44.2: {rows}")
    return report


def compare(reference: dict, current: dict, logit_tol=1e-3, feature_tol=1e-4):
    """Per input: max |logit diff|, frames whose argmax differs, feature diff.
    Passes only if every argmax matches and all differences are within tolerance."""
    rows, ok = [], True
    for key in sorted(k for k in reference if k.endswith("_logits")):
        name = key[: -len("_logits")]
        ref, cur = reference[key], current.get(key)
        if cur is None or cur.shape != ref.shape:
            rows.append({"input": name, "error": f"shape {None if cur is None else cur.shape} vs {ref.shape}"})
            ok = False
            continue
        row = {"input": name, "frames": int(ref.shape[0]),
               "max_abs_logit_diff": float(np.max(np.abs(ref - cur))),
               "argmax_mismatch_frames": int(np.sum(ref.argmax(-1) != cur.argmax(-1)))}
        fkey = f"{name}_features"
        if fkey in reference:
            row["max_abs_feature_diff"] = float(np.max(np.abs(reference[fkey] - current[fkey])))
            ok &= row["max_abs_feature_diff"] <= feature_tol
        ok &= row["argmax_mismatch_frames"] == 0 and row["max_abs_logit_diff"] <= logit_tol
        rows.append(row)
    return ok, rows
