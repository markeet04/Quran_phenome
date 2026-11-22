"""Crash-isolated inference for the one-shot test run (NB16).

The first NB16 run died with "CUDA error: an illegal memory access" on every model, on both
GPUs, while caching test logits (the answer key was never loaded). A CUDA error poisons the
whole process, so each model now runs in a child process, clips shortest first, writing the
clip it is about to run to a progress file. clip_logits ends with .cpu(), which synchronises,
so an error surfaces on the clip that caused it. If the child dies, that clip is excluded and
the child is restarted on the rest; excluded clips are run on CPU with the same checkpoint
(float32, single clip, the same code path). The number of CPU clips is reported.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

FE_NAME = "facebook/w2v-bert-2.0"


def save_audio(audio: dict, path):
    """{clip id: float32 16 kHz array} -> one .npz (ids may not be valid identifiers: use an index)."""
    ids = list(audio)
    np.savez(path, __ids__=np.array(ids, dtype=object), **{f"a{i}": np.asarray(audio[c], np.float32)
                                                          for i, c in enumerate(ids)})


def load_audio(path):
    z = np.load(path, allow_pickle=True)
    ids = list(z["__ids__"])
    return {c: z[f"a{i}"] for i, c in enumerate(ids)}


def _child(model_dir, audio_path, out_path, progress_path, exclude_path, device, attn="default"):
    import torch
    from transformers import AutoFeatureExtractor
    from .model import load_model
    from .posteriors import clip_logits
    audio = load_audio(audio_path)
    exclude = set(json.load(open(exclude_path)))
    model, info = load_model(model_dir, device, attn_implementation=None if attn == "default" else attn)
    if any(info.values()):
        raise RuntimeError(f"{model_dir} did not load cleanly: {info}")
    fe = AutoFeatureExtractor.from_pretrained(FE_NAME)
    out = {}
    for cid in sorted(audio, key=lambda c: (len(audio[c]), c)):
        if cid in exclude:
            continue
        with open(progress_path, "w") as f:
            f.write(cid)
        if os.environ.get("IQRA_TEST_CRASH_ON") and cid in os.environ["IQRA_TEST_CRASH_ON"].split(","):
            os._exit(3)                                   # tests only: simulate a hard GPU crash
        out[cid] = torch.from_numpy(np.ascontiguousarray(clip_logits(model, fe, audio[cid], device)))
    torch.save(out, out_path)
    with open(progress_path, "w") as f:
        f.write("")


def run_model(name, model_dir, audio_path, workdir, gpu_index=None, max_restarts=25, log=print,
              child_device="cuda:0", attn="eager", storm_after=3):
    """{clip: float32 logits [T, V]} for every clip, plus a report. GPU in a child process with
    crash isolation when gpu_index is given; CPU in-process for excluded clips."""
    import torch
    os.makedirs(workdir, exist_ok=True)
    out_path = os.path.join(workdir, f"{name}.pt")
    prog = os.path.join(workdir, f"{name}.progress")
    excl_path = os.path.join(workdir, f"{name}.exclude.json")
    exclude, restarts, t0 = [], 0, time.time()
    logits = {}
    if gpu_index is not None:
        src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # the child must import iqra
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index), PYTHONUNBUFFERED="1",
                   PYTHONPATH=os.pathsep.join([src] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
        while True:
            json.dump(exclude, open(excl_path, "w"))
            open(prog, "w").close()
            r = subprocess.run([sys.executable, "-m", "iqra.safe_infer", model_dir, audio_path, out_path, prog,
                                excl_path, child_device, attn], env=env, capture_output=True, text=True)
            if r.returncode == 0:
                logits = {k: v.numpy() for k, v in torch.load(out_path, weights_only=True).items()}
                break
            bad = open(prog).read().strip()
            if not bad or bad in exclude or restarts >= max_restarts:
                raise RuntimeError(f"{name}: child failed outside a clip (restarts {restarts}):\n{r.stderr[-3000:]}")
            restarts += 1
            if restarts >= storm_after:       # repeated crashes: this clip and every longer one go to CPU
                audio = load_audio(audio_path)
                n = len(audio[bad])
                more = [c for c in audio if len(audio[c]) >= n and c not in exclude]
                exclude.extend(more)
                log(f"  [{name}] GPU crashed on {bad} ({n / 16000:.1f} s), crash {restarts}: "
                    f"{len(more)} clips this long or longer go to CPU")
            else:
                exclude.append(bad)
                log(f"  [{name}] GPU crashed on clip {bad}; restarting without it ({restarts})")
    cpu_ids = exclude if gpu_index is not None else None
    if cpu_ids is None or cpu_ids:
        from transformers import AutoFeatureExtractor
        from .model import load_model
        from .posteriors import clip_logits
        audio = load_audio(audio_path)
        model, info = load_model(model_dir, "cpu", attn_implementation=None if attn == "default" else attn)
        if any(info.values()):
            raise RuntimeError(f"{model_dir} did not load cleanly: {info}")
        fe = AutoFeatureExtractor.from_pretrained(FE_NAME)
        for cid in (cpu_ids if cpu_ids is not None else list(audio)):
            logits[cid] = clip_logits(model, fe, audio[cid], "cpu")
        del model
    return logits, {"attn": attn, "gpu_restarts": restarts, "cpu_clips": list(exclude) if gpu_index is not None else "all",
                    "minutes": round((time.time() - t0) / 60, 2)}


if __name__ == "__main__":
    _child(*sys.argv[1:8])
