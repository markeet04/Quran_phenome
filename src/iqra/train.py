"""Stage-2 training (SPEC T5): runs BASE, A and B from checkpoint-5000.

An explicit loop instead of the HF Trainer: NB2/NB8 used Trainer under transformers
4.44.2, Kaggle now has 5.x, and Trainer internals (sampler hooks, compute_loss signature)
differ between them. The loop reproduces NB8's settings exactly: AdamW (lr 5e-5, betas
0.9/0.999, eps 1e-8, no weight decay), grad-norm clip 1.0, cosine schedule with 200
warmup steps, fp16 autocast with a GradScaler, batch 8 x 2 accumulation, gradient
checkpointing (non-reentrant). The model's own forward computes the CTC loss.

Recipe v2 (runs A and B) adds layer fusion (fusion logits at lr 1e-2, decision 30), the
consistency loss, weighted sampling from NB12 and SpecAugment; BASE adds none of them.
Model selection: greedy decode of Extra-dev every 500 steps with the official scorer;
the best 3 checkpoints by F1 are kept, the rest deleted (disk).
"""
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, posteriors, scorer
from .align import accuracy_counts
from .decode import greedy_decode
from .losses import ctc_with_consistency
from .model import FUSION_FLAG, add_layer_fusion, fusion_weights


@dataclass
class RunConfig:
    name: str
    extra_folds: tuple
    fusion: bool
    consistency_lambda: float
    sample_weights: bool
    spec_augment: bool
    learning_rate: float = 5e-5
    fusion_lr: float = 1e-2              # decision 30
    warmup_steps: int = 200
    max_steps: int = 8000
    batch_size: int = 8
    grad_accum: int = 2
    max_grad_norm: float = 1.0
    extra_repeat: int = 20
    eval_every: int = 500
    keep_best: int = 3
    consistency_every: int = 1           # 2 = every other step (time fallback, SPEC risks)
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_feature_prob: float = 0.0
    dev_slice_eval_clips: int = 250      # fixed subset of the 1 h slice for the drift PER
    num_workers: int = 2
    fp16: bool = True
    seed: int = 13
    hours_budget: float = 11.0           # abort after the probe window if projected over


RUNS = {
    "BASE": RunConfig("BASE", ("fold_a", "fold_b"), fusion=False, consistency_lambda=0.0,
                      sample_weights=False, spec_augment=False),
    "A": RunConfig("A", ("fold_a",), fusion=True, consistency_lambda=1.0,
                   sample_weights=True, spec_augment=True),
    "B": RunConfig("B", ("fold_b",), fusion=True, consistency_lambda=1.0,
                   sample_weights=True, spec_augment=True),
}


# --------------------------------------------------------------------------- data
@dataclass
class Source:
    name: str
    ds: object                 # datasets.Dataset without an indices mapping
    rows: list
    labels: list               # label string per entry of `rows`
    weights: list = field(default=None)


class CTCDataset:
    """Clips from several sources; audio decoded with soundfile from the raw Arrow
    bytes (datasets-version independent), resampled to 16 kHz, featurised per item."""

    def __init__(self, sources, feature_extractor, tok2id, unk_id):
        self.sources, self.fe, self.tok2id, self.unk = sources, feature_extractor, tok2id, unk_id
        self.index = [(s, k) for s, src in enumerate(sources) for k in range(len(src.rows))]
        for src in sources:
            if getattr(src.ds, "_indices", None) is not None:
                raise ValueError(f"{src.name}: pass the full dataset and row numbers, not a selection")

    def __len__(self):
        return len(self.index)

    def weights(self):
        return np.array([1.0 if self.sources[s].weights is None else self.sources[s].weights[k]
                         for s, k in self.index], dtype=np.float64)

    def __getitem__(self, i):
        s, k = self.index[i]
        src = self.sources[s]
        cell = src.ds.data.slice(src.rows[k], 1).column("audio").to_pylist()[0]
        audio = data.to_16k(*data.decode_audio(cell))
        feats = self.fe(audio, sampling_rate=16_000).input_features[0]
        labels = [self.tok2id.get(t, self.unk) for t in str(src.labels[k]).split()]
        return {"input_features": feats, "labels": labels}


class Collator:
    def __init__(self, feature_extractor):
        self.fe = feature_extractor

    def __call__(self, items):
        import torch
        batch = self.fe.pad([{"input_features": it["input_features"]} for it in items], return_tensors="pt")
        n = max(1, max(len(it["labels"]) for it in items))
        labels = torch.full((len(items), n), -100, dtype=torch.long)
        for r, it in enumerate(items):
            if it["labels"]:
                labels[r, :len(it["labels"])] = torch.tensor(it["labels"])
        return {"input_features": batch["input_features"], "attention_mask": batch["attention_mask"],
                "labels": labels}


def build_sources(run: RunConfig, train_ds, train_labels, tts_ds, tts_labels, extra_ds, extra_ids_by_row,
                  extra_labels_by_id, splits, sample_weights=None):
    """Training mix for one run: all of Iqra_train (label phoneme_ref), all of Iqra_TTS
    (label phoneme_mis), and the run's Extra folds repeated `extra_repeat` times (label
    phoneme_mis). Extra-dev never enters. Recipe v2 attaches NB12's per-row weights to
    Iqra_train and Iqra_TTS; Extra rows keep weight 1 on top of their repeat."""
    row_of = {cid: r for r, cid in enumerate(extra_ids_by_row)}
    extra_ids = [cid for fold in run.extra_folds for cid in splits["extra"][fold]]
    leak = set(extra_ids) & set(splits["extra"]["extra_dev"])
    if leak:
        raise AssertionError(f"{len(leak)} Extra-dev clips in the training mix")
    w_train = w_tts = None
    if run.sample_weights:
        if sample_weights is None:
            raise ValueError("recipe v2 needs sample_weights.parquet from NB12")
        sw = sample_weights.set_index(["dataset", "row"])["weight"]
        w_train = [float(sw[("iqra_train", r)]) for r in range(len(train_ds))]
        w_tts = [float(sw[("iqra_tts", r)]) for r in range(len(tts_ds))]
    rep_ids = extra_ids * run.extra_repeat
    return [
        Source("iqra_train", train_ds, list(range(len(train_ds))), list(train_labels), w_train),
        Source("iqra_tts", tts_ds, list(range(len(tts_ds))), list(tts_labels), w_tts),
        Source("extra", extra_ds, [row_of[c] for c in rep_ids], [extra_labels_by_id[c] for c in rep_ids],
               [1.0] * len(rep_ids) if run.sample_weights else None),
    ]


def make_loader(dataset, run: RunConfig):
    """BASE: shuffled without replacement (as NB8). Recipe v2: WeightedRandomSampler."""
    import torch
    from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
    g = torch.Generator().manual_seed(run.seed)
    if run.sample_weights:
        sampler = WeightedRandomSampler(torch.as_tensor(dataset.weights()), num_samples=len(dataset),
                                        replacement=True, generator=g)
    else:
        sampler = RandomSampler(dataset, generator=g)
    return DataLoader(dataset, batch_size=run.batch_size, sampler=sampler, collate_fn=Collator(dataset.fe),
                      num_workers=run.num_workers, drop_last=True,
                      persistent_workers=run.num_workers > 0)


def forever(loader):
    while True:
        for b in loader:
            yield b


# --------------------------------------------------------------------------- schedule
def cosine_factor(step, warmup, total):
    """Linear warmup then cosine decay to 0 (as transformers' get_cosine_schedule_with_warmup)."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


# --------------------------------------------------------------------------- evaluation
class DevEvaluator:
    """Official F1 on Extra-dev (greedy, single-clip float32, as NB5/NB14) and PER on a
    fixed subset of the Iqra dev slice (drift on correct speech)."""

    def __init__(self, dev_audio, truth_df, slice_audio, slice_refs, fe, id2tok, blank, device, clusters):
        self.dev_audio, self.truth = dev_audio, truth_df
        self.slice_audio, self.slice_refs = slice_audio, slice_refs
        self.fe, self.id2tok, self.blank, self.device, self.clusters = fe, id2tok, blank, device, clusters

    def __call__(self, model):
        was_training = model.training
        model.eval()
        try:
            preds = [{"ID": cid, "Prediction": " ".join(greedy_decode(
                posteriors.clip_logits(model, self.fe, a, self.device), self.id2tok, self.blank)[0])}
                for cid, a in self.dev_audio.items()]
            pred_df = pd.DataFrame(preds)
            if (pred_df["Prediction"].str.strip() == "").any():
                return {"f1": 0.0, "precision": float("nan"), "recall": 0.0, "per_slice": float("nan"),
                        "empty_predictions": int((pred_df["Prediction"].str.strip() == "").sum())}
            res = scorer.official_score(self.truth, pred_df)
            edits = total = 0
            for cid, a in self.slice_audio.items():
                toks = greedy_decode(posteriors.clip_logits(model, self.fe, a, self.device), self.id2tok, self.blank)[0]
                e, n = accuracy_counts(self.slice_refs[cid].split(), toks)
                edits += e; total += n
            return {"f1": res["f1"], "precision": res["precision"], "recall": res["recall"],
                    "per_dev": res["per"], "per_slice": edits / max(total, 1), "pred_df": pred_df}
        finally:
            if was_training:
                model.train()


class CheckpointKeeper:
    """Keeps the best k checkpoints by dev F1; deletes the others immediately."""

    def __init__(self, out_dir, k, feature_extractor=None):
        self.out, self.k, self.fe = Path(out_dir), k, feature_extractor
        self.kept = []                                  # (f1, step, path)

    def offer(self, model, step, f1):
        if len(self.kept) >= self.k and f1 <= min(x[0] for x in self.kept):
            return False
        path = self.out / f"ckpt_step{step:05d}"
        model.save_pretrained(path, safe_serialization=True)
        if self.fe is not None:
            self.fe.save_pretrained(path)
        self.kept.append((f1, step, str(path)))
        self.kept.sort(key=lambda x: (-x[0], x[1]))
        while len(self.kept) > self.k:
            _, _, drop = self.kept.pop()
            shutil.rmtree(drop, ignore_errors=True)
        (self.out / "kept_checkpoints.json").write_text(json.dumps(
            [{"f1": f, "step": s, "path": p} for f, s, p in self.kept], indent=1))
        return True


# --------------------------------------------------------------------------- loop
def configure(model, run: RunConfig):
    """Apply the run's SpecAugment/fusion settings; returns the active config to print."""
    c = model.config
    c.apply_spec_augment = bool(run.spec_augment)
    c.mask_time_prob = run.mask_time_prob if run.spec_augment else 0.0
    c.mask_time_length = run.mask_time_length
    c.mask_feature_prob = run.mask_feature_prob if run.spec_augment else 0.0
    if run.fusion:
        add_layer_fusion(model)
    return {"run": run.name, "apply_spec_augment": c.apply_spec_augment, "mask_time_prob": c.mask_time_prob,
            "mask_time_length": c.mask_time_length, "mask_feature_prob": c.mask_feature_prob,
            "layerdrop": getattr(c, "layerdrop", None), "fusion": bool(getattr(c, FUSION_FLAG, False)),
            "consistency_lambda": run.consistency_lambda, "consistency_every": run.consistency_every,
            "sample_weights": run.sample_weights, "extra_folds": list(run.extra_folds),
            "lr": run.learning_rate, "fusion_lr": run.fusion_lr if run.fusion else None,
            "batch": f"{run.batch_size}x{run.grad_accum}", "max_steps": run.max_steps, "fp16": run.fp16}


def train(run: RunConfig, model, loader, evaluator, out_dir, device, probe_steps=None, log=print):
    """Train; returns a summary. probe_steps: stop after that many optimiser steps and
    report timing (no checkpoints). Otherwise abort early if the projection at step
    200 exceeds run.hours_budget."""
    import torch
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(run.seed)
    model.to(device).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    fusion_params = [p for n, p in model.named_parameters() if "layer_fusion" in n]
    other = [p for n, p in model.named_parameters() if "layer_fusion" not in n and p.requires_grad]
    groups = [{"params": other, "lr": run.learning_rate}]
    if fusion_params:
        groups.append({"params": fusion_params, "lr": run.fusion_lr})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    base_lrs = [g["lr"] for g in opt.param_groups]
    use_amp = run.fp16 and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    keeper = CheckpointKeeper(out, run.keep_best, getattr(evaluator, "fe", None))
    rows, evals = [], []
    kl_skipped = kl_tried = 0           # adapter layerdrop: passes at different frame rates
    batches = forever(loader)
    total_steps = probe_steps or run.max_steps
    t_start = t_window = time.time()

    def evaluate(step):
        t0 = time.time()
        r = evaluator(model)
        r = {k: v for k, v in r.items() if k != "pred_df"}
        r.update(step=step, eval_minutes=round((time.time() - t0) / 60, 2))
        fw = fusion_weights(model)
        if fw is not None:
            r["fusion_last_weight"] = round(float(fw[-1]), 4)
            r["fusion_top3_layers"] = [int(i) for i in np.argsort(fw)[::-1][:3]]
        if probe_steps is None and step > 0:
            r["kept"] = keeper.offer(model, step, r["f1"])
        evals.append(r)
        log(f"[eval] step {step}: F1 {r['f1']:.4f} P {r.get('precision', float('nan')):.4f} "
            f"R {r.get('recall', float('nan')):.4f} | dev PER {r.get('per_dev', float('nan')):.4f} "
            f"| slice PER {r.get('per_slice', float('nan')):.4f} | {r['eval_minutes']} min"
            + (f" | fusion last-layer weight {r['fusion_last_weight']}" if fw is not None else ""))
        pd.DataFrame(evals).to_csv(out / "evals.csv", index=False)
        return r

    step0 = evaluate(0)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    t_start = t_window = time.time()
    for step in range(1, total_steps + 1):
        f = cosine_factor(step - 1, run.warmup_steps, run.max_steps)
        for g, lr0 in zip(opt.param_groups, base_lrs):
            g["lr"] = lr0 * f
        lam = run.consistency_lambda if (step % run.consistency_every == 0) else 0.0
        ctc_sum = kl_sum = 0.0
        for _ in range(run.grad_accum):
            batch = {k: v.to(device) for k, v in next(batches).items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                total, ctc, kl, skipped = ctc_with_consistency(model, batch, lam)
            kl_skipped += int(skipped)
            kl_tried += int(lam > 0)
            scaler.scale(total / run.grad_accum).backward()
            ctc_sum += float(ctc) / run.grad_accum
            kl_sum += float(kl) / run.grad_accum
        scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], run.max_grad_norm))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == total_steps:
            dt = (time.time() - t_window) / (50 if step % 50 == 0 else step % 50)
            t_window = time.time()
            mem = torch.cuda.max_memory_allocated() / 2**30 if use_amp else 0.0
            rows.append({"step": step, "ctc": round(ctc_sum, 4), "kl": round(kl_sum, 5),
                         "kl_skipped_frac": round(kl_skipped / max(kl_tried, 1), 3), "lr": opt.param_groups[0]["lr"],
                         "grad_norm": round(gnorm, 3), "sec_per_step": round(dt, 3), "peak_mem_gb": round(mem, 2)})
            log(f"step {step}/{total_steps} | ctc {ctc_sum:.4f} kl {kl_sum:.5f} | lr {opt.param_groups[0]['lr']:.2e} "
                f"| grad {gnorm:.2f} | {dt:.2f} s/step | peak {mem:.1f} GB")
            pd.DataFrame(rows).to_csv(out / "train_log.csv", index=False)
        if probe_steps is None and step == 200:
            sec_step = (time.time() - t_start) / step
            eval_min = step0["eval_minutes"]
            projected = (sec_step * run.max_steps + 60 * eval_min * (run.max_steps // run.eval_every + 1)) / 3600
            log(f"projection at step 200: {projected:.1f} h for {run.max_steps} steps incl. evals")
            if projected > run.hours_budget:
                raise RuntimeError(f"projected {projected:.1f} h > budget {run.hours_budget} h: stop and "
                                   "re-plan (SPEC risks: consistency every other step, or fewer steps)")
        if probe_steps is None and (step % run.eval_every == 0 or step == run.max_steps):
            evaluate(step)
    elapsed = time.time() - t_start
    summary = {"run": run.name, "steps": total_steps, "hours": round(elapsed / 3600, 3),
               "kl_skipped_frac": round(kl_skipped / max(kl_tried, 1), 4),
               "sec_per_step": round(elapsed / total_steps, 3), "step0_eval": step0,
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if use_amp else None,
               "evals": evals, "kept": keeper.kept}
    if probe_steps:
        summary["projected_hours_full_run"] = round(
            (summary["sec_per_step"] * run.max_steps
             + 60 * step0["eval_minutes"] * (run.max_steps // run.eval_every + 1)) / 3600, 2)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def run_config_dict(run: RunConfig):
    return asdict(run)
