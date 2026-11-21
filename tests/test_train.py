"""T5 training loop on CPU with a tiny w2v-BERT and fake data."""
import io
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")
sf = pytest.importorskip("soundfile")

from conftest import PHONEMES, VOCAB_TOKENS  # noqa: E402
from iqra import model as M  # noqa: E402
from iqra import train as T  # noqa: E402
from test_model import _tiny  # noqa: E402

TOK2ID = {t: i for i, t in enumerate(VOCAB_TOKENS)}
FE = transformers.SeamlessM4TFeatureExtractor()


def _wav(seconds, seed, sr=16000):
    x = np.random.default_rng(seed).normal(0, 0.05, int(seconds * sr)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    return {"bytes": buf.getvalue(), "path": None}, x


def _ds(n, seed, extra_cols=None):
    rng = np.random.default_rng(seed)
    cells = [_wav(rng.uniform(0.6, 1.5), seed * 1000 + i)[0] for i in range(n)]
    cols = {"audio": cells}
    feats = {"audio": {"bytes": datasets.Value("binary"), "path": datasets.Value("string")}}
    for name, values in (extra_cols or {}).items():
        cols[name] = values
        feats[name] = datasets.Value("string")
    ds = datasets.Dataset.from_dict(cols, features=datasets.Features(feats))
    return ds.cast_column("audio", datasets.Audio(decode=False))


def _phon(rng, n):
    return " ".join(PHONEMES[i] for i in rng.integers(0, len(PHONEMES), n))


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    train_ds = _ds(12, 1)
    tts_ds = _ds(10, 2)
    extra_ids = [f"e{i}" for i in range(9)]
    extra_ds = _ds(9, 3, {"id": extra_ids})
    splits = {"extra": {"ids_by_row": extra_ids, "fold_a": extra_ids[:3], "fold_b": extra_ids[3:6],
                        "extra_dev": extra_ids[6:]}}
    labels = {c: _phon(rng, 4) for c in extra_ids}
    sw = pd.DataFrame({"dataset": ["iqra_train"] * 12 + ["iqra_tts"] * 10,
                       "row": list(range(12)) + list(range(10)),
                       "weight": [0.2] + [1.0] * 11 + [3.0] + [1.0] * 9})
    return dict(train_ds=train_ds, train_labels=[_phon(rng, 5) for _ in range(12)], tts_ds=tts_ds,
                tts_labels=[_phon(rng, 5) for _ in range(10)], extra_ds=extra_ds, extra_ids_by_row=extra_ids,
                extra_labels_by_id=labels, splits=splits, sample_weights=sw)


def test_cosine_factor():
    assert T.cosine_factor(0, 200, 8000) == 0.0
    assert T.cosine_factor(100, 200, 8000) == 0.5
    assert T.cosine_factor(200, 200, 8000) == 1.0
    assert T.cosine_factor(8000, 200, 8000) == pytest.approx(0.0, abs=1e-12)


def test_build_sources_and_dataset(setup):
    run = replace(T.RUNS["A"], extra_repeat=20)
    srcs = T.build_sources(run, **setup)
    assert [s.name for s in srcs] == ["iqra_train", "iqra_tts", "extra"]
    assert len(srcs[2].rows) == 3 * 20                             # fold A only, x20
    ds = T.CTCDataset(srcs, FE, TOK2ID, TOK2ID["<unk>"])
    w = ds.weights()
    assert w[0] == 0.2 and w[12] == 3.0 and len(w) == 12 + 10 + 60
    item = ds[0]
    assert item["input_features"].shape[1] == 160 and len(item["labels"]) == 5
    batch = T.Collator(FE)([ds[0], ds[13], ds[30]])               # 5, 5 and 4 label tokens
    assert batch["labels"].shape == (3, 5)
    assert int((batch["labels"][2] == -100).sum()) == 1 and int((batch["labels"][0] == -100).sum()) == 0
    assert batch["attention_mask"].shape[:2] == batch["input_features"].shape[:2]
    base = T.build_sources(T.RUNS["BASE"], **setup)
    assert len(base[2].rows) == 6 * 20 and base[0].weights is None   # folds A+B, unweighted


def test_build_sources_refuses_extra_dev(setup):
    bad = dict(setup)
    bad["splits"] = {"extra": dict(setup["splits"]["extra"], fold_a=setup["splits"]["extra"]["extra_dev"])}
    with pytest.raises(AssertionError):
        T.build_sources(T.RUNS["A"], **bad)


def test_checkpoint_keeper_keeps_best_three(tmp_path):
    m = _tiny()
    keeper = T.CheckpointKeeper(tmp_path, 3, FE)
    for step, f1 in [(500, 0.30), (1000, 0.35), (1500, 0.32), (2000, 0.31), (2500, 0.40), (3000, 0.10)]:
        keeper.offer(m, step, f1)
    assert [s for _, s, _ in keeper.kept] == [2500, 1000, 1500]
    assert sorted(p.name for p in tmp_path.glob("ckpt_step*")) == ["ckpt_step01000", "ckpt_step01500", "ckpt_step02500"]


def _evaluator(setup):
    from iqra import data
    dev_ids = setup["splits"]["extra"]["extra_dev"]
    rows = {c: r for r, c in enumerate(setup["extra_ids_by_row"])}
    dev_audio = {c: data.to_16k(*data.decode_audio(
        setup["extra_ds"].data.slice(rows[c], 1).column("audio").to_pylist()[0])) for c in dev_ids}
    ref = setup["extra_labels_by_id"]
    said = {c: ref[c].split()[:-1] + ["a"] for c in dev_ids}        # one real error per clip
    truth = pd.DataFrame({"ID": dev_ids, "Reference_phn": [ref[c] for c in dev_ids],
                          "Annotation_phn": [" ".join(said[c]) for c in dev_ids]})
    slice_audio = {f"s{i}": data.to_16k(*data.decode_audio(setup["train_ds"].data.slice(i, 1).column("audio").to_pylist()[0]))
                   for i in range(3)}
    slice_refs = {f"s{i}": setup["train_labels"][i] for i in range(3)}
    return T.DevEvaluator(dev_audio, truth, slice_audio, slice_refs, FE, VOCAB_TOKENS, 0, "cpu",
                          dict(zip(truth["ID"], truth["Reference_phn"])))


class _Always:
    """Evaluator stub returning a scripted F1 sequence (the tiny model's own F1 is ~0)."""
    def __init__(self, f1s, fe):
        self.f1s, self.fe, self.k = list(f1s), fe, 0

    def __call__(self, model):
        f = self.f1s[min(self.k, len(self.f1s) - 1)]
        self.k += 1
        return {"f1": f, "precision": f, "recall": f, "per_dev": 0.5, "per_slice": 0.5}


def test_end_to_end_tiny_training(setup, tmp_path):
    run = replace(T.RUNS["A"], max_steps=6, eval_every=2, warmup_steps=2, batch_size=2, grad_accum=2,
                  num_workers=0, fp16=False, keep_best=2, extra_repeat=2)
    m = _tiny()
    active = T.configure(m, run)
    assert active["fusion"] and active["apply_spec_augment"] and active["mask_time_prob"] == 0.05
    w0 = M.fusion_weights(m).copy()
    ds = T.CTCDataset(T.build_sources(run, **setup), FE, TOK2ID, TOK2ID["<unk>"])
    summary = T.train(run, m, T.make_loader(ds, run), _Always([0.0, 0.2, 0.5, 0.3], FE), tmp_path, "cpu",
                      log=lambda *a: None)
    assert [e["step"] for e in summary["evals"]] == [0, 2, 4, 6]
    assert [s for _, s, _ in summary["kept"]] == [4, 6]             # best 2 by F1
    log = pd.read_csv(tmp_path / "train_log.csv")
    assert np.isfinite(log["ctc"]).all() and (log["kl"] >= 0).all()
    assert np.abs(M.fusion_weights(m) - w0).max() > 0               # fusion weights move (lr 1e-2)
    reloaded, info = M.load_model(tmp_path / "ckpt_step00004")
    assert not any(info.values()) and getattr(reloaded.config, M.FUSION_FLAG)


def test_real_evaluator_and_probe(setup, tmp_path):
    run = replace(T.RUNS["BASE"], max_steps=50, batch_size=2, grad_accum=1, num_workers=0, fp16=False,
                  extra_repeat=1)
    m = _tiny()
    T.configure(m, run)
    ds = T.CTCDataset(T.build_sources(run, **setup), FE, TOK2ID, TOK2ID["<unk>"])
    summary = T.train(run, m, T.make_loader(ds, run), _evaluator(setup), tmp_path, "cpu",
                      probe_steps=3, log=lambda *a: None)
    assert summary["steps"] == 3 and summary["kept"] == []           # probe keeps nothing
    assert "projected_hours_full_run" in summary and "f1" in summary["step0_eval"]
