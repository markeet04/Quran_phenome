"""T6: run discovery, cache plan, out-of-fold guard, ensembles, LM rescoring."""
import json
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import PHONEMES, VOCAB_TOKENS  # noqa: E402
from iqra import ensemble as E  # noqa: E402
from iqra import lm as L  # noqa: E402
from iqra.decode import greedy_decode, log_softmax  # noqa: E402
from iqra.posteriors import make_entry  # noqa: E402

BLANK = 0


def _run(root, sub, name, kept, folds):
    d = root / sub / f"nb13_{name.lower()}"
    d.mkdir(parents=True)
    for _, step, _ in kept:
        c = d / f"ckpt_step{step:05d}"
        c.mkdir()
        (c / "config.json").write_text("{}")
    (d / "run_summary.json").write_text(json.dumps(
        {"run": name, "kept": [[f, s, f"/kaggle/working/nb13_{name.lower()}/ckpt_step{s:05d}"] for f, s, _ in kept]}))
    (d / "run_config.json").write_text(json.dumps({"run": {"name": name, "extra_folds": list(folds)},
                                                   "active": {}, "repo_commit": "x"}))
    return d


A_KEPT = [(0.3201, 5500, 0), (0.3154, 7500, 0), (0.3149, 8000, 0)]
B_KEPT = [(0.3393, 7000, 0), (0.3363, 7500, 0), (0.336, 8000, 0)]
BASE_KEPT = [(0.3353, 6000, 0), (0.33, 7000, 0), (0.32, 8000, 0)]


@pytest.fixture
def runs_root(tmp_path):
    _run(tmp_path, "run-a", "A", A_KEPT, ["fold_a"])
    _run(tmp_path, "nb-b", "B", B_KEPT, ["fold_b"])
    _run(tmp_path, "base", "BASE", BASE_KEPT, ["fold_a", "fold_b"])
    return tmp_path


SPLITS = {"extra": {"fold_a": ["a1", "a2"], "fold_b": ["b1", "b2"], "extra_dev": ["d1"]}}


def test_find_runs_and_plan(runs_root):
    _run(runs_root, "dup-of-b", "B", B_KEPT, ["fold_b"])          # same run attached twice: fine
    probe = runs_root / "probe" / "nb13_probe" / "base"          # PROBE output: nothing kept,
    probe.mkdir(parents=True)                                    # no run_config.json -> ignored
    (probe / "run_summary.json").write_text(json.dumps({"run": "BASE", "kept": []}))
    runs = E.find_runs(runs_root)
    assert [m.step for m in runs["A"]["members"]] == [5500, 7500, 8000]
    assert runs["B"]["members"][0].name == "B07000"
    plan = E.cache_plan(runs)
    assert [(m.name, s) for m, s in plan] == [
        ("A05500", ["fold_b", "extra_dev", "iqra_dev_slice"]), ("A07500", ["fold_b", "extra_dev", "iqra_dev_slice"]),
        ("A08000", ["fold_b", "extra_dev", "iqra_dev_slice"]), ("B07000", ["fold_a", "extra_dev", "iqra_dev_slice"]),
        ("B07500", ["fold_a", "extra_dev", "iqra_dev_slice"]), ("B08000", ["fold_a", "extra_dev", "iqra_dev_slice"]),
        ("BASE06000", ["extra_dev"])]
    E.assert_out_of_fold(plan, runs, SPLITS)


def test_find_runs_over_several_roots(tmp_path_factory):
    one, two = tmp_path_factory.mktemp("input"), tmp_path_factory.mktemp("kagglehub")
    _run(one, "a", "A", A_KEPT, ["fold_a"])
    _run(one, "b", "B", B_KEPT, ["fold_b"])
    assert "BASE" not in E.find_runs([one], required=())               # probe without raising
    _run(two, "v5", "BASE", BASE_KEPT, ["fold_a", "fold_b"])          # e.g. a notebook version
    assert sorted(E.find_runs([one, two])) == ["A", "B", "BASE"]


def test_find_runs_rejects_conflicts_and_missing(runs_root, tmp_path_factory):
    _run(runs_root, "other-b", "B", [(0.30, 6000, 0)], ["fold_b"])
    with pytest.raises(AssertionError, match="two different outputs"):
        E.find_runs(runs_root)
    lonely = tmp_path_factory.mktemp("lonely")
    _run(lonely, "a", "A", A_KEPT, ["fold_a"])
    with pytest.raises(FileNotFoundError, match="not attached"):
        E.find_runs(lonely)


def test_wrong_folds_and_leaks_are_caught(runs_root):
    runs = E.find_runs(runs_root)
    bad = dict(runs, A=dict(runs["A"], extra_folds=("fold_b",)))
    with pytest.raises(AssertionError, match="trained on"):
        E.cache_plan(bad)
    plan = [(runs["A"]["members"][0], ["fold_a"])]                 # A on its own training fold
    with pytest.raises(AssertionError, match="which it trained on"):
        E.assert_out_of_fold(plan, runs, SPLITS)
    leaky = {"extra": dict(SPLITS["extra"], extra_dev=["a1"])}
    with pytest.raises(AssertionError, match="Extra-dev"):
        E.assert_out_of_fold(E.cache_plan(runs), runs, leaky)


def _entry(rng, T=40, V=len(VOCAB_TOKENS), scale=4.0):
    return make_entry(rng.normal(0, scale, (T, V)).astype(np.float32), VOCAB_TOKENS, BLANK)


def test_ensemble_entry_is_mean_probability():
    rng = np.random.default_rng(0)
    a, b = _entry(rng), _entry(rng)
    ens = E.ensemble_entry([a, b], VOCAB_TOKENS, BLANK)
    pa, pb = np.exp(log_softmax(a["logits"].numpy())), np.exp(log_softmax(b["logits"].numpy()))
    np.testing.assert_allclose(np.exp(ens["logits"].numpy()), (pa + pb) / 2, rtol=1e-5, atol=1e-7)
    assert ens["tokens"] == greedy_decode(np.log((pa + pb) / 2), VOCAB_TOKENS, BLANK)[0]
    same = E.ensemble_entry([a, a, a], VOCAB_TOKENS, BLANK)
    assert same["tokens"] == a["tokens"] and same["spans"] == a["spans"]


def test_ensemble_split_and_frame_check():
    rng = np.random.default_rng(1)
    caches = {"m1": {"c1": _entry(rng), "c2": _entry(rng, T=30)},
              "m2": {"c1": _entry(rng), "c2": _entry(rng, T=30)}}
    assert E.frames_agree(caches, ["m1", "m2"]) == []
    ens = E.ensemble_split(caches, ["m1", "m2"], VOCAB_TOKENS, BLANK)
    assert sorted(ens) == ["c1", "c2"] and ens["c2"]["logits"].shape[0] == 30
    caches["m2"]["c2"] = _entry(rng, T=31)
    assert E.frames_agree(caches, ["m1", "m2"]) == [("c2", [30, 31])]
    del caches["m2"]["c1"]
    with pytest.raises(AssertionError, match="different clips"):
        E.ensemble_split(caches, ["m1", "m2"], VOCAB_TOKENS, BLANK)


def test_clip_final_deletions():
    import pandas as pd
    df = pd.DataFrame({"op": ["D", "D", "D", "S"], "is_last": [1, 1, 0, 1], "label": [0, 1, 0, 0]})
    r = E.clip_final_deletions(df)
    assert (r["deletions"], r["false_alarm_deletions"], r["clip_final_deletions"]) == (3, 2, 2)
    assert (r["clip_final_among_false_alarms"], r["clip_final_real"], r["share_of_false_alarm_deletions"]) == (1, 1, 0.5)


# --------------------------------------------------------------------------- LM
REAL = [p for p in PHONEMES if p in VOCAB_TOKENS]


def test_nbest_greedy_first_and_alpha_zero_is_greedy():
    rng = np.random.default_rng(2)
    for _ in range(10):
        lp = log_softmax(rng.normal(0, 2.5, (60, len(VOCAB_TOKENS))))
        cands = L.nbest(lp, VOCAB_TOKENS, BLANK)
        assert cands[0][1] == greedy_decode(lp, VOCAB_TOKENS, BLANK)[0]
        assert 1 <= len(cands) <= 17
        assert all(t not in ("<sil>", "<pad>", "<unk>") for _, toks in cands for t in toks)
    tied = np.full((12, len(VOCAB_TOKENS)), -50.0)                 # exact ties for the top
    tied[:, [31, 33]] = 0.0                                          # 'b' (31) and 'd' (33) tie
    tied[6:, :] = -50.0
    tied[6:, [0, 45]] = 0.0                                          # blank vs 'k': blank wins
    tied = log_softmax(tied)
    assert L.nbest(tied, VOCAB_TOKENS, BLANK)[0][1] == greedy_decode(tied, VOCAB_TOKENS, BLANK)[0] == ["b"]
    lm = L.NgramLM().train([" ".join(rng.choice(REAL, 20)) for _ in range(50)])
    by_clip = {"c": cands}
    assert L.rescore(by_clip, lm, 0.0)["c"] == cands[0][1]


def test_lm_prefers_its_corpus_and_corpus_excludes_dev():
    rnd = random.Random(0)
    lines = [" ".join(rnd.choice(REAL[:10]) for _ in range(15)) for _ in range(200)]
    corpus, removed = L.build_corpus(lines + ["  b  a  "], exclude=["b a", "nothing"], vocab_symbols=set(REAL))
    assert removed == 1 and "b a" not in corpus and len(corpus) == 200
    corpus2, _ = L.build_corpus(["b <sil> a q"], exclude=[], vocab_symbols={"a", "b"})
    assert corpus2 == ["b a"]
    lm = L.NgramLM().train(corpus)
    toks = corpus[0].split()
    shuffled = toks[:]
    random.Random(1).shuffle(shuffled)
    foreign = [REAL[-1]] * len(toks)
    assert lm.score_seq(toks) > lm.score_seq(foreign)
    assert lm.score_seq(toks) >= lm.score_seq(shuffled) - 1e-9


def test_choose_alpha_recall_guard():
    rows = [{"alpha": 0.0, "f1": 0.30, "recall": 0.70}, {"alpha": 1.0, "f1": 0.31, "recall": 0.7003},
            {"alpha": 4.0, "f1": 0.35, "recall": 0.65}]
    assert L.choose_alpha(rows, 0.70)["alpha"] == 1.0              # 4.0 drops recall: refused
    assert L.choose_alpha(rows[:1] + rows[2:], 0.70)["alpha"] == 0.0
