"""Scorer parity, count parity, perfect and null predictors (SPEC T0)."""
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import make_synthetic
from iqra import scorer
from iqra.align import mdd_eval_dir, scorer_env


def _by_hand(truth_df, pred_df, tmp_path):
    """Run the two official scripts the way NB5 does: relative paths, cwd = work dir."""
    truth_df.to_csv(tmp_path / "truth.csv", index=False)
    pred_df.to_csv(tmp_path / "pred.csv", index=False)
    d = mdd_eval_dir()
    r1 = subprocess.run([sys.executable, str(d / "align_data.py"), "--truth-file", "truth.csv",
                         "--pred-file", "pred.csv", "--output-dir", "aligned"],
                        cwd=tmp_path, capture_output=True, text=True, check=True,
                        env=scorer_env())
    r2 = subprocess.run([sys.executable, str(d / "ins_del_cor_sub_analysis.py"),
                         "--align-folder", "aligned/"],
                        cwd=tmp_path, capture_output=True, text=True, check=True,
                        env=scorer_env())
    grab = lambda pat, s: re.search(pat, s).group(1)
    out = r2.stdout
    return {
        "precision": float(grab(r"Precision: ([0-9.]+)", out)),
        "recall": float(grab(r"Recall: ([0-9.]+)", out)),
        "f1": float(grab(r"f1-score: ([0-9.]+)", out)),
        "TA": int(grab(r"True Acception: [0-9.]+ \| (\d+)", out)),
        "FR": int(grab(r"False Rejection: [0-9.]+ \| (\d+)", out)),
        "FA": int(grab(r"False Acceptance: [0-9.]+ \| (\d+)", out)),
        "TR": int(grab(r"Correct Diagnosis: [0-9.]+ \| (\d+)", out))
              + int(grab(r"Error Diagnosis: [0-9.]+ \| (\d+)", out)),
        "accuracy": float(grab(r"Accuracy: ([0-9.]+)", r1.stdout)),
        "raw": out,
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scorer_parity(seed, tmp_path):
    truth, pred = make_synthetic(50, seed)
    hand = _by_hand(truth, pred, tmp_path)
    ours = scorer.official_score(truth, pred)
    for k in ("precision", "recall", "f1", "TA", "FR", "FA", "TR"):
        assert ours[k] == hand[k], (k, ours[k], hand[k])
    assert ours["per"] == round(1 - hand["accuracy"], 4)
    print(f"seed {seed}: P={ours['precision']} R={ours['recall']} F1={ours['f1']} "
          f"TR={ours['TR']} FR={ours['FR']} FA={ours['FA']} TA={ours['TA']} PER={ours['per']}")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_count_parity(seed):
    truth, pred = make_synthetic(50, seed)
    official = scorer.official_score(truth, pred)
    counts = scorer.per_clip_counts(truth, pred)
    assert len(counts) == 50
    sums = counts[["TR", "FR", "FA", "TA"]].sum()
    for k in ("TR", "FR", "FA", "TA"):
        assert int(sums[k]) == official[k], (k, int(sums[k]), official[k])
    m = scorer.summed_metrics(counts)
    assert round(m["precision"], 4) == official["precision"]
    assert round(m["recall"], 4) == official["recall"]
    assert round(m["f1"], 4) == official["f1"]
    print(f"seed {seed}: per-clip sums TR={int(sums['TR'])} FR={int(sums['FR'])} "
          f"FA={int(sums['FA'])} TA={int(sums['TA'])} == official")


@pytest.mark.parametrize("seed", range(10, 16))
def test_count_parity_every_branch(seed, tmp_path):
    """Mixed predictions plus very short clips: every official counter fires, and the
    per-clip sums still equal the official totals exactly."""
    long_t, long_p = make_synthetic(150, seed, mixed=True)
    short_t, short_p = make_synthetic(50, seed + 100, mixed=True, min_len=1, max_len=3)
    short_t["ID"] = short_t["ID"].str.replace("clip", "short")
    short_p["ID"] = short_p["ID"].str.replace("clip", "short")
    truth = pd.concat([long_t, short_t], ignore_index=True)
    pred = pd.concat([long_p, short_p], ignore_index=True)

    counts = scorer.per_clip_counts(truth, pred)
    totals = counts[list(scorer.COUNT_KEYS)].sum()
    assert (totals > 0).all(), totals[totals == 0].to_dict()

    official = scorer.official_score(truth, pred)
    hand = _by_hand(truth, pred, tmp_path)
    for k in ("TR", "FR", "FA", "TA"):
        assert int(counts[k].sum()) == official[k] == hand[k], k
    m = scorer.summed_metrics(counts)
    for k in ("precision", "recall", "f1"):
        assert round(m[k], 4) == official[k] == hand[k], k
    print(f"seed {seed}: 200 clips, all 11 counters > 0, min {int(totals.min())}; "
          f"TR={official['TR']} FR={official['FR']} FA={official['FA']} TA={official['TA']} "
          f"F1={official['f1']} parity exact")


def test_per_parsed_when_accuracy_is_negative():
    analysis = ("Recall: 0.5000\nPrecision: 0.5000\nf1-score: 0.5000\nTrue Acception: 0.5 | 1\n"
                "False Rejection: 0.5 | 1\nFalse Acceptance: 0.5 | 1\nCorrect Diagnosis: 1.0 | 1\n"
                "Error Diagnosis: 0.0 | 0\n")
    out = scorer.parse_official("** MD&D Evaluation **\nCorrect Rate: -0.2\nAccuracy: -7.6054\n", analysis)
    assert out["per"] == 8.6054


def test_perfect_predictor(synthetic):
    truth, _ = synthetic
    pred = truth.rename(columns={"Annotation_phn": "Prediction"})[["ID", "Prediction"]]
    res = scorer.official_score(truth, pred)
    assert res["f1"] == 1.0 and res["precision"] == 1.0 and res["recall"] == 1.0
    assert res["FR"] == 0 and res["FA"] == 0


def test_null_predictor(synthetic):
    truth, _ = synthetic
    pred = truth.rename(columns={"Reference_phn": "Prediction"})[["ID", "Prediction"]]
    counts = scorer.per_clip_counts(truth, pred)
    assert int(counts["TR"].sum()) == 0
    assert scorer.summed_metrics(counts)["recall"] == 0.0
    if int(counts["FR"].sum()) == 0:
        # TR + FR == 0: the official script divides by zero computing precision.
        with pytest.raises(scorer.ScorerUndefined):
            scorer.official_score(truth, pred)
        print("null predictor: TR=0, FR=0 -> official script raises ZeroDivisionError")
    else:
        assert scorer.official_score(truth, pred)["recall"] == 0.0


def test_rejects_inputs_the_official_scripts_mishandle(synthetic):
    truth, pred = synthetic
    with pytest.raises(scorer.ScorerInputError):
        scorer.official_score(truth, pred.iloc[:-1])               # silently dropped clip
    empty = pred.copy(); empty.loc[0, "Prediction"] = ""
    with pytest.raises(scorer.ScorerInputError):
        scorer.official_score(truth, empty)                        # NaN crash
    bad = truth.copy(); bad.loc[0, "ID"] = "xrefx"
    bad_pred = pred.copy(); bad_pred.loc[0, "ID"] = "xrefx"
    with pytest.raises(scorer.ScorerInputError):
        scorer.official_score(bad, bad_pred)                       # line-parser substring


def test_paired_bootstrap(synthetic):
    truth, pred = synthetic
    counts = scorer.per_clip_counts(truth, pred)
    clusters = {i: f"s{k % 10}" for k, i in enumerate(counts.index)}      # 10 "sentences"
    same = scorer.paired_bootstrap_gain(counts, counts, clusters, n_resamples=200)
    assert same["gain"] == 0.0 and same["lo"] == 0.0 and same["hi"] == 0.0
    assert same["n_clusters"] == 10
    perfect = truth.rename(columns={"Annotation_phn": "Prediction"})[["ID", "Prediction"]]
    better = scorer.paired_bootstrap_gain(counts, scorer.per_clip_counts(truth, perfect),
                                          clusters, n_resamples=200)
    assert better["gain"] > 0 and better["excludes_zero"]
    with pytest.raises(ValueError):
        scorer.paired_bootstrap_gain(counts, counts, {}, n_resamples=10)


def test_cluster_bootstrap_is_wider_on_correlated_data():
    """10 readings per sentence that share errors: resampling clips overstates certainty."""
    rng = np.random.default_rng(0)
    rows_b, rows_n, clusters = [], [], {}
    for s in range(27):
        # the new system helps on some sentences and hurts on others, consistently per sentence
        effect = rng.normal(0, 3)
        for r in range(10):
            cid = f"s{s}_r{r}"
            clusters[cid] = f"s{s}"
            base = dict(TR=10, FR=6, FA=4)
            new = dict(TR=10, FR=max(0, int(round(6 - effect))), FA=4)
            rows_b.append({"ID": cid, **base}); rows_n.append({"ID": cid, **new})
    b = pd.DataFrame(rows_b).set_index("ID"); n = pd.DataFrame(rows_n).set_index("ID")
    by_clip = scorer.paired_bootstrap_gain(b, n, {i: i for i in b.index}, n_resamples=500)
    by_sentence = scorer.paired_bootstrap_gain(b, n, clusters, n_resamples=500)
    assert by_clip["gain"] == by_sentence["gain"]
    assert (by_sentence["hi"] - by_sentence["lo"]) > 2 * (by_clip["hi"] - by_clip["lo"])


def test_bootstrap_f1_interval(synthetic):
    truth, pred = synthetic
    counts = scorer.per_clip_counts(truth, pred)
    res = scorer.bootstrap_f1_interval(counts, {i: i for i in counts.index}, n_resamples=200)
    assert res["lo"] <= res["f1"] <= res["hi"]
    assert res["f1"] == pytest.approx(scorer.summed_metrics(counts)["f1"])
