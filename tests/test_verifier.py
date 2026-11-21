"""Candidate features, error types and the LightGBM verifier on synthetic posteriors."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightgbm")

from conftest import PHONEMES, VOCAB_TOKENS  # noqa: E402
from iqra import scorer, verifier  # noqa: E402
from iqra.candidates import candidate_features, error_type  # noqa: E402
from iqra.config import Verifier  # noqa: E402
from iqra.decode import greedy_decode  # noqa: E402

TOK2ID = {t: i for i, t in enumerate(VOCAB_TOKENS)}
REAL = [p for p in PHONEMES if p in TOK2ID]


def test_error_types():
    assert error_type("D", "b", "") == "deletion"
    assert error_type("S", "l", "ll") == "shadda"
    assert error_type("S", "a", "aa") == "vowel_length"
    assert error_type("S", "a", "i") == "vowel_quality"
    assert error_type("S", "S", "s") == "emphatic"
    assert error_type("S", "SS", "s") == "emphatic"
    assert error_type("S", "<", "a") == "hamza"
    assert error_type("S", "q", "k") == "consonant"
    assert error_type("S", "b", "a") == "vowel_consonant"


def _clip(rng, said, expected, noise):
    """Logits that spell `said`, plus random confusions (false alarms) at rate `noise`."""
    frames, spoken = [], []
    for tok in said:
        out = tok
        if rng.random() < noise:
            out = REAL[rng.integers(len(REAL))]
        spoken.append(out)
        frames += [TOK2ID[out]] * int(rng.integers(2, 5)) + [0]
    T, V = len(frames), len(VOCAB_TOKENS)
    logits = rng.normal(0, 1, (T, V))
    logits[np.arange(T), frames] += 6.0
    toks, spans = greedy_decode(logits, VOCAB_TOKENS, 0)
    return {"logits": torch.from_numpy(logits.astype(np.float32)), "tokens": toks,
            "spans": [list(s) for s in spans]}


def _table(n, seed, noise=0.15):
    rng = np.random.default_rng(seed)
    rows, truth, heard = [], [], {}
    for k in range(n):
        expected = [REAL[i] for i in rng.integers(0, len(REAL), rng.integers(6, 14))]
        said = list(expected)
        if rng.random() < 0.7:
            said[rng.integers(len(said))] = REAL[rng.integers(len(REAL))]
        entry = _clip(rng, said, expected, noise)
        cid = f"c{seed}_{k}"
        rows += candidate_features(cid, expected, entry, TOK2ID, VOCAB_TOKENS, 0, said=said)
        truth.append({"ID": cid, "Reference_phn": " ".join(expected), "Annotation_phn": " ".join(said)})
        heard[cid] = entry["tokens"]
    return pd.DataFrame(rows), pd.DataFrame(truth), heard


def test_features_are_complete_and_labelled():
    df, _, _ = _table(30, 0)
    assert set(verifier.FEATURES) <= set(df.columns)
    assert set(df["op"]) <= {"S", "D"}
    assert df["label"].isin([0, 1]).all() and df["label"].sum() > 0
    s = df[df["op"] == "S"]
    assert s["heard_lp_mean"].notna().all() and (s["span_len"] >= 1).all()


def test_verifier_trains_cross_validates_and_sweeps():
    cfg = Verifier(max_rounds=60, early_stopping=10, cv_folds=3)
    train_df, _, _ = _table(120, 1)
    cv = verifier.cross_validate(train_df, list(train_df["clip"]), cfg)
    assert np.isfinite(cv["oof"]).all() and 0 <= cv["auc"] <= 1
    booster, enc = verifier.train(train_df, int(np.mean(cv["best_iterations"])), cfg)
    dev_df, truth, heard = _table(40, 2)
    probs = verifier.predict(booster, enc, dev_df)
    sweep = verifier.sweep_tau(dev_df, probs, truth, heard, [0.0, 0.3, 0.6, 0.95])
    # tau = 0 keeps every flag: identical to scoring the raw decode
    raw = pd.DataFrame({"ID": list(heard), "Prediction": [" ".join(v) for v in heard.values()]})
    assert sweep.iloc[0]["f1"] == pytest.approx(scorer.summed_metrics(scorer.per_clip_counts(truth, raw))["f1"])
    assert verifier.robust_tau(sweep) in set(sweep["tau"])


@pytest.mark.parametrize("name", list(verifier.FEATURE_SETS))
def test_feature_subsets_train_and_predict(name):
    cfg = Verifier(max_rounds=30, early_stopping=5, cv_folds=3)
    feats = verifier.FEATURE_SETS[name]
    train_df, _, _ = _table(60, 3)
    cv = verifier.cross_validate(train_df, list(train_df["clip"]), cfg, features=feats)
    booster, enc = verifier.train(train_df, max(1, int(np.median(cv["best_iterations"]))), cfg, features=feats)
    assert booster.num_feature() == len(feats)
    assert len(verifier.predict(booster, enc, _table(10, 4)[0])) > 0


def test_primary_design_reverts_every_deletion():
    """Decision 29: S-only verifier; D candidates are reverted whatever tau is."""
    cfg = Verifier(max_rounds=30, early_stopping=5, cv_folds=3)
    train_df, _, _ = _table(80, 5)
    train_df["sentence"] = train_df["clip"]
    booster, enc, cv = verifier.train_primary(train_df, cfg)
    assert booster.num_feature() == len(verifier.FEATURE_SETS["primary"])
    dev_df, truth, heard = _table(30, 6)
    probs = verifier.predict_primary(booster, enc, dev_df)
    assert np.isnan(probs[dev_df["op"] == "D"]).all() and np.isfinite(probs[dev_df["op"] == "S"]).all()
    expected = dict(zip(truth["ID"], truth["Reference_phn"].str.split()))
    keep_all_s = verifier.apply(dev_df, probs, expected, heard, tau=0.0, revert_deletions=True)
    from iqra.candidates import revert
    for cid, h in heard.items():
        d_pos = {int(p): False for c, p, op in dev_df[["clip", "pos", "op"]].itertuples(index=False)
                 if c == cid and op == "D"}
        assert keep_all_s[cid] == revert(expected[cid], h, d_pos)


def test_identity_free_set_has_no_identity_features():
    assert not set(verifier.IDENTITY) & set(verifier.FEATURE_SETS["no_identity"])


def test_auc_and_precision_at_recall():
    assert verifier.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert verifier.auc([0, 1], [0.5, 0.5]) == 0.5
    assert verifier.precision_at_recall([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 1.0) == pytest.approx(2 / 3)
