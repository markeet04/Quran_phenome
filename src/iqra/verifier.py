"""LightGBM flag verifier: train, grouped CV, predict, tau sweep (SPEC section 5).

Cross-validation groups by canonical sentence, not by clip: Extra sentences are read
~10 times, so clip-grouped folds would put readings of the same sentence on both sides
and overstate AUC (same reasoning as decision 25).
"""
import numpy as np
import pandas as pd

from .candidates import revert
from . import scorer

CATEGORICAL = ["op", "tok_exp", "tok_heard", "exp_prev1", "exp_prev2", "exp_next1", "exp_next2",
               "pair", "err_type", "runner_up"]
NUMERIC = ["fa_failed", "span_len", "exp_lp_mean", "exp_lp_max", "gop", "entropy_mean", "blank_mean",
           "n_flags_per_phone", "n_frames", "n_expected", "rel_pos", "heard_lp_mean", "heard_lp_max",
           "margin_exp_minus_heard", "heard_peak_prob", "runner_up_prob", "runner_up_is_expected",
           "heard_span_len"]
FEATURES = CATEGORICAL + NUMERIC

# Feature groups for the T3 diagnostics (ablations are reported, never used to tune)
IDENTITY = ["n_expected", "n_frames", "rel_pos", "n_flags_per_phone"]   # utterance length / position
ACOUSTIC = ["fa_failed", "span_len", "exp_lp_mean", "exp_lp_max", "gop", "entropy_mean", "blank_mean",
            "heard_lp_mean", "heard_lp_max", "margin_exp_minus_heard", "heard_peak_prob",
            "runner_up_prob", "runner_up_is_expected", "heard_span_len"]
FEATURE_SETS = {
    "full": FEATURES,
    "no_identity": [f for f in FEATURES if f not in IDENTITY],
    "acoustic_only": ["op"] + ACOUSTIC,
    "pair_only": ["op", "pair"],
}


def params(cfg_verifier, seed=None):
    v = cfg_verifier
    return {"objective": "binary", "learning_rate": v.learning_rate, "num_leaves": v.num_leaves,
            "min_data_in_leaf": v.min_data_in_leaf, "seed": v.seed if seed is None else seed,
            "deterministic": True, "force_row_wise": True, "num_threads": 4, "verbose": -1}


class Encoder:
    """Fixes category levels on the training table; unseen levels become NaN."""

    def __init__(self, features=None):
        self.features = list(features or FEATURES)
        self.categorical = [c for c in self.features if c in CATEGORICAL]

    def fit(self, df):
        self.levels = {c: sorted(df[c].astype(str).unique()) for c in self.categorical}
        return self

    def transform(self, df):
        X = df[self.features].copy()
        for c in self.features:
            if c in self.categorical:
                X[c] = pd.Categorical(X[c].astype(str), categories=self.levels[c])
            else:
                X[c] = X[c].astype(float)
        return X


def auc(y, p):
    """ROC AUC by the rank-sum formula (ties get average ranks)."""
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def precision_at_recall(y, p, target=0.9):
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p, kind="stable")
    tp = np.cumsum(y[order] == 1)
    recall = tp / max(int((y == 1).sum()), 1)
    k = int(np.argmax(recall >= target))
    return float(tp[k] / (k + 1))


def cross_validate(train_df, groups, cfg_verifier, features=None):
    """Grouped K-fold: out-of-fold probabilities, best iteration per fold, AUC, P@R0.9."""
    import lightgbm as lgb
    y = train_df["label"].astype(int).to_numpy()
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(cfg_verifier.seed)
    rng.shuffle(uniq)
    fold_of = {g: k % cfg_verifier.cv_folds for k, g in enumerate(uniq)}
    folds = np.array([fold_of[g] for g in groups])
    oof = np.full(len(y), np.nan)
    best = []
    for k in range(cfg_verifier.cv_folds):
        tr, va = folds != k, folds == k
        enc = Encoder(features).fit(train_df[tr])
        dtr = lgb.Dataset(enc.transform(train_df[tr]), y[tr], categorical_feature=enc.categorical)
        dva = lgb.Dataset(enc.transform(train_df[va]), y[va], categorical_feature=enc.categorical,
                          reference=dtr)
        booster = lgb.train(params(cfg_verifier), dtr, num_boost_round=cfg_verifier.max_rounds,
                            valid_sets=[dva], callbacks=[lgb.early_stopping(cfg_verifier.early_stopping,
                                                                            verbose=False)])
        oof[va] = booster.predict(enc.transform(train_df[va]), num_iteration=booster.best_iteration)
        best.append(int(booster.best_iteration or cfg_verifier.max_rounds))
    return {"oof": oof, "best_iterations": best, "auc": auc(y, oof),
            "precision_at_recall_0.9": precision_at_recall(y, oof), "folds": folds}


def train(train_df, n_rounds, cfg_verifier, features=None):
    import lightgbm as lgb
    enc = Encoder(features).fit(train_df)
    d = lgb.Dataset(enc.transform(train_df), train_df["label"].astype(int).to_numpy(),
                    categorical_feature=enc.categorical)
    booster = lgb.train(params(cfg_verifier), d, num_boost_round=int(n_rounds))
    return booster, enc


def predict(booster, enc, df):
    return booster.predict(enc.transform(df))


def apply(candidates_df, probs, expected_by_clip, heard_by_clip, tau):
    """New prediction per clip: keep candidates with P(real) >= tau, revert the rest."""
    keep = {}
    for (clip, pos), p in zip(candidates_df[["clip", "pos"]].itertuples(index=False), probs):
        keep.setdefault(clip, {})[int(pos)] = bool(p >= tau)
    return {clip: revert(expected_by_clip[clip], heard, keep.get(clip, {}))
            for clip, heard in heard_by_clip.items()}


def sweep_tau(candidates_df, probs, truth_df, heard_by_clip, taus):
    """Exact F1 per tau from per-clip counts (parity-tested against the official scorer)."""
    expected = dict(zip(truth_df["ID"], truth_df["Reference_phn"].str.split()))
    rows = []
    for tau in taus:
        new = apply(candidates_df, probs, expected, heard_by_clip, tau)
        pred = pd.DataFrame({"ID": list(new), "Prediction": [" ".join(v) for v in new.values()]})
        m = scorer.summed_metrics(scorer.per_clip_counts(truth_df, pred))
        rows.append({"tau": tau, **m})
    return pd.DataFrame(rows)


def robust_tau(sweep: pd.DataFrame, band=0.005):
    """Middle of the tau range whose F1 is within `band` of the best (SPEC section 5)."""
    best = sweep["f1"].max()
    near = sweep[sweep["f1"] >= best - band]["tau"].to_numpy()
    return float(near[len(near) // 2])
