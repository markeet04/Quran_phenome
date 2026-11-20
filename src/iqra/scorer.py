"""Official IqraEval scoring, plus per-clip counts for the paired bootstrap.

official_score() runs the two unmodified official scripts as subprocesses, each call
in a fresh temp directory (ins_del_cor_sub_analysis.py joins the align folder with
no separator, so the folder is always passed with a trailing '/'). Reported numbers
come only from here (SPEC rule 4).

per_clip_counts() re-implements the counting loops of ins_del_cor_sub_analysis.py
for one clip at a time, on the official Align/get_op. Its sums must equal the
official totals exactly (tests/test_scorer.py, count parity).

Behaviours of the official scripts worth knowing (read from the pinned source):
  - only IDs present in BOTH files are scored; we require identical ID sets instead.
  - an empty prediction is read by pandas as NaN and crashes .split(); we reject it.
  - lines are parsed with substring tests ("ref" in line, "hyp" in line, " op "),
    so an ID containing those breaks the official parse; we reject such IDs.
  - model insertions relative to the reciter are never counted (no FR for them).
  - if TR + FR == 0 the official script divides by zero.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .align import EPS, align_ops, mdd_eval_dir, scorer_env

COUNT_KEYS = ("cor_cor", "cor_nocor", "sub_sub", "sub_sub1", "sub_nosub",
              "ins_ins", "ins_ins1", "ins_noins", "del_del", "del_del1", "del_nodel")


class ScorerInputError(ValueError):
    """Inputs the official scripts would silently mis-score or crash on."""


class ScorerUndefined(RuntimeError):
    """The official script divided by zero (TR + FR == 0, or no reference errors)."""


# --------------------------------------------------------------------------- inputs
def _validate(truth_df: pd.DataFrame, pred_df: pd.DataFrame):
    for col in ("ID", "Reference_phn", "Annotation_phn"):
        if col not in truth_df.columns:
            raise ScorerInputError(f"truth is missing column {col}")
    for col in ("ID", "Prediction"):
        if col not in pred_df.columns:
            raise ScorerInputError(f"predictions are missing column {col}")
    t_ids, p_ids = list(truth_df["ID"]), list(pred_df["ID"])
    if len(set(t_ids)) != len(t_ids) or len(set(p_ids)) != len(p_ids):
        raise ScorerInputError("duplicate IDs")
    if set(t_ids) != set(p_ids):
        missing = sorted(set(t_ids) - set(p_ids))[:5]
        extra = sorted(set(p_ids) - set(t_ids))[:5]
        raise ScorerInputError(f"ID sets differ (missing preds e.g. {missing}, extra e.g. {extra})")
    for i in t_ids:
        s = str(i)
        if re.search(r"\s", s) or "ref" in s or "hyp" in s:
            raise ScorerInputError(f"ID {s!r} would break the official line parser")
    for col, df in (("Reference_phn", truth_df), ("Annotation_phn", truth_df),
                    ("Prediction", pred_df)):
        bad = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
        if len(bad):
            raise ScorerInputError(f"{len(bad)} empty {col} values (e.g. ID {bad['ID'].iloc[0]})")


# --------------------------------------------------------------------------- official
_PATTERNS = {
    "recall": r"Recall:\s*([0-9.]+)",
    "precision": r"Precision:\s*([0-9.]+)",
    "f1": r"f1-score:\s*([0-9.]+)",
    "TA": r"True Acception:\s*[0-9.]+\s*\|\s*(\d+)",
    "FR": r"False Rejection:\s*[0-9.]+\s*\|\s*(\d+)",
    "FA": r"False Acceptance:\s*[0-9.]+\s*\|\s*(\d+)",
    "CD": r"Correct Diagnosis:\s*[0-9.]+\s*\|\s*(\d+)",
    "ED": r"Error Diagnosis:\s*[0-9.]+\s*\|\s*(\d+)",
}


def run_official(truth_csv, pred_csv, workdir):
    """Run align_data.py then ins_del_cor_sub_analysis.py exactly as NB5 does.

    Returns (align stdout, analysis stdout, align folder). `workdir` must be fresh.
    """
    folder = mdd_eval_dir()
    aligned = Path(workdir) / "aligned"
    r1 = subprocess.run([sys.executable, str(folder / "align_data.py"),
                         "--truth-file", str(truth_csv), "--pred-file", str(pred_csv),
                         "--output-dir", str(aligned)],
                        capture_output=True, text=True, env=scorer_env())
    if r1.returncode != 0:
        raise RuntimeError(f"align_data.py failed:\n{r1.stderr[-2000:]}")
    r2 = subprocess.run([sys.executable, str(folder / "ins_del_cor_sub_analysis.py"),
                         "--align-folder", str(aligned) + "/"],
                        capture_output=True, text=True, env=scorer_env())
    if r2.returncode != 0:
        if "ZeroDivisionError" in r2.stderr:
            raise ScorerUndefined("official scorer divided by zero (TR+FR == 0 or no errors)")
        raise RuntimeError(f"ins_del_cor_sub_analysis.py failed:\n{r2.stderr[-2000:]}")
    return r1.stdout, r2.stdout, aligned


def parse_official(align_stdout: str, analysis_stdout: str) -> dict:
    out = {}
    for key, pat in _PATTERNS.items():
        m = re.search(pat, analysis_stdout)
        if not m:
            raise RuntimeError(f"could not find {key} in scorer output:\n{analysis_stdout}")
        out[key] = float(m.group(1)) if key in ("recall", "precision", "f1") else int(m.group(1))
    out["TR"] = out["CD"] + out["ED"]
    m = re.search(r"Accuracy:\s*(-?[0-9.]+)", align_stdout)     # negative when PER > 1
    out["per"] = round(1.0 - float(m.group(1)), 4) if m else float("nan")
    return out


def official_score(truth_df: pd.DataFrame, pred_df: pd.DataFrame, keep_dir=None) -> dict:
    """Score predictions with the official scripts in a fresh temp dir.

    truth_df: ID, Reference_phn, Annotation_phn.  pred_df: ID, Prediction.
    Returns precision/recall/f1 (official 4-dp values), TA/FR/FA/TR/CD/ED counts, per,
    and raw stdout. If keep_dir is given, inputs and alignment files are copied there.
    """
    _validate(truth_df, pred_df)
    with tempfile.TemporaryDirectory(prefix="iqra_score_") as tmp:
        tmp = Path(tmp)
        truth_df[["ID", "Reference_phn", "Annotation_phn"]].to_csv(tmp / "truth.csv", index=False)
        pred_df[["ID", "Prediction"]].to_csv(tmp / "pred.csv", index=False)
        a_out, s_out, aligned = run_official(tmp / "truth.csv", tmp / "pred.csv", tmp)
        result = parse_official(a_out, s_out)
        result["stdout"] = a_out + s_out
        if keep_dir is not None:
            keep = Path(keep_dir)
            keep.mkdir(parents=True, exist_ok=True)
            for p in [tmp / "truth.csv", tmp / "pred.csv", *aligned.iterdir()]:
                (keep / p.name).write_bytes(p.read_bytes())
            (keep / "scorer_stdout.txt").write_text(result["stdout"], encoding="utf-8")
    return result


# --------------------------------------------------------------------------- per clip
def clip_counts(canon, trans, pred) -> dict:
    """Official counting for one clip (mirrors ins_del_cor_sub_analysis.py lines 154-246).

    canon = Reference_phn tokens, trans = Annotation_phn tokens, pred = Prediction tokens.
    """
    c = dict.fromkeys(COUNT_KEYS, 0)
    rh_ref, rh_hum, rh_op = align_ops(canon, trans)     # ref_human_detail: arr[0], arr[1], arr[2]
    ho_hum, ho_our, ho_op = align_ops(trans, pred)      # human_our_detail: arr[3], arr[4], arr[5]
    ro_ref, ro_our, ro_op = align_ops(canon, pred)      # ref_our_detail:   arr[6], arr[7], arr[8]

    # deletion detection: walk canonical tokens in both canonical alignments
    flag = 0
    for i in range(len(rh_ref)):
        if rh_ref[i] == EPS:
            continue
        while flag < len(ro_ref) and ro_ref[flag] == EPS:
            flag += 1
        if flag < len(ro_ref) and rh_ref[i] == ro_ref[flag] and rh_ref[i] != EPS:
            if rh_op[i] == "D" and ro_op[flag] == "D":
                c["del_del"] += 1
            elif rh_op[i] == "D" and ro_op[flag] != "D" and ro_op[flag] != "C":
                c["del_del1"] += 1
            elif rh_op[i] == "D" and ro_op[flag] != "D" and ro_op[flag] == "C":
                c["del_nodel"] += 1
            flag += 1

    # correct / substitution / insertion detection: walk the reciter's tokens
    flag = 0
    for i in range(len(rh_hum)):
        if rh_hum[i] == EPS:
            continue
        while ho_hum[flag] == EPS:       # unbounded in the official code too
            flag += 1
        if rh_hum[i] == ho_hum[flag] and rh_hum[i] != EPS:
            if rh_op[i] == "C" and ho_op[flag] == "C":
                c["cor_cor"] += 1
            elif rh_op[i] == "C" and ho_op[flag] != "C":
                c["cor_nocor"] += 1

            if rh_op[i] == "S" and ho_op[flag] == "C":
                c["sub_sub"] += 1
            elif rh_op[i] == "S" and ho_op[flag] != "C" and rh_ref[i] != ho_our[flag]:
                c["sub_sub1"] += 1
            elif rh_op[i] == "S" and ho_op[flag] != "C" and rh_ref[i] == ho_our[flag]:
                c["sub_nosub"] += 1

            if rh_op[i] == "I" and ho_op[flag] == "C":
                c["ins_ins"] += 1
            elif rh_op[i] == "I" and ho_op[flag] != "C" and ho_op[flag] != "D":
                c["ins_ins1"] += 1
            elif rh_op[i] == "I" and ho_op[flag] != "C" and ho_op[flag] == "D":
                c["ins_noins"] += 1
            flag += 1
    return _derive(c)


def _derive(c: dict) -> dict:
    c = dict(c)
    c["TR"] = c["sub_sub"] + c["sub_sub1"] + c["del_del1"] + c["del_del"] + c["ins_ins1"] + c["ins_ins"]
    c["FR"] = c["cor_nocor"]
    c["FA"] = c["sub_nosub"] + c["ins_noins"] + c["del_nodel"]
    c["TA"] = c["cor_cor"]
    c["CD"] = c["sub_sub"] + c["ins_ins"] + c["del_del"]
    c["ED"] = c["sub_sub1"] + c["ins_ins1"] + c["del_del1"]
    return c


def per_clip_counts(truth_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """One row per clip with every official count. Same validation as official_score."""
    _validate(truth_df, pred_df)
    df = truth_df[["ID", "Reference_phn", "Annotation_phn"]].merge(
        pred_df[["ID", "Prediction"]], on="ID")
    rows = []
    for _, r in df.iterrows():
        cnt = clip_counts(str(r["Reference_phn"]).split(), str(r["Annotation_phn"]).split(),
                          str(r["Prediction"]).split())
        cnt["ID"] = r["ID"]
        rows.append(cnt)
    return pd.DataFrame(rows).set_index("ID")


def metrics_from_counts(TR, FR, FA) -> dict:
    """P, R, F1 from summed counts, as the official script computes them (unrounded).

    0/0 gives NaN instead of the official ZeroDivisionError; F1 is 0 when TR == 0.
    """
    precision = TR / (TR + FR) if (TR + FR) else float("nan")
    recall = TR / (TR + FA) if (TR + FA) else float("nan")
    if TR == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def summed_metrics(counts: pd.DataFrame) -> dict:
    s = counts[["TR", "FR", "FA"]].sum()
    return metrics_from_counts(int(s["TR"]), int(s["FR"]), int(s["FA"]))


# --------------------------------------------------------------------------- bootstrap
def paired_bootstrap_gain(counts_base: pd.DataFrame, counts_new: pd.DataFrame, clusters,
                          n_resamples=1000, interval=0.90, seed=13) -> dict:
    """F1(new) - F1(base) with a paired cluster-bootstrap interval (SPEC section 5).

    Both frames come from per_clip_counts on the same clips. `clusters` maps each clip
    ID to its cluster: on Extra-dev, the canonical sentence (phoneme_ref), because every
    sentence is read ~10 times and recordings of one sentence are not independent.
    Each resample draws whole clusters with replacement (every clip of a drawn cluster
    comes along), sums TR/FR/FA per system and recomputes P, R, F1, so the interval
    reflects how many independent sentences the dev set really has. Pass
    clusters={id: id} for a plain clip bootstrap.
    """
    ids = list(counts_base.index)
    if set(ids) != set(counts_new.index):
        raise ValueError("paired bootstrap needs the same clips in both systems")
    missing = [i for i in ids if i not in clusters]
    if missing:
        raise ValueError(f"{len(missing)} clips have no cluster (e.g. {missing[:3]})")
    keys = pd.Series([clusters[i] for i in ids], index=ids)
    cols = ["TR", "FR", "FA"]
    b = counts_base.loc[ids, cols].groupby(keys.values).sum().to_numpy(dtype=np.int64)
    n = counts_new.loc[ids, cols].groupby(keys.values).sum().to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    gains = np.empty(n_resamples)
    n_clusters = len(b)
    for k in range(n_resamples):
        idx = rng.integers(0, n_clusters, n_clusters)
        fb = metrics_from_counts(*b[idx].sum(axis=0))["f1"]
        fn = metrics_from_counts(*n[idx].sum(axis=0))["f1"]
        gains[k] = fn - fb
    lo, hi = np.quantile(gains, [(1 - interval) / 2, 1 - (1 - interval) / 2])
    point = (metrics_from_counts(*n.sum(axis=0))["f1"] - metrics_from_counts(*b.sum(axis=0))["f1"])
    return {"gain": point, "lo": float(lo), "hi": float(hi), "n_clusters": n_clusters,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def bootstrap_f1_interval(counts: pd.DataFrame, clusters, n_resamples=1000, interval=0.90,
                          seed=13) -> dict:
    """F1 of one system with a cluster-bootstrap interval (for ladder rows)."""
    ids = list(counts.index)
    keys = [clusters[i] for i in ids]
    c = counts.loc[ids, ["TR", "FR", "FA"]].groupby(keys).sum().to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    f1s = np.array([metrics_from_counts(*c[rng.integers(0, len(c), len(c))].sum(axis=0))["f1"]
                    for _ in range(n_resamples)])
    lo, hi = np.quantile(f1s, [(1 - interval) / 2, 1 - (1 - interval) / 2])
    return {"f1": metrics_from_counts(*c.sum(axis=0))["f1"], "lo": float(lo), "hi": float(hi),
            "n_clusters": len(c)}
