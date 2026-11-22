"""Deletion rule and the base-rate projection used to pick the T8 headline (SPEC decision 37).

The verifier and its tau were tuned on Extra-dev, where ~21% of flags are real; on the test
set the baseline's flags are ~63% real (row 0: P 0.634, R 0.703). An edit that removes a
fixed share of real errors (TR) and of false alarms (FR) changes F1 very differently at the
two base rates, so each candidate system is projected to the test baseline by applying its
dev keep fractions. A heuristic: it assumes the keep fractions transfer.
"""
from .candidates import extract_candidates, revert

TEST_BASELINE = {"precision": 0.634, "recall": 0.703}      # row 0 on test (SPEC section 1)


def deletion_rule(expected, heard):
    """Revert every D candidate (output the expected phoneme); keep every S and insertion."""
    return revert(expected, heard, {c.pos: False for c in extract_candidates(expected, heard) if c.op == "D"})


def keep_fractions(counts_plain, counts_new):
    """Share of the plain decode's TR and FR that survive the edit (summed over clips)."""
    tr0, fr0 = int(counts_plain["TR"].sum()), int(counts_plain["FR"].sum())
    tr1, fr1 = int(counts_new["TR"].sum()), int(counts_new["FR"].sum())
    return {"keep_tr": tr1 / tr0, "keep_fr": fr1 / fr0}


def projected(keep_tr, keep_fr, base_precision=TEST_BASELINE["precision"], base_recall=TEST_BASELINE["recall"]):
    """P, R, F1 after keeping `keep_tr` of the baseline's TR and `keep_fr` of its FR.
    The number of real errors (TR + FA) is fixed, so recall scales with keep_tr."""
    tr = base_recall * keep_tr
    fr = base_recall * (1 - base_precision) / base_precision * keep_fr
    p = tr / (tr + fr) if tr + fr else float("nan")
    r = tr
    f1 = 2 * p * r / (p + r) if tr else 0.0
    return {"precision": p, "recall": r, "f1": f1}
