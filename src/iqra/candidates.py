"""Candidate flags, their labels, and the revert rule (SPEC section 5).

The verifier scores only S and D positions in the official alignment of expected vs heard:
  S: expected index j was heard as something else (heard index h)
  D: expected index j was not heard
Insertions (heard phonemes with no expected counterpart) are never candidates and are
never removed. In the official scorer a model insertion is never a false rejection; at a
gap where the reciter also inserted it is a true rejection (ins_ins / ins_ins1). So
removing one can only lose TR. remove_insertions() exists only for the dev check that
confirms this on Extra-dev.
Features are added in T3 (verifier); this module only fixes keys, labels and edits.
"""
from dataclasses import dataclass

import numpy as np

from .align import align_ops


@dataclass(frozen=True)
class Candidate:
    op: str          # "S" or "D"
    pos: int         # position in the alignment
    j: int           # expected index
    h: int           # heard index (S); -1 for D
    expected: str    # expected phoneme
    heard: str       # heard phoneme ("" for D)


def alignment(expected, heard):
    """Official alignment of expected vs heard with expected/heard indices per position.

    Returns a list of (op, j, h, expected_tok, heard_tok). For D, h = -1; for I, j is the
    gap index (the insertion sits before expected index j).
    """
    a1, a2, ops = align_ops(list(expected), list(heard))
    rows, j, h = [], 0, 0
    for e, d, op in zip(a1, a2, ops):
        if op in ("C", "S"):
            rows.append((op, j, h, e, d)); j += 1; h += 1
        elif op == "D":
            rows.append((op, j, -1, e, "")); j += 1
        else:  # "I"
            rows.append((op, j, h, "", d)); h += 1
    assert j == len(expected) and h == len(heard)
    return rows


def extract_candidates(expected, heard):
    """S and D positions only."""
    return [Candidate(op, pos, j, h, e, d)
            for pos, (op, j, h, e, d) in enumerate(alignment(expected, heard))
            if op in ("S", "D")]


def label_candidates(expected, said, candidates):
    """1 if the human also deviates at expected index j (substituted or deleted it), else 0.

    `said` is what was actually spoken (phoneme_mis / Annotation_phn). A kept flag where
    the human deviated is a true rejection; where the human was correct, a false rejection.
    """
    deviated = {j for op, j, _, _, _ in alignment(expected, said) if op in ("S", "D")}
    return [int(c.j in deviated) for c in candidates]


def revert(expected, heard, keep):
    """Rebuild the prediction after the verifier: reject = revert, never drop.

    `keep` maps alignment position -> bool for S/D candidates (missing = keep).
      C: heard phoneme
      S: keep -> heard phoneme,  reject -> expected phoneme
      D: keep -> nothing,        reject -> expected phoneme
      I: always the heard phoneme (insertions are never removed)
    """
    out = []
    for pos, (op, _, _, e, d) in enumerate(alignment(expected, heard)):
        kept = keep.get(pos, True)
        if op in ("C", "I"):
            out.append(d)
        elif op == "S":
            out.append(d if kept else e)
        elif not kept:  # "D"
            out.append(e)
    return out


def remove_insertions(expected, heard):
    """The prediction with every insertion dropped. Dev alignment-shift check only."""
    return [d for op, _, _, _, d in alignment(expected, heard) if op in ("C", "S")]


def official_label_check(expected, said, heard):
    """What the official counting implies about each S/D candidate (label-rule audit).

    For every candidate, revert only that flag and recount with scorer.clip_counts:
    TR drops (FR unchanged) -> the flag was a real error (implied 1); FR drops (TR
    unchanged) -> a false alarm (implied 0); anything else -> ambiguous (reverting it
    shifted the alignment). Returns [(candidate, label_rule, implied or None)].
    """
    from .scorer import clip_counts
    cands = extract_candidates(expected, heard)
    labels = label_candidates(expected, said, cands)
    base = clip_counts(expected, said, heard)
    out = []
    for c, y in zip(cands, labels):
        k = clip_counts(expected, said, revert(expected, heard, {c.pos: False}))
        d_tr, d_fr = base["TR"] - k["TR"], base["FR"] - k["FR"]
        implied = 1 if (d_tr > 0 and d_fr == 0) else 0 if (d_fr > 0 and d_tr == 0) else None
        out.append((c, y, implied))
    return out


# --------------------------------------------------------------------------- features
VOWELS = {"a", "i", "u", "A", "I", "U", "aa", "ii", "uu", "AA", "II", "UU"}
# emphatic vs plain counterpart (ص/س, ض/د, ط/ت, ظ/ذ, ظ/ز) and ض/ظ; other consonant swaps
# such as ق/ك fall under "consonant"
EMPHATIC_PAIRS = {frozenset(p) for p in (("S", "s"), ("D", "d"), ("T", "t"), ("Z", "*"), ("Z", "z"),
                                         ("D", "Z"))}


def _base(tok):
    """Consonant without gemination: 'll' -> 'l'; vowels unchanged."""
    return tok[0] if (len(tok) == 2 and tok[0] == tok[1] and tok not in VOWELS) else tok


def error_type(op, expected, heard):
    """Coarse error category for per-type analysis (reviewers' question)."""
    if op == "D":
        return "deletion"
    if "<" in (expected[:1], heard[:1]):
        return "hamza"
    if expected in VOWELS and heard in VOWELS:
        if expected[0] == heard[0] and len(expected) != len(heard):
            return "vowel_length"
        return "vowel_quality"
    if expected in VOWELS or heard in VOWELS:
        return "vowel_consonant"
    if _base(expected) == _base(heard):
        return "shadda"
    if frozenset((_base(expected), _base(heard))) in EMPHATIC_PAIRS:
        return "emphatic"
    return "consonant"


def candidate_features(clip_id, expected, entry, tok2id, id2tok, blank, said=None):
    """One feature row per S/D candidate of a cached clip (DESIGN.md, candidate table).

    expected: canonical tokens; entry: posterior-cache entry (logits, tokens, spans);
    said: what was actually spoken (labels), or None for unlabelled clips.
    Single-model features; ensemble votes are added in T6.
    """
    from .decode import log_softmax
    from .forced_align import token_spans

    logits = entry["logits"].numpy() if hasattr(entry["logits"], "numpy") else np.asarray(entry["logits"])
    lp = log_softmax(logits)
    probs = np.exp(lp)
    heard, heard_spans = entry["tokens"], entry["spans"]
    cands = extract_candidates(expected, heard)
    if not cands:
        return []
    exp_ids = [tok2id[t] for t in expected]
    fa_spans, fa_failed = token_spans(lp, exp_ids, blank)
    labels = label_candidates(expected, said, cands) if said is not None else [None] * len(cands)
    entropy = -(probs * lp).sum(axis=1)
    T, L = lp.shape[0], len(expected)
    pad = lambda k: expected[k] if 0 <= k < L else ("<bos>" if k < 0 else "<eos>")
    rows = []
    for c, y in zip(cands, labels):
        a, b = fa_spans[c.j]
        fr = slice(a, b + 1)
        e_id = tok2id[c.expected]
        row = {
            "clip": clip_id, "pos": c.pos, "op": c.op, "j": c.j,
            "tok_exp": c.expected, "tok_heard": c.heard if c.op == "S" else "<del>",
            "exp_prev1": pad(c.j - 1), "exp_prev2": pad(c.j - 2),
            "exp_next1": pad(c.j + 1), "exp_next2": pad(c.j + 2),
            "pair": f"{c.expected}>{c.heard if c.op == 'S' else '<del>'}",
            "err_type": error_type(c.op, c.expected, c.heard),
            # evidence from the expected text: forced-alignment span of expected token j
            "fa_failed": int(fa_failed), "span_len": b - a + 1,
            "exp_lp_mean": float(lp[fr, e_id].mean()), "exp_lp_max": float(lp[fr, e_id].max()),
            "gop": float((lp[fr, e_id] - lp[fr].max(axis=1)).mean()),
            "entropy_mean": float(entropy[fr].mean()), "blank_mean": float(probs[fr, blank].mean()),
            # utterance
            "n_flags_per_phone": len(cands) / max(L, 1), "n_frames": T, "n_expected": L,
            "rel_pos": c.j / max(L - 1, 1), "is_last": int(c.j == L - 1),
            "label": y,
        }
        if c.op == "S":
            h_id = tok2id[c.heard]
            row["heard_lp_mean"] = float(lp[fr, h_id].mean())
            row["heard_lp_max"] = float(lp[fr, h_id].max())
            row["margin_exp_minus_heard"] = row["exp_lp_mean"] - row["heard_lp_mean"]
            ha, hb = heard_spans[c.h]
            hfr = probs[ha:hb + 1]
            peak_t = ha + int(hfr[:, h_id].argmax())
            row["heard_peak_prob"] = float(probs[peak_t, h_id])
            order = np.argsort(probs[peak_t])[::-1]
            runner = int(order[1] if order[0] == h_id else order[0])
            row["runner_up"] = id2tok[runner] if runner < len(id2tok) else "<unk>"
            row["runner_up_prob"] = float(probs[peak_t, runner])
            row["runner_up_is_expected"] = int(runner == e_id)
            row["heard_span_len"] = hb - ha + 1
        else:
            for k in ("heard_lp_mean", "heard_lp_max", "margin_exp_minus_heard", "heard_peak_prob",
                      "runner_up_prob", "runner_up_is_expected", "heard_span_len"):
                row[k] = np.nan
            row["runner_up"] = "<none>"
        rows.append(row)
    return rows
