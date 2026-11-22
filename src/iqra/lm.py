"""Phoneme n-gram LM and N-best rescoring (SPEC section 5, ladder row 5).

Ported from NB10 without changing the method: an interpolated order-5 n-gram with
add-k smoothing (k = 0.1), and N-best rescoring where the candidates are the greedy
path plus swaps to the runner-up phoneme at the most uncertain emission steps
(score = acoustic log-prob + alpha x LM log10-prob). Differences from NB10, all from
the spec: the corpus is training canonical text only (Iqra_train and Extra folds A+B
phoneme_ref) minus every line equal to an Extra-dev canonical sentence (decision 14),
never test text; it runs on the ensemble log-posteriors; alpha is chosen on Extra-dev
with NB10's recall guard (no setting may lower recall below the plain decode's).
"""
import itertools
import math
from collections import defaultdict

import numpy as np

from .config import DROP_TOKENS
from .decode import log_softmax


class NgramLM:
    """Interpolated n-gram, add-k smoothing; log10 P(token | history). As NB10."""

    def __init__(self, order=5, k=0.1):
        self.order, self.k = order, k
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]
        self.ctx_tot = [defaultdict(int) for _ in range(order)]
        self.vocab = set()
        self.V = 0

    def train(self, lines):
        for ln in lines:
            toks = ["<s>"] + ln.split() + ["</s>"]
            self.vocab.update(toks)
            for n in range(self.order):
                for i in range(n, len(toks)):
                    ctx = tuple(toks[i - n:i])
                    self.counts[n][ctx][toks[i]] += 1
                    self.ctx_tot[n][ctx] += 1
        self.V = len(self.vocab)
        return self

    def _p_order(self, n, ctx, w):
        c = self.counts[n][ctx].get(w, 0) if ctx in self.counts[n] else 0
        tot = self.ctx_tot[n].get(ctx, 0)
        return (c + self.k) / (tot + self.k * self.V) if (tot or self.k) else 0.0

    def logp(self, w, history):
        p = wsum = 0.0
        for n in range(self.order):
            if n > len(history):
                break
            ctx = tuple(history[len(history) - n:]) if n else tuple()
            weight = n + 1
            p += weight * self._p_order(n, ctx, w)
            wsum += weight
        p = p / wsum if wsum else 1e-10
        return math.log10(max(p, 1e-10))

    def score_seq(self, toks):
        s, hist = 0.0, []
        for w in list(toks) + ["</s>"]:
            s += self.logp(w, hist)
            hist.append(w)
        return s


def build_corpus(lines, exclude, vocab_symbols):
    """Clean canonical lines (vocabulary phonemes only) minus every line equal to one in
    `exclude` (Extra-dev canonical sentences, decision 14). Returns (corpus, n_removed)."""
    norm = lambda s: " ".join(str(s).split())
    excl = {norm(s) for s in exclude}
    out, removed = [], 0
    for s in lines:
        if norm(s) in excl:
            removed += 1
            continue
        toks = [t for t in str(s).split() if t in vocab_symbols]
        if toks:
            out.append(" ".join(toks))
    return out, removed


def nbest(log_probs, id2tok, blank_id, drop=DROP_TOKENS, max_variants=16, margin=0.45, max_swap_pos=6):
    """NB10's candidate list: [(acoustic log-prob, tokens)], the greedy path first.

    `log_probs` are frame log-posteriors (an ensemble's, here). At each collapsed CTC
    emission step keep the top-1 and top-2 symbols; steps whose top-2 is a real phoneme
    within `margin` nats are swap candidates; the most uncertain `max_swap_pos` of them
    are swapped singly and in pairs, up to `max_variants` variants.
    """
    lp = np.log(np.clip(np.exp(log_softmax(log_probs)), 1e-9, 1.0))      # as NB10
    sym = lambda i: id2tok[i] if 0 <= i < len(id2tok) else "<unk>"
    steps, prev = [], None
    for t in range(lp.shape[0]):
        # stable descending order: on an exact tie the lowest id comes first, as argmax picks
        # it in the greedy decode (NB10's argsort()[::-1] took the highest; found by the dry run)
        order = np.argsort(-lp[t], kind="stable")
        top1 = int(order[0])
        if top1 != prev and top1 != blank_id:
            top2 = int(order[1])
            steps.append((top1, float(lp[t, top1]), top2, float(lp[t, top2])))
        prev = top1
    base = [(i1, l1) for (i1, l1, _, _) in steps]
    cands = [(sum(l for _, l in base), [sym(i) for i, _ in base if sym(i) not in drop])]
    gaps = sorted((l1 - l2, idx) for idx, (_, l1, i2, l2) in enumerate(steps)
                  if sym(i2) not in drop and i2 != blank_id and (l1 - l2) < margin)
    swap_idx = [g[1] for g in gaps[:max_swap_pos]]
    combos = []
    for r in (1, 2):
        for combo in itertools.combinations(swap_idx, r):
            combos.append(combo)
            if len(combos) >= max_variants:
                break
        if len(combos) >= max_variants:
            break
    for combo in combos:
        seq = list(base)
        for j in combo:
            _, _, i2, l2 = steps[j]
            seq[j] = (i2, l2)
        cands.append((sum(l for _, l in seq), [sym(i) for i, _ in seq if sym(i) not in drop]))
    return cands


def rescore(cands_by_clip, lm, alpha):
    """{clip: tokens} choosing argmax of acoustic + alpha x LM per clip."""
    out = {}
    for cid, cands in cands_by_clip.items():
        best, best_s = None, -1e18
        for ac, toks in cands:
            s = ac + alpha * (lm.score_seq(toks) if toks else -1e9)
            if s > best_s:
                best_s, best = s, toks
        out[cid] = best or []
    return out


def choose_alpha(sweep_rows, base_recall, eps=0.0005):
    """NB10's decision rule: among settings with recall >= base recall - eps, the highest
    F1 (alpha 0 = the plain decode is always eligible). Returns the chosen row."""
    safe = [r for r in sweep_rows if r["recall"] >= base_recall - eps]
    return max(safe, key=lambda r: (r["f1"], -r["alpha"]))
