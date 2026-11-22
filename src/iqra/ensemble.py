"""T6: find the trained runs, plan the posterior caches, build ensembles (SPEC T6).

Runs are found by content, never by dataset or notebook name: every NB13 output holds
run_summary.json ("run": "A" / "B" / "BASE", the kept checkpoints) and run_config.json
(the run's Extra folds). The same run attached twice (a notebook output and a dataset
made from it) is accepted once; two different outputs claiming the same run are an error.

Cache plan (SPEC compute budget, priority 4): only what is used.
  run-A members: fold B (verifier training), Extra-dev, Iqra dev slice
  run-B members: fold A (verifier training), Extra-dev, Iqra dev slice
  BASE (its best checkpoint): Extra-dev only (ladder row 1 with a bootstrap interval)
A member is never cached on a fold it trained on (rule 6, out-of-fold only).
"""
import json
from dataclasses import dataclass
from pathlib import Path

from .decode import ensemble_log_posteriors, log_softmax
from .posteriors import logits_of, make_entry

EXPECTED_RUNS = ("A", "B", "BASE")
VERIFIER_FOLD = {"A": "fold_b", "B": "fold_a"}       # each run's out-of-fold Extra split


@dataclass(frozen=True)
class Member:
    run: str
    step: int
    f1: float          # official Extra-dev F1 logged in T5
    path: str          # local checkpoint folder

    @property
    def name(self):
        return f"{self.run}{self.step:05d}"


def find_runs(search_root="/kaggle/input", required=EXPECTED_RUNS):
    """{run name: {"dir", "extra_folds", "members": [Member, ...] best first}}.
    `search_root` may be one folder or a list of folders."""
    roots = [search_root] if isinstance(search_root, (str, Path)) else list(search_root)
    found = {}
    summaries = sorted({p for r in roots for p in Path(r).glob("**/run_summary.json")})
    for summ in summaries:
        d = summ.parent
        s = json.loads(summ.read_text())
        name = s["run"]
        if not s.get("kept"):
            continue                                   # a timing probe keeps nothing
        cfg_path = d / "run_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"{d}: run_summary.json without run_config.json")
        cfg = json.loads(cfg_path.read_text())["run"]   # NB13 writes {"run": RunConfig, "active", ...}
        members = []
        for f1, step, p in s["kept"]:
            local = d / Path(p).name
            if not (local / "config.json").exists():
                raise FileNotFoundError(f"run {name}: kept checkpoint {Path(p).name} missing under {d}")
            members.append(Member(name, int(step), float(f1), str(local)))
        members.sort(key=lambda m: (-m.f1, m.step))
        entry = {"dir": str(d), "extra_folds": tuple(cfg["extra_folds"]), "members": members}
        if name in found:
            same = [(m.step, m.f1) for m in found[name]["members"]] == [(m.step, m.f1) for m in members]
            if not same:
                raise AssertionError(f"two different outputs claim run {name}: {found[name]['dir']} and {d}")
            continue
        found[name] = entry
    missing = [r for r in required if r not in found]
    if missing:
        raise FileNotFoundError(f"runs not attached: {missing} (found {sorted(found)}). Attach the NB13 "
                                "outputs of BASE, A and B (notebook outputs or datasets made from them).")
    return found


def check_folds(runs):
    """The runs must have trained on the folds the design says."""
    expect = {"A": ("fold_a",), "B": ("fold_b",), "BASE": ("fold_a", "fold_b")}
    for r, folds in expect.items():
        if tuple(runs[r]["extra_folds"]) != folds:
            raise AssertionError(f"run {r} trained on {runs[r]['extra_folds']}, expected {folds}")


def cache_plan(runs):
    """[(member, [splits])]: A and B members (all kept), then BASE's best checkpoint."""
    check_folds(runs)
    plan = []
    for r in ("A", "B"):
        for m in runs[r]["members"]:
            plan.append((m, [VERIFIER_FOLD[r], "extra_dev", "iqra_dev_slice"]))
    plan.append((runs["BASE"]["members"][0], ["extra_dev"]))
    return plan


def assert_out_of_fold(plan, runs, splits):
    """Rule 6: no member is cached on Extra clips it trained on, except Extra-dev, which no
    run trained on (checked too)."""
    for m, split_names in plan:
        trained = {c for f in runs[m.run]["extra_folds"] for c in splits["extra"][f]}
        if trained & set(splits["extra"]["extra_dev"]):
            raise AssertionError(f"run {m.run} trained on Extra-dev clips")
        for s in split_names:
            if s in ("fold_a", "fold_b") and trained & set(splits["extra"][s]):
                raise AssertionError(f"{m.name} would produce candidates on {s}, which it trained on")


def cache_name(member_name, split):
    return f"post_{member_name}_{split}.pt"


# --------------------------------------------------------------------------- ensembles
def ensemble_entry(member_entries, id2tok, blank_id):
    """Mean of member probabilities (log-sum-exp minus log K), as a cache entry.

    The entry's "logits" are the ensemble log-posteriors (log_softmax of them is the
    identity), so every function that reads cache entries works on ensembles unchanged.
    """
    lps = [log_softmax(logits_of(e)) for e in member_entries]
    return make_entry(ensemble_log_posteriors(lps), id2tok, blank_id)


def ensemble_split(caches, member_names, id2tok, blank_id):
    """{clip: ensemble entry} over the clips every member has (must be identical sets)."""
    sets = [set(caches[n]) for n in member_names]
    if any(s != sets[0] for s in sets):
        raise AssertionError(f"members cover different clips: {[len(s) for s in sets]}")
    return {cid: ensemble_entry([caches[n][cid] for n in member_names], id2tok, blank_id)
            for cid in sorted(sets[0])}


def frames_agree(caches, member_names):
    """Every member gives the same T per clip (SPEC: assert before ensembling)."""
    bad = []
    for cid in caches[member_names[0]]:
        ts = {int(caches[n][cid]["logits"].shape[0]) for n in member_names}
        if len(ts) != 1:
            bad.append((cid, sorted(ts)))
    return bad


def clip_final_deletions(cands_df):
    """Diagnostic (decision 29): deletions of the clip's last expected phoneme, among
    false alarms and among real errors."""
    d = cands_df[cands_df["op"] == "D"]
    last = d[d["is_last"] == 1]
    fa = d[d["label"] == 0]
    return {"deletions": int(len(d)), "false_alarm_deletions": int(len(fa)),
            "clip_final_deletions": int(len(last)),
            "clip_final_among_false_alarms": int((last["label"] == 0).sum()),
            "clip_final_real": int((last["label"] == 1).sum()),
            "share_of_false_alarm_deletions": round(float((last["label"] == 0).sum()) / max(len(fa), 1), 4)}


def summarise_members(runs):
    return {r: [{"step": m.step, "f1_t5": m.f1} for m in runs[r]["members"]] for r in runs}
