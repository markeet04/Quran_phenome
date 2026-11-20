"""All paths, dataset names, columns, hyperparameters and seeds in one place.

Values mirror the existing pipeline (NB1, NB2, NB5, NB8) unless SPEC.md says otherwise.
"""
from dataclasses import dataclass, field

# Official scorer: github.com/Iqra-Eval/interspeech_IqraEval, folder mdd_eval, pinned.
SCORER_REPO = "https://github.com/Iqra-Eval/interspeech_IqraEval.git"
SCORER_COMMIT = "12ebde7829ad6a97a0f131457c2efb7fc095b257"

# NB5's drop-set, applied after CTC collapse.
DROP_TOKENS = frozenset({"<sil>", "sil", "<pad>", "<unk>"})

SAMPLING_RATE = 16_000


@dataclass
class Datasets:
    train: str = "IqraEval/Iqra_train"
    tts: str = "IqraEval/Iqra_TTS"
    extra: str = "IqraEval/Iqra_Extra_IS26"
    test_audio: str = "IqraEval/QuranMB.v2"
    test_truth: str = "IqraEval/IqraEval_Test_GT"


@dataclass
class Columns:
    canonical: str = "phoneme_ref"        # Iqra_train, Iqra_TTS, Iqra_Extra_IS26
    said_errors: str = "phoneme_mis"      # what was actually said (TTS, Extra)
    test_canonical: str = "Reference_phn"
    test_annotation: str = "Annotation_phn"


def find_cache_root(default="/kaggle/input/datasets/qasimmaajid/iqra-cache/iqra_prep",
                    search_root="/kaggle/input"):
    """The attached iqra-cache folder (the one holding phoneme_vocab.json).

    Uses the path NB2/NB8 used if it exists; otherwise searches /kaggle/input, since
    Kaggle has mounted datasets under different paths over time.
    """
    import glob
    import os
    if os.path.exists(os.path.join(default, "phoneme_vocab.json")):
        return default
    hits = sorted(glob.glob(os.path.join(search_root, "**", "phoneme_vocab.json"), recursive=True))
    hits = [os.path.dirname(h) for h in hits
            if os.path.isdir(os.path.join(os.path.dirname(h), "iqra_train_arrow"))]
    if not hits:
        raise FileNotFoundError("iqra-cache not found under /kaggle/input: attach the "
                                "'iqra-cache' dataset (Add Input -> Your Datasets)")
    return hits[0]


def find_input(marker, search_root="/kaggle/input", what=None):
    """Directory of the first file matching `marker` (a glob relative to any depth under
    `search_root`), e.g. find_input("splits.json") or find_input("checkpoint-5000/config.json").
    Raises with a clear message naming the dataset to attach."""
    import glob
    import os
    hits = sorted(glob.glob(os.path.join(search_root, "**", marker), recursive=True))
    if not hits:
        raise FileNotFoundError(f"{marker} not found under {search_root}: attach "
                                f"{what or 'the dataset that contains it'}")
    return os.path.dirname(hits[0])


@dataclass
class KagglePaths:
    cache_root: str = "/kaggle/input/datasets/qasimmaajid/iqra-cache/iqra_prep"
    stage1_ckpt: str = ("/kaggle/input/datasets/qasimmaajid/quran-phoneme-model/"
                        "w2vbert-quran-phoneme/checkpoint-5000")
    nb8_final: str = ("/kaggle/input/datasets/qasimmaajid/model-3/"
                      "w2vbert-quran-phoneme-real/final")
    working: str = "/kaggle/working"

    @property
    def train_arrow(self) -> str:
        return f"{self.cache_root}/iqra_train_arrow"

    @property
    def tts_arrow(self) -> str:
        return f"{self.cache_root}/iqra_tts_arrow"

    @property
    def vocab(self) -> str:
        return f"{self.cache_root}/phoneme_vocab.json"


@dataclass
class Splits:
    seed: int = 13
    extra_dev_frac: float = 0.20
    iqra_dev_slice_hours: float = 1.0


@dataclass
class Stage2:
    """NB8 settings; recipe v2 toggles default off (BASE)."""
    learning_rate: float = 5e-5
    lr_scheduler: str = "cosine"
    warmup_steps: int = 200
    batch_size: int = 8
    grad_accum: int = 2
    max_steps: int = 8000
    extra_repeat: int = 20
    eval_every: int = 500
    keep_best: int = 3
    fusion: bool = False
    consistency_lambda: float = 0.0
    sample_weights: bool = False
    spec_augment: bool = False
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_feature_prob: float = 0.0
    seed: int = 13


@dataclass
class Verifier:
    num_leaves: int = 31
    learning_rate: float = 0.05
    max_rounds: int = 1000
    early_stopping: int = 50
    min_data_in_leaf: int = 20
    cv_folds: int = 5
    tau_grid: tuple = tuple(round(0.05 * i, 2) for i in range(20))   # 0.00 .. 0.95
    robust_tau_band: float = 0.005
    seed: int = 13


@dataclass
class Bootstrap:
    n_resamples: int = 1000
    interval: float = 0.90
    seed: int = 13


@dataclass
class Config:
    datasets: Datasets = field(default_factory=Datasets)
    columns: Columns = field(default_factory=Columns)
    paths: KagglePaths = field(default_factory=KagglePaths)
    splits: Splits = field(default_factory=Splits)
    stage2: Stage2 = field(default_factory=Stage2)
    verifier: Verifier = field(default_factory=Verifier)
    bootstrap: Bootstrap = field(default_factory=Bootstrap)
