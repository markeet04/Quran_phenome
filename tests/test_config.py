"""find_cache_root: default path first, else search the input folder."""
import pytest

from iqra.config import find_cache_root


def _make_cache(root):
    root.mkdir(parents=True)
    (root / "phoneme_vocab.json").write_text("{}")
    (root / "iqra_train_arrow").mkdir()
    return root


def test_default_path_wins(tmp_path):
    default = _make_cache(tmp_path / "datasets" / "me" / "iqra-cache" / "iqra_prep")
    _make_cache(tmp_path / "other" / "iqra_prep")
    assert find_cache_root(str(default), str(tmp_path)) == str(default)


def test_finds_cache_mounted_elsewhere(tmp_path):
    mounted = _make_cache(tmp_path / "iqra-cache" / "iqra_prep")
    (tmp_path / "decoy").mkdir()
    (tmp_path / "decoy" / "phoneme_vocab.json").write_text("{}")      # no iqra_train_arrow
    assert find_cache_root(str(tmp_path / "missing"), str(tmp_path)) == str(mounted)


def test_clear_error_when_not_attached(tmp_path):
    with pytest.raises(FileNotFoundError, match="attach the 'iqra-cache' dataset"):
        find_cache_root(str(tmp_path / "missing"), str(tmp_path))
