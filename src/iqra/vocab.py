"""Phoneme vocabulary from NB1's phoneme_vocab.json (<pad>=0 is the CTC blank, <unk>=1)."""
import hashlib
import json
from dataclasses import dataclass

from .config import DROP_TOKENS


@dataclass(frozen=True)
class Vocab:
    tok2id: dict

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @property
    def id2tok(self) -> list:
        """Index -> token; ids missing from the json decode as <unk>, like NB5's id2tok.get."""
        out = ["<unk>"] * (max(self.tok2id.values()) + 1)
        for tok, i in self.tok2id.items():
            out[i] = tok
        return out

    @property
    def blank_id(self) -> int:
        return self.tok2id["<pad>"]

    @property
    def drop(self) -> frozenset:
        return DROP_TOKENS

    def phonemes(self) -> set:
        """Real phoneme symbols (everything the decoder can output)."""
        return {t for t in self.tok2id if t not in DROP_TOKENS}

    @property
    def hash(self) -> str:
        """Stable fingerprint, stored in posterior caches and checked when ensembling."""
        blob = json.dumps(sorted(self.tok2id.items()), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
