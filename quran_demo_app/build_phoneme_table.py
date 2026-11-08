"""
build_phoneme_table.py  —  OPTIONAL pre-build step / self-check.

Phonetises every verse in quran_text.SURAHS (whole-verse, in context) and
writes `expected_phonemes.json`. Prints each verse's per-word phonemes so you
can eyeball them, and checks every symbol is in the model vocab.

The app also phonetises on demand, so this is only needed to pre-build the
cache or to check the references.

Usage:
    python build_phoneme_table.py
"""
import os, sys, json
from quran_text import SURAHS, PhonemeLookup

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

lookup = PhonemeLookup()
all_syms = set()
for name, verses in SURAHS.items():
    print(f"\n=== {name}")
    for i, v in enumerate(verses, 1):
        pairs = lookup.verse(v)
        print(f"  {i}: " + "  |  ".join(" ".join(ph) for _, ph in pairs))
        all_syms.update(p for _, ph in pairs for p in ph)

lookup.dirty = True
lookup.save()
print(f"\nwrote {lookup.path}  ({len(lookup.cache)} verses)")

try:
    from app import resolve_model_dir
    vocab_path = os.path.join(resolve_model_dir() or "", "vocab.json")
    vocab = set(json.load(open(vocab_path, encoding="utf-8")).keys())
    print("symbols not in model vocab (should be empty):", sorted(all_syms - vocab))
except Exception as e:
    print("(skipped vocab check — model not found:", e, ")")
