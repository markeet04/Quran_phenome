"""
Shared Qur'an text + expected-phoneme helpers (used by app.py and
build_phoneme_table.py).

Expected phonemes are computed by running the MSA phonetiser over each WHOLE
verse — the same way the training labels (phoneme_ref) were made — so that
context rules apply: hamzat-wasl / "al-" kept at the start of a verse and elided
mid-verse, sun-letter assimilation, etc. The verse result is then split back
into words using the phonetiser's word-boundary output.
"""
import os, re, io, sys, json, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
PHONETISER_DIR = os.path.join(HERE, "MSA_phonetiser")
TABLE_PATH = os.path.join(HERE, "expected_phonemes.json")

# ---- the surahs offered in the dropdown (diacritised, one string per verse) --
SURAHS = {
    "1  Al-Fatiha (الفاتحة)": [
        "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        "الرَّحْمَٰنِ الرَّحِيمِ",
        "مَالِكِ يَوْمِ الدِّينِ",
        "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
        "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
        "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ",
    ],
    "103  Al-Asr (العصر)": [
        "وَالْعَصْرِ",
        "إِنَّ الْإِنسَانَ لَفِي خُسْرٍ",
        "إِلَّا الَّذِينَ آمَنُوا وَعَمِلُوا الصَّالِحَاتِ وَتَوَاصَوْا بِالْحَقِّ وَتَوَاصَوْا بِالصَّبْرِ",
    ],
    "108  Al-Kawthar (الكوثر)": [
        "إِنَّا أَعْطَيْنَاكَ الْكَوْثَرَ",
        "فَصَلِّ لِرَبِّكَ وَانْحَرْ",
        "إِنَّ شَانِئَكَ هُوَ الْأَبْتَرُ",
    ],
    "112  Al-Ikhlas (الإخلاص)": [
        "قُلْ هُوَ اللَّهُ أَحَدٌ",
        "اللَّهُ الصَّمَدُ",
        "لَمْ يَلِدْ وَلَمْ يُولَدْ",
        "وَلَمْ يَكُن لَّهُ كُفُوًا أَحَدٌ",
    ],
    "113  Al-Falaq (الفلق)": [
        "قُلْ أَعُوذُ بِرَبِّ الْفَلَقِ",
        "مِن شَرِّ مَا خَلَقَ",
        "وَمِن شَرِّ غَاسِقٍ إِذَا وَقَبَ",
        "وَمِن شَرِّ النَّفَّاثَاتِ فِي الْعُقَدِ",
        "وَمِن شَرِّ حَاسِدٍ إِذَا حَسَدَ",
    ],
    "114  An-Nas (الناس)": [
        "قُلْ أَعُوذُ بِرَبِّ النَّاسِ",
        "مَلِكِ النَّاسِ",
        "إِلَٰهِ النَّاسِ",
        "مِن شَرِّ الْوَسْوَاسِ الْخَنَّاسِ",
        "الَّذِي يُوَسْوِسُ فِي صُدُورِ النَّاسِ",
        "مِنَ الْجِنَّةِ وَالنَّاسِ",
    ],
}


# ---- text clean-up before phonetising ----------------------------------------
_SHADDA = "ّ"
_HARAKAT = "ًٌٍَُِْ"   # tanween, short vowels, sukun

def clean_text(text):
    """The phonetiser expects shadda DIRECTLY after its letter ('r~a'); most
    Qur'an text stores the vowel first ('ra~'), which makes it double the vowel
    (r aa) instead of the consonant (rr a). Also drop Uthmani-only marks."""
    t = text.replace("ٱ", "ا")                 # alif wasla -> alif
    t = re.sub("[ۖ-ۭ]", "", t)                 # Qur'anic annotation marks
    t = re.sub(f"([{_HARAKAT}]+){_SHADDA}", _SHADDA + r"\1", t)
    return " ".join(t.split())


# ---- normalisation of phonetiser output to the model alphabet ---------------
# Same as NB3 except '<' (hamza) is KEPT: it is a real phoneme in the model
# vocab, and with whole-verse context it only appears where it is pronounced.
def _norm_one(p):
    if p in ("sil", "<sil>"):
        return None
    m = re.fullmatch(r"([A-Za-z]+)0([A-Za-z]+)0", p)   # i0i0 -> ii
    if m and m.group(1) == m.group(2):
        base = m.group(1)
        return base if len(base) > 1 else base + base
    m = re.fullmatch(r"([A-Za-z]+)[0-9]", p)           # i0 -> i, ii0 -> ii
    if m:
        return m.group(1)
    return p

def normalise(seq):
    out = []
    for p in seq:
        q = _norm_one(p)
        if q:
            out.append(q)
    return out


# ---- phonetiser (optional at runtime; only needed for words not cached) -----
_phonetise = None
_phonetiser_error = None

def _load_phonetiser():
    global _phonetise, _phonetiser_error
    if _phonetise or _phonetiser_error:
        return _phonetise
    try:
        if PHONETISER_DIR not in sys.path:
            sys.path.insert(0, PHONETISER_DIR)
        from phonetiser import phonetise_Arabic
        _phonetise = phonetise_Arabic.phonetise
    except Exception as e:
        _phonetiser_error = (f"MSA_phonetiser not available ({type(e).__name__}: {e}). "
                             f"Clone it into {PHONETISER_DIR}")
    return _phonetise

def phonetiser_available():
    return _load_phonetiser() is not None

def verse_word_phonemes(verse):
    """Phonetise one whole verse in context -> [(word, [phonemes]), ...].

    phonetise() returns two views of the same pronunciation:
      flat:   'b i0 s m i0 ll a h i0 ...'   (phonemes, words not marked)
      bounds: "bi0s'mi0 lla'hi0 ..."        (one token per word, stress marks)
    Each bounds token is that word's phonemes concatenated, so we consume flat
    phonemes until they spell the token — giving per-word phonemes."""
    fn = _load_phonetiser()
    if fn is None:
        raise RuntimeError(_phonetiser_error)
    verse = clean_text(verse)
    words = verse.split()
    with contextlib.redirect_stdout(io.StringIO()):     # it prints debug output
        bounds, flat, _ = fn(verse)
    b_tokens, f_tokens = bounds[0].split(), flat[0].split()

    per_word, pos = [], 0
    for tok in b_tokens:
        target = tok.replace("'", "")
        if target == "sil":
            if pos < len(f_tokens) and f_tokens[pos] == "sil":
                pos += 1
            continue
        acc = []
        while pos < len(f_tokens) and len("".join(acc)) < len(target):
            acc.append(f_tokens[pos]); pos += 1
        if "".join(acc) != target:
            raise RuntimeError(f"could not split phonetiser output at '{tok}' for: {verse}")
        per_word.append(acc)
    if len(per_word) != len(words):
        raise RuntimeError(f"phonetiser gave {len(per_word)} words, text has "
                           f"{len(words)}: {verse}")
    return [(w, normalise(ph)) for w, ph in zip(words, per_word) if normalise(ph)]


# ---- verse -> per-word phonemes cache ----------------------------------------
CACHE_VERSION = 2   # v1 (per-isolated-word, shadda bug) caches are ignored

class PhonemeLookup:
    """Looks verses up in expected_phonemes.json, falling back to the live
    phonetiser (and remembering the result)."""

    def __init__(self, path=TABLE_PATH):
        self.path = path
        self.cache = {}
        self.dirty = False
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("version") == CACHE_VERSION:
                    self.cache = data["verses"]
            except Exception:
                pass

    def verse(self, text):
        key = clean_text(text)
        if key not in self.cache:
            self.cache[key] = [[w, ph] for w, ph in verse_word_phonemes(key)]
            self.dirty = True
        return [(w, ph) for w, ph in self.cache[key]]

    def verses(self, texts):
        """-> list of (word, phonemes) over several verses (each phonetised as
        its own utterance, i.e. the reciter restarts at each verse)."""
        return [p for t in texts for p in self.verse(t)]

    def save(self):
        if not self.dirty:
            return
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"version": CACHE_VERSION, "verses": self.cache}, f,
                      ensure_ascii=False, indent=1)
        self.dirty = False
