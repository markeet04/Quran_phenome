"""
Quran Recitation — Model Test Bench  (local Tkinter desktop app)

Test the phoneme model on your own voice or on audio files:
  1. pick what is being recited (a surah / single verse, custom text, or nothing)
  2. Record from the mic OR open an audio file
  3. the model transcribes the audio to phonemes; the app aligns them against
     the expected phonemes and grades each word (green / amber / red)

Re-run the same audio against a different reference with "Analyse again".
Every test is appended to results_log.csv (History tab).

The model only outputs phonemes; alignment + severity grading run here (same
rules as NB3). Expected phonemes come from expected_phonemes.json and/or the
MSA_phonetiser (cloned next to this file).

Run:  python app.py        Setup: see README.md
"""
import os, sys, csv, json, time, glob, threading, queue, datetime
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from quran_text import SURAHS, PhonemeLookup, phonetiser_available

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "app_config.json")
LOG_PATH = os.path.join(HERE, "results_log.csv")
SAMPLE_RATE = 16_000

# severity grading config (mirrors NB3) -------------------------------------
_DOUBLED = {"ll","rr","nn","ss","SS","dd","DD","tt","TT","bb","mm","kk",
            "ff","qq","jj","zz","ZZ","hh","HH","xx","gg","**","cc","yy","ww"}
_VOWEL_LEN_PAIRS = {("a","aa"),("aa","a"),("i","ii"),("ii","i"),
                    ("u","uu"),("uu","u"),("A","AA"),("AA","A"),
                    ("I","II"),("II","I"),("U","UU"),("UU","U")}
DROP = {"<sil>", "sil", "<pad>", "<unk>", "<s>", "</s>", "|", ""}


# ===========================================================================
# Config / model folder
# ===========================================================================
def load_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def find_model_in(path):
    """Accept the model folder itself, or a parent of it (e.g. the folder
    that contains final/ or checkpoint-XXXX/)."""
    if not path or not os.path.isdir(path):
        return None
    if os.path.exists(os.path.join(path, "config.json")):
        return path
    for sub in ("final",):
        p = os.path.join(path, sub)
        if os.path.exists(os.path.join(p, "config.json")):
            return p
    hits = sorted(glob.glob(os.path.join(path, "*", "config.json")) +
                  glob.glob(os.path.join(path, "*", "*", "config.json")))
    return os.path.dirname(hits[0]) if hits else None

def resolve_model_dir():
    for cand in (os.environ.get("QURAN_MODEL_DIR"),
                 load_config().get("model_dir"),
                 os.path.join(HERE, "model")):
        found = find_model_in(cand)
        if found:
            return found
    return None


# ===========================================================================
# Model wrapper
# ===========================================================================
class PhonemeModel:
    def __init__(self):
        self.ready = False
        self.error = None
        self.model_dir = None

    def load(self, model_dir):
        self.ready, self.error, self.model_dir = False, None, model_dir
        try:
            import torch
            from transformers import (AutoModelForCTC, AutoFeatureExtractor,
                                      Wav2Vec2CTCTokenizer)
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCTC.from_pretrained(model_dir).to(self.device).eval()
            self.fe = AutoFeatureExtractor.from_pretrained(model_dir)
            self.tok = Wav2Vec2CTCTokenizer.from_pretrained(model_dir)
            id2tok = {i: t for t, i in self.tok.get_vocab().items()}
            self.labels = [id2tok.get(i, "") for i in range(self.model.config.vocab_size)]
            self.vocab = set(self.tok.get_vocab().keys())
            self.blank = self.tok.pad_token_id
            self.n_params = sum(p.numel() for p in self.model.parameters())
            self.ready = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def transcribe(self, audio_f32):
        """audio float32 @16k mono -> list of phoneme tokens (greedy CTC)."""
        feats = self.fe(np.asarray(audio_f32, dtype=np.float32),
                        sampling_rate=SAMPLE_RATE, return_tensors="pt")
        feats = {k: v.to(self.device) for k, v in feats.items()}
        with self.torch.no_grad():
            ids = self.model(**feats).logits[0].argmax(-1).tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and i != self.blank:
                sym = (self.labels[i] if i < len(self.labels) else "").strip()
                if sym not in DROP:
                    out.append(sym)
            prev = i
        return out


# ===========================================================================
# Grading (align heard vs expected, assign severity) — same rules as NB3
# ===========================================================================
def align(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if ref[i-1]==hyp[j-1] else 1):
            ops.append(("ok" if ref[i-1]==hyp[j-1] else "sub", ref[i-1], hyp[j-1])); i-=1; j-=1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(("del", ref[i-1], None)); i-=1
        else:
            ops.append(("ins", None, hyp[j-1])); j-=1
    return ops[::-1]

def _collapse(seq):
    return [p[0] if p in _DOUBLED else p for p in seq]

def _severity(op, ref_sym, hyp_sym):
    if op == "sub":
        return "minor" if (ref_sym, hyp_sym) in _VOWEL_LEN_PAIRS else "major"
    if op == "del":
        return "major"
    return None

def per_word_grade(words_phon, heard):
    """words_phon: list of (word_text, [phonemes]) ; heard: phoneme list.
    Returns per-word verdicts, overall phoneme accuracy, and the alignment."""
    ref_flat, ref_word_of = [], []
    for wi, (_, ph) in enumerate(words_phon):
        ref_flat += ph
        ref_word_of += [wi]*len(ph)
    ops = align(_collapse(ref_flat), _collapse(heard))
    errors = {wi: [] for wi in range(len(words_phon))}
    rp = 0
    for op, r, h in ops:
        if op in ("ok","sub","del"):
            wi = ref_word_of[rp] if rp < len(ref_word_of) else len(words_phon)-1
            orig = ref_flat[rp] if rp < len(ref_flat) else r
            sev = _severity(op, orig, h)
            if sev:
                errors[wi].append((sev, op, orig, h))
            rp += 1
    report = []
    for wi, (w, ph) in enumerate(words_phon):
        errs = errors[wi]
        verdict = ("major" if any(s=="major" for s,*_ in errs)
                   else "minor" if errs else "pass")
        report.append({"word": w, "verdict": verdict, "errors": errs, "phon": ph})
    d = sum(1 for op,_,_ in ops if op != "ok")
    acc = 1 - d / max(len(_collapse(ref_flat)), 1)
    return report, max(acc, 0.0), ops


# ===========================================================================
# Audio: mic recording + file loading
# ===========================================================================
class Recorder:
    def __init__(self):
        self.frames, self.recording, self.level, self.error = [], False, 0.0, None
        try:
            import sounddevice as sd
            self.sd = sd
        except Exception as e:
            self.sd = None
            self.error = f"sounddevice not available: {e}"

    def start(self):
        self.frames, self.level = [], 0.0
        self.stream = self.sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                          dtype="float32", callback=self._cb)
        self.stream.start()
        self.recording = True

    def _cb(self, indata, frames, t, status):
        if self.recording:
            self.frames.append(indata.copy())
            self.level = float(np.sqrt(np.mean(indata**2)))

    def stop(self):
        self.recording = False
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass
        if not self.frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.frames, axis=0).flatten().astype(np.float32)

def load_audio_file(path):
    """Any format libsndfile reads (wav/flac/ogg/mp3) -> float32 mono @16k."""
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio, sr


# ===========================================================================
# The Tkinter UI
# ===========================================================================
BG, PANEL, FG, DIM = "#1e1e2e", "#181825", "#cdd6f4", "#6c7086"
GREEN, AMBER, RED, BLUE = "#a6e3a1", "#f9e2af", "#f38ba8", "#89b4fa"
ARABIC_FONT = ("Segoe UI", 18)
MODES = ("Surah / verse", "Custom text", "No reference (transcribe only)")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Quran Recitation — Model Test Bench")
        self.geometry("1000x780")
        self.minsize(820, 620)
        self.configure(bg=BG)

        self.model = PhonemeModel()
        self.recorder = Recorder()
        self.lookup = PhonemeLookup()
        self.ui_q = queue.Queue()          # bg threads -> UI thread
        self.audio = None                  # current clip (float32 @16k)
        self.audio_source = ""
        self.busy = False
        self.rec_started = 0.0

        self._style()
        self._build_ui()
        self._load_history()
        self._on_mode_change()
        self.after(50, self._pump)

        model_dir = resolve_model_dir()
        if model_dir:
            self._start_model_load(model_dir)
        else:
            self.model_lbl.config(text="No model found — click 'Choose model folder…' "
                                       "(the folder with config.json + model.safetensors)",
                                  fg=RED)

    # ------------------------------------------------------------------ style
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=PANEL)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("TRadiobutton", background=BG, foreground=FG)
        s.map("TRadiobutton", background=[("active", BG)])
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=PANEL, foreground=FG, padding=(12, 4))
        s.map("TNotebook.Tab", background=[("selected", "#313244")])
        s.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG,
                    arrowcolor=FG)
        s.map("TCombobox", fieldbackground=[("readonly", PANEL)],
              foreground=[("readonly", FG)])
        s.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG,
                    rowheight=22)
        s.configure("Treeview.Heading", background="#313244", foreground=FG)
        s.configure("Level.Horizontal.TProgressbar", troughcolor=PANEL, background=GREEN)
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", FG)

    def _btn(self, parent, text, cmd, color=BLUE, **kw):
        return tk.Button(parent, text=text, command=cmd, bg=color, fg="#11111b",
                         activebackground=color, relief="flat", padx=10, pady=3,
                         font=("Segoe UI", 10, "bold"), **kw)

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        pad = {"padx": 14}

        # --- model row
        row = tk.Frame(self, bg=BG); row.pack(fill="x", pady=(10, 4), **pad)
        tk.Label(row, text="Model", bg=BG, fg=BLUE, font=("Segoe UI", 10, "bold"),
                 width=10, anchor="w").pack(side="left")
        self.model_lbl = tk.Label(row, text="", bg=BG, fg=FG, anchor="w",
                                  font=("Segoe UI", 9))
        self.model_lbl.pack(side="left", fill="x", expand=True)
        self.choose_btn = self._btn(row, "Choose model folder…", self._choose_model)
        self.choose_btn.pack(side="right")

        # --- reference row
        row = tk.Frame(self, bg=BG); row.pack(fill="x", pady=4, **pad)
        tk.Label(row, text="Reference", bg=BG, fg=BLUE, font=("Segoe UI", 10, "bold"),
                 width=10, anchor="w").pack(side="left")
        self.mode = tk.StringVar(value=MODES[0])
        for m in MODES:
            ttk.Radiobutton(row, text=m, value=m, variable=self.mode,
                            command=self._on_mode_change).pack(side="left", padx=(0, 12))

        row = tk.Frame(self, bg=BG); row.pack(fill="x", pady=4, **pad)
        tk.Label(row, text="", bg=BG, width=10).pack(side="left")
        self.surah_var = tk.StringVar(value=list(SURAHS)[0])
        self.surah_cb = ttk.Combobox(row, textvariable=self.surah_var, values=list(SURAHS),
                                     state="readonly", width=26, font=("Segoe UI", 10))
        self.surah_cb.pack(side="left")
        self.surah_cb.bind("<<ComboboxSelected>>", lambda e: self._on_surah_change())
        self.verse_label = tk.Label(row, text="  Verse", bg=BG, fg=FG)
        self.verse_label.pack(side="left")
        self.verse_var = tk.StringVar()
        self.verse_cb = ttk.Combobox(row, textvariable=self.verse_var, state="readonly",
                                     width=14, font=("Segoe UI", 10))
        self.verse_cb.pack(side="left", padx=6)
        self.verse_cb.bind("<<ComboboxSelected>>", lambda e: self._show_reference())
        self.custom_var = tk.StringVar()
        self.custom_entry = tk.Entry(row, textvariable=self.custom_var, justify="right",
                                     bg=PANEL, fg=FG, insertbackground=FG, relief="flat",
                                     font=("Segoe UI", 13))
        self.custom_var.trace_add("write", lambda *a: self._show_reference())

        self.verse_lbl = tk.Label(self, text="", font=ARABIC_FONT, bg=BG, fg=FG,
                                  wraplength=960, justify="right", anchor="e")
        self.verse_lbl.pack(fill="x", pady=(4, 6), **pad)

        # --- audio row
        row = tk.Frame(self, bg=BG); row.pack(fill="x", pady=4, **pad)
        tk.Label(row, text="Audio", bg=BG, fg=BLUE, font=("Segoe UI", 10, "bold"),
                 width=10, anchor="w").pack(side="left")
        self.rec_btn = self._btn(row, "● Record", self._toggle_record, GREEN, width=10)
        self.rec_btn.pack(side="left")
        self.open_btn = self._btn(row, "Open audio file…", self._open_file)
        self.open_btn.pack(side="left", padx=6)
        self.play_btn = self._btn(row, "▶ Play", self._play, "#cba6f7")
        self.play_btn.pack(side="left")
        self.save_btn = self._btn(row, "Save WAV…", self._save_wav, "#cba6f7")
        self.save_btn.pack(side="left", padx=6)
        self.again_btn = self._btn(row, "↻ Analyse again", self._analyse_current, AMBER)
        self.again_btn.pack(side="left")
        self.level = ttk.Progressbar(row, style="Level.Horizontal.TProgressbar",
                                     length=120, maximum=1.0)
        self.level.pack(side="right")
        self.audio_lbl = tk.Label(row, text="no audio yet", bg=BG, fg=DIM,
                                  font=("Segoe UI", 9))
        self.audio_lbl.pack(side="right", padx=8)

        # --- score row
        row = tk.Frame(self, bg=BG); row.pack(fill="x", pady=(8, 2), **pad)
        self.score_lbl = tk.Label(row, text="", font=("Segoe UI", 22, "bold"),
                                  bg=BG, fg=AMBER)
        self.score_lbl.pack(side="left")
        self.summary_lbl = tk.Label(row, text="", font=("Segoe UI", 11), bg=BG, fg=FG)
        self.summary_lbl.pack(side="left", padx=16)

        # --- result tabs
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, pady=(4, 6), **pad)
        self.words_txt = self._text_tab("Per word")
        self.align_txt = self._text_tab("Alignment")
        self.raw_txt = self._text_tab("Raw phonemes")
        hist = tk.Frame(self.tabs, bg=PANEL)
        self.tabs.add(hist, text="History")
        cols = ("time", "audio", "reference", "secs", "infer", "acc", "major", "minor")
        self.hist = ttk.Treeview(hist, columns=cols, show="headings")
        widths = (130, 180, 260, 55, 55, 60, 55, 55)
        for c, w in zip(cols, widths):
            self.hist.heading(c, text=c)
            self.hist.column(c, width=w, anchor="w" if c in ("audio", "reference") else "center")
        sb = ttk.Scrollbar(hist, orient="vertical", command=self.hist.yview)
        self.hist.configure(yscrollcommand=sb.set)
        self.hist.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")

        self.status = tk.StringVar(value="")
        tk.Label(self, textvariable=self.status, font=("Segoe UI", 9), bg="#11111b",
                 fg="#bac2de", anchor="w", padx=8).pack(fill="x", side="bottom")
        self._refresh_buttons()

    def _text_tab(self, title):
        frame = tk.Frame(self.tabs, bg=PANEL)
        self.tabs.add(frame, text=title)
        t = tk.Text(frame, font=("Consolas", 11), bg=PANEL, fg=FG, wrap="word",
                    relief="flat", padx=10, pady=10, insertbackground=FG)
        sb = ttk.Scrollbar(frame, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=sb.set)
        t.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        for tag, col in (("major", RED), ("minor", AMBER), ("pass", GREEN),
                         ("ok", GREEN), ("sub", RED), ("del", RED), ("ins", AMBER),
                         ("dim", DIM), ("head", BLUE)):
            t.tag_config(tag, foreground=col)
        t.tag_config("head", font=("Consolas", 11, "bold"))
        t.tag_config("arabic", font=ARABIC_FONT)
        return t

    # ------------------------------------------------------------------ threading
    def _pump(self):
        """Run callbacks queued by background threads on the UI thread."""
        try:
            while True:
                self.ui_q.get_nowait()()
        except queue.Empty:
            pass
        if self.recorder.recording:
            self.level["value"] = min(self.recorder.level * 8, 1.0)
            self.audio_lbl.config(text=f"recording… {time.time()-self.rec_started:.1f}s",
                                  fg=RED)
        self.after(50, self._pump)

    def _bg(self, fn, *args):
        threading.Thread(target=fn, args=args, daemon=True).start()

    def _ui(self, fn):
        self.ui_q.put(fn)

    def _refresh_buttons(self):
        ready = self.model.ready and not self.busy
        recording = self.recorder.recording
        # never swap the model while it is loading or in use (crashes torch)
        self.choose_btn.config(state="disabled" if (self.busy or recording) else "normal")
        self.rec_btn.config(state="normal" if (ready or recording) else "disabled")
        for b in (self.open_btn, self.again_btn):
            b.config(state="normal" if ready and not recording else "disabled")
        if self.audio is None:
            self.again_btn.config(state="disabled")
        has_audio = self.audio is not None and not recording
        self.play_btn.config(state="normal" if has_audio and self.recorder.sd else "disabled")
        self.save_btn.config(state="normal" if has_audio else "disabled")

    # ------------------------------------------------------------------ model
    def _choose_model(self):
        d = filedialog.askdirectory(title="Select the model folder (contains config.json)")
        if not d:
            return
        found = find_model_in(d)
        if not found:
            messagebox.showerror("Not a model folder",
                                 f"No config.json found in\n{d}\n(or its sub-folders).")
            return
        cfg = load_config(); cfg["model_dir"] = found; save_config(cfg)
        self._start_model_load(found)

    def _start_model_load(self, model_dir):
        if self.busy:
            return
        self.busy = True
        self._refresh_buttons()
        self.model_lbl.config(text=f"Loading {model_dir} …", fg=AMBER)
        self.status.set("Loading model… first load on CPU can take ~30-60s")

        def work():
            new = PhonemeModel()          # load separately, swap in when done
            new.load(model_dir)
            def done():
                self.busy = False
                if new.ready:
                    self.model = new
                    self.model_lbl.config(
                        text=f"{model_dir}   ·   {self.model.n_params/1e6:.0f}M params   ·   "
                             f"{len(self.model.vocab)} tokens   ·   {self.model.device}",
                        fg=GREEN)
                    self.status.set("Model ready. Pick a reference, then Record or open a file.")
                    self._show_reference()
                else:
                    self.model_lbl.config(text=f"FAILED: {new.error}", fg=RED)
                    self.status.set("Model failed to load.")
                    messagebox.showerror("Model load failed", new.error)
                self._refresh_buttons()
            self._ui(done)
        self._bg(work)

    # ------------------------------------------------------------------ reference
    def _on_mode_change(self):
        m = self.mode.get()
        for w in (self.surah_cb, self.verse_label, self.verse_cb, self.custom_entry):
            w.pack_forget()
        if m == MODES[0]:
            self.surah_cb.pack(side="left")
            self.verse_label.pack(side="left")
            self.verse_cb.pack(side="left", padx=6)
            self._on_surah_change()
            return
        if m == MODES[1]:
            self.custom_entry.pack(side="left", fill="x", expand=True)
            self.custom_entry.focus_set()
        self._show_reference()

    def _on_surah_change(self):
        n = len(SURAHS[self.surah_var.get()])
        values = ["All verses"] + [f"Verse {i}" for i in range(1, n + 1)]
        self.verse_cb.config(values=values)
        self.verse_var.set("Verse 1")
        self._show_reference()

    def reference_text(self):
        """-> (label, [verse, ...]) or (label, None) for transcribe-only.
        Each verse is phonetised as its own utterance."""
        m = self.mode.get()
        if m == MODES[2]:
            return "(none)", None
        if m == MODES[1]:
            t = self.custom_var.get().strip()
            return ("custom: " + t[:40]), ([t] if t else None)
        surah = self.surah_var.get()
        verses = SURAHS[surah]
        short = " ".join(surah.split("(")[0].split())
        v = self.verse_var.get()
        if v.startswith("Verse "):
            i = int(v.split()[1])
            return f"{short}:{i}", [verses[i - 1]]
        return f"{short} (all)", list(verses)

    def _show_reference(self):
        label, verses = self.reference_text()
        text = " ۝ ".join(verses) if verses else None
        self.verse_lbl.config(text=text or ("Type diacritised Arabic above…"
                                            if self.mode.get() == MODES[1] else
                                            "No reference — the app will only show what the model hears."),
                              fg=FG if text else DIM)

    # ------------------------------------------------------------------ audio in
    def _toggle_record(self):
        if not self.recorder.recording:
            if self.recorder.error:
                messagebox.showerror("Mic error", self.recorder.error); return
            try:
                self.recorder.start()
            except Exception as e:
                messagebox.showerror("Mic error", f"{type(e).__name__}: {e}"); return
            self.rec_started = time.time()
            self.rec_btn.config(text="■ Stop", bg=RED, activebackground=RED)
            self.status.set("Recording… recite now, then press Stop.")
        else:
            audio = self.recorder.stop()
            self.level["value"] = 0
            self.rec_btn.config(text="● Record", bg=GREEN, activebackground=GREEN)
            if audio.size < SAMPLE_RATE // 2:
                self.status.set("Recording too short — try again.")
                self._refresh_buttons()
                return
            self._set_audio(audio, f"mic {datetime.datetime.now():%H:%M:%S}")
            self._analyse_current()
        self._refresh_buttons()

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open a recitation clip",
            filetypes=[("Audio", "*.wav *.flac *.ogg *.mp3"), ("All files", "*.*")])
        if not path:
            return
        try:
            audio, sr = load_audio_file(path)
        except Exception as e:
            messagebox.showerror("Could not read audio", f"{type(e).__name__}: {e}")
            return
        self._set_audio(audio, os.path.basename(path))
        self.status.set(f"Loaded {os.path.basename(path)} (orig {sr} Hz → 16 kHz mono)")
        self._analyse_current()

    def _set_audio(self, audio, source):
        self.audio, self.audio_source = audio, source
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        warn = "  ⚠ very quiet" if peak < 0.02 else "  ⚠ clipping" if peak >= 0.999 else ""
        self.audio_lbl.config(text=f"{source} · {len(audio)/SAMPLE_RATE:.1f}s · peak {peak:.2f}{warn}",
                              fg=AMBER if warn else FG)
        self._refresh_buttons()

    def _play(self):
        if self.audio is not None and self.recorder.sd:
            self.recorder.sd.stop()
            self.recorder.sd.play(self.audio, SAMPLE_RATE)

    def _save_wav(self):
        if self.audio is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav",
                                            filetypes=[("WAV", "*.wav")],
                                            initialfile="recitation.wav")
        if path:
            import soundfile as sf
            sf.write(path, self.audio, SAMPLE_RATE)
            self.status.set(f"Saved {path}")

    # ------------------------------------------------------------------ analyse
    def _analyse_current(self):
        if self.audio is None or not self.model.ready or self.busy:
            return
        label, text = self.reference_text()
        if self.mode.get() == MODES[1] and not text:
            messagebox.showinfo("No text", "Type the diacritised verse first, "
                                           "or choose 'No reference'.")
            return
        self.busy = True
        self._refresh_buttons()
        secs = len(self.audio) / SAMPLE_RATE
        self.status.set(f"Analysing {secs:.1f}s clip… "
                        + (f"(on CPU expect ~{secs*2:.0f}s)" if self.model.device == "cpu" else ""))
        audio, source = self.audio, self.audio_source

        def work():
            try:
                words_phon = None
                if text:
                    try:
                        words_phon = self.lookup.verses(text)
                        self.lookup.save()
                    except RuntimeError as e:
                        raise RuntimeError(
                            "Couldn't get the expected phonemes for this verse "
                            f"(phonetiser problem):\n{e}")
                t0 = time.time()
                heard = self.model.transcribe(audio)
                infer = time.time() - t0
                res = {"source": source, "label": label, "secs": len(audio) / SAMPLE_RATE,
                       "infer": infer, "heard": heard, "words_phon": words_phon}
                if words_phon:
                    res["report"], res["acc"], res["ops"] = per_word_grade(words_phon, heard)
                self._ui(lambda: self._finish(res))
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                def fail():
                    self.busy = False
                    self._refresh_buttons()
                    self.status.set("Error during analysis.")
                    messagebox.showerror("Analyse error", msg)
                self._ui(fail)
        self._bg(work)

    def _finish(self, r):
        self.busy = False
        self._refresh_buttons()
        self._render(r)
        self._log(r)
        rtf = r["infer"] / max(r["secs"], 1e-6)
        self.status.set(f"Done — {r['secs']:.1f}s audio, inference {r['infer']:.1f}s "
                        f"({rtf:.2f}× real-time) on {self.model.device}.")

    # ------------------------------------------------------------------ render
    def _render(self, r):
        for t in (self.words_txt, self.align_txt, self.raw_txt):
            t.delete("1.0", "end")
        heard = r["heard"]

        # raw tab (always)
        R = self.raw_txt
        R.insert("end", f"MODEL HEARD ({len(heard)} phonemes)\n", "head")
        R.insert("end", (" ".join(heard) or "(nothing — silence or noise?)") + "\n\n")
        if r["words_phon"]:
            flat = [p for _, ph in r["words_phon"] for p in ph]
            R.insert("end", f"EXPECTED ({len(flat)} phonemes)\n", "head")
            R.insert("end", " ".join(flat) + "\n\n", "dim")
            unknown = sorted({p for p in flat if p not in self.model.vocab})
            if unknown:
                R.insert("end", "⚠ expected symbols NOT in the model vocab (reference/"
                                "alphabet mismatch): " + " ".join(unknown) + "\n", "major")

        if not r["words_phon"]:
            self.score_lbl.config(text="Transcribed", fg=BLUE)
            self.summary_lbl.config(text=f"{len(heard)} phonemes heard — no reference to grade against")
            self.words_txt.insert("end", "No reference selected — see the Raw phonemes tab.\n", "dim")
            self.tabs.select(2)
            return

        acc, report = r["acc"], r["report"]
        col = GREEN if acc >= 0.85 else AMBER if acc >= 0.6 else RED
        self.score_lbl.config(text=f"{acc*100:.1f}%", fg=col)
        n = {k: sum(1 for w in report if w["verdict"] == k) for k in ("pass", "minor", "major")}
        self.summary_lbl.config(
            text=f"phoneme accuracy   ·   🟢 {n['pass']} correct   🟡 {n['minor']} minor   "
                 f"🔴 {n['major']} major   of {len(report)} words")

        # per-word tab
        W = self.words_txt
        W.insert("end", "🟢 correct   🟡 minor (vowel length)   🔴 major (wrong / missing letter)\n\n", "dim")
        for w in report:
            tag = w["verdict"]
            icon = {"pass": "🟢", "minor": "🟡", "major": "🔴"}[tag]
            W.insert("end", f"{icon}  ", tag)
            W.insert("end", w["word"], ("arabic", tag))
            W.insert("end", f"    /{' '.join(w['phon'])}/\n", "dim")
            for sev, op, rr, hh in w["errors"]:
                desc = (f"expected '{rr}', heard '{hh}'" if op == "sub"
                        else f"'{rr}' missing")
                W.insert("end", f"        {sev}: {desc}\n", sev)

        # alignment tab
        A = self.align_txt
        A.insert("end", "Phoneme-by-phoneme alignment (doubled consonants compared as single).\n"
                        "green = match · red = substituted/missing · amber = extra sound heard\n\n", "dim")
        ops = r["ops"]
        for start in range(0, len(ops), 14):
            chunk = ops[start:start + 14]
            cells = [((rr or "·"), (hh or "·"), op) for op, rr, hh in chunk]
            A.insert("end", "exp  ", "head")
            for rr, hh, op in cells:
                A.insert("end", rr.ljust(max(len(rr), len(hh)) + 2), op)
            A.insert("end", "\nhrd  ", "head")
            for rr, hh, op in cells:
                A.insert("end", hh.ljust(max(len(rr), len(hh)) + 2), op)
            A.insert("end", "\n\n")
        self.tabs.select(0)

    # ------------------------------------------------------------------ history
    def _log(self, r):
        row = {
            "time": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            "audio": r["source"], "reference": r["label"],
            "secs": f"{r['secs']:.1f}", "infer": f"{r['infer']:.1f}",
            "acc": f"{r['acc']*100:.1f}" if "acc" in r else "",
            "major": sum(w["verdict"] == "major" for w in r["report"]) if "report" in r else "",
            "minor": sum(w["verdict"] == "minor" for w in r["report"]) if "report" in r else "",
            "heard": " ".join(r["heard"]),
        }
        new = not os.path.exists(LOG_PATH)
        try:
            with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:
            self.status.set(f"(could not write log: {e})")
        self.hist.insert("", 0, values=[row[c] for c in self.hist["columns"]])

    def _load_history(self):
        if not os.path.exists(LOG_PATH):
            return
        try:
            with open(LOG_PATH, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.hist.insert("", 0, values=[row.get(c, "") for c in self.hist["columns"]])
        except Exception:
            pass


if __name__ == "__main__":
    # native crashes (e.g. inside torch) give no Python traceback; this writes
    # the stack of every thread to crash_log.txt so they can be diagnosed
    import faulthandler
    _crash_log = open(os.path.join(HERE, "crash_log.txt"), "w")
    faulthandler.enable(file=_crash_log, all_threads=True)
    App().mainloop()
