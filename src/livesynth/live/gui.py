"""LiveSynth real-time instrument (PyQt6).

    livesynth-live                      # or: python -m livesynth.live
    livesynth-live --quant 8            # MLX weight-only int8 (faster on small Macs)

Play with a MIDI keyboard or the computer keyboard:
    A W S E D F T G Y H U J K   one octave of keys (white row A..K, black row W..U)
    Z / X                       octave down / up
    C / V                       velocity down / up
    Space                       panic (all notes off, reset)
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QProgressBar, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

from livesynth.live.embedder import EmbedderProcess
from livesynth.live.host import SynthHost

AUDIO_EXT = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3", ".m4a", ".caf"}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEYMAP = {Qt.Key.Key_A: 0, Qt.Key.Key_W: 1, Qt.Key.Key_S: 2, Qt.Key.Key_E: 3, Qt.Key.Key_D: 4,
          Qt.Key.Key_F: 5, Qt.Key.Key_T: 6, Qt.Key.Key_G: 7, Qt.Key.Key_Y: 8, Qt.Key.Key_H: 9,
          Qt.Key.Key_U: 10, Qt.Key.Key_J: 11, Qt.Key.Key_K: 12}

STYLE = """
QWidget { background: #15171c; color: #d8dbe2; font-size: 13px; }
QGroupBox, QFrame#card { background: #1d2027; border: 1px solid #2a2e37; border-radius: 10px; }
QLabel#title { font-size: 20px; font-weight: 600; color: #f1f3f7; }
QLabel#muted { color: #8a90a0; }
QLabel#slotname { color: #9ecbff; }
QComboBox, QLineEdit, QDoubleSpinBox {
    background: #262a33; border: 1px solid #343946; border-radius: 6px; padding: 4px 8px; }
QPushButton { background: #2c313c; border: 1px solid #3a404d; border-radius: 6px; padding: 5px 12px; }
QPushButton:hover { background: #363c49; }
QPushButton#panic { background: #4a2328; border-color: #6b2f37; }
QCheckBox::indicator { width: 16px; height: 16px; }
QSlider::groove:horizontal { height: 6px; background: #2c313c; border-radius: 3px; }
QSlider::handle:horizontal { background: #9ecbff; width: 18px; margin: -7px 0; border-radius: 9px; }
QProgressBar { background: #262a33; border: none; border-radius: 4px; height: 8px; }
QProgressBar::chunk { background: #58c48b; border-radius: 4px; }
"""


class PianoWidget(QWidget):
    """Three-octave on-screen keyboard; highlights held notes, playable by mouse."""

    pressed = pyqtSignal(int)
    released = pyqtSignal(int)
    WHITE = [0, 2, 4, 5, 7, 9, 11]

    def __init__(self) -> None:
        super().__init__()
        self.base = 48
        self.held: set[int] = set()
        self._mouse_note: int | None = None
        self.setMinimumHeight(90)

    def _layout(self):
        whites = [self.base + 12 * o + s for o in range(3) for s in self.WHITE] + [self.base + 36]
        w = self.width() / len(whites)
        rects = {n: QRectF(i * w, 0, w, self.height()) for i, n in enumerate(whites)}
        blacks = {}
        for i, n in enumerate(whites[:-1]):
            if (n + 1) % 12 in (1, 3, 6, 8, 10):
                blacks[n + 1] = QRectF((i + 0.65) * w, 0, w * 0.7, self.height() * 0.6)
        return rects, blacks

    def paintEvent(self, _ev) -> None:                        # noqa: N802
        p = QPainter(self)
        whites, blacks = self._layout()
        for n, r in whites.items():
            p.fillRect(r.adjusted(1, 0, -1, 0), QColor("#9ecbff" if n in self.held else "#e9ebf0"))
            if n % 12 == 0:
                p.setPen(QPen(QColor("#6b7080")))
                p.setFont(QFont("", 8))
                p.drawText(r.adjusted(0, 0, 0, -4), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                           f"C{n // 12 - 1}")
        for n, r in blacks.items():
            p.fillRect(r, QColor("#4f8fd6" if n in self.held else "#20232a"))

    def _note_at(self, pos) -> int | None:
        whites, blacks = self._layout()
        for n, r in blacks.items():
            if r.contains(pos):
                return n
        for n, r in whites.items():
            if r.contains(pos):
                return n
        return None

    def mousePressEvent(self, ev) -> None:                    # noqa: N802
        n = self._note_at(ev.position())
        if n is not None:
            self._mouse_note = n
            self.pressed.emit(n)

    def mouseReleaseEvent(self, _ev) -> None:                 # noqa: N802
        if self._mouse_note is not None:
            self.released.emit(self._mouse_note)
            self._mouse_note = None


class TimbreSlot(QFrame):
    """Preset / reference audio / text prompt for one timbre slot."""

    def __init__(self, title: str, slot: int, window: "MainWindow") -> None:
        super().__init__()
        self.setObjectName("card")
        self.slot, self.win = slot, window
        self.setAcceptDrops(True)
        lay = QGridLayout(self)
        head = QLabel(title)
        head.setFont(QFont("", 13, QFont.Weight.DemiBold))
        self.name = QLabel("—")
        self.name.setObjectName("slotname")
        lay.addWidget(head, 0, 0)
        lay.addWidget(self.name, 0, 1, 1, 3)
        self.preset = QComboBox()
        self.preset.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.preset.setMinimumWidth(220)
        self.preset.activated.connect(self._on_preset)
        audio_btn = QPushButton("Reference audio…")
        audio_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        audio_btn.clicked.connect(self._pick_audio)
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("or describe it: the sound of an acoustic string")
        self.prompt.returnPressed.connect(self._on_prompt)
        # How the CLAP text embedding is mapped onto the audio embeddings the
        # model was trained on (the modality gap): an orthogonal Procrustes
        # rotation, or the raw text embedding.
        self.align = QComboBox()
        self.align.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for label, key in (("Procrustes", "procrustes"), ("No alignment", "none")):
            self.align.addItem(label, key)
        self.align.setToolTip("Text-to-audio alignment of the CLAP embedding")
        self.align.activated.connect(self._on_align)
        self._last_prompt: str | None = None
        lay.addWidget(self.preset, 1, 0, 1, 2)
        lay.addWidget(audio_btn, 1, 2)
        lay.addWidget(self.prompt, 2, 0, 1, 3)
        lay.addWidget(self.align, 2, 3)
        hint = QLabel("drop an audio file here")
        hint.setObjectName("muted")
        lay.addWidget(hint, 1, 3)

    def fill_presets(self, names: list[str]) -> None:
        self.preset.clear()
        self.preset.addItem("Preset…")
        self.preset.addItems(names)

    def _on_preset(self, idx: int) -> None:
        if idx > 0:
            name = self.preset.itemText(idx)
            self.win.apply_embedding(self.slot, self.win.host.presets[name], name, "preset")

    def _pick_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Reference recording", str(Path.home()),
                                              "Audio (*" + " *".join(sorted(AUDIO_EXT)) + ")")
        if path:
            self.win.embed_audio(self.slot, path)

    def _on_prompt(self) -> None:
        text = self.prompt.text().strip()
        if text:
            self._last_prompt = text
            self.win.embed_text(self.slot, text, self.align.currentData())
        self.win.setFocus()

    def _on_align(self, _idx: int) -> None:
        # Re-embed the current prompt with the new alignment, if this slot holds one.
        if self._last_prompt is not None and self.win.slot_source[self.slot] == "text":
            self.win.embed_text(self.slot, self._last_prompt, self.align.currentData())

    def dragEnterEvent(self, ev) -> None:                     # noqa: N802
        urls = ev.mimeData().urls()
        if urls and Path(urls[0].toLocalFile()).suffix.lower() in AUDIO_EXT:
            ev.acceptProposedAction()

    def dropEvent(self, ev) -> None:                          # noqa: N802
        self.win.embed_audio(self.slot, ev.mimeData().urls()[0].toLocalFile())
        ev.acceptProposedAction()


class MainWindow(QMainWindow):
    def __init__(self, host: SynthHost, embedder: EmbedderProcess) -> None:
        super().__init__()
        self.host, self.embedder = host, embedder
        self.setWindowTitle("LiveSynth")
        self.setStyleSheet(STYLE)
        self.resize(760, 560)
        self.kb_base, self.kb_velocity = 60, 100
        self.kb_down: dict[int, int] = {}
        self.held: set[int] = set()
        self._held_lock = threading.Lock()
        self.midi_in = None
        self._booted = False
        self.slot_source = ["", ""]              # "preset" | "audio" | "text" per slot
        self._pending: dict[int, str] = {}       # embedder job id -> source kind

        root = QWidget()
        v = QVBoxLayout(root)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)
        top = QHBoxLayout()
        title = QLabel("LiveSynth")
        title.setObjectName("title")
        self.status = QLabel("loading model…")
        self.status.setObjectName("muted")
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.status)
        v.addLayout(top)

        self.slots = [TimbreSlot("Timbre A", 0, self), TimbreSlot("Timbre B", 1, self)]
        for s in self.slots:
            v.addWidget(s)

        morph = QFrame()
        morph.setObjectName("card")
        mh = QHBoxLayout(morph)
        mh.addWidget(QLabel("Morph   A"))
        self.morph = QSlider(Qt.Orientation.Horizontal)
        self.morph.setRange(0, 1000)
        self.morph.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.morph.valueChanged.connect(lambda x: self.host.set_morph(x / 1000))
        mh.addWidget(self.morph, 1)
        mh.addWidget(QLabel("B"))
        v.addWidget(morph)

        perf = QFrame()
        perf.setObjectName("card")
        ph = QHBoxLayout(perf)
        self.keep = QCheckBox("Keep playing when I stop")
        self.keep.setChecked(True)
        self.keep.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.keep.toggled.connect(lambda on: setattr(self.host, "keep_playing", on))
        self.grace = QDoubleSpinBox()
        self.grace.setRange(0.0, 5.0)
        self.grace.setSingleStep(0.1)
        self.grace.setValue(host.grace_s)
        self.grace.setSuffix(" s")
        self.grace.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.grace.valueChanged.connect(lambda x: setattr(self.host, "grace_s", x))
        self.auto = QCheckBox("Autonomous")
        self.auto.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.auto.toggled.connect(lambda on: setattr(self.host, "autonomous", on))
        panic = QPushButton("Panic")
        panic.setObjectName("panic")
        panic.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        panic.clicked.connect(self._panic)
        ph.addWidget(self.keep)
        ph.addWidget(QLabel("after"))
        ph.addWidget(self.grace)
        ph.addSpacing(16)
        ph.addWidget(self.auto)
        ph.addStretch(1)
        ph.addWidget(panic)
        v.addWidget(perf)

        inp = QFrame()
        inp.setObjectName("card")
        ih = QHBoxLayout(inp)
        ih.addWidget(QLabel("MIDI in"))
        self.midi_combo = QComboBox()
        self.midi_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.midi_combo.setMinimumWidth(220)
        self.midi_combo.activated.connect(lambda i: self._open_midi(self.midi_combo.itemText(i)))
        refresh = QPushButton("↻")
        refresh.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        refresh.clicked.connect(self._fill_midi)
        self.kb_label = QLabel()
        self.kb_label.setObjectName("muted")
        ih.addWidget(self.midi_combo)
        ih.addWidget(refresh)
        ih.addStretch(1)
        ih.addWidget(self.kb_label)
        v.addWidget(inp)

        self.piano = PianoWidget()
        self.piano.pressed.connect(lambda n: self._note_on(n, self.kb_velocity))
        self.piano.released.connect(self._note_off)
        v.addWidget(self.piano)

        lv = QHBoxLayout()
        lv.addWidget(QLabel("Output"))
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        lv.addWidget(self.meter, 1)
        v.addLayout(lv)

        self.setCentralWidget(root)
        self._update_kb_label()
        self._fill_midi()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)
        self.setFocus()

    # -- notes ------------------------------------------------------------------

    def _note_on(self, n: int, vel: int) -> None:
        self.host.note_on(n, vel)
        with self._held_lock:
            self.held.add(n)

    def _note_off(self, n: int) -> None:
        self.host.note_off(n)
        with self._held_lock:
            self.held.discard(n)

    def _panic(self) -> None:
        self.host.panic()
        self.kb_down.clear()
        with self._held_lock:
            self.held.clear()

    def _update_kb_label(self) -> None:
        name = NOTE_NAMES[self.kb_base % 12] + str(self.kb_base // 12 - 1)
        self.kb_label.setText(f"keys A–K = {name}…  Z/X octave  C/V velocity {self.kb_velocity}")
        self.piano.base = max(0, self.kb_base - 12)

    def keyPressEvent(self, ev) -> None:                      # noqa: N802
        if ev.isAutoRepeat():
            return
        k = ev.key()
        if k == Qt.Key.Key_Z:
            self.kb_base = max(12, self.kb_base - 12)
        elif k == Qt.Key.Key_X:
            self.kb_base = min(108, self.kb_base + 12)
        elif k == Qt.Key.Key_C:
            self.kb_velocity = max(10, self.kb_velocity - 15)
        elif k == Qt.Key.Key_V:
            self.kb_velocity = min(127, self.kb_velocity + 15)
        elif k == Qt.Key.Key_Space:
            self._panic()
        elif k in KEYMAP and k not in self.kb_down:
            n = self.kb_base + KEYMAP[k]
            if n <= 127:
                self.kb_down[k] = n
                self._note_on(n, self.kb_velocity)
            return
        else:
            super().keyPressEvent(ev)
            return
        self._update_kb_label()

    def keyReleaseEvent(self, ev) -> None:                    # noqa: N802
        if ev.isAutoRepeat():
            return
        n = self.kb_down.pop(ev.key(), None)
        if n is not None:
            self._note_off(n)
        else:
            super().keyReleaseEvent(ev)

    # -- MIDI -------------------------------------------------------------------

    def _fill_midi(self) -> None:
        try:
            import mido
            names = mido.get_input_names()
        except Exception:                                     # noqa: BLE001
            names = []
        self.midi_combo.clear()
        self.midi_combo.addItem("(none)")
        self.midi_combo.addItems(names)
        self.midi_combo.addItem("Virtual port: LiveSynth In")
        if names:
            self.midi_combo.setCurrentIndex(1)
            self._open_midi(names[0])

    def _open_midi(self, name: str) -> None:
        import mido
        if self.midi_in is not None:
            self.midi_in.close()
            self.midi_in = None
        if name == "(none)":
            return

        def cb(msg) -> None:
            if msg.type == "note_on" and msg.velocity > 0:
                self._note_on(msg.note, msg.velocity)
            elif msg.type in ("note_off", "note_on"):
                self._note_off(msg.note)
            elif msg.type == "control_change" and msg.control == 123:
                self._panic()

        try:
            if name.startswith("Virtual port"):
                self.midi_in = mido.open_input("LiveSynth In", virtual=True, callback=cb)
            else:
                self.midi_in = mido.open_input(name, callback=cb)
        except Exception as exc:                              # noqa: BLE001
            self.status.setText(f"MIDI error: {exc}")

    # -- timbre -----------------------------------------------------------------

    def apply_embedding(self, slot: int, emb, label: str, source: str) -> None:
        self.host.set_slot(slot, emb)
        self.slot_source[slot] = source
        self.slots[slot].name.setText(label)
        if slot == 0 and self.slots[1].name.text() == "—":
            self.slots[1].name.setText(label)
            self.slot_source[1] = source

    def embed_audio(self, slot: int, path: str) -> None:
        self.slots[slot].name.setText(f"embedding {Path(path).name}…")
        self._pending[self.embedder.submit_audio(slot, path, Path(path).stem)] = "audio"

    def embed_text(self, slot: int, prompt: str, align: str = "procrustes") -> None:
        self.slots[slot].name.setText(f'embedding "{prompt}"…')
        self._pending[self.embedder.submit_text(slot, prompt, align)] = "text"

    # -- periodic -----------------------------------------------------------------

    def on_engine_ready(self) -> None:
        names = list(self.host.presets)
        for s in self.slots:
            s.fill_presets(names)
        if names:
            self.apply_embedding(0, self.host.presets[names[0]], names[0], "preset")

    def _tick(self) -> None:
        if not self._booted:
            if self.host.error:
                self.status.setText(f"engine failed: {self.host.error}")
                self._booted = True
            elif self.host.is_ready():
                self._booted = True
                self.on_engine_ready()
        for r in self.embedder.poll():
            if r.embedding is None:
                self.slots[r.slot].name.setText(f"failed: {r.error}")
            else:
                self.apply_embedding(r.slot, r.embedding, r.label, self._pending.pop(r.job_id, "audio"))
        st = self.host.stats()
        if st.frames:
            mode = "continuing" if st.absent else "playing"
            self.status.setText(f"{st.backend} · {st.frame_ms_mean:.1f} ms/frame (p99 {st.frame_ms_p99:.1f}) · "
                                f"underruns {st.underruns} · {mode}")
        if self.embedder.error:
            self.status.setText(self.embedder.error)
        self.meter.setValue(int(min(1.0, st.level * 4) * 100))
        with self._held_lock:
            held = set(self.held)
        if held != self.piano.held:
            self.piano.held = held
            self.piano.update()

    def closeEvent(self, ev) -> None:                         # noqa: N802
        self.timer.stop()
        if self.midi_in is not None:
            self.midi_in.close()
        self.host.stop()
        self.embedder.close()
        ev.accept()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="LiveSynth real-time instrument")
    ap.add_argument("--model-dir", default=None, help="local weights (default: download from the Hub)")
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "cuda", "cpu"])
    ap.add_argument("--quant", type=int, default=0, choices=[0, 4, 8],
                    help="MLX weight-only quantisation (0 = bf16)")
    ap.add_argument("--buffer", type=int, default=2, help="audio queue depth in 10-ms frames")
    ap.add_argument("--output-device", default=None, help="sounddevice output device name or index")
    args = ap.parse_args(argv)

    app = QApplication(sys.argv)
    host = SynthHost(model_dir=args.model_dir, backend=args.backend, quant=args.quant,
                     buffer_frames=args.buffer, output_device=args.output_device)
    embedder = EmbedderProcess(args.model_dir)
    embedder.start()
    win = MainWindow(host, embedder)
    win.show()

    host.start(wait=False)            # the window polls host.is_ready()
    code = app.exec()
    host.stop()
    embedder.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
