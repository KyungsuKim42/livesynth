"""Real-time synthesis host: engine thread + audio output + performer state.

The host owns one streaming engine (MLX on Apple Silicon, PyTorch/CUDA
otherwise) and runs it in a dedicated thread at 100 frames per second. The
thread is paced by a small queue drained by the audio callback. Everything the
performer controls is thread-safe and takes effect on the next frame:

* notes (``note_on`` / ``note_off``), from MIDI or the computer keyboard;
* two timbre slots A and B and a morph position between them — the engine's
  timbre is the spherical interpolation, smoothed over ~50 ms;
* *keep playing*: when every key is released, the MIDI condition becomes absent
  after a short grace period and the model continues the performance;
* *autonomous*: the MIDI condition is always absent (the model improvises).

The engine is created inside its thread because MLX streams are thread-local.
"""

from __future__ import annotations

import math
import platform
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from livesynth.midi import NoteTracker

SR = 48_000
HOP = 480


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = a / max(np.linalg.norm(a), 1e-8)
    b = b / max(np.linalg.norm(b), 1e-8)
    dot = float(np.clip(a @ b, -1 + 1e-6, 1 - 1e-6))
    th = math.acos(dot)
    return ((math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)).astype(np.float32)


def default_backend() -> str:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import mlx.core  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@dataclass
class HostStats:
    backend: str = ""
    frame_ms_mean: float = 0.0
    frame_ms_p99: float = 0.0
    underruns: int = 0
    level: float = 0.0
    absent: bool = False
    frames: int = 0


class SynthHost:
    """Args:
        model_dir: directory with the released weights (``None`` = Hub download).
        backend: ``"auto"``, ``"mlx"``, ``"cuda"`` or ``"cpu"``.
        quant: MLX only — 0 (bf16, default), 8 or 4 for weight-only quantisation.
        buffer_frames: audio queue depth in 10-ms frames (latency vs. safety).
        audio: open an audio output stream (``False`` for headless use/tests;
            then read frames with :meth:`pull`).
    """

    def __init__(self, model_dir: str | None = None, backend: str = "auto", quant: int = 0,
                 buffer_frames: int = 2, audio: bool = True, output_device=None,
                 seed: int | None = None) -> None:
        self.model_dir = model_dir
        self.backend = default_backend() if backend == "auto" else backend
        self.quant = quant
        self.audio = audio
        self.output_device = output_device
        self.seed = seed
        self.tracker = NoteTracker()
        self._lock = threading.Lock()
        self._slots: list[np.ndarray | None] = [None, None]
        self._morph_target = 0.0
        self._morph = 0.0
        self.keep_playing = True
        self.autonomous = False
        self.grace_s = 1.0          # rests shorter than this never trigger autonomy
        self._idle_frames = 0
        self._has_played = False
        self._reset_req = False
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max(1, buffer_frames))
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: str | None = None
        self._times: deque[float] = deque(maxlen=500)
        self._underruns = 0
        self._level = 0.0
        self._absent = False
        self._frames = 0
        self._thread: threading.Thread | None = None
        self._stream = None
        self.presets: dict[str, np.ndarray] = {}

    # -- lifecycle -------------------------------------------------------------

    def start(self, wait: bool = True, timeout: float = 600.0) -> None:
        """Start the engine thread. With ``wait`` (default) block until the model
        is loaded and compiled; otherwise poll :meth:`is_ready` / :attr:`error`."""
        self._thread = threading.Thread(target=self._run, name="livesynth-engine", daemon=True)
        self._thread.start()
        if not wait:
            return
        if not self._ready.wait(timeout):
            raise TimeoutError("engine did not start")
        if self._error:
            raise RuntimeError(self._error)

    def is_ready(self) -> bool:
        return self._ready.is_set() and self._error is None

    @property
    def error(self) -> str | None:
        return self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    # -- performer controls (any thread) -----------------------------------------

    def note_on(self, pitch: int, velocity: int = 100) -> None:
        with self._lock:
            self.tracker.note_on(pitch, velocity)
            self._has_played = True

    def note_off(self, pitch: int) -> None:
        with self._lock:
            self.tracker.note_off(pitch)

    def all_notes_off(self) -> None:
        with self._lock:
            self.tracker.all_notes_off()

    def panic(self) -> None:
        """Release every note and clear the model state (silence)."""
        with self._lock:
            self.tracker.all_notes_off()
            self._has_played = False
            self._reset_req = True

    def set_slot(self, i: int, embedding: np.ndarray) -> None:
        with self._lock:
            self._slots[i] = np.asarray(embedding, np.float32).reshape(-1)
            if self._slots[1 - i] is None:
                self._slots[1 - i] = self._slots[i].copy()

    def set_morph(self, x: float) -> None:
        with self._lock:
            self._morph_target = float(min(1.0, max(0.0, x)))

    def stats(self) -> HostStats:
        t = np.asarray(self._times) if self._times else np.zeros(1)
        return HostStats(self.backend, float(t.mean()), float(np.percentile(t, 99)),
                         self._underruns, self._level, self._absent, self._frames)

    def pull(self, timeout: float = 5.0) -> np.ndarray:
        """Headless mode: next 480-sample block."""
        return self._q.get(timeout=timeout)

    # -- engine thread -------------------------------------------------------------

    def _load_presets(self, d: Path) -> None:
        import json
        from safetensors.numpy import load_file
        names = json.loads((d / "config.json").read_text())["presets"]
        emb = load_file(str(d / "presets.safetensors"))["embeddings"]
        self.presets = {n: emb[i].astype(np.float32) for i, n in enumerate(names)}

    def _make_engine(self):
        from livesynth.hub import resolve_model_dir
        d = resolve_model_dir(local_dir=self.model_dir)
        self._load_presets(d)
        if self.backend == "mlx":
            from livesynth.mlx_engine import MLXStreamingEngine
            eng = MLXStreamingEngine(d, seed=self.seed)
            if self.quant:
                eng.quantize(self.quant)
            return eng
        from livesynth.synth import LiveSynth
        synth = LiveSynth.from_pretrained(local_dir=d, device="cuda" if self.backend == "cuda" else "cpu")
        eng = synth.streaming_engine(use_graph=self.backend == "cuda", seed=self.seed)
        return eng

    def _timbre(self) -> np.ndarray | None:
        a, b = self._slots
        if a is None:
            return None
        alpha = 1.0 - math.exp(-0.01 / 0.05)             # 50-ms smoothing per frame
        self._morph += alpha * (self._morph_target - self._morph)
        if abs(self._morph) < 1e-4 or b is None:
            return a
        if abs(1.0 - self._morph) < 1e-4:
            return b
        return _slerp(a, b, self._morph)

    def _open_audio(self) -> None:
        import sounddevice as sd
        leftover = [np.zeros(0, np.float32)]

        def cb(out, nframes, _t, _status) -> None:
            buf = leftover[0]
            while buf.size < nframes:
                try:
                    buf = np.concatenate([buf, self._q.get_nowait()])
                except queue.Empty:
                    self._underruns += 1
                    buf = np.concatenate([buf, np.zeros(nframes - buf.size, np.float32)])
            out[:, 0] = buf[:nframes]
            leftover[0] = buf[nframes:]

        self._stream = sd.OutputStream(samplerate=SR, channels=1, blocksize=HOP, dtype="float32",
                                       latency="low", callback=cb, device=self.output_device)
        self._stream.start()

    def _run(self) -> None:
        try:
            if sys.platform == "darwin":                  # macOS: user-interactive QoS
                try:
                    import ctypes
                    ctypes.CDLL(None).pthread_set_qos_class_self_np(0x21, 0)
                except Exception:                         # noqa: BLE001
                    pass
            eng = self._make_engine()
            if self._slots[0] is None and self.presets:
                self.set_slot(0, next(iter(self.presets.values())))
            self._set_engine_timbre(eng, self._timbre())
            zero = np.zeros(128, np.int64)
            for _ in range(3):                            # compile / warm up
                self._step(eng, zero, zero, False)
            self._reset(eng)
            if self.audio:
                self._open_audio()
        except Exception as exc:                          # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
            self._ready.set()
            return
        self._ready.set()
        while not self._stop.is_set():
            with self._lock:
                if self._reset_req:
                    self._reset_req = False
                    self._reset(eng)
                state, age = self.tracker.frame()
                held = bool((state >= 2).any())
                self._idle_frames = 0 if held else self._idle_frames + 1
                grace = int(round(self.grace_s * 100))      # read live (GUI may change it)
                absent = self.autonomous or (self.keep_playing and self._has_played
                                             and self._idle_frames > grace)
                timbre = self._timbre()
            self._set_engine_timbre(eng, timbre)
            t0 = time.perf_counter()
            block = self._step(eng, state, age, absent)
            self._times.append((time.perf_counter() - t0) * 1e3)
            self._absent = absent
            self._frames += 1
            self._level = 0.9 * self._level + 0.1 * float(np.sqrt(np.mean(block ** 2)))
            while not self._stop.is_set():
                try:
                    self._q.put(block, timeout=0.2)       # blocks when full = real-time pacing
                    break
                except queue.Full:
                    continue
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def _reset(self, eng) -> None:
        eng.reset()

    def _set_engine_timbre(self, eng, timbre) -> None:
        if timbre is None:
            return
        if self.backend == "mlx":
            eng.set_timbre(timbre)
        else:
            import torch
            eng.set_timbre(torch.from_numpy(timbre))

    def _step(self, eng, state, age, absent) -> np.ndarray:
        return np.asarray(eng.step(state, age, absent), np.float32).reshape(-1)
