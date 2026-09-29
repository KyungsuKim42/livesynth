"""LiveSynth: a real-time streaming neural instrument.

>>> from livesynth import LiveSynth
>>> synth = LiveSynth.from_pretrained()
>>> audio = synth.render("song.mid", synth.embed_audio("cello.wav"))

Heavy dependencies are imported lazily, so ``livesynth.mlx_engine`` and
``livesynth.midi.NoteTracker`` work without PyTorch.
"""

from __future__ import annotations

__version__ = "0.1.0"

_LAZY = {
    "LiveSynth": ("livesynth.synth", "LiveSynth"),
    "StreamingEngine": ("livesynth.stream", "StreamingEngine"),
    "NoteTracker": ("livesynth.midi", "NoteTracker"),
    "load_notes": ("livesynth.midi", "load_notes"),
    "notes_to_frames": ("livesynth.midi", "notes_to_frames"),
    "slerp": ("livesynth.timbre", "slerp"),
}
__all__ = [*_LAZY, "__version__"]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module 'livesynth' has no attribute {name!r}")
