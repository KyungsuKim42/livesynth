"""LiveSynth: a real-time streaming neural instrument.

>>> from livesynth import LiveSynth
>>> synth = LiveSynth.from_pretrained()
>>> audio = synth.render("song.mid", synth.embed_audio("cello.wav"))
"""

from livesynth.midi import NoteTracker, load_notes, notes_to_frames
from livesynth.synth import LiveSynth
from livesynth.timbre import slerp

__version__ = "0.1.0"
__all__ = ["LiveSynth", "NoteTracker", "load_notes", "notes_to_frames", "slerp", "__version__"]
