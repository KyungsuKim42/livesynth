# Python API

All methods live on `LiveSynth` (`synth = LiveSynth.from_pretrained()`).

| Method | What it does |
|---|---|
| `render(midi, timbre)` | Render a MIDI performance in one timbre. |
| `render_batch(midis, timbres)` | Render many items in mini-batches (item `i` uses seed `seed + i`). |
| `morph(midi, timbre_a, timbre_b, start, end)` | Move from one timbre to another along the spherical path, updated every frame. |
| `render_timbre_path(midi, path)` | Render with an arbitrary per-frame timbre path (`[N, 512]` tensor or a function of time). |
| `embed_audio(path_or_array, sr)` | CLAP embedding of a reference recording (the loudest 10 s are used). |
| `embed_text(prompt, align)` | CLAP text embedding, raw by default (`align="none"`) or rotated toward the audio embeddings with an orthogonal Procrustes map (`align="procrustes"`). |
| `presets`, `preset(name)` | The 53 built-in timbres (held-out NSynth instruments). |
| `preset_audio(name)` | The 10-s reference recording a preset was computed from (downloaded on first use). |
| `streaming_engine(timbre)` | A frame-by-frame engine for real-time use (see below). |
| `generate(midi_state, midi_age, timbre, midi_absent)` | Low-level batched generation on frame-level conditions. |

## Inputs and outputs

* **MIDI**: a file path, a `pretty_midi.PrettyMIDI` object, or a note array of
  shape `[K, 4]` with rows `(start_s, end_s, pitch, velocity)`. Drum tracks are
  ignored and the sustain pedal extends notes.
* **Timbre**: an embedding, a preset name, or the path of a reference recording.
* **Audio**: mono `float32` arrays at 48 kHz.

## Streaming

`synth.streaming_engine()` (PyTorch, with a CUDA graph on NVIDIA GPUs) and
`livesynth.mlx_engine.MLXStreamingEngine` (Apple Silicon) expose the same
interface: `step(state, age, absent) -> 480 samples`, one 10-ms frame per call.
Feed them from `livesynth.NoteTracker`:

```python
from livesynth import LiveSynth, NoteTracker

synth = LiveSynth.from_pretrained()
engine = synth.streaming_engine(timbre="keyboard_acoustic_004")
tracker = NoteTracker()
tracker.note_on(60, 100)
block = engine.step(*tracker.frame())       # float32 [480]
```

`engine.set_timbre(embedding)` may be called between any two frames. To check
whether a machine keeps up in real time, run `python -m livesynth.live.bench`.
