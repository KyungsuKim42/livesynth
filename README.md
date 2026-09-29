# LiveSynth

**A streaming neural synthesizer for instrument cloning and text-to-instrument.**

LiveSynth turns MIDI into 48-kHz audio in the timbre of any instrument, given
either a short reference recording or a text prompt. It generates one 10-ms
frame at a time with no lookahead, so the same model that renders offline also
runs as a playable real-time instrument. Because the timbre condition can
change every frame and the MIDI condition can be withdrawn, the model also
morphs smoothly between instruments and keeps playing on its own when the
performer stops.

> Kyungsu Kim, Yejin Kim, Kyogu Lee (Seoul National University).
> *LiveSynth: A Streaming Neural Synthesizer for Instrument Cloning and Text-to-Instrument.*

## Installation

```bash
pip install git+https://github.com/KyungsuKim42/livesynth.git
```

Python 3.10 or newer and PyTorch 2.1 or newer are required. A CUDA GPU is
recommended for fast offline rendering; Apple Silicon (MPS) and CPU also work.
The model weights (about 400 MB) and the CLAP timbre encoder are downloaded
from the Hugging Face Hub the first time the model is loaded.

## Quickstart

```python
import soundfile as sf
from livesynth import LiveSynth

synth = LiveSynth.from_pretrained()

# Zero-shot instrument cloning from a reference recording
timbre = synth.embed_audio("cello.wav")
audio = synth.render("song.mid", timbre)
sf.write("cello_song.wav", audio, synth.sample_rate)

# Text-to-instrument
audio = synth.render("song.mid", synth.embed_text("the sound of an acoustic string"))

# Built-in presets (held-out NSynth instruments)
print(synth.presets)
audio = synth.render("song.mid", "keyboard_acoustic_004")
```

Or run the bundled script, which writes one example of every feature:

```bash
python quickstart.py song.mid [reference.wav]
```

## Features

| Method | What it does |
|---|---|
| `render(midi, timbre)` | Render a MIDI performance in one timbre. |
| `render_batch(midis, timbres)` | Render many items in mini-batches (item `i` uses seed `seed + i`). |
| `morph(midi, timbre_a, timbre_b, start, end)` | Move from one timbre to another along the spherical path, updated every frame. |
| `render_timbre_path(midi, path)` | Render with an arbitrary per-frame timbre path (`[N, 512]` tensor or a function of time). |
| `continue_performance(midi, timbre, prefix, length)` | Play the MIDI up to `prefix` seconds, then continue autonomously for `length` seconds. With `midi=None` the model improvises. |
| `embed_audio(path_or_array, sr)` | CLAP embedding of a reference recording (the loudest 10 s are used). |
| `embed_text(prompt, align)` | CLAP text embedding mapped onto the audio region (`"procrustes"`, `"translation"` or `"none"`). |
| `generate(midi_state, midi_age, timbre, midi_absent)` | Low-level batched generation on frame-level conditions. |

MIDI inputs can be a file path, a `pretty_midi.PrettyMIDI` object, or a note
array of shape `[K, 4]` with rows `(start_s, end_s, pitch, velocity)`. Timbre
inputs can be an embedding, a preset name, or the path of a reference recording.
All outputs are mono `float32` arrays at 48 kHz.

## How it works

* **Causal VAE codec.** Audio is represented as 128-dimensional continuous
  latents at 100 Hz (48 kHz, 480-sample hop). Only the lightweight causal
  decoder (8 M parameters) is needed for synthesis.
* **Feedback-free generator.** A 190 M-parameter causal Transformer maps
  per-frame Gaussian noise, MIDI and timbre to latents. It never reads its own
  previous output, so it runs as one parallel pass offline or frame by frame
  with a bounded 5-s key/value cache in real time; both paths compute exactly
  the same function.
* **Conditioning.** MIDI enters once at the input as a per-pitch state
  embedding with note velocity and note age; timbre (a LAION-CLAP embedding)
  modulates every block through adaLN-Zero and may differ per frame. A learned
  *absent* MIDI embedding lets the model continue without MIDI.

## Roadmap

- [x] Offline inference API (rendering, batching, morphing, continuation)
- [ ] Automatic weight download from the Hugging Face Hub
- [ ] Real-time GUI with MIDI controller and computer-keyboard input
- [ ] VST3 / AU plug-in and standalone app

See [docs/ROADMAP.md](docs/ROADMAP.md) for details.

## License

The code is released under the MIT License. The license of the model weights
will be announced with the public release.
