"""LiveSynth quickstart: clone a timbre, describe one in text, render in batch and morph.

    python quickstart.py path/to/song.mid [path/to/reference.wav]

Outputs are written to ./outputs. The first run downloads the model weights
(about 400 MB) and the CLAP timbre encoder from the Hugging Face Hub.
"""

import sys
from pathlib import Path

import soundfile as sf

from livesynth import LiveSynth


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    midi = sys.argv[1]
    reference = sys.argv[2] if len(sys.argv) > 2 else None
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    synth = LiveSynth.from_pretrained()
    print(f"device={synth.device}, {len(synth.presets)} presets, e.g. {synth.presets[:3]}")

    # 1. Zero-shot instrument cloning: timbre from a reference recording
    #    (or a built-in preset when no reference is given).
    timbre = synth.embed_audio(reference) if reference else synth.preset("keyboard_acoustic_004")
    audio = synth.render(midi, timbre, seed=0)
    sf.write(out / "render.wav", audio, synth.sample_rate)

    # 2. Text-to-instrument.
    strings = synth.embed_text("the sound of an acoustic string")
    sf.write(out / "text_strings.wav", synth.render(midi, strings, seed=0), synth.sample_rate)

    # 3. Batch rendering: several timbres for the same MIDI.
    names = ["string_acoustic_014", "organ_electronic_057", "brass_acoustic_059"]
    for name, a in zip(names, synth.render_batch([midi] * len(names), names, seed=0)):
        sf.write(out / f"batch_{name}.wav", a, synth.sample_rate)

    # 4. Timbre morphing: from the piano preset to a plucked-string preset
    #    between 3 s and 13 s, updated every 10-ms frame.
    morph = synth.morph(midi, "keyboard_acoustic_004", "string_acoustic_056", start=3.0, end=13.0, seed=0)
    sf.write(out / "morph.wav", morph, synth.sample_rate)

    print(f"wrote {sorted(p.name for p in out.glob('*.wav'))} to {out}/")


if __name__ == "__main__":
    main()
