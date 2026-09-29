# Roadmap

## Phase 1 — Offline inference package (done)

* `livesynth` Python package with `LiveSynth.from_pretrained()`.
* `render`, `render_batch`, `morph`, `render_timbre_path`, `continue_performance`,
  `embed_audio`, `embed_text`, low-level `generate`.
* Parallel pass with chunked banded attention (memory linear in length), and
  a frame-level `step` API on the same weights for streaming.
* Tests: parallel pass equals frame-by-frame streaming; causality; decoder
  streaming equals the full pass; offline MIDI conversion equals the live
  `NoteTracker`; released weights reproduce the research implementation
  bit-exactly in bf16.

## Phase 2 — Weight distribution

* Private Hugging Face repository `KyungsuKim/LiveSynth` with
  `config.json`, `generator.safetensors` (Linear weights bf16, rest fp32),
  `decoder.safetensors`, `text_align.safetensors`, `presets.safetensors`.
* Download on first use via `huggingface_hub.snapshot_download`; local override
  with `LIVESYNTH_WEIGHTS`.

## Phase 3 — Real-time instrument (Python)

* Streaming engine: backbone `step` with a ring-buffer KV cache + streaming
  decoder, one 480-sample block per 10 ms.
  * CUDA path: static shapes + CUDA graphs.
  * Apple Silicon path: MLX engine with int8 weights.
* GUI (`livesynth-live`): MIDI device input (mido / python-rtmidi) and
  computer-keyboard input, preset browser, drag-and-drop reference audio, text
  prompt, A/B morph slider, "keep playing" (continuation) toggle, level meter,
  latency / underrun display.

## Phase 4 — Plug-in and standalone app

* JUCE (C++) project producing VST3, AU and a standalone app.
* Inference: export the per-frame step (backbone + decoder) to an `.mlxfn`
  function with MLX and call it from C++ on Apple Silicon; a cross-platform
  backend (ONNX Runtime or LibTorch) for Windows / Linux follows.
* Timbre in the plug-in: presets and user embeddings computed by the Python
  package first; an exported CLAP audio encoder for drag-and-drop later.
* UI: preset / reference slot pair with a morph knob, text prompt field,
  continuation toggle, output level, host MIDI input.
