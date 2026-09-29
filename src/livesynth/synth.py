"""High-level offline inference API."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from livesynth.hub import DEFAULT_REPO, resolve_clap_checkpoint, resolve_model_dir
from livesynth.midi import FRAME_RATE, MidiLike, load_notes, notes_end, notes_to_frames
from livesynth.nn.backbone import BackboneConfig, LiveSynthBackbone
from livesynth.nn.decoder import LatentDecoder
from livesynth.timbre import TimbreEncoder, slerp

TimbreLike = Any   # torch.Tensor [512] | np.ndarray [512] | preset name | audio file path


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LiveSynth:
    """A streaming neural instrument: MIDI + timbre -> 48-kHz audio.

    Create with :meth:`from_pretrained`. All rendering methods return mono
    ``float32`` NumPy arrays at :attr:`sample_rate`.

    Timbre arguments accept a CLAP embedding (tensor or array of shape [512]),
    the name of a built-in preset (see :attr:`presets`), or the path of a
    reference recording, which is embedded with :meth:`embed_audio`. Use
    :meth:`embed_text` for text prompts.
    """

    def __init__(self, backbone: LiveSynthBackbone, decoder: LatentDecoder, config: dict,
                 presets: torch.Tensor, text_align: dict[str, torch.Tensor],
                 device: torch.device, precision: str = "auto",
                 cache_dir: str | os.PathLike | None = None) -> None:
        self.backbone = backbone.eval()
        self.decoder = decoder.eval()
        self.config = config
        self.device = device
        self._preset_names: list[str] = list(config.get("presets", []))
        self._presets = presets.to(device)
        self._text_align = text_align
        self._cache_dir = cache_dir
        self._timbre_encoder: TimbreEncoder | None = None
        if precision == "auto":
            precision = "bf16" if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else "fp32"
        if precision not in ("bf16", "fp32"):
            raise ValueError("precision must be 'auto', 'bf16' or 'fp32'")
        self.precision = precision

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_pretrained(cls, repo_id: str = DEFAULT_REPO, *, device: str | torch.device | None = None,
                        revision: str | None = None, local_dir: str | os.PathLike | None = None,
                        cache_dir: str | os.PathLike | None = None,
                        precision: str = "auto") -> "LiveSynth":
        """Load the released model, downloading the weights on first use.

        Args:
            repo_id: Hugging Face model repository.
            device: ``"cuda"``, ``"mps"``, ``"cpu"``; default picks the best available.
            local_dir: load from a local directory instead of the Hub
                (also settable with the ``LIVESYNTH_WEIGHTS`` environment variable).
            precision: ``"bf16"`` autocast (default on CUDA) or ``"fp32"``.
        """
        from safetensors.torch import load_file
        dev = torch.device(device) if device is not None else _auto_device()
        d = resolve_model_dir(repo_id, revision=revision, local_dir=local_dir, cache_dir=cache_dir)
        cfg = json.loads((Path(d) / "config.json").read_text())

        backbone = LiveSynthBackbone(BackboneConfig(**cfg["backbone"]))
        sd = {k: v.float() for k, v in load_file(str(Path(d) / "generator.safetensors")).items()}
        backbone.load_state_dict(sd)

        dc = cfg["decoder"]
        decoder = LatentDecoder(latent_dim=dc["latent_dim"], d_latent=dc["d_latent"], dim=dc["dim"],
                                depth=dc["depth"], intermediate_mult=dc["intermediate_mult"],
                                kernel_size=dc["kernel_size"], n_fft=dc["n_fft"],
                                hop=cfg["hop_length"])
        decoder.load_state_dict(LatentDecoder.remap_state_dict(
            load_file(str(Path(d) / "decoder.safetensors"))))

        presets = load_file(str(Path(d) / "presets.safetensors"))["embeddings"]
        text_align = load_file(str(Path(d) / "text_align.safetensors"))
        for p in list(backbone.parameters()) + list(decoder.parameters()):
            p.requires_grad_(False)
        return cls(backbone.to(dev), decoder.to(dev), cfg, presets, text_align, dev,
                   precision=precision, cache_dir=cache_dir)

    # -- properties -----------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return int(self.config["sample_rate"])

    @property
    def frame_rate(self) -> int:
        return int(self.config["frame_rate"])

    @property
    def hop_length(self) -> int:
        return int(self.config["hop_length"])

    @property
    def presets(self) -> list[str]:
        """Names of the built-in timbre presets (held-out NSynth instruments)."""
        return list(self._preset_names)

    # -- timbre ---------------------------------------------------------------

    def _encoder(self) -> TimbreEncoder:
        if self._timbre_encoder is None:
            self._timbre_encoder = TimbreEncoder(resolve_clap_checkpoint(self._cache_dir),
                                                 self.device, self._text_align)
        return self._timbre_encoder

    def preset(self, name: str) -> torch.Tensor:
        """Embedding of a built-in preset [512]."""
        try:
            return self._presets[self._preset_names.index(name)]
        except ValueError:
            raise KeyError(f"unknown preset {name!r}; see LiveSynth.presets") from None

    def embed_audio(self, audio: Any, sr: int | None = None, crop: str = "loudest") -> torch.Tensor:
        """Reference recording (file path or array with ``sr``) -> timbre embedding [512].

        Recordings longer than 10 s are cropped (``crop``: ``"loudest"``,
        ``"center"`` or ``"start"``). A few seconds of the instrument playing
        on its own works best."""
        return self._encoder().embed_audio(audio, sr=sr, crop=crop)

    def embed_text(self, prompt: str, align: str = "procrustes") -> torch.Tensor:
        """Text prompt (e.g. ``"the sound of an acoustic string"``) -> timbre embedding [512].

        ``align`` maps the CLAP text embedding toward the audio embeddings the
        model was trained on: ``"procrustes"`` (default), ``"translation"`` or
        ``"none"``. Text prompts reliably select the instrument family; the
        finer timbre is better specified with a reference recording."""
        return self._encoder().embed_text(prompt, align=align)

    def timbre(self, t: TimbreLike) -> torch.Tensor:
        """Resolve any timbre argument to a unit-norm embedding [512] on the device."""
        if isinstance(t, torch.Tensor) or isinstance(t, np.ndarray):
            e = torch.as_tensor(t, dtype=torch.float32, device=self.device)
            if e.shape[-1] != self.config["backbone"]["timbre_dim"]:
                raise ValueError(f"timbre embedding must have size "
                                 f"{self.config['backbone']['timbre_dim']}, got {tuple(e.shape)}")
            return F.normalize(e, dim=-1)
        if isinstance(t, (str, Path)):
            if isinstance(t, str) and t in self._preset_names:
                return self.preset(t)
            if Path(t).exists():
                return self.embed_audio(t)
            raise ValueError(f"{t!r} is neither a preset name nor an existing audio file "
                             f"(use embed_text() for text prompts)")
        raise TypeError(f"unsupported timbre type {type(t).__name__}")

    # -- low-level generation -------------------------------------------------

    def _autocast(self):
        if self.precision == "bf16":
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return torch.autocast(device_type=self.device.type, enabled=False)

    def _noise(self, n_frames: int, seed: int | None) -> torch.Tensor:
        g = None
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(int(seed))
        return torch.randn(n_frames, self.backbone.cfg.noise_dim, device=self.device, generator=g)

    @torch.no_grad()
    def generate(self, midi_state: torch.Tensor, midi_age: torch.Tensor, timbre: torch.Tensor,
                 midi_absent: torch.Tensor | None = None, noise: torch.Tensor | None = None,
                 seed: int | None = 0) -> torch.Tensor:
        """Low-level batched generation.

        Args:
            midi_state:  [B, N, 128] long (see :func:`livesynth.midi.notes_to_frames`)
            midi_age:    [B, N, 128] long
            timbre:      [B, 512] or per-frame [B, N, 512]
            midi_absent: [B, N] bool; frames where the MIDI condition is absent
                         (the model then continues on its own)
            noise:       [B, N, 128]; drawn from ``seed + i`` per item when omitted
        Returns:
            audio [B, N * hop_length] on the device (float32)
        """
        b, n = midi_state.shape[:2]
        if noise is None:
            noise = torch.stack([self._noise(n, None if seed is None else seed + i) for i in range(b)])
        with self._autocast():
            z = self.backbone(noise.to(self.device), midi_state.to(self.device),
                              timbre.to(self.device),
                              midi_absent=None if midi_absent is None else midi_absent.to(self.device),
                              midi_age=midi_age.to(self.device))
        return self.decoder(z.float())

    def _frames_for(self, notes: np.ndarray, duration: float | None, tail: float) -> int:
        seconds = duration if duration is not None else notes_end(notes) + tail
        return max(1, int(math.ceil(seconds * self.frame_rate - 1e-6)))

    # -- rendering ------------------------------------------------------------

    def render(self, midi: MidiLike, timbre: TimbreLike, duration: float | None = None,
               tail: float = 1.0, seed: int | None = 0) -> np.ndarray:
        """Render a MIDI performance with one timbre.

        Args:
            midi: MIDI file path, ``pretty_midi.PrettyMIDI``, or note array
                ``[K, 4]`` of ``(start_s, end_s, pitch, velocity)``.
            timbre: embedding, preset name, or reference audio path.
            duration: output length in seconds (default: last note-off + ``tail``).
            seed: noise seed; ``None`` for a random draw.
        """
        return self.render_batch([midi], [timbre], durations=[duration], tail=tail, seed=seed)[0]

    def render_batch(self, midis: Sequence[MidiLike], timbres: Sequence[TimbreLike] | TimbreLike,
                     durations: Sequence[float | None] | None = None, tail: float = 1.0,
                     seed: int | None = 0, batch_size: int = 8) -> list[np.ndarray]:
        """Render many performances; item ``i`` uses noise seed ``seed + i``.

        ``timbres`` is either one timbre for all items or one per item. Items
        are padded to the longest in each mini-batch; since the model is
        causal, padding never changes the audio of shorter items.

        Repeated calls are deterministic. In ``bf16`` precision, however, GPU
        kernels depend on the batch size, so an item rendered in a batch differs
        slightly from the same item rendered alone (about 30 dB SNR); with
        ``precision="fp32"`` the two agree to numerical precision."""
        items = list(midis)
        if not isinstance(timbres, (list, tuple)):
            timbres = [timbres] * len(items)
        if len(timbres) != len(items):
            raise ValueError("timbres must be a single timbre or one per MIDI item")
        durations = list(durations) if durations is not None else [None] * len(items)
        notes = [load_notes(m) for m in items]
        frames = [self._frames_for(nt, d, tail) for nt, d in zip(notes, durations)]
        embs = [self.timbre(t) for t in timbres]
        out: list[np.ndarray] = [np.zeros(0, np.float32)] * len(items)
        for b0 in range(0, len(items), batch_size):
            idx = list(range(b0, min(len(items), b0 + batch_size)))
            n = max(frames[i] for i in idx)
            conds = [notes_to_frames(notes[i], n, self.frame_rate) for i in idx]
            state = torch.stack([c[0] for c in conds])
            age = torch.stack([c[1] for c in conds])
            noise = torch.stack([self._noise(n, None if seed is None else seed + i) for i in idx])
            audio = self.generate(state, age, torch.stack([embs[i] for i in idx]), noise=noise)
            for j, i in enumerate(idx):
                out[i] = audio[j, : frames[i] * self.hop_length].cpu().numpy()
        return out

    def render_timbre_path(self, midi: MidiLike, path: torch.Tensor | Callable[[np.ndarray], Any],
                           duration: float | None = None, tail: float = 1.0,
                           seed: int | None = 0) -> np.ndarray:
        """Render with a timbre that changes every frame.

        ``path`` is a ``[N, 512]`` tensor of per-frame embeddings (its length
        sets the output length unless ``duration`` is given), or a function
        mapping frame times in seconds ``[N]`` to such a tensor."""
        notes = load_notes(midi)
        if isinstance(path, torch.Tensor):
            # The tensor defines the length unless ``duration`` is given; the
            # last embedding is held if the output is longer than the path.
            if path.dim() != 2:
                raise ValueError("a tensor timbre path must have shape [N, 512]")
            n = path.shape[0] if duration is None else self._frames_for(notes, duration, tail)
            emb = F.normalize(path.float().to(self.device), dim=-1)
            if emb.shape[0] < n:
                emb = torch.cat([emb, emb[-1:].expand(n - emb.shape[0], -1)])
            emb = emb[:n]
        else:
            n = self._frames_for(notes, duration, tail)
            times = np.arange(n) / self.frame_rate
            emb = F.normalize(torch.as_tensor(path(times), dtype=torch.float32,
                                              device=self.device), dim=-1)
        state, age = notes_to_frames(notes, n, self.frame_rate)
        audio = self.generate(state[None], age[None], emb[None],
                              noise=self._noise(n, seed)[None])
        return audio[0].cpu().numpy()

    def morph(self, midi: MidiLike, timbre_a: TimbreLike, timbre_b: TimbreLike,
              start: float = 0.0, end: float | None = None, duration: float | None = None,
              tail: float = 1.0, seed: int | None = 0) -> np.ndarray:
        """Morph from ``timbre_a`` to ``timbre_b`` while playing ``midi``.

        The embedding stays at A until ``start`` seconds, moves along the
        spherical path to B until ``end`` (default: end of the output), and
        stays at B afterwards. The timbre is updated every 10-ms frame."""
        a, b = self.timbre(timbre_a), self.timbre(timbre_b)
        notes = load_notes(midi)
        total = self._frames_for(notes, duration, tail) / self.frame_rate
        stop = total if end is None else end
        if stop <= start:
            raise ValueError("morph end must be after start")

        def path(times: np.ndarray) -> torch.Tensor:
            t = torch.as_tensor(np.clip((times - start) / (stop - start), 0.0, 1.0),
                                dtype=torch.float32, device=self.device)[:, None]
            return slerp(a[None], b[None], t)

        return self.render_timbre_path(notes, path, duration=total, tail=tail, seed=seed)

    def continue_performance(self, midi: MidiLike | None, timbre: TimbreLike,
                             prefix: float | None = None, length: float = 5.0,
                             seed: int | None = 0) -> np.ndarray:
        """Play ``midi`` up to ``prefix`` seconds, then keep playing without MIDI.

        After ``prefix`` (default: the last note-off) the MIDI condition is
        marked *absent* and the model continues the performance on its own for
        ``length`` seconds, following the key and texture of what came before.
        With ``midi=None`` the model improvises from the start. The returned
        audio covers prefix and continuation."""
        notes = load_notes(midi) if midi is not None else np.zeros((0, 4))
        cut = notes_end(notes) if prefix is None else float(prefix)
        n_pre = int(round(cut * self.frame_rate))
        n = n_pre + int(round(length * self.frame_rate))
        state, age = notes_to_frames(notes, n, self.frame_rate)
        absent = torch.zeros(1, n, dtype=torch.bool)
        absent[:, n_pre:] = True
        audio = self.generate(state[None], age[None], self.timbre(timbre)[None],
                              midi_absent=absent, noise=self._noise(n, seed)[None])
        return audio[0].cpu().numpy()
