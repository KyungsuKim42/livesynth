"""Timbre conditioning: CLAP embeddings from audio or text, and interpolation.

LiveSynth is conditioned on a 512-d LAION-CLAP embedding
(``music_audioset_epoch_15_esc_90.14``). An embedding can come from

* a reference recording (:meth:`TimbreEncoder.embed_audio`) — zero-shot
  instrument cloning, or
* a text prompt (:meth:`TimbreEncoder.embed_text`) — text-to-instrument. CLAP
  text and audio embeddings occupy different regions of the shared space (the
  modality gap). The raw text embedding is used by default; an orthogonal
  Procrustes rotation fitted on NSynth can map it onto the audio region.

CLAP consumes at most 10 s; longer references are cropped to the loudest 10-s
window (the model was trained on 10-s references of the playing instrument).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

CLAP_SR = 48_000
CLAP_SECONDS = 10.0


def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
    """Spherical interpolation between unit vectors ``a`` and ``b`` [..., D].

    ``t`` may be a scalar or a tensor broadcastable to ``a[..., :1]`` (for
    example ``[N, 1]`` for a per-frame path).
    """
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    t = torch.as_tensor(t, dtype=torch.float32, device=a.device)
    dot = (a * b).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    th = torch.acos(dot)
    return (torch.sin((1 - t) * th) * a + torch.sin(t * th) * b) / torch.sin(th)


def _load_audio(audio: Any, sr: int | None) -> np.ndarray:
    """Path / array -> mono float32 at 48 kHz."""
    if isinstance(audio, (str, Path)):
        import soundfile as sf
        x, sr = sf.read(str(audio), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    else:
        x = audio.detach().cpu().numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio)
        x = x.astype(np.float32)
        if x.ndim == 2:                                   # [C, T] or [T, C]
            x = x.mean(axis=0 if x.shape[0] < x.shape[1] else 1)
        if sr is None:
            raise ValueError("sample rate `sr` is required for array input")
    if sr != CLAP_SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), CLAP_SR)
        x = resample_poly(x, CLAP_SR // g, int(sr) // g).astype(np.float32)
    return x


def _crop(x: np.ndarray, mode: str) -> np.ndarray:
    n = int(CLAP_SECONDS * CLAP_SR)
    if len(x) <= n:
        return x
    if mode == "start":
        return x[:n]
    if mode == "center":
        s = (len(x) - n) // 2
        return x[s: s + n]
    if mode == "loudest":                     # 10-s window with the most energy, 0.1-s hop
        c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
        starts = np.arange(0, len(x) - n + 1, CLAP_SR // 10)
        s = int(starts[np.argmax(c[starts + n] - c[starts])])
        return x[s: s + n]
    raise ValueError(f"unknown crop mode {mode!r}")


class TimbreEncoder:
    """Frozen LAION-CLAP wrapper. Created lazily by :class:`livesynth.LiveSynth`."""

    def __init__(self, ckpt_path: str, device: torch.device,
                 text_align: dict[str, torch.Tensor] | None = None) -> None:
        import laion_clap
        self.device = device
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.clap.load_ckpt(ckpt_path, verbose=False)
        self.clap = self.clap.to(device).eval()
        self.text_align = {k: v.to(device) for k, v in (text_align or {}).items()}

    @torch.no_grad()
    def embed_audio(self, audio: Any, sr: int | None = None, crop: str = "loudest"
                    ) -> torch.Tensor:
        """Reference recording (path or array) -> unit-norm embedding [512]."""
        x = _crop(_load_audio(audio, sr), crop)
        t = torch.from_numpy(x).to(self.device)[None]
        e = self.clap.get_audio_embedding_from_data(x=t, use_tensor=True)
        return F.normalize(e.float(), dim=-1)[0]

    @torch.no_grad()
    def embed_text(self, prompt: str, align: str = "none") -> torch.Tensor:
        """Text prompt -> unit-norm embedding [512].

        ``align``: ``"none"`` (the raw CLAP text embedding, default) or
        ``"procrustes"`` (orthogonal rotation fitted on NSynth prompt/audio pairs).
        """
        e = self.clap.get_text_embedding([prompt], use_tensor=True).float()
        e = F.normalize(e, dim=-1)[0]
        if align == "none":
            return e
        if align == "procrustes":
            return F.normalize(e @ self.text_align["rotation"], dim=-1)
        raise ValueError(f"unknown align mode {align!r}; use 'procrustes' or 'none'")
