"""Real-time streaming engine: one 10-ms frame (480 samples) per call.

:class:`StreamingEngine` runs the generator and the decoder frame by frame with
*static* state — a ring-buffer key/value cache per block, fixed convolution
contexts and a fixed overlap-add buffer — so that on CUDA the whole frame can
be captured once in a CUDA graph and replayed with a single launch. The
arithmetic is the same as the offline path (same modules, same precision
policy), which the tests check: a streamed performance equals
:meth:`livesynth.LiveSynth.generate` on the same noise.

Typical use::

    synth = LiveSynth.from_pretrained()
    eng = synth.streaming_engine(timbre="keyboard_acoustic_004")
    tracker = NoteTracker()
    tracker.note_on(60, 100)
    while playing:
        state, age = tracker.frame()
        block = eng.step(state, age)          # np.float32 [480]

``set_timbre`` may be called between any two frames (per-frame morphing);
``absent=True`` marks the frame's MIDI condition as absent, so the model keeps
playing on its own.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from livesynth.constants import LEAD_IN_FRAMES
from livesynth.nn.backbone import LiveSynthBackbone, rotary_tables
from livesynth.nn.decoder import LatentDecoder

_LINEAR_TYPES = (nn.Linear,)


class StreamingEngine:
    """Frame-by-frame synthesiser with static state and optional CUDA graph.

    Args:
        backbone, decoder: trained modules (copied; the originals are untouched).
        device: torch device.
        precision: ``"bf16"`` (autocast, like offline rendering on CUDA) or ``"fp32"``.
        use_graph: capture the frame in a CUDA graph (CUDA only).
        max_pos: RoPE table length; the cache is re-based before it is reached,
            so sessions can run indefinitely.
        seed: noise seed (``None`` for random).
        forget_on_resume: when the MIDI condition returns after absent frames
            (the performer resumes after the model played on its own), remove the
            absent frames from the attention window. Without this the model keeps
            improvising on top of the resumed performance for up to one window
            (5 s): on 53 held-out instruments it doubled the number of notes and
            cut note F1 after resuming from 0.188 to 0.126; with it F1 is 0.199,
            and the audio matches a quiet rest again ~0.1 s after resuming.
    """

    def __init__(self, backbone: LiveSynthBackbone, decoder: LatentDecoder,
                 device: torch.device, precision: str = "bf16", use_graph: bool = True,
                 max_pos: int = 65536, seed: int | None = None,
                 forget_on_resume: bool = True) -> None:
        self.device = torch.device(device)
        self.forget_on_resume = forget_on_resume
        self.precision = precision
        self.use_graph = bool(use_graph and self.device.type == "cuda")
        self.bb = copy.deepcopy(backbone).to(self.device).eval()
        self.dec = copy.deepcopy(decoder).to(self.device).eval()
        if precision == "bf16":
            # Linear weights are exactly bf16-representable (released that way);
            # holding them in bf16 makes autocast's casts no-ops inside the graph.
            for m in self.bb.modules():
                if isinstance(m, _LINEAR_TYPES):
                    m.to(torch.bfloat16)
        cfg = self.bb.cfg
        self.window = cfg.attn_window
        self.hop = self.dec.hop
        self.n_fft = self.dec.vocos.head.n_fft
        self.noise_dim, self.timbre_dim = cfg.noise_dim, cfg.timbre_dim
        n_heads, dh = cfg.n_heads, cfg.d_model // cfg.n_heads
        d = self.device

        cos, sin = rotary_tables(torch.arange(max_pos, device=d), dh, cfg.rope_base)
        self._rope_cos, self._rope_sin = cos, sin
        self._rebase_at = (max_pos // 2 // self.window) * self.window
        self._rebase_by = (self._rebase_at * 5 // 6 // self.window) * self.window

        # graph inputs
        self.midi_buf = torch.zeros(1, 1, 128, dtype=torch.long, device=d)
        self.age_buf = torch.zeros(1, 1, 128, dtype=torch.long, device=d)
        self.absent_buf = torch.zeros(1, 1, dtype=torch.bool, device=d)
        self.noise_buf = torch.zeros(1, self.noise_dim, device=d)
        self.timbre_buf = torch.zeros(1, self.timbre_dim, device=d)
        self.cos_buf = torch.zeros(1, dh // 2, device=d)
        self.sin_buf = torch.zeros(1, dh // 2, device=d)
        self.slot_buf = torch.zeros(1, dtype=torch.long, device=d)
        self.bias_buf = torch.full((self.window,), float("-inf"), device=d)
        self._visible = np.zeros(self.window, bool)        # host mirror of bias_buf == 0
        self._slot_absent = np.zeros(self.window, bool)    # slot holds an absent frame
        self._prev_absent = False
        v_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.k_ring = [torch.zeros(1, n_heads, self.window, dh, device=d) for _ in self.bb.blocks]
        self.v_ring = [torch.zeros(1, n_heads, self.window, dh, dtype=v_dtype, device=d)
                       for _ in self.bb.blocks]
        vocos = self.dec.vocos
        dim, k = vocos.blocks[0].dwconv.conv.in_channels, vocos.blocks[0].dwconv.conv.kernel_size[0]
        self.ctx = [torch.zeros(1, dim, k - 1, device=d) for _ in vocos.blocks]
        self.ola = torch.zeros(1, self.n_fft, device=d)
        self.audio_out = torch.zeros(1, self.hop, device=d)

        self._gen = torch.Generator(device=d)
        if seed is not None:
            self._gen.manual_seed(int(seed))
        else:
            self._gen.seed()
        self._graph: Any = None
        self._frame = 0
        self.reset()

    # -- control ------------------------------------------------------------

    def set_timbre(self, timbre: torch.Tensor) -> None:
        """Timbre embedding [512] for the next frame(s)."""
        self.timbre_buf.copy_(F.normalize(timbre.float().reshape(1, -1), dim=-1))

    @torch.no_grad()
    def reset(self, seed: int | None = None) -> None:
        """Clear all streaming state (keeps the current timbre)."""
        for kr, vr in zip(self.k_ring, self.v_ring):
            kr.zero_()
            vr.zero_()
        self.bias_buf.fill_(float("-inf"))
        self._visible[:] = False
        self._slot_absent[:] = False
        self._prev_absent = False
        for c in self.ctx:
            c.zero_()
        self.ola.zero_()
        self._frame = 0
        self._lead_pending = True          # silent lead-in frames run before the next real frame
        if seed is not None:
            self._gen.manual_seed(int(seed))
        # Overlap-add warm-up: decode the same zero-latent frames the offline
        # decoder prepends, and discard their audio.
        vocos = self.dec.vocos
        z0 = torch.zeros(1, vocos.in_proj.conv.in_channels, 1, device=self.device)
        for _ in range(vocos.warmup):
            self._decode(z0)

    # -- in-graph maths --------------------------------------------------------

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                              enabled=self.precision == "bf16", cache_enabled=False)

    def _decode(self, zp: torch.Tensor) -> None:
        """Projected latent [1, d_latent, 1] -> one hop into ``audio_out``."""
        vocos = self.dec.vocos
        x = vocos.in_norm(vocos.in_proj.conv(zp).transpose(1, 2))           # [1, 1, dim]
        for blk, ctx in zip(vocos.blocks, self.ctx):
            ext = torch.cat([ctx, x.transpose(1, 2)], dim=-1)                # [1, dim, k]
            h = blk.dwconv.conv(ext).transpose(1, 2)
            ctx.copy_(ext[..., 1:])
            x = x + blk._mlp(h)
        head = vocos.head
        frame = head._frames(vocos.final_norm(x))[:, :, 0]                  # [1, n_fft]
        acc = self.ola + frame
        self.audio_out.copy_((acc[:, : self.hop] / head.env).clamp(-1.0, 1.0))
        self.ola.copy_(F.pad(acc[:, self.hop:], (0, self.hop)))

    def _frame_step(self) -> None:
        bb = self.bb
        with self._autocast():
            x = bb.noise_proj(self.noise_buf).unsqueeze(1) + bb.midi_enc(
                self.midi_buf, self.absent_buf, self.age_buf)
            cond = bb.cond_proj(self.timbre_buf)
            for blk, kr, vr in zip(bb.blocks, self.k_ring, self.v_ring):
                x = blk.step_ring(x, cond, kr, vr, self.slot_buf, self.bias_buf,
                                  self.cos_buf, self.sin_buf)
            z = bb.head(bb.final_norm(x)).transpose(1, 2)                   # [1, latent, 1]
        self._decode(self.dec._project(z.float()))

    @torch.no_grad()
    def _capture(self) -> None:
        snap = ([k.clone() for k in self.k_ring], [v.clone() for v in self.v_ring],
                [c.clone() for c in self.ctx], self.ola.clone(), self.bias_buf.clone())

        def restore():
            for dst, src in zip(self.k_ring + self.v_ring + self.ctx + [self.ola, self.bias_buf],
                                snap[0] + snap[1] + snap[2] + [snap[3], snap[4]]):
                dst.copy_(src)

        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            for _ in range(3):                     # autotune outside the capture
                self._frame_step()
        torch.cuda.current_stream(self.device).wait_stream(s)
        restore()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._frame_step()
        restore()

    @torch.no_grad()
    def _rebase(self) -> None:
        """Rotate cached keys back by Δ positions and rewind the frame counter.
        Attention scores depend only on position differences, so this is exact
        up to rounding; Δ is a multiple of the window so ring slots stay aligned."""
        dlt = self._rebase_by
        c = self._rope_cos[dlt][None, None, None]
        s = -self._rope_sin[dlt][None, None, None]
        h = c.shape[-1]
        for kr in self.k_ring:
            x1, x2 = kr[..., :h].clone(), kr[..., h:].clone()
            kr[..., :h].copy_(x1 * c - x2 * s)
            kr[..., h:].copy_(x1 * s + x2 * c)
        self._frame -= dlt

    # -- per frame ---------------------------------------------------------------

    @torch.no_grad()
    def step_tensor(self, state: torch.Tensor, age: torch.Tensor, absent: bool = False,
                    noise: torch.Tensor | None = None) -> torch.Tensor:
        """One frame from device tensors ``state``/``age`` [128]. Returns audio [1, 480]
        on the device (valid until the next call)."""
        if self._lead_pending:             # same silent frames as offline rendering, audio dropped
            self._lead_pending = False
            none = torch.zeros(128, dtype=torch.long, device=self.device)
            quiet = torch.zeros(self.noise_dim, device=self.device)
            for _ in range(LEAD_IN_FRAMES):
                self._advance(none, none, False, quiet)
        return self._advance(state, age, absent, noise)

    def _advance(self, state: torch.Tensor, age: torch.Tensor, absent: bool,
                 noise: torch.Tensor | None) -> torch.Tensor:
        if self._frame >= self._rebase_at:
            self._rebase()
        pos = self._frame
        slot = pos % self.window
        self.midi_buf.copy_(state.reshape(1, 1, -1))
        self.age_buf.copy_(age.reshape(1, 1, -1))
        self.absent_buf.fill_(absent)
        if noise is None:
            torch.randn(self.noise_buf.shape, generator=self._gen, device=self.device,
                        out=self.noise_buf)
        else:
            self.noise_buf.copy_(noise.reshape(1, -1))
        self.cos_buf.copy_(self._rope_cos[pos:pos + 1])
        self.sin_buf.copy_(self._rope_sin[pos:pos + 1])
        self.slot_buf.fill_(slot)
        absent = bool(absent)
        if self.forget_on_resume and self._prev_absent and not absent:
            hide = np.where(self._visible & self._slot_absent)[0]
            if len(hide):
                self.bias_buf[torch.from_numpy(hide).to(self.device)] = float("-inf")
                self._visible[hide] = False
        if not self._visible[slot]:                       # this slot now holds the current frame
            self.bias_buf[slot] = 0.0
            self._visible[slot] = True
        self._slot_absent[slot] = absent
        self._prev_absent = absent
        if self.use_graph:
            if self._graph is None:
                self._capture()
            self._graph.replay()
        else:
            self._frame_step()
        self._frame += 1
        return self.audio_out

    def step(self, state: np.ndarray, age: np.ndarray, absent: bool = False) -> np.ndarray:
        """One frame from NumPy ``state``/``age`` [128] (e.g. from
        :class:`livesynth.NoteTracker`). Returns ``float32`` audio [480]."""
        t = torch.from_numpy(np.asarray(state, dtype=np.int64)).to(self.device, non_blocking=True)
        a = torch.from_numpy(np.asarray(age, dtype=np.int64)).to(self.device, non_blocking=True)
        return self.step_tensor(t, a, absent).reshape(-1).cpu().numpy()
