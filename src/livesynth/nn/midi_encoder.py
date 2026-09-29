"""Per-pitch MIDI state encoder.

Every 10-ms frame is described by one categorical state per MIDI pitch:

    0          none
    1          note-off in this frame
    2 .. 6     note-on in this frame, velocity level 0..4
    7 .. 11    note held (sustain), velocity level 0..4

Velocity is quantised to the five levels ``VEL_LEVELS``. Each (pitch, state)
pair owns a learned embedding; the embeddings of all active pitches are summed,
a learned baseline is added (so silence is a learned code rather than zero),
and the result is RMS-normalised so that chords and single notes have the same
scale. Sustained notes additionally carry their *age* (frames since note-on),
injected by rotating a 64-d subspace of the embedding (RoPE-style), which lets
the model know how long a note has been held even after its onset left the
attention window. When MIDI is *absent* (the performer let go), the whole
vector is replaced by a learned ``absent`` embedding — distinct from "MIDI says
be silent".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from livesynth.constants import (  # noqa: F401  (re-exported)
    AGE_MAX_PERIOD, AGE_MIN_PERIOD, AGE_ROT_DIM, MAX_NOTE_AGE, N_STATES, N_VEL, NONE,
    OFFSET, ONSET_BASE, SUSTAIN_BASE, VEL_LEVELS, velocity_level,
)


class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x * self.weight


def age_angles(age: torch.Tensor) -> torch.Tensor:
    """Frame age [...] (long) -> rotation angles [..., AGE_ROT_DIM // 2]."""
    a = age.float().clamp(max=float(MAX_NOTE_AGE)).unsqueeze(-1)
    n_pair = AGE_ROT_DIM // 2
    ratio = (AGE_MAX_PERIOD / AGE_MIN_PERIOD) ** (
        torch.arange(n_pair, device=age.device, dtype=torch.float32) / (n_pair - 1))
    periods = AGE_MIN_PERIOD * ratio
    return 2.0 * torch.pi * a / periods


class MidiEncoder(nn.Module):
    """[B, N, 128] states (+ age, absent mask) -> [B, N, d_model] additive embedding."""

    def __init__(self, d_model: int, n_pitches: int = 128, n_states: int = N_STATES) -> None:
        super().__init__()
        assert d_model >= AGE_ROT_DIM
        self.n_pitches = n_pitches
        self.n_states = n_states
        self.table = nn.Embedding(n_pitches * n_states, d_model)
        self.absent_emb = nn.Parameter(torch.zeros(d_model))
        self.baseline = nn.Parameter(torch.zeros(d_model))
        self.out_norm = _RMSNorm(d_model)
        self.register_buffer(
            "pitch_offset", (torch.arange(n_pitches) * n_states).view(1, 1, n_pitches),
            persistent=False)

    def forward(self, state: torch.Tensor, absent: torch.Tensor | None = None,
                age: torch.Tensor | None = None) -> torch.Tensor:
        """``state`` [B, N, P] long, ``absent`` [B, N] bool, ``age`` [B, N, P] long."""
        state = state.long()
        emb = self.table(self.pitch_offset + state)                    # [B, N, P, D]
        active = (state != NONE).unsqueeze(-1).to(emb.dtype)           # [B, N, P, 1]
        if age is not None:
            sus = (state >= SUSTAIN_BASE) & (state < SUSTAIN_BASE + N_VEL)
            ang = age_angles(age.long() * sus)                         # [B, N, P, 32]
            cos, sin = ang.cos().to(emb.dtype), ang.sin().to(emb.dtype)
            h = AGE_ROT_DIM // 2
            e1, e2 = emb[..., :h], emb[..., h:AGE_ROT_DIM]
            s1 = ((e1 * cos - e2 * sin) * active).sum(dim=2)
            s2 = ((e1 * sin + e2 * cos) * active).sum(dim=2)
            rest = (emb[..., AGE_ROT_DIM:] * active).sum(dim=2)
            out = torch.cat([s1, s2, rest], dim=-1)
        else:
            out = (emb * active).sum(dim=2)
        out = self.out_norm(out + self.baseline.to(out.dtype))
        if absent is not None:
            out = torch.where(absent.unsqueeze(-1), self.absent_emb.to(out.dtype).expand_as(out), out)
        return out
