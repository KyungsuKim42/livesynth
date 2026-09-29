"""Feedback-free causal Transformer generator.

The generator maps per-frame Gaussian noise, MIDI and timbre to a sequence of
(standardised) VAE latents. It never consumes its own previous output: temporal
dependency is carried only by causal self-attention over hidden states, so the
same weights run either as one parallel pass over a whole sequence
(:meth:`LiveSynthBackbone.forward`) or one frame at a time with a bounded
key/value cache (:meth:`LiveSynthBackbone.step`). Both paths implement the same
banded causal attention: frame ``t`` attends to frames ``t - W + 1 .. t``.

Conditioning:
  * MIDI is added once to the projected noise at the input.
  * Timbre (a CLAP embedding) modulates every block through adaLN-Zero
    (shift, scale and gate for attention and FFN). It may differ per frame,
    which is what enables timbre morphing.
  * A per-head learned scalar attention sink (a zero key/value column) keeps
    streaming stable; it carries no content and never occupies a cache slot.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from livesynth.nn.midi_encoder import N_STATES, MidiEncoder


@dataclass
class BackboneConfig:
    d_model: int = 896
    n_layers: int = 13
    n_heads: int = 14
    ffn_mult: float = 2.6667
    noise_dim: int = 128
    timbre_dim: int = 512
    latent_dim: int = 128
    n_pitches: int = 128
    midi_states: int = N_STATES
    rope_base: float = 10_000.0
    attn_window: int = 500


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x * self.weight if self.weight is not None else x


def rotary_tables(positions: torch.Tensor, head_dim: int, base: float
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim))
    f = positions.float()[:, None] * inv[None, :]
    return f.cos(), f.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, H, T, Dh]; cos/sin [T, Dh/2] (half-split rotation)."""
    # cos/sin stay fp32 (as in training): under bf16 autocast the rotation is
    # computed in fp32 and attention then runs in bf16.
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    cos, sin = cos[None, None], sin[None, None]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalAttention(nn.Module):
    #: query chunk length for long sequences in the parallel path
    chunk: int = 1000

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.window = int(cfg.attn_window)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.dh)
        self.k_norm = RMSNorm(self.dh)
        self.sink = nn.Parameter(torch.zeros(self.h))

    def _qkv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
        q, k, v = (y.view(b, t, self.h, self.dh).transpose(1, 2) for y in (q, k, v))
        q, k = self.q_norm(q), self.k_norm(k)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Attention with a scalar sink: one extra zero key/value column whose
        logit is the learned per-head ``sink``. ``mask`` [Tq, Tk] additive."""
        b = q.shape[0]
        z = torch.zeros(b, self.h, 1, self.dh, dtype=q.dtype, device=q.device)
        tq, tk = mask.shape
        full = torch.cat([mask.expand(1, self.h, tq, tk),
                          self.sink.view(1, self.h, 1, 1).to(q.dtype).expand(1, self.h, tq, 1)],
                         dim=-1)
        return F.scaled_dot_product_attention(
            q, torch.cat([k, z], dim=2), torch.cat([v, z], dim=2), attn_mask=full)

    def _band_mask(self, q0: int, q1: int, k0: int, k1: int, dtype, device) -> torch.Tensor:
        qi = torch.arange(q0, q1, device=device)[:, None]
        kj = torch.arange(k0, k1, device=device)[None, :]
        ok = (kj <= qi) & (qi - kj < self.window)
        return torch.zeros(ok.shape, dtype=dtype, device=device).masked_fill(~ok, float("-inf"))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self._qkv(x, cos, sin)
        outs = []
        # Query chunks with their exact key range: memory O(chunk * (chunk + W))
        # instead of O(T^2), identical results.
        for q0 in range(0, t, self.chunk):
            q1 = min(t, q0 + self.chunk)
            k0 = max(0, q0 - self.window + 1)
            m = self._band_mask(q0, q1, k0, q1, q.dtype, q.device)
            outs.append(self._attend(q[:, :, q0:q1], k[:, :, k0:q1], v[:, :, k0:q1], m))
        o = torch.cat(outs, dim=2).transpose(1, 2).reshape(b, t, -1)
        return self.out(o)

    def step(self, x: torch.Tensor, cache: dict, cos: torch.Tensor, sin: torch.Tensor
             ) -> torch.Tensor:
        """One frame ``x`` [B, 1, d] with a rolling cache of the last W keys/values."""
        b = x.shape[0]
        q, k, v = self._qkv(x, cos, sin)
        if cache.get("k") is None:
            cache["k"], cache["v"] = k, v
        else:
            cache["k"] = torch.cat([cache["k"], k], dim=2)[:, :, -self.window:]
            cache["v"] = torch.cat([cache["v"], v], dim=2)[:, :, -self.window:]
        mask = torch.zeros(1, cache["k"].shape[2], dtype=q.dtype, device=q.device)
        o = self._attend(q, cache["k"], cache["v"], mask)
        return self.out(o.transpose(1, 2).reshape(b, 1, -1))


    def step_ring(self, x: torch.Tensor, k_ring: torch.Tensor, v_ring: torch.Tensor,
                  slot: torch.Tensor, bias: torch.Tensor, cos: torch.Tensor,
                  sin: torch.Tensor) -> torch.Tensor:
        """One frame with a fixed-size ring cache (for CUDA-graph capture).

        ``k_ring``/``v_ring`` [B, H, W, Dh] are updated in place at ``slot``
        (a 1-element long tensor); ``bias`` [W] is 0 for filled slots and -inf
        for empty ones. Attention does not depend on key order (positions are
        encoded by RoPE), so this equals :meth:`step`."""
        b = x.shape[0]
        q, k, v = self._qkv(x, cos, sin)
        k_ring.index_copy_(2, slot, k.to(k_ring.dtype))
        v_ring.index_copy_(2, slot, v.to(v_ring.dtype))
        o = self._attend(q, k_ring, v_ring, bias.to(q.dtype).view(1, -1))
        return self.out(o.transpose(1, 2).reshape(b, 1, -1))


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(d, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


def _per_frame(m: torch.Tensor) -> torch.Tensor:
    """Modulation [B, d] (global) or [B, N, d] (per frame) -> broadcastable [B, *, d]."""
    return m.unsqueeze(1) if m.dim() == 2 else m


class Block(nn.Module):
    """Pre-norm Transformer block with adaLN-Zero timbre modulation."""

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, affine=False)
        self.attn = CausalAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, affine=False)
        self.ffn = SwiGLU(cfg.d_model, round(cfg.d_model * cfg.ffn_mult))
        self.adaln = nn.Linear(cfg.d_model, 6 * cfg.d_model)

    def _mods(self, cond: torch.Tensor):
        return [_per_frame(m) for m in self.adaln(cond).chunk(6, dim=-1)]

    def forward(self, x, cond, cos, sin):
        s1, c1, g1, s2, c2, g2 = self._mods(cond)
        x = x + g1 * self.attn(self.norm1(x) * (1 + c1) + s1, cos, sin)
        return x + g2 * self.ffn(self.norm2(x) * (1 + c2) + s2)

    def step(self, x, cond, cache, cos, sin):
        s1, c1, g1, s2, c2, g2 = self._mods(cond)
        x = x + g1 * self.attn.step(self.norm1(x) * (1 + c1) + s1, cache, cos, sin)
        return x + g2 * self.ffn(self.norm2(x) * (1 + c2) + s2)

    def step_ring(self, x, cond, k_ring, v_ring, slot, bias, cos, sin):
        s1, c1, g1, s2, c2, g2 = self._mods(cond)
        x = x + g1 * self.attn.step_ring(self.norm1(x) * (1 + c1) + s1, k_ring, v_ring,
                                         slot, bias, cos, sin)
        return x + g2 * self.ffn(self.norm2(x) * (1 + c2) + s2)


class LiveSynthBackbone(nn.Module):
    """(noise, MIDI, timbre) -> standardised VAE latent. Causal and feedback-free."""

    def __init__(self, cfg: BackboneConfig | None = None) -> None:
        super().__init__()
        self.cfg = c = cfg or BackboneConfig()
        self.head_dim = c.d_model // c.n_heads
        self.midi_enc = MidiEncoder(c.d_model, n_pitches=c.n_pitches, n_states=c.midi_states)
        self.noise_proj = nn.Linear(c.noise_dim, c.d_model, bias=False)
        self.cond_proj = nn.Linear(c.timbre_dim, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.final_norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.latent_dim)

    def forward(self, noise: torch.Tensor, midi_state: torch.Tensor, timbre: torch.Tensor,
                midi_absent: torch.Tensor | None = None,
                midi_age: torch.Tensor | None = None) -> torch.Tensor:
        """Parallel pass.

        Args:
            noise:       [B, N, noise_dim]
            midi_state:  [B, N, 128] long
            timbre:      [B, timbre_dim] (fixed) or [B, N, timbre_dim] (per frame)
            midi_absent: [B, N] bool, frames whose MIDI condition is absent
            midi_age:    [B, N, 128] long, frames since note-on of sustained pitches
        Returns:
            latent [B, latent_dim, N]
        """
        x = self.noise_proj(noise) + self._encode_midi(midi_state, midi_absent, midi_age)
        cond = self.cond_proj(timbre)
        cos, sin = rotary_tables(torch.arange(x.shape[1], device=x.device),
                                 self.head_dim, self.cfg.rope_base)
        for blk in self.blocks:
            x = blk(x, cond, cos, sin)
        return self.head(self.final_norm(x)).transpose(1, 2).contiguous()

    def _encode_midi(self, state, absent, age) -> torch.Tensor:
        # The per-pitch lookup materialises [B, T, 128, d_model]; encode in time
        # chunks (frames are independent) to bound memory on long sequences.
        b, n = state.shape[:2]
        step = max(1, 1000 // max(1, b))
        if n <= step:
            return self.midi_enc(state, absent, age)
        return torch.cat([self.midi_enc(state[:, i:i + step],
                                        None if absent is None else absent[:, i:i + step],
                                        None if age is None else age[:, i:i + step])
                          for i in range(0, n, step)], dim=1)

    # -- streaming ---------------------------------------------------------

    def init_stream(self) -> dict:
        return {"caches": [{} for _ in self.blocks], "pos": 0}

    @torch.no_grad()
    def step(self, state: dict, noise_t: torch.Tensor, midi_t: torch.Tensor,
             timbre_t: torch.Tensor, absent_t: torch.Tensor | None = None,
             age_t: torch.Tensor | None = None) -> torch.Tensor:
        """One frame. ``noise_t`` [B, noise_dim], ``midi_t`` [B, 128],
        ``timbre_t`` [B, timbre_dim], ``absent_t`` [B] bool, ``age_t`` [B, 128].
        Returns latent [B, latent_dim, 1]."""
        me = self.midi_enc(midi_t.unsqueeze(1),
                           absent_t.view(-1, 1) if absent_t is not None else None,
                           age_t.unsqueeze(1) if age_t is not None else None)
        x = self.noise_proj(noise_t).unsqueeze(1) + me
        cond = self.cond_proj(timbre_t)
        cos, sin = rotary_tables(torch.tensor([state["pos"]], device=x.device),
                                 self.head_dim, self.cfg.rope_base)
        for blk, cache in zip(self.blocks, state["caches"]):
            x = blk.step(x, cond, cache, cos, sin)
        state["pos"] += 1
        return self.head(self.final_norm(x)).transpose(1, 2).contiguous()
