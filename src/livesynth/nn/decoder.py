"""Latent-to-waveform decoder (the decoding half of the causal VAE codec).

The generator predicts standardised 128-d latents at 100 Hz. This module
de-standardises them, projects them to the decoder width and synthesises
48-kHz audio with a lightweight causal Vocos-style network: a ConvNeXt stack at
the frame rate followed by one inverse STFT per frame (n_fft 1920, hop 480) and
causal overlap-add. Every operation is causal, so the same weights run over a
whole sequence (:meth:`forward`) or one frame at a time (:meth:`stream`); the two
match exactly after the fixed overlap-add warm-up, which both paths discard.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Left-padded 1-D convolution (stride 1). ``stream`` keeps the left context."""

    def __init__(self, cin: int, cout: int, kernel_size: int, groups: int = 1) -> None:
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(cin, cout, kernel_size, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))

    def stream(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if self.pad == 0:
            return self.conv(x)
        buf = state.get(self)
        if buf is None:
            buf = x.new_zeros(x.shape[0], x.shape[1], self.pad)
        xx = torch.cat([buf, x], dim=-1)
        state[self] = xx[..., -self.pad:]
        return self.conv(xx)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, mult: int = 3, kernel_size: int = 7) -> None:
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel_size, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, mult * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mult * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def _mlp(self, h: torch.Tensor) -> torch.Tensor:
        return self.gamma * self.pwconv2(self.act(self.pwconv1(self.norm(h))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x [B, T, dim]
        return x + self._mlp(self.dwconv(x.transpose(1, 2)).transpose(1, 2))

    def stream(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        return x + self._mlp(self.dwconv.stream(x.transpose(1, 2), state).transpose(1, 2))


class ISTFTHead(nn.Module):
    """Per-frame (log-magnitude, phase) prediction -> waveform by overlap-add."""

    def __init__(self, dim: int, n_fft: int, hop: int) -> None:
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.n_bins = n_fft // 2 + 1
        self.out = nn.Linear(dim, 2 * self.n_bins)
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window, persistent=False)
        env = (window ** 2).reshape(n_fft // hop, hop).sum(dim=0)
        self.register_buffer("env", env.clamp_min(1e-8), persistent=False)
        dc = torch.ones(self.n_bins)
        dc[0] = 0.0                                   # zero-mean output
        self.register_buffer("dc_mask", dc, persistent=False)

    def _frames(self, x: torch.Tensor) -> torch.Tensor:
        o = self.out(x).float()
        log_mag, phase = o.chunk(2, dim=-1)
        mag = torch.exp(log_mag).clamp(max=1e2) * self.dc_mask
        spec = torch.polar(mag, phase).transpose(1, 2)            # [B, bins, T]
        return torch.fft.irfft(spec, n=self.n_fft, dim=1) * self.window.view(1, -1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = self._frames(x)                                   # [B, n_fft, T]
        n = frames.shape[-1]
        out_len = (n - 1) * self.hop + self.n_fft
        audio = F.fold(frames, (1, out_len), (1, self.n_fft), stride=(1, self.hop)
                       ).reshape(frames.shape[0], out_len)
        win_sq = (self.window ** 2).view(1, self.n_fft, 1).expand(1, self.n_fft, n)
        env = F.fold(win_sq, (1, out_len), (1, self.n_fft), stride=(1, self.hop)
                     ).reshape(1, out_len).clamp_min(1e-8)
        return (audio / env)[:, : n * self.hop]

    def stream(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        frames = self._frames(x)
        b, _, n = frames.shape
        ola = state.get((self, "ola"))
        if ola is None:
            ola = frames.new_zeros(b, self.n_fft)
        outs = []
        for t in range(n):
            ola = ola + frames[:, :, t]
            outs.append(ola[:, : self.hop] / self.env)
            ola = torch.cat([ola[:, self.hop:], frames.new_zeros(b, self.hop)], dim=1)
        state[(self, "ola")] = ola
        return torch.cat(outs, dim=1)


class VocosDecoder(nn.Module):
    def __init__(self, d_latent: int = 256, dim: int = 384, depth: int = 8,
                 intermediate_mult: int = 3, kernel_size: int = 7,
                 n_fft: int = 1920, hop: int = 480) -> None:
        super().__init__()
        self.hop = hop
        self.in_proj = CausalConv1d(d_latent, dim, 1)
        self.in_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(ConvNeXtBlock(dim, intermediate_mult, kernel_size)
                                    for _ in range(depth))
        self.final_norm = nn.LayerNorm(dim)
        self.head = ISTFTHead(dim, n_fft, hop)
        self.warmup = n_fft // hop - 1                 # overlap-add warm-up frames

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z [B, d_latent, T] -> audio [B, T * hop]."""
        z = F.pad(z, (self.warmup, 0))
        x = self.in_norm(self.in_proj(z).transpose(1, 2))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.final_norm(x))[:, self.warmup * self.hop:]

    def stream(self, z: torch.Tensor, state: dict) -> torch.Tensor:
        prime = 0 if state.get((self, "primed")) else self.warmup
        state[(self, "primed")] = True
        if prime:
            z = F.pad(z, (prime, 0))
        x = self.in_norm(self.in_proj.stream(z, state).transpose(1, 2))
        for blk in self.blocks:
            x = blk.stream(x, state)
        audio = self.head.stream(self.final_norm(x), state)
        return audio[:, prime * self.hop:] if prime else audio


class LatentDecoder(nn.Module):
    """Standardised latent [B, 128, T] -> 48-kHz audio [B, T * 480] in [-1, 1]."""

    def __init__(self, latent_dim: int = 128, d_latent: int = 256, dim: int = 384,
                 depth: int = 8, intermediate_mult: int = 3, kernel_size: int = 7,
                 n_fft: int = 1920, hop: int = 480) -> None:
        super().__init__()
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))
        self.out_proj = nn.Linear(latent_dim, d_latent)
        self.vocos = VocosDecoder(d_latent, dim, depth, intermediate_mult, kernel_size, n_fft, hop)
        self.hop = hop

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float() * self.latent_std.view(1, -1, 1) + self.latent_mean.view(1, -1, 1)
        return self.out_proj(z.transpose(1, 2)).transpose(1, 2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.vocos(self._project(z)).clamp(-1.0, 1.0)

    def stream(self, z: torch.Tensor, state: dict) -> torch.Tensor:
        return self.vocos.stream(self._project(z), state).clamp(-1.0, 1.0)

    @staticmethod
    def remap_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Released checkpoint keys (``vae.out_proj.*``, ``decoder.*``) -> module keys."""
        out = {}
        for k, v in sd.items():
            if k.startswith("vae.out_proj."):
                out["out_proj." + k[len("vae.out_proj."):]] = v
            elif k.startswith("decoder."):
                out["vocos." + k[len("decoder."):]] = v
            else:
                out[k] = v
        return out
