"""MLX streaming engine for Apple Silicon (single stream, no PyTorch needed).

Same interface and arithmetic as :class:`livesynth.stream.StreamingEngine`:
one ``step(state, age, absent)`` call produces one 10-ms block (480 samples).
Weights are read directly from the released ``*.safetensors`` files. The frame
function is a pure function of explicit state and is wrapped in
``mx.compile``; the inverse STFT is a precomputed inverse-DFT matrix product.

Precision: ``bfloat16`` by default (the model was trained with bf16 autocast;
float16 overflows). ``quantize(bits=8)`` applies weight-only quantisation to
the large matrices, which roughly halves the frame time on an M3.

    eng = MLXStreamingEngine.from_pretrained()          # downloads on first use
    eng.set_timbre(embedding)                           # np.ndarray [512]
    eng.quantize(8)
    block = eng.step(state, age)                        # np.float32 [480]
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

from livesynth.constants import (
    AGE_MAX_PERIOD, AGE_MIN_PERIOD, AGE_ROT_DIM, MAX_NOTE_AGE, N_VEL, SUSTAIN_BASE,
)


def _rms(x: mx.array, w: mx.array | None, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    y = (x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + eps)).astype(x.dtype)
    return y * w if w is not None else y


def _layernorm(x: mx.array, w: mx.array, b: mx.array, eps: float = 1e-5) -> mx.array:
    x32 = x.astype(mx.float32)
    m = mx.mean(x32, axis=-1, keepdims=True)
    v = mx.var(x32, axis=-1, keepdims=True)
    return ((x32 - m) * mx.rsqrt(v + eps)).astype(x.dtype) * w + b


def _gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1 + mx.erf(x / mx.sqrt(mx.array(2.0, dtype=x.dtype))))


class MLXStreamingEngine:
    _QUANT_SUFFIXES = (".qkv", ".out", ".w12", ".w3", ".adaln", ".pw1", ".pw2")

    def __init__(self, model_dir: str | os.PathLike, dtype=mx.bfloat16,
                 use_compile: bool = True, max_pos: int = 65536, seed: int | None = None) -> None:
        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text())
        bc, dc = cfg["backbone"], cfg["decoder"]
        self.config = cfg
        self.dtype = dtype
        self.use_compile = use_compile
        self.n_layers, self.n_heads = bc["n_layers"], bc["n_heads"]
        self.d_model = bc["d_model"]
        self.dh = self.d_model // self.n_heads
        self.noise_dim, self.timbre_dim = bc["noise_dim"], bc["timbre_dim"]
        self.window = bc["attn_window"]
        self.n_states, self.n_pitches = bc["midi_states"], bc["n_pitches"]
        self.hop, self.n_fft = cfg["hop_length"], dc["n_fft"]
        self.dec_depth, self.dec_k = dc["depth"], dc["kernel_size"]
        self.dec_dim, self.d_latent = dc["dim"], dc["d_latent"]
        self._rng = np.random.default_rng(seed)

        self.w: dict[str, mx.array] = {}
        self.q: dict[str, tuple] = {}
        self._load(d)

        f = np.arange(max_pos)[:, None] * (
            1.0 / (bc["rope_base"] ** (np.arange(0, self.dh, 2) / self.dh)))[None, :]
        self.rope_cos = mx.array(np.cos(f).astype(np.float32))
        self.rope_sin = mx.array(np.sin(f).astype(np.float32))
        self._rebase_at = (max_pos // 2 // self.window) * self.window
        self._rebase_by = (self._rebase_at * 5 // 6 // self.window) * self.window

        n, nb = self.n_fft, self.n_fft // 2 + 1
        k, t = np.arange(nb), np.arange(n)
        wk = np.full(nb, 2.0)
        wk[0] = 1.0
        if n % 2 == 0:
            wk[-1] = 1.0
        ang = 2 * np.pi * t[:, None] * k[None, :] / n
        self.idft_c = mx.array((np.cos(ang) * wk / n).astype(np.float32))
        self.idft_s = mx.array((-np.sin(ang) * wk / n).astype(np.float32))
        win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)
        self.window_fn = mx.array(win)
        self.env = mx.array(np.maximum((win ** 2).reshape(n // self.hop, self.hop).sum(0), 1e-8))
        dcm = np.ones(nb, np.float32)
        dcm[0] = 0.0
        self.dc_mask = mx.array(dcm)
        self.pitch_off = mx.array((np.arange(self.n_pitches) * self.n_states).astype(np.int32))
        n_pair = AGE_ROT_DIM // 2
        self.age_periods = (AGE_MIN_PERIOD * (AGE_MAX_PERIOD / AGE_MIN_PERIOD) ** (
            np.arange(n_pair) / (n_pair - 1))).astype(np.float32)

        self._compiled = None
        self.timbre = mx.zeros((1, self.timbre_dim), dtype=dtype)
        self.reset()

    @classmethod
    def from_pretrained(cls, repo_id: str | None = None, local_dir: str | None = None,
                        **kw) -> "MLXStreamingEngine":
        from livesynth.hub import DEFAULT_REPO, resolve_model_dir
        return cls(resolve_model_dir(repo_id or DEFAULT_REPO, local_dir=local_dir), **kw)

    # -- weights ----------------------------------------------------------------

    def _load(self, d: Path) -> None:
        g = mx.load(str(d / "generator.safetensors"))
        dec = mx.load(str(d / "decoder.safetensors"))
        dt = self.dtype
        w = self.w
        w["midi_table"] = g["midi_enc.table.weight"].astype(dt)
        w["midi_absent"] = g["midi_enc.absent_emb"].astype(dt)
        w["midi_baseline"] = g["midi_enc.baseline"].astype(dt)
        w["midi_norm"] = g["midi_enc.out_norm.weight"].astype(dt)
        w["noise_proj"] = g["noise_proj.weight"].astype(dt)
        w["cond_proj"], w["cond_proj.b"] = g["cond_proj.weight"].astype(dt), g["cond_proj.bias"].astype(dt)
        w["final_norm"] = g["final_norm.weight"].astype(dt)
        w["head"], w["head.b"] = g["head.weight"].astype(dt), g["head.bias"].astype(dt)
        for i in range(self.n_layers):
            p = f"blocks.{i}."
            w[f"b{i}.adaln"], w[f"b{i}.adaln.b"] = g[p + "adaln.weight"].astype(dt), g[p + "adaln.bias"].astype(dt)
            w[f"b{i}.qkv"] = g[p + "attn.qkv.weight"].astype(dt)
            w[f"b{i}.out"] = g[p + "attn.out.weight"].astype(dt)
            w[f"b{i}.qn"] = g[p + "attn.q_norm.weight"].astype(dt)
            w[f"b{i}.kn"] = g[p + "attn.k_norm.weight"].astype(dt)
            w[f"b{i}.sink"] = g[p + "attn.sink"].astype(dt)
            w[f"b{i}.w12"] = g[p + "ffn.w12.weight"].astype(dt)
            w[f"b{i}.w3"] = g[p + "ffn.w3.weight"].astype(dt)
        w["lat_mean"] = dec["latent_mean"].astype(mx.float32)
        w["lat_std"] = dec["latent_std"].astype(mx.float32)
        w["out_proj"], w["out_proj.b"] = dec["vae.out_proj.weight"].astype(dt), dec["vae.out_proj.bias"].astype(dt)
        w["in_proj"] = dec["decoder.in_proj.conv.weight"][:, :, 0].astype(dt)
        w["in_proj.b"] = dec["decoder.in_proj.conv.bias"].astype(dt)
        w["in_norm"], w["in_norm.b"] = dec["decoder.in_norm.weight"].astype(dt), dec["decoder.in_norm.bias"].astype(dt)
        for j in range(self.dec_depth):
            p = f"decoder.blocks.{j}."
            w[f"d{j}.dw"] = dec[p + "dwconv.conv.weight"][:, 0, :].T.astype(dt)      # [k, dim]
            w[f"d{j}.dw.b"] = dec[p + "dwconv.conv.bias"].astype(dt)
            w[f"d{j}.norm"], w[f"d{j}.norm.b"] = dec[p + "norm.weight"].astype(dt), dec[p + "norm.bias"].astype(dt)
            w[f"d{j}.pw1"], w[f"d{j}.pw1.b"] = dec[p + "pwconv1.weight"].astype(dt), dec[p + "pwconv1.bias"].astype(dt)
            w[f"d{j}.pw2"], w[f"d{j}.pw2.b"] = dec[p + "pwconv2.weight"].astype(dt), dec[p + "pwconv2.bias"].astype(dt)
            w[f"d{j}.gamma"] = dec[p + "gamma"].astype(dt)
        w["final_ln"], w["final_ln.b"] = dec["decoder.final_norm.weight"].astype(dt), dec["decoder.final_norm.bias"].astype(dt)
        w["ihead"], w["ihead.b"] = dec["decoder.head.out.weight"].astype(dt), dec["decoder.head.out.bias"].astype(dt)
        mx.eval(list(w.values()))

    def quantize(self, bits: int = 8, group_size: int = 32) -> None:
        """Weight-only quantisation of the large matrices (``bits`` 4 or 8).
        The SwiGLU width (2389) is zero-padded to a multiple of ``group_size``,
        which leaves the output unchanged."""
        names = [k for k in self.w if k.endswith(self._QUANT_SUFFIXES)]
        names += ["head", "ihead", "cond_proj", "out_proj", "in_proj", "noise_proj"]
        hid = self.w["b0.w3"].shape[1]
        pad = (-hid) % group_size
        for name in sorted(set(names)):
            m = self.w[name]
            if name.endswith(".w3") and pad:
                m = mx.concatenate([m, mx.zeros((m.shape[0], pad), dtype=m.dtype)], axis=1)
            if name.endswith(".w12") and pad:
                a, b = mx.split(m, 2, axis=0)
                z = mx.zeros((pad, m.shape[1]), dtype=m.dtype)
                m = mx.concatenate([a, z, b, z], axis=0)
            wq, sc, bs = mx.quantize(m.astype(self.dtype), group_size, bits)
            self.q[name] = (wq, sc, bs, group_size, bits)
            del self.w[name]
        self._compiled = None
        mx.eval([t for v in self.q.values() for t in v[:3]])

    def _mm(self, x: mx.array, name: str) -> mx.array:
        if name in self.q:
            wq, sc, bs, gs, bits = self.q[name]
            return mx.quantized_matmul(x, wq, scales=sc, biases=bs, transpose=True,
                                       group_size=gs, bits=bits)
        return x @ self.w[name].T

    # -- state --------------------------------------------------------------------

    def set_timbre(self, timbre) -> None:
        t = np.asarray(timbre, np.float32).reshape(1, -1)
        t = t / max(float(np.linalg.norm(t)), 1e-8)
        self.timbre = mx.array(t).astype(self.dtype)

    def reset(self, seed: int | None = None) -> None:
        dt = self.dtype
        # one extra always-zero slot per cache: the scalar attention sink column
        self.k_cache = [mx.zeros((1, self.n_heads, self.window + 1, self.dh), dtype=dt)
                        for _ in range(self.n_layers)]
        self.v_cache = [mx.zeros_like(k) for k in self.k_cache]
        self.ctx = [mx.zeros((1, self.dec_k - 1, self.dec_dim), dtype=dt)
                    for _ in range(self.dec_depth)]
        self.ola = mx.zeros((self.n_fft,), dtype=mx.float32)
        self.bias = np.full((1, 1, 1, self.window), -np.inf, np.float32)
        self._frame = 0
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        z = mx.zeros((1, 1, self.d_latent), dtype=dt)
        for _ in range(self.n_fft // self.hop - 1):         # overlap-add warm-up
            _, self.ctx, self.ola = self._decode(z, self.ctx, self.ola)
        mx.eval(self.ctx + [self.ola])

    # -- maths ------------------------------------------------------------------

    def _rope(self, x, cos, sin):
        h = self.dh // 2
        x1, x2 = x[..., :h], x[..., h:]
        c, s = cos[None, None].astype(x.dtype), sin[None, None].astype(x.dtype)
        return mx.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)

    def _block(self, i, x, cond, kc, vc, cos, sin, slot, bias):
        w = self.w
        s1, c1, g1, s2, c2, g2 = mx.split(self._mm(cond, f"b{i}.adaln") + w[f"b{i}.adaln.b"], 6, axis=-1)
        h = _rms(x, None) * (1 + c1[:, None]) + s1[:, None]
        q, k, v = mx.split(self._mm(h, f"b{i}.qkv"), 3, axis=-1)
        q, k, v = (y.reshape(1, 1, self.n_heads, self.dh).transpose(0, 2, 1, 3) for y in (q, k, v))
        q = self._rope(_rms(q, w[f"b{i}.qn"]), cos, sin)
        k = self._rope(_rms(k, w[f"b{i}.kn"]), cos, sin)
        idx = mx.broadcast_to(slot.reshape(1, 1, 1, 1), (1, self.n_heads, 1, self.dh))
        kc = mx.put_along_axis(kc, idx, k, axis=2)
        vc = mx.put_along_axis(vc, idx, v, axis=2)
        mask = mx.concatenate([mx.broadcast_to(bias, (1, self.n_heads, 1, self.window)),
                               w[f"b{i}.sink"].reshape(1, self.n_heads, 1, 1)], axis=-1)
        o = mx.fast.scaled_dot_product_attention(q, kc, vc, scale=self.dh ** -0.5,
                                                 mask=mask.astype(q.dtype))
        x = x + g1[:, None] * self._mm(o.transpose(0, 2, 1, 3).reshape(1, 1, -1), f"b{i}.out")
        h2 = _rms(x, None) * (1 + c2[:, None]) + s2[:, None]
        a, b = mx.split(self._mm(h2, f"b{i}.w12"), 2, axis=-1)
        return x + g2[:, None] * self._mm(a * mx.sigmoid(a) * b, f"b{i}.w3"), kc, vc

    def _decode(self, zp, ctx, ola):
        """Projected latent [1, 1, d_latent] -> (audio [hop], ctx, ola)."""
        w = self.w
        x = _layernorm(self._mm(zp, "in_proj") + w["in_proj.b"], w["in_norm"], w["in_norm.b"])
        new_ctx = []
        for j in range(self.dec_depth):
            ext = mx.concatenate([ctx[j], x], axis=1)                          # [1, k, dim]
            h = (ext * w[f"d{j}.dw"]).sum(axis=1, keepdims=True) + w[f"d{j}.dw.b"]
            new_ctx.append(ext[:, 1:])
            hh = _gelu(self._mm(_layernorm(h, w[f"d{j}.norm"], w[f"d{j}.norm.b"]), f"d{j}.pw1")
                       + w[f"d{j}.pw1.b"])
            x = x + w[f"d{j}.gamma"] * (self._mm(hh, f"d{j}.pw2") + w[f"d{j}.pw2.b"])
        x = _layernorm(x, w["final_ln"], w["final_ln.b"])
        o = (self._mm(x, "ihead") + w["ihead.b"]).astype(mx.float32)[0, 0]
        nb = self.n_fft // 2 + 1
        mag = mx.minimum(mx.exp(o[:nb]), 1e2) * self.dc_mask
        frame = (self.idft_c @ (mag * mx.cos(o[nb:])) + self.idft_s @ (mag * mx.sin(o[nb:]))) * self.window_fn
        ola = ola + frame
        audio = mx.clip(ola[: self.hop] / self.env, -1.0, 1.0)
        return audio, new_ctx, mx.concatenate([ola[self.hop:], mx.zeros((self.hop,), dtype=mx.float32)])

    def _midi(self, state, age_cos, age_sin, absent):
        w = self.w
        emb = w["midi_table"][self.pitch_off + state]                         # [P, d]
        act = (state != 0)[:, None].astype(self.dtype)
        h = AGE_ROT_DIM // 2
        e1, e2 = emb[:, :h], emb[:, h:2 * h]
        s1 = ((e1 * age_cos - e2 * age_sin) * act).sum(axis=0)
        s2 = ((e1 * age_sin + e2 * age_cos) * act).sum(axis=0)
        rest = (emb[:, 2 * h:] * act).sum(axis=0)
        out = _rms(mx.concatenate([s1, s2, rest])[None] + w["midi_baseline"], w["midi_norm"])
        return mx.where(absent, w["midi_absent"][None], out)                  # [1, d]

    def _frame_fn(self, noise, state, age_cos, age_sin, absent, timbre, cos, sin, slot, bias,
                  k_caches, v_caches, ctx, ola):
        w = self.w
        cond = self._mm(timbre, "cond_proj") + w["cond_proj.b"]
        x = (self._mm(noise, "noise_proj") + self._midi(state, age_cos, age_sin, absent))[:, None]
        new_k, new_v = [], []
        for i in range(self.n_layers):
            x, kc, vc = self._block(i, x, cond, k_caches[i], v_caches[i], cos, sin, slot, bias)
            new_k.append(kc)
            new_v.append(vc)
        z = self._mm(_rms(x, w["final_norm"]), "head") + w["head.b"]           # [1, 1, latent]
        zr = z.astype(mx.float32) * w["lat_std"] + w["lat_mean"]
        zp = self._mm(zr.astype(self.dtype), "out_proj") + w["out_proj.b"]
        audio, ctx, ola = self._decode(zp, ctx, ola)
        return audio, new_k, new_v, ctx, ola

    # -- per frame -----------------------------------------------------------------

    def step(self, state: np.ndarray, age: np.ndarray, absent: bool = False,
             noise: np.ndarray | None = None) -> np.ndarray:
        """One frame. ``state``/``age`` [128] (from :class:`livesynth.NoteTracker`).
        Returns ``float32`` audio [480]."""
        if self._frame >= self._rebase_at:
            dlt = self._rebase_by
            c, s = self.rope_cos[dlt:dlt + 1], -self.rope_sin[dlt:dlt + 1]
            self.k_cache = [self._rope(k, c, s) for k in self.k_cache]
            self._frame -= dlt
        pos = self._frame
        slot = pos % self.window
        if pos < self.window:
            self.bias[..., slot] = 0.0
        st = np.asarray(state, np.int64).reshape(-1)
        sus = (st >= SUSTAIN_BASE) & (st < SUSTAIN_BASE + N_VEL)
        a = np.minimum(np.asarray(age, np.int64).reshape(-1) * sus, MAX_NOTE_AGE).astype(np.float32)
        ang = 2.0 * np.pi * a[:, None] / self.age_periods[None, :]
        if noise is None:
            noise = self._rng.standard_normal(self.noise_dim).astype(np.float32)
        fn = self._frame_fn
        if self.use_compile:
            if self._compiled is None:
                self._compiled = mx.compile(self._frame_fn)
            fn = self._compiled
        audio, self.k_cache, self.v_cache, self.ctx, self.ola = fn(
            mx.array(np.asarray(noise, np.float32).reshape(1, -1)).astype(self.dtype),
            mx.array(st.astype(np.int32)),
            mx.array(np.cos(ang)).astype(self.dtype), mx.array(np.sin(ang)).astype(self.dtype),
            mx.array(np.array([bool(absent)])), self.timbre,
            self.rope_cos[pos:pos + 1], self.rope_sin[pos:pos + 1],
            mx.array([slot]), mx.array(self.bias),
            self.k_cache, self.v_cache, self.ctx, self.ola)
        self._frame += 1
        mx.eval(audio)
        return np.asarray(audio)
