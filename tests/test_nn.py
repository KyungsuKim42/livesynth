"""Structural tests with random weights (no download needed)."""

import numpy as np
import pytest
import torch

from livesynth.midi import NoteTracker, notes_to_frames
from livesynth.nn.backbone import BackboneConfig, CausalAttention, LiveSynthBackbone
from livesynth.nn.decoder import LatentDecoder

torch.manual_seed(0)


def small_backbone(window: int = 8) -> LiveSynthBackbone:
    cfg = BackboneConfig(d_model=64, n_layers=3, n_heads=4, noise_dim=16, timbre_dim=32,
                         latent_dim=8, attn_window=window)
    m = LiveSynthBackbone(cfg).eval()
    with torch.no_grad():                   # adaLN-Zero is zero at init; make it non-trivial
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m


def random_inputs(b: int, n: int, cfg: BackboneConfig):
    noise = torch.randn(b, n, cfg.noise_dim)
    state = torch.randint(0, cfg.midi_states, (b, n, cfg.n_pitches))
    state[torch.rand(b, n, cfg.n_pitches) < 0.9] = 0
    age = torch.randint(0, 300, (b, n, cfg.n_pitches))
    absent = torch.rand(b, n) < 0.3
    timbre = torch.randn(b, n, cfg.timbre_dim)
    return noise, state, age, absent, timbre


@pytest.mark.parametrize("chunk", [1000, 7])
def test_parallel_matches_streaming(chunk):
    m = small_backbone(window=8)
    CausalAttention.chunk = chunk
    try:
        cfg = m.cfg
        noise, state, age, absent, timbre = random_inputs(2, 30, cfg)
        with torch.no_grad():
            par = m(noise, state, timbre, midi_absent=absent, midi_age=age)
            st = m.init_stream()
            seq = torch.cat([m.step(st, noise[:, t], state[:, t], timbre[:, t],
                                    absent[:, t], age[:, t]) for t in range(30)], dim=-1)
        assert torch.allclose(par, seq, atol=1e-4), (par - seq).abs().max()
    finally:
        CausalAttention.chunk = 1000


def test_causal():
    m = small_backbone(window=8)
    noise, state, age, absent, timbre = random_inputs(1, 24, m.cfg)
    with torch.no_grad():
        a = m(noise, state, timbre, absent, age)
        noise2 = noise.clone()
        noise2[:, 15:] += 1.0
        b = m(noise2, state, timbre, absent, age)
    assert torch.allclose(a[..., :15], b[..., :15], atol=1e-6)
    assert not torch.allclose(a[..., 15:], b[..., 15:])


def test_global_timbre_equals_constant_per_frame():
    m = small_backbone()
    noise, state, age, absent, timbre = random_inputs(1, 12, m.cfg)
    g = timbre[:, 0]
    with torch.no_grad():
        a = m(noise, state, g, absent, age)
        b = m(noise, state, g[:, None].expand(-1, 12, -1), absent, age)
    assert torch.allclose(a, b, atol=1e-6)


def test_midi_encoder_chunking():
    m = small_backbone()
    noise, state, age, absent, _ = random_inputs(2, 1200, m.cfg)   # > chunk of 500 frames
    with torch.no_grad():
        full = m.midi_enc(state, absent, age)
        chunked = m._encode_midi(state, absent, age)
    assert torch.allclose(full, chunked, atol=1e-6)


def test_decoder_stream_matches_forward():
    dec = LatentDecoder(latent_dim=8, d_latent=16, dim=32, depth=2, n_fft=64, hop=16).eval()
    with torch.no_grad():
        for p in dec.parameters():
            p.add_(0.05 * torch.randn_like(p))
        z = torch.randn(1, 8, 20)
        full = dec(z)
        st: dict = {}
        streamed = torch.cat([dec.stream(z[..., t:t + 1], st) for t in range(20)], dim=-1)
    assert full.shape == (1, 20 * 16)
    assert torch.allclose(full, streamed, atol=1e-5), (full - streamed).abs().max()


def test_note_tracker_matches_offline():
    # (start, end, pitch, velocity) on a 10-ms grid
    notes = np.array([[0.00, 0.30, 60, 100], [0.10, 0.20, 64, 30],
                      [0.30, 0.50, 60, 127], [0.25, 0.60, 67, 80]])
    n = 70
    ref_state, ref_age = notes_to_frames(notes, n)
    tr = NoteTracker()
    states, ages = [], []
    for f in range(n):
        t = f / 100
        for s, e, p, v in notes:            # note-offs before note-ons, like a MIDI stream
            if round(e * 100) == f:
                tr.note_off(int(p))
        for s, e, p, v in notes:
            if round(s * 100) == f:
                tr.note_on(int(p), int(v))
        st, ag = tr.frame()
        states.append(st)
        ages.append(ag)
    states = torch.from_numpy(np.stack(states))
    ages = torch.from_numpy(np.stack(ages))
    assert torch.equal(states, ref_state)
    sus = (ref_state >= 7)
    assert torch.equal(ages[sus], ref_age[sus])


def _engine_pair(window: int = 8, max_pos: int = 65536):
    from livesynth.stream import StreamingEngine
    m = small_backbone(window=window)
    dec = LatentDecoder(latent_dim=8, d_latent=16, dim=32, depth=2, n_fft=64, hop=16).eval()
    with torch.no_grad():
        for p in dec.parameters():
            p.add_(0.05 * torch.randn_like(p))
    eng = StreamingEngine(m, dec, torch.device("cpu"), precision="fp32", use_graph=False,
                          max_pos=max_pos)
    return m, dec, eng


def test_streaming_engine_matches_offline():
    m, dec, eng = _engine_pair()
    noise, state, age, absent, timbre = random_inputs(1, 40, m.cfg)
    g = torch.nn.functional.normalize(timbre[:, 0], dim=-1)   # the engine normalises timbre
    with torch.no_grad():
        ref = dec(m(noise, state, g, absent, age))[0]
    eng.set_timbre(g[0])
    out = torch.cat([eng.step_tensor(state[0, t], age[0, t], bool(absent[0, t]),
                                     noise=noise[0, t]).reshape(-1).clone() for t in range(40)])
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max()


def test_streaming_engine_rope_rebase():
    torch.manual_seed(1)
    m, dec, eng = _engine_pair(window=8, max_pos=64)          # re-bases every 32 frames
    _, _, eng_long = _engine_pair(window=8, max_pos=4096)
    eng_long.bb.load_state_dict(eng.bb.state_dict())
    eng_long.dec.load_state_dict(eng.dec.state_dict())
    eng_long.reset()
    noise, state, age, absent, timbre = random_inputs(1, 120, m.cfg)
    for e in (eng, eng_long):
        e.set_timbre(timbre[0, 0])
    a = [eng.step_tensor(state[0, t], age[0, t], False, noise=noise[0, t]).clone() for t in range(120)]
    b = [eng_long.step_tensor(state[0, t], age[0, t], False, noise=noise[0, t]).clone() for t in range(120)]
    assert torch.allclose(torch.cat(a, -1), torch.cat(b, -1), atol=1e-4)
