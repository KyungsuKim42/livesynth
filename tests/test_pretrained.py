"""Tests with the released weights. Skipped unless weights are available locally
(``LIVESYNTH_WEIGHTS=/path/to/dir``) or ``LIVESYNTH_TEST_HUB=1`` allows a download."""

import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (os.environ.get("LIVESYNTH_WEIGHTS") or os.environ.get("LIVESYNTH_TEST_HUB")),
    reason="set LIVESYNTH_WEIGHTS or LIVESYNTH_TEST_HUB=1 to run tests with the released weights")

NOTES = np.array([[0.0, 0.5, 60, 100], [0.5, 1.0, 64, 90], [1.0, 1.5, 67, 80],
                  [1.5, 2.5, 72, 110], [0.0, 2.5, 48, 70]])


@pytest.fixture(scope="module")
def synth():
    from livesynth import LiveSynth
    return LiveSynth.from_pretrained(precision="fp32")


def test_render_shape_and_level(synth):
    a = synth.render(NOTES, synth.presets[0], seed=0)
    assert a.dtype == np.float32
    assert a.shape == (int(np.ceil((2.5 + 1.0) * 100)) * synth.hop_length,)
    assert 0.01 < float(np.sqrt((a ** 2).mean())) < 0.9          # audible, not clipped noise
    assert np.array_equal(a, synth.render(NOTES, synth.presets[0], seed=0))


def test_batch_matches_single_in_fp32(synth):
    names = synth.presets[:2]
    batch = synth.render_batch([NOTES, NOTES[:3]], names, seed=5)
    single = [synth.render(NOTES, names[0], seed=5), synth.render(NOTES[:3], names[1], seed=6)]
    for b, s in zip(batch, single):
        assert b.shape == s.shape
        assert np.abs(b - s).max() < 1e-3


def test_morph_endpoints(synth):
    a, b = synth.presets[0], synth.presets[1]
    m = synth.morph(NOTES, a, b, start=10.0, end=11.0, seed=0)   # morph after the audio ends
    assert np.abs(m - synth.render(NOTES, a, seed=0)).max() < 1e-4


def test_continuation_keeps_prefix(synth):
    t = synth.presets[0]
    full = synth.render(NOTES, t, duration=4.0, seed=0)
    cont = synth.continue_performance(NOTES, t, prefix=2.0, length=2.0, seed=0)
    assert cont.shape == full.shape
    n = 2 * synth.sample_rate
    assert np.abs(cont[:n] - full[:n]).max() < 1e-4                # causal: prefix unchanged
    assert float(np.sqrt((cont[n:] ** 2).mean())) > 0.01           # keeps playing


def test_timbre_resolution(synth):
    e = synth.timbre(synth.presets[3])
    assert e.shape == (512,) and abs(float(e.norm()) - 1) < 1e-4
    with pytest.raises(ValueError):
        synth.timbre("definitely not a preset or file")
    with pytest.raises(ValueError):
        synth.timbre(torch.zeros(7))


def test_preset_is_embedding_of_its_reference(synth):
    name = synth.presets[3]
    path = synth.preset_audio(name)
    assert path.suffix == ".flac"
    e = synth.embed_audio(str(path)).to(synth.device)
    assert float(e @ synth.preset(name)) > 0.999
