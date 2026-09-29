"""Headless tests of the real-time host (no audio device). Need released weights."""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("LIVESYNTH_WEIGHTS"),
                                reason="set LIVESYNTH_WEIGHTS to run host tests")


@pytest.fixture(scope="module")
def host():
    import torch
    from livesynth.live.host import SynthHost
    h = SynthHost(model_dir=os.environ["LIVESYNTH_WEIGHTS"],
                  backend="cuda" if torch.cuda.is_available() else "cpu",
                  audio=False, buffer_frames=1, seed=0)
    h.start()
    yield h
    h.stop()


def frames(h, n):
    return [h.pull() for _ in range(n)]


def test_keep_playing_after_release(host):
    host.keep_playing, host.autonomous, host.grace_s = True, False, 0.2
    host.panic()
    frames(host, 5)
    host.note_on(60, 100)
    host.note_on(64, 90)
    blocks = frames(host, 60)
    assert all(b.shape == (480,) for b in blocks)
    assert float(np.sqrt(np.mean(np.concatenate(blocks[20:]) ** 2))) > 0.01
    assert not host.stats().absent
    host.note_off(60)
    host.note_off(64)
    frames(host, 10)                      # within the 0.2-s grace period
    assert not host.stats().absent
    frames(host, 20)
    assert host.stats().absent            # the model now continues on its own
    host.note_on(67, 100)
    frames(host, 3)
    assert not host.stats().absent        # playing again takes over immediately
    host.note_off(67)


def test_stop_when_keep_playing_off(host):
    host.keep_playing, host.autonomous = False, False
    host.panic()
    host.note_on(60, 100)
    frames(host, 30)
    host.note_off(60)
    tail = frames(host, 250)
    assert not host.stats().absent
    assert float(np.sqrt(np.mean(np.concatenate(tail[-50:]) ** 2))) < 0.02   # decays to silence


def test_morph_and_slots(host):
    names = list(host.presets)
    host.set_slot(0, host.presets[names[0]])
    host.set_slot(1, host.presets[names[1]])
    host.set_morph(1.0)
    host.note_on(60, 100)
    frames(host, 40)
    assert abs(host._morph - 1.0) < 0.01  # smoothed position reached B
    host.set_morph(0.0)
    host.note_off(60)
    frames(host, 40)
    st = host.stats()
    assert st.frames > 0 and np.isfinite(st.frame_ms_mean)
