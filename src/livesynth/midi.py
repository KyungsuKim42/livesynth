"""MIDI handling: load note lists and convert them to per-frame model conditions.

A *note list* is a float array ``[K, 4]`` of ``(start_s, end_s, pitch, velocity)``.
:func:`load_notes` accepts a MIDI file path, a ``pretty_midi.PrettyMIDI`` object,
a ``[K, 3]`` / ``[K, 4]`` array, or a list of tuples, and returns that array.
:func:`notes_to_frames` turns it into the ``(state, age)`` tensors the model
consumes, at 100 frames per second.
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from livesynth.nn.midi_encoder import (
    MAX_NOTE_AGE, N_VEL, OFFSET, ONSET_BASE, SUSTAIN_BASE, velocity_level,
)

FRAME_RATE = 100
N_PITCHES = 128

MidiLike = Any   # str | Path | pretty_midi.PrettyMIDI | np.ndarray | Sequence[tuple]


def _from_pretty_midi(pm, drop_drums: bool, sustain_pedal: bool,
                      programs: Sequence[int] | None) -> np.ndarray:
    end_time = pm.get_end_time()
    rows: list[tuple[float, float, int, int]] = []
    for inst in pm.instruments:
        if drop_drums and inst.is_drum:
            continue
        if programs is not None and inst.program not in programs:
            continue
        downs: list[tuple[float, float]] = []
        if sustain_pedal:
            cur = None
            for c in sorted((c for c in inst.control_changes if c.number == 64),
                            key=lambda c: c.time):
                if c.value >= 64 and cur is None:
                    cur = c.time
                elif c.value < 64 and cur is not None:
                    downs.append((cur, c.time))
                    cur = None
            if cur is not None:
                downs.append((cur, end_time))
        starts: dict[int, list[float]] = {}
        for n in inst.notes:
            starts.setdefault(n.pitch, []).append(n.start)
        for v in starts.values():
            v.sort()
        for n in inst.notes:
            end = n.end
            for td, tu in downs:                     # extend to pedal release ...
                if td <= n.end < tu:
                    end = tu
                    break
            if end > n.end:                          # ... but not past a re-strike
                s = starts[n.pitch]
                j = bisect.bisect_right(s, n.start)
                if j < len(s):
                    end = min(end, max(n.end, s[j]))
            rows.append((n.start, end, n.pitch, n.velocity))
    return np.asarray(sorted(rows), dtype=np.float64).reshape(-1, 4)


def load_notes(midi: MidiLike, drop_drums: bool = True, sustain_pedal: bool = True,
               programs: Sequence[int] | None = None) -> np.ndarray:
    """Return a ``[K, 4]`` note array ``(start_s, end_s, pitch, velocity)``.

    For MIDI files all non-drum tracks are merged (select tracks with
    ``programs``). The sustain pedal (CC64) lengthens notes up to the pedal
    release or the next strike of the same pitch, matching how the training
    audio was rendered. Arrays without a velocity column get velocity 100.
    """
    if isinstance(midi, (str, Path)):
        import pretty_midi
        return _from_pretty_midi(pretty_midi.PrettyMIDI(str(midi)), drop_drums,
                                 sustain_pedal, programs)
    if type(midi).__name__ == "PrettyMIDI":
        return _from_pretty_midi(midi, drop_drums, sustain_pedal, programs)
    arr = np.asarray(midi, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 4))
    if arr.ndim != 2 or arr.shape[1] not in (3, 4):
        raise ValueError("note array must have shape [K, 3] or [K, 4]: "
                         "(start_s, end_s, pitch[, velocity])")
    if arr.shape[1] == 3:
        arr = np.concatenate([arr, np.full((len(arr), 1), 100.0)], axis=1)
    return arr[np.argsort(arr[:, 0], kind="stable")]


def notes_end(notes: np.ndarray) -> float:
    return float(notes[:, 1].max()) if len(notes) else 0.0


def notes_to_frames(notes: np.ndarray, n_frames: int, frame_rate: int = FRAME_RATE
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Note array -> ``state`` [N, 128] long and ``age`` [N, 128] long.

    Per pitch and frame the state is one of none / note-off / note-on (5
    velocity levels) / sustain (5 levels); when two events fall in the same
    frame the priority is note-on > note-off > sustain. ``age`` counts frames
    since the note-on of the note occupying that pitch (used for sustains only).
    """
    state = torch.zeros(n_frames, N_PITCHES, dtype=torch.long)
    age = torch.zeros(n_frames, N_PITCHES, dtype=torch.long)
    if len(notes) == 0 or n_frames == 0:
        return state, age
    sustains, offsets, onsets = [], [], []
    for s, e, p, vel in notes:
        p = int(p)
        if not 0 <= p < N_PITCHES:
            continue
        v = velocity_level(float(vel))
        on, off = int(s * frame_rate), int(e * frame_rate)
        if on == off or on >= n_frames or off < 0:
            continue
        lo, hi = max(on, 0), min(off, n_frames - 1)
        if hi >= lo:
            sustains.append((lo, hi, p, v, on))
        if 0 <= off < n_frames:
            offsets.append((off, p))
        if 0 <= on < n_frames:
            onsets.append((on, p, v))
    for lo, hi, p, v, on in sustains:
        state[lo:hi + 1, p] = SUSTAIN_BASE + v
        age[lo:hi + 1, p] = torch.clamp(torch.arange(lo, hi + 1) - on, max=MAX_NOTE_AGE)
    for f, p in offsets:
        state[f, p] = OFFSET
    for f, p, v in onsets:
        state[f, p] = ONSET_BASE + v
    return state, age


class NoteTracker:
    """Live note state for streaming: ``note_on`` / ``note_off`` events in,
    one ``(state[128], age[128])`` frame out per 10-ms step."""

    def __init__(self) -> None:
        self._held: dict[int, list[int]] = {}      # pitch -> [velocity level, age]
        self._onsets: dict[int, int] = {}          # pitch -> level, note-on this frame
        self._offsets: set[int] = set()

    def note_on(self, pitch: int, velocity: int = 100) -> None:
        if velocity <= 0:
            self.note_off(pitch)
            return
        lv = velocity_level(velocity)
        self._onsets[pitch] = lv
        self._held[pitch] = [lv, 0]

    def note_off(self, pitch: int) -> None:
        if pitch in self._held or pitch in self._onsets:
            self._held.pop(pitch, None)
            self._offsets.add(pitch)

    def all_notes_off(self) -> None:
        for p in list(self._held):
            self.note_off(p)

    @property
    def active(self) -> bool:
        return bool(self._held or self._onsets or self._offsets)

    def frame(self) -> tuple[np.ndarray, np.ndarray]:
        state = np.zeros(N_PITCHES, dtype=np.int64)
        age = np.zeros(N_PITCHES, dtype=np.int64)
        for p, (lv, a) in self._held.items():
            state[p] = SUSTAIN_BASE + lv
            age[p] = min(a, MAX_NOTE_AGE)
        for p in self._offsets:
            state[p] = OFFSET
        for p, lv in self._onsets.items():
            state[p] = ONSET_BASE + lv
            age[p] = 0
        for v in self._held.values():
            v[1] += 1
        self._onsets.clear()
        self._offsets.clear()
        return state, age


__all__ = ["FRAME_RATE", "N_VEL", "NoteTracker", "load_notes", "notes_end", "notes_to_frames"]
