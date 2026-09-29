"""Timbre embedding in a separate process.

Computing a CLAP embedding (audio resampling + a Transformer forward pass)
inside the synthesis process holds the Python GIL long enough to starve the
audio callback and cause dropouts, so it runs in a spawned child process. The
GUI submits jobs and polls for results.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import queue
from dataclasses import dataclass

import numpy as np


@dataclass
class EmbedResult:
    job_id: int
    slot: int
    label: str
    embedding: np.ndarray | None
    error: str | None = None


def _worker(model_dir: str | None, jobs, results) -> None:
    try:
        import torch
        from safetensors.torch import load_file
        from livesynth.hub import resolve_clap_checkpoint, resolve_model_dir
        from livesynth.timbre import TimbreEncoder
        d = resolve_model_dir(local_dir=model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc = TimbreEncoder(resolve_clap_checkpoint(), device,
                            load_file(str(d / "text_align.safetensors")))
        results.put(("ready", None))
    except Exception as exc:                                  # noqa: BLE001
        results.put(("fatal", f"timbre encoder unavailable: {type(exc).__name__}: {exc}"))
        return
    while True:
        job = jobs.get()
        if job is None:
            return
        job_id, slot, kind, payload, label = job
        try:
            if kind == "audio":
                e = enc.embed_audio(payload)
            else:
                e = enc.embed_text(payload, align="procrustes")
            results.put(("ok", EmbedResult(job_id, slot, label, e.cpu().numpy().astype(np.float32))))
        except Exception as exc:                              # noqa: BLE001
            results.put(("ok", EmbedResult(job_id, slot, label, None, f"{type(exc).__name__}: {exc}")))


class EmbedderProcess:
    def __init__(self, model_dir: str | None = None) -> None:
        ctx = mp.get_context("spawn")
        self._jobs = ctx.Queue()
        self._results = ctx.Queue()
        self._proc = ctx.Process(target=_worker, args=(model_dir, self._jobs, self._results),
                                 daemon=True)
        self._ids = itertools.count()
        self.ready = False
        self.error: str | None = None

    def start(self) -> None:
        self._proc.start()

    def submit_audio(self, slot: int, path: str, label: str) -> int:
        i = next(self._ids)
        self._jobs.put((i, slot, "audio", path, label))
        return i

    def submit_text(self, slot: int, prompt: str) -> int:
        i = next(self._ids)
        self._jobs.put((i, slot, "text", prompt, f'"{prompt}"'))
        return i

    def poll(self) -> list[EmbedResult]:
        out = []
        while True:
            try:
                kind, payload = self._results.get_nowait()
            except queue.Empty:
                return out
            if kind == "ready":
                self.ready = True
            elif kind == "fatal":
                self.error = payload
            else:
                out.append(payload)

    def close(self) -> None:
        try:
            self._jobs.put(None)
            self._proc.join(timeout=2)
        finally:
            if self._proc.is_alive():
                self._proc.terminate()
