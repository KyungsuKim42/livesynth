"""Measure the per-frame synthesis time of a streaming engine (no audio device).

    python -m livesynth.live.bench                 # auto backend, bf16
    python -m livesynth.live.bench --quant 8       # MLX int8
    python -m livesynth.live.bench --frames 6000   # sustained run (1 minute of audio)

Real time requires a mean well below 10 ms per frame (one 480-sample block).
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np

from livesynth.live.host import SynthHost


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "cuda", "cpu"])
    ap.add_argument("--quant", type=int, default=0, choices=[0, 4, 8])
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--out", default=None, help="write the result as JSON")
    args = ap.parse_args()

    host = SynthHost(model_dir=args.model_dir, backend=args.backend, quant=args.quant, audio=False)
    eng = host._make_engine()                                  # same thread as the steps
    if host.presets:
        host._set_engine_timbre(eng, next(iter(host.presets.values())))
    rng = np.random.default_rng(0)
    state = np.zeros(128, np.int64)
    age = np.zeros(128, np.int64)
    times = []
    for f in range(args.warmup + args.frames):
        if rng.random() < 0.05:                               # a few note events, like playing
            p = int(rng.integers(48, 84))
            state[:] = 0
            state[p] = 2 + 3
            age[:] = 0
        elif state.any():
            on = state >= 2
            state[on & (state <= 6)] += 5                      # onset -> sustain
            age[state >= 7] += 1
        t0 = time.perf_counter()
        host._step(eng, state, age, False)
        if f >= args.warmup:
            times.append((time.perf_counter() - t0) * 1e3)
    t = np.asarray(times)
    res = {"backend": host.backend, "quant": args.quant, "frames": args.frames,
           "mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
           "p95_ms": float(np.percentile(t, 95)), "p99_ms": float(np.percentile(t, 99)),
           "max_ms": float(t.max()), "real_time_factor": float(t.mean() / 10.0),
           "machine": platform.platform(), "processor": platform.processor()}
    try:
        import mlx.core as mx
        res["mlx"] = mx.__version__
    except ImportError:
        pass
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
