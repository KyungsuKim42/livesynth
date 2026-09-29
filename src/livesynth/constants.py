"""Model constants shared by the PyTorch and MLX code paths (no torch import)."""

VEL_LEVELS = (25, 50, 75, 100, 127)
N_VEL = len(VEL_LEVELS)
NONE = 0
OFFSET = 1
ONSET_BASE = 2                     # onset at level i  -> 2 + i
SUSTAIN_BASE = 2 + N_VEL           # sustain at level i -> 7 + i
N_STATES = 2 + 2 * N_VEL           # 12

MAX_NOTE_AGE = 4095                # frames (~41 s at 100 Hz)
AGE_ROT_DIM = 64                   # rotated subspace (32 pairs)
AGE_MIN_PERIOD = 20.0              # frames (0.2 s)
AGE_MAX_PERIOD = 5120.0            # frames (51.2 s)

FRAME_RATE = 100
N_PITCHES = 128


def velocity_level(velocity: float) -> int:
    """MIDI velocity (1..127) -> nearest quantisation level index 0..4."""
    return min(range(N_VEL), key=lambda i: abs(VEL_LEVELS[i] - velocity))
