"""Speech protection: stop the AI cleanup riding the volume of loud speech.

DeepFilterNet decides, many times a second, how much of the signal is noise.
On loud, slightly distorted syllables it sometimes decides "a lot" and ducks
the voice by 10-20 dB for a moment, which sounds like someone turning the
volume dial down and up.

But where the voice is far above the room's noise floor the original
recording is already clean -- the noise is masked. So on those moments we
crossfade to the (declipped) original, and only use the AI's output where
it's needed: pauses, breaths and quiet speech. The crossfade follows the
voice level smoothly, so there's no switching.
"""

from __future__ import annotations

import numpy as np

from .audio import SR
from .guard import _local_floor

FRAME = SR // 100  # 10 ms


def _levels(x: np.ndarray) -> np.ndarray:
    n = x.size // FRAME
    f = x[: n * FRAME].reshape(n, FRAME).astype(np.float64)
    db = 10 * np.log10((f ** 2).mean(1) + 1e-12)
    return np.convolve(db, np.ones(5) / 5, mode="same")  # ~50 ms


def protect(original: np.ndarray, cleaned: np.ndarray, *, clean_above_db: float = 22.0,
            width_db: float = 6.0, attack_ms: float = 20.0, release_ms: float = 150.0) -> tuple[np.ndarray, float]:
    """Returns the blended audio and the fraction of time taken from the original."""
    lo = _levels(original)
    floor = _local_floor(lo)
    # 0 at floor+clean_above-width, 1 at floor+clean_above+width
    target = np.clip((lo - (floor + clean_above_db - width_db)) / (2 * width_db), 0.0, 1.0)
    a = 1 - np.exp(-10.0 / attack_ms)
    r = 1 - np.exp(-10.0 / release_ms)
    w = np.empty_like(target)
    cur = 0.0
    for i, t in enumerate(target):
        cur += (t - cur) * (a if t > cur else r)
        w[i] = cur
    ws = np.interp(np.arange(original.size), (np.arange(w.size) + 0.5) * FRAME, w,
                   left=float(w[0]) if w.size else 0.0, right=float(w[-1]) if w.size else 0.0)
    out = ws * original + (1 - ws) * cleaned
    return out.astype(np.float32), float(ws.mean())
