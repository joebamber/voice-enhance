"""Repair hard clipping before cleanup.

Where the recorder ran out of headroom the waveform is flat-topped at the
ceiling. Those flat runs are rebuilt with a cubic spline through the good
samples either side, so the peak is rounded off the way it would have been.
Only short runs (up to ~2 ms) are touched; that's how clipping on speech
looks. The result can go past 0 dBFS, which is fine: everything downstream
is floating point, and the final gain stage leaves 1 dB of headroom.
"""

from __future__ import annotations

import numpy as np

MAX_RUN = 96      # samples (2 ms at 48 kHz)
CONTEXT = 12      # good samples used either side of a run


def find_clipping(x: np.ndarray, rel: float = 0.999) -> tuple[float, list[tuple[int, int]]]:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0:
        return 0.0, []
    at_ceiling = np.abs(x) >= peak * rel
    # a real clipping ceiling is hit many times; a single loud peak isn't clipping
    if at_ceiling.sum() < 8 or peak < 0.5:
        return peak, []
    d = np.diff(np.concatenate([[0], at_ceiling.astype(np.int8), [0]]))
    runs = [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    return peak, runs


def declip(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Returns the repaired audio and how many clipped runs were fixed."""
    from scipy.interpolate import CubicSpline
    peak, runs = find_clipping(x)
    if not runs:
        return x, 0
    y = x.astype(np.float32).copy()
    fixed = 0
    for a, b in runs:
        if b - a > MAX_RUN or a < CONTEXT or b + CONTEXT > x.size:
            continue
        idx = np.r_[a - CONTEXT:a, b:b + CONTEXT]
        good = idx[np.abs(x[idx]) < peak * 0.999]
        if good.size < 6:
            continue
        seg = CubicSpline(good, x[good].astype(np.float64))(np.arange(a, b))
        sign = np.sign(x[a:b])
        # keep the polarity and never go below the clipped level
        y[a:b] = (sign * np.clip(np.abs(seg), peak, peak * 4)).astype(np.float32)
        fixed += 1
    return y, fixed
