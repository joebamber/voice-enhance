"""Background gate: silence what's far quieter than the main speaker.

DeepFilterNet removes steady noise but keeps anything voice-like, and it
can't tell the person at the mic from faint voices in the background. Those
sit 30+ dB below the speaker, so after cleanup a level gate takes them out:
anything more than `below_db` under the speaker's level is faded to silence.
It opens slightly early (so the first consonant isn't clipped) and holds for
a moment after each sound (so word endings and short gaps stay natural).
"""

from __future__ import annotations

import numpy as np

from .audio import SR
from .backends import speech_level_db

FRAME = SR // 100  # 10 ms


def gate(audio: np.ndarray, *, below_db: float = 30.0, width_db: float = 6.0, range_db: float = 80.0,
         lookahead_ms: float = 30.0, hold_ms: float = 200.0) -> tuple[np.ndarray, float]:
    """Returns the gated audio and the fraction of time it was closed."""
    n = audio.size // FRAME
    if n < 10:
        return audio, 0.0
    f = audio[: n * FRAME].reshape(n, FRAME).astype(np.float64)
    # level over ~50 ms: a syllable opens the gate, a single click doesn't
    lv = 10 * np.log10(np.convolve((f ** 2).mean(1), np.ones(5) / 5, mode="same") + 1e-14)
    thr = speech_level_db(audio) - below_db
    # 0 at thr - width, 1 at thr
    open_ = np.clip((lv - (thr - width_db)) / width_db, 0.0, 1.0)
    ahead, hold = int(lookahead_ms / 10), int(hold_ms / 10)
    padded = np.pad(open_, (hold, ahead))
    open_ = np.lib.stride_tricks.sliding_window_view(padded, hold + ahead + 1).max(1)
    open_ = np.convolve(open_, np.ones(5) / 5, mode="same")  # ~50 ms fades
    floor = 10 ** (-range_db / 20)
    g = floor + (1 - floor) * open_
    gs = np.interp(np.arange(audio.size), (np.arange(n) + 0.5) * FRAME, g, left=g[0], right=g[-1])
    return (audio * gs).astype(np.float32), float((open_ < 0.5).mean())
