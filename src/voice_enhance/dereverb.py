"""Tail suppression ("dereverb").

Room reverb is what's left ringing after each word. A peak follower tracks
the recent level of the voice and lets it fall no faster than 40 dB/s; real
reverb dies away faster than that, so as soon as the signal drops more than
10 dB under the follower it's a tail, and it's pushed down (2:1, up to 12 dB).
Word onsets are always at the follower, so speech itself is untouched.

Measured on the EV Cafe test clip, this takes the drop 60-200 ms after each
word from ~20 dB (raw / DeepFilterNet) to ~27 dB, the same as Adobe.
"""

from __future__ import annotations

import numpy as np

from .audio import SR

FRAME = SR // 100  # 10 ms


def dereverb(audio: np.ndarray, strength: float = 1.0, *, thresh_db: float = 10.0, ratio: float = 2.0,
             max_att_db: float = 12.0, decay_db_s: float = 40.0, attack_ms: float = 3.0,
             release_ms: float = 40.0) -> np.ndarray:
    if strength <= 0:
        return audio
    n = audio.size // FRAME
    if n < 2:
        return audio
    frames = audio[: n * FRAME].reshape(n, FRAME)
    level = 10 * np.log10((frames.astype(np.float64) ** 2).mean(1) + 1e-12)

    follower = np.empty(n)
    p, step = -120.0, decay_db_s * FRAME / SR
    for i in range(n):
        p = max(level[i], p - step)
        follower[i] = p

    below = np.maximum(0.0, (follower - thresh_db) - level)
    target = np.minimum(max_att_db * strength, below * (ratio - 1) * strength)

    a = 1 - np.exp(-(FRAME / SR) / (attack_ms / 1000))
    r = 1 - np.exp(-(FRAME / SR) / (release_ms / 1000))
    att = np.empty(n)
    cur = 0.0
    for i in range(n):
        cur += (target[i] - cur) * (a if target[i] > cur else r)
        att[i] = cur

    gain_db = np.interp(np.arange(audio.size), (np.arange(n) + 0.5) * FRAME, -att)
    return (audio * 10 ** (gain_db / 20)).astype(np.float32)
