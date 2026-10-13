"""Finishing touches on the finished voice, after Studio mic: even out the level,
then shape the tone. Tuned by ear against Waves Voice ReGen on nathan-test.wav
(its loudness range was 4.8 LU against Sidon's 14, with ~4 dB more presence,
~3 dB less below 125 Hz and nothing above 12 kHz).

All filters are zero-phase (run forwards and backwards), in 60 s pieces cross-faded
together so an hour-long file doesn't need gigabytes of float64.
"""

from __future__ import annotations

import numpy as np

from .audio import SR, stitch

FRAME = SR // 50  # 20 ms
LEVEL_WINDOW = 20  # frames (400 ms) the level is measured over
LEVEL_SMOOTH = 10  # frames (200 ms) the gain moves over
LEVEL_SPEECH_DB = 30.0  # speech: within this of the loudest speech (its 95th percentile)
LEVEL_SILENCE_DB = 40.0  # further below than this, a held boost is dropped (cuts are kept)
LEVEL_MAX_CUT = 8.0  # dB
PRESENCE_HZ, PRESENCE_Q = 2800.0, 1.0
SHELF_HZ = 120.0
TOP_ORDER = 10  # doubled by running both ways: like Voice ReGen's cliff
PIECE_S, PAD_S, FADE_S = 60, 1, 0.05


def active(level_db: float, presence_db: float, shelf_db: float, top_hz: float) -> bool:
    return level_db > 0 or presence_db > 0 or shelf_db > 0 or top_hz > 0


def run(audio: np.ndarray, *, level_db: float = 0.0, presence_db: float = 0.0,
        shelf_db: float = 0.0, top_hz: float = 0.0) -> np.ndarray:
    """`level_db`: raise quieter speech by up to this much (and lower louder speech by
    up to LEVEL_MAX_CUT) towards its median; `presence_db`: boost around 2.8 kHz;
    `shelf_db`: cut below 120 Hz; `top_hz`: steep low-pass. 0 = off for each."""
    gain = _level_gain(audio, level_db) if level_db > 0 else None
    sos = _eq(presence_db, shelf_db, top_hz)

    def piece(a, b, _):
        from scipy.signal import sosfiltfilt
        x = audio[a:b].astype(np.float64)
        if gain is not None:
            x = x * np.interp(np.arange(a, b), (np.arange(gain.size) + 0.5) * FRAME, gain)
        if sos is not None:
            x = sosfiltfilt(sos, x)
        return x.astype(np.float32)

    return stitch(audio.size, PIECE_S * SR, PAD_S * SR, int(FADE_S * SR), piece)


def _level_gain(audio: np.ndarray, max_boost: float) -> np.ndarray:
    """Per-frame linear gain riding speech towards its median level. Outside speech the
    last speech gain is held, so the pauses aren't pumped up between words."""
    from scipy.ndimage import uniform_filter1d
    m = audio.size // FRAME
    if m < LEVEL_WINDOW:
        return np.ones(max(m, 1))
    p = (audio[: m * FRAME].astype(np.float64).reshape(m, FRAME) ** 2).mean(1)
    lv = 10 * np.log10(uniform_filter1d(p, LEVEL_WINDOW) + 1e-12)
    top = np.percentile(lv, 95)
    speech = lv > top - LEVEL_SPEECH_DB
    if not speech.any():
        return np.ones(m)
    g = np.clip(np.median(lv[speech]) - lv, -LEVEL_MAX_CUT, max_boost)
    last = 0.0
    for i in range(m):
        if speech[i]:
            last = g[i]
        else:
            g[i] = min(last, 0.0) if lv[i] < top - LEVEL_SILENCE_DB else last
    return 10 ** (uniform_filter1d(g, LEVEL_SMOOTH) / 20)


def _eq(presence_db: float, shelf_db: float, top_hz: float):
    """One second-order-sections filter for the tone settings that are on, or None."""
    from scipy.signal import butter, tf2sos
    parts = []
    if presence_db > 0:  # RBJ peaking
        A, w = 10 ** (presence_db / 40), 2 * np.pi * PRESENCE_HZ / SR
        al, c = np.sin(w) / (2 * PRESENCE_Q), np.cos(w)
        parts.append(tf2sos([1 + al * A, -2 * c, 1 - al * A], [1 + al / A, -2 * c, 1 - al / A]))
    if shelf_db > 0:  # RBJ low shelf, slope 1, cutting
        A, w = 10 ** (-shelf_db / 40), 2 * np.pi * SHELF_HZ / SR
        al, c, r = np.sin(w) / 2 * np.sqrt(2), np.cos(w), 2 * np.sqrt(10 ** (-shelf_db / 40))
        parts.append(tf2sos([A * ((A + 1) - (A - 1) * c + r * al), 2 * A * ((A - 1) - (A + 1) * c),
                             A * ((A + 1) - (A - 1) * c - r * al)],
                            [(A + 1) + (A - 1) * c + r * al, -2 * ((A - 1) + (A + 1) * c),
                             (A + 1) + (A - 1) * c - r * al]))
    if top_hz > 0:
        parts.append(butter(TOP_ORDER, min(top_hz, 0.45 * SR), "low", fs=SR, output="sos"))
    return np.vstack(parts) if parts else None
