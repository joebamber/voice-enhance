"""De-hiss: take the recording's steady treble hiss out from under the speech.

Above ~5 kHz DeepFilterNet only has coarse band gains, and while someone
speaks it opens those bands to keep the S's and T's, hiss and all: on
gurjeet-2 it took ~1 dB off the 8-12 kHz hiss during speech (5 dB between
words). Lower down the voice masks it; up there it doesn't, so it's heard.

The hiss is steady, so its spectrum is learnt from the original between words
(where the 0-4 kHz band is quietest), and each treble bin is turned down as
far as it's no louder than that hiss. Sibilants sit well above it and pass.
Plain spectral processing, no AI; below DEHISS_FROM_HZ nothing changes.
"""

from __future__ import annotations

import numpy as np

from .audio import SR
from .backends import chunked

N_FFT = 1024
DEHISS_FROM_HZ = 4500.0  # fades in from here...
DEHISS_FULL_HZ = 6000.0  # ...to full strength here
OVER = 1.5  # treat bins up to this many times the hiss's power as hiss


def _stft(x: np.ndarray):
    from scipy.signal import stft
    return stft(x, SR, nperseg=N_FFT)


def hiss_profile(audio: np.ndarray, block_s: float = 60.0, gap_pct: float = 15.0) -> np.ndarray:
    """Mean power per bin in the quietest `gap_pct`% of frames (by 0-4 kHz level),
    averaged over blocks so memory stays flat on long files."""
    n = int(block_s * SR)
    total, count = None, 0
    for a in range(0, audio.size, n):
        seg = audio[a:a + n]
        if seg.size < N_FFT * 8:
            continue
        f, _, Z = _stft(seg)
        P = np.abs(Z) ** 2
        low = P[f < 4000].sum(0)
        gaps = low <= np.percentile(low, gap_pct)
        s = P[:, gaps].sum(1)
        total = s if total is None else total + s
        count += int(gaps.sum())
    if total is None:
        return np.zeros(N_FFT // 2 + 1)
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(total / count, 9)  # ~400 Hz smoothing across frequency


def dehiss(audio: np.ndarray, ref: np.ndarray, max_db: float = 12.0) -> np.ndarray:
    """Turn the treble down by up to `max_db` where it's no louder than the hiss
    measured in `ref` (the original, before cleanup). Same length as `audio`."""
    from scipy.ndimage import uniform_filter
    from scipy.signal import istft
    noise = hiss_profile(ref)
    floor = 10 ** (-max_db / 20)

    def run(seg: np.ndarray) -> np.ndarray:
        f, _, Z = _stft(seg)
        # power smoothed over ~3 bins x ~3 frames so the gain doesn't flicker (musical noise)
        P = uniform_filter(np.abs(Z) ** 2, size=(3, 3), mode="nearest")
        g = np.clip(1.0 - OVER * noise[:, None] / (P + 1e-20), floor, 1.0)
        g = uniform_filter(g, size=(3, 5), mode="nearest")
        w = np.clip((f - DEHISS_FROM_HZ) / (DEHISS_FULL_HZ - DEHISS_FROM_HZ), 0.0, 1.0)[:, None]
        _, y = istft(Z * (1.0 - w * (1.0 - g)), SR, nperseg=N_FFT)
        return y[: seg.size]

    return chunked(run, audio).astype(np.float32)
