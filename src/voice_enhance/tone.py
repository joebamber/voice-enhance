"""Tone matching: nudge the cleaned voice toward a target long-term spectrum.

'warm' is the average spectrum of Adobe Enhance Speech v2 output, measured on
speech in the EV Cafe Peter Jones episode. Compared with a straight denoise it
has a big proximity-style low end (+15 dB around 80-125 Hz) and a softened
2-6 kHz presence region, which is most of why it sounds 'richer / warmer'.

Because the correction is computed per file (target minus what this file
actually has), a thin laptop mic gets more low end than a mic that is
already full. Boosts are capped so nothing gets silly.
"""

from __future__ import annotations

import numpy as np

from .audio import SR, load

BANDS = [40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
         1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000]

# dB relative to 1 kHz, speech frames only
PROFILES = {
    "warm": [-13.6, -11.6, -7.9, 4.7, 19.6, 20.4, 15.3, 17.1, 17.5, 14.9, 16.1, 14.1, 9.6, 1.5, 0.0,
             1.3, 0.3, -4.1, -5.2, -7.5, -9.2, -9.0, -11.0, -14.5, -20.3, -26.0, -36.6],
}

MAX_BOOST, MAX_CUT = 14.0, 12.0


def profile(audio: np.ndarray) -> np.ndarray:
    """1/3-octave long-term spectrum of the louder (speech) frames, dB re 1 kHz."""
    from scipy.signal import welch
    f0 = SR // 10
    n = audio.size // f0
    if n < 10:
        raise ValueError("need at least a second of audio to measure tone")
    frames = audio[: n * f0].reshape(n, f0)
    lv = 10 * np.log10((frames ** 2).mean(1) + 1e-12)
    speech = frames[lv > np.percentile(lv, 40)].ravel()
    f, p = welch(speech, SR, nperseg=16384)
    b = np.array([10 * np.log10(p[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].mean() + 1e-20) for c in BANDS])
    return b - b[BANDS.index(1000)]


def target_from(ref: str) -> np.ndarray:
    if ref in PROFILES:
        return np.array(PROFILES[ref])
    from pathlib import Path
    return profile(load(Path(ref).expanduser()))


# Where clarity lives: full weight 3-8 kHz (consonants, 'air' of the voice),
# nothing below 1.6 kHz so it doesn't turn nasal/honky.
_PRES_W = np.interp(np.log10(BANDS), np.log10([1600, 3000, 8000, 12500, 16000]), [0.0, 1.0, 1.0, 0.6, 0.3])


# Where warmth/boom lives: full weight up to 200 Hz, gone by 600 Hz.
_WARM_W = np.interp(np.log10(BANDS), np.log10([200, 600]), [1.0, 0.0])


def with_presence(target: np.ndarray, presence_db: float, warmth_db: float = 0.0) -> np.ndarray:
    """Adjust a tone target: presence_db lifts 3-8 kHz (clarity), warmth_db
    raises or lowers everything below ~400 Hz (body vs. boom)."""
    return target + presence_db * _PRES_W + warmth_db * _WARM_W


def correction(audio: np.ndarray, target: np.ndarray, amount: float = 1.0) -> list[tuple[int, float]]:
    diff = target - profile(audio)
    diff = np.convolve(np.pad(diff, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")  # smooth across bands
    diff -= diff[BANDS.index(1000)]
    diff = np.clip(diff * amount, -MAX_CUT, MAX_BOOST)
    # never boost below 70 Hz: that's rumble/handling territory, not voice
    diff = np.where((np.array(BANDS) < 70) & (diff > 0), 0.0, diff)
    return [(b, float(round(g, 2))) for b, g in zip(BANDS, diff)]


def apply_eq(audio: np.ndarray, curve: list[tuple[int, float]], taps: int = 8191) -> np.ndarray:
    """Linear-phase FIR EQ (delay-compensated, so it stays sample-aligned)."""
    from scipy.signal import fftconvolve, firwin2
    fb = np.array([f for f, _ in curve], dtype=float)
    gb = np.array([g for _, g in curve], dtype=float)
    grid = np.linspace(0, SR / 2, 2049)
    lg = np.interp(np.log10(np.maximum(grid, 1.0)), np.log10(fb), gb)  # flat beyond the end bands
    h = firwin2(taps, grid, 10 ** (lg / 20), fs=SR)
    y = fftconvolve(audio, h, mode="full")[taps // 2: taps // 2 + audio.size]
    return y.astype(np.float32)


def voice_weight(audio: np.ndarray, above_db: float = 12.0, width_db: float = 6.0,
                 attack_ms: float = 30.0, release_ms: float = 150.0) -> np.ndarray:
    """0..1 per sample: how much voice is present (level relative to the local
    noise floor), smoothed so it follows words and phrases, not waveforms."""
    from .guard import _local_floor
    frame = SR // 100
    n = audio.size // frame
    if n < 2:
        return np.ones(audio.size, dtype=np.float32)
    f = audio[: n * frame].reshape(n, frame).astype(np.float64)
    lv = np.convolve(10 * np.log10((f ** 2).mean(1) + 1e-12), np.ones(3) / 3, mode="same")
    floor = _local_floor(lv)
    target = np.clip((lv - (floor + above_db - width_db)) / (2 * width_db), 0.0, 1.0)
    a, r = 1 - np.exp(-10.0 / attack_ms), 1 - np.exp(-10.0 / release_ms)
    w = np.empty(n)
    cur = 0.0
    for i, t in enumerate(target):
        cur += (t - cur) * (a if t > cur else r)
        w[i] = cur
    return np.interp(np.arange(audio.size), (np.arange(n) + 0.5) * frame, w,
                     left=float(w[0]), right=float(w[-1])).astype(np.float32)


LOW_SPLIT_HZ = 300


def apply_eq_voice_only_lows(audio: np.ndarray, curve: list[tuple[int, float]],
                             pause_highpass_hz: float = 120.0) -> np.ndarray:
    """The warm curve's low-end boost is for the *voice*. Applied statically it
    also lifts room rumble, bumps and plosive thumps in the pauses (measured
    +15 dB). So: full curve while someone is speaking; in pauses the same curve
    with no boost below ~300 Hz and a gentle high-pass, crossfaded smoothly."""
    from scipy.signal import butter, sosfiltfilt
    warm = apply_eq(audio, curve)
    flat_curve = [(f, min(g, 0.0) if f < LOW_SPLIT_HZ else g) for f, g in curve]
    flat = apply_eq(audio, flat_curve)
    sos = butter(4, pause_highpass_hz, "high", fs=SR, output="sos")
    flat = sosfiltfilt(sos, flat).astype(np.float32)
    w = voice_weight(audio)
    return (w * warm + (1 - w) * flat).astype(np.float32)
