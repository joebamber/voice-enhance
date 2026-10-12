"""StuPASE: generative speech enhancement built to keep the speaker's own voice.

Cisco's model (cisco-ai/stupase, Apache 2.0; its code is vendored in
stupase_model/): WavLM fine-tuned to predict clean phonetic features, a
flow-matching transformer that draws a clean mel spectrogram from those features
*and* the recording's own spectrum, and a Vocos vocoder. Seeing the recording's
spectrum is what keeps it from drifting off the speaker's voice
("low hallucination"), where Sidon works from phonetic features alone.

It works at 16 kHz, so it hears and makes only up to 8 kHz, and its coarse mel
spectrogram blurs the 4-8 kHz band where an "s" lives (it comes out like an "f").
So by default it only supplies the voice below a crossover (~5 kHz) and the
recording supplies everything above: sibilants and breath are noise-like, so
the two don't need to line up in phase there. With the crossover off, the
rebuild regenerates the top end after it instead. Its output is a new waveform,
like Sidon's, so it can't be layered full-band with the original.
Weights (~2.2 GB) come from Hugging Face on first use; CPU runs at ~5x real time
(the Mac GPU was slower).
"""

from __future__ import annotations

import contextlib
import io
import warnings
from functools import lru_cache

import numpy as np

from .audio import SR

REPO = "cisco-ai/stupase"
FS = 16_000
# The model takes at most 120 s; chunks are cut in pauses, so each one is a fresh
# generation and the joins land where there's nothing to hear.
CHUNK_S = 30
SEARCH_S = 5  # each cut goes at the quietest 20 ms in the last SEARCH_S of a chunk
FADE = 160  # 10 ms each side of a cut
STEPS, CFG, SWAY = 8, 0.5, -1.0  # Cisco's defaults: flow steps, guidance strength, step spacing
SEED = 0  # the flow starts from random noise: fixed, so re-renders sound the same
TOP_HZ = 7500.0  # with no crossover, the rebuild regenerates everything above this
CROSSOVER_HZ = 5000.0


def available() -> bool:
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in ("x_transformers", "torchdiffeq"))


@lru_cache(maxsize=1)
def _load():
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download(REPO, f) for f in ("DeWavLM-R.pt", "CFM.pt", "Vocoder_Mel-16k.pt")]
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        from .stupase_model.stupase import StuPASE
        return StuPASE(*paths).eval()


_last: dict = {}  # one-entry cache: moving the crossover or the output level in the app doesn't re-run the model


def run(audio: np.ndarray, crossover_hz: float = CROSSOVER_HZ, rebuild_top: bool = True) -> np.ndarray:
    """48 kHz mono in, 48 kHz mono out, time-aligned with the input.
    StuPASE below `crossover_hz` and the input above it; with no crossover (0), all
    StuPASE, and with `rebuild_top` it moves on to the "rebuild" progress stage to
    regenerate above 8 kHz."""
    import hashlib
    from . import progress
    key = hashlib.sha1(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()
    if _last.get("key") != key:
        _last.clear()
        _last.update(key=key, out=_synthesise(audio))
    progress.fraction(1.0)
    y = _last["out"]
    if crossover_hz > 0:
        return _cross(y, audio, crossover_hz)
    if rebuild_top:
        progress.stage("rebuild")
        if "rebuilt" not in _last:
            from .backends import rebuild
            _last["rebuilt"], _ = rebuild(y, above_hz=TOP_HZ)
        y = _last["rebuilt"]
        progress.fraction(1.0)
    return y.copy()


def _cross(low: np.ndarray, high: np.ndarray, hz: float) -> np.ndarray:
    """`low` below `hz`, `high` above. Zero-phase and complementary (high minus its own
    lowpass), so with the same signal on both sides it gives back exactly that signal.
    StuPASE's band is matched to the level of the recording's below the crossover."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, hz, "low", fs=SR, output="sos")
    lo, hi_lo = sosfiltfilt(sos, low), sosfiltfilt(sos, high)
    r_lo, r_ref = np.sqrt(np.mean(lo ** 2)), np.sqrt(np.mean(hi_lo ** 2))
    if r_lo > 0:
        lo *= r_ref / r_lo
    return (lo + high - hi_lo).astype(np.float32)


def _cuts(x: np.ndarray) -> list[int]:
    """Chunk boundaries (16 kHz samples), each in the quietest 20 ms near CHUNK_S."""
    hop = FS // 50
    energy = np.convolve(x ** 2, np.ones(hop), "valid")[::hop]  # 20 ms windows on a 20 ms grid
    cuts, pos = [0], 0
    while x.size - pos > (CHUNK_S + SEARCH_S) * FS:
        lo, hi = (pos + (CHUNK_S - SEARCH_S) * FS) // hop, (pos + CHUNK_S * FS) // hop
        pos = (lo + int(np.argmin(energy[lo:hi]))) * hop + hop // 2
        cuts.append(pos)
    return cuts + [x.size]


def _synthesise(audio: np.ndarray) -> np.ndarray:
    import torch
    import torchaudio.functional as AF
    from . import progress
    peak = float(np.abs(audio).max())
    if peak == 0:
        return audio.copy()
    model = _load()
    x = AF.resample(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)), SR, FS)
    x = (0.9 * x / x.abs().max()).numpy()
    cuts = _cuts(x)
    y = np.zeros_like(x)
    ramp = np.linspace(0.0, 1.0, 2 * FADE, dtype=np.float32)
    with torch.inference_mode():
        for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
            s, e = max(0, a - FADE), min(x.size, b + FADE)
            torch.manual_seed(SEED)
            seg = model(torch.from_numpy(x[s:e]).view(1, -1), steps=STEPS, cfg_strength=CFG,
                        sway_sampling_coef=SWAY).view(-1).numpy()
            # crossfade over the 20 ms around each cut, which sits in a pause
            if a > 0:
                seg[: 2 * FADE] *= ramp
            if b < x.size:
                seg[-2 * FADE:] *= ramp[::-1]
            y[s:e] += seg
            progress.fraction((i + 1) / (len(cuts) - 1))
    y = AF.resample(torch.from_numpy(y), FS, SR).numpy()
    y = np.pad(y, (0, max(0, audio.size - y.size)))[: audio.size].astype(np.float32)
    from .sidon import _lag
    y = np.roll(y, -_lag(audio, y))
    rms_in, rms_out = np.sqrt(np.mean(audio ** 2)), np.sqrt(np.mean(y ** 2))
    return y * (rms_in / rms_out) if rms_out > 0 else y
