"""Sidon: speech restoration that re-synthesises the voice as close-mic studio speech.

Two TorchScript models from sarulab-speech/sidon-v0.1 (MIT): w2v-BERT 2.0's first
layers fine-tuned to predict clean speech features from the recording, and a
vocoder that builds 48 kHz audio from them. It only hears up to 8 kHz (16 kHz
input) and generates everything above, so it replaces the rebuild step.

It runs last, on the finished voice. Its output is a new waveform, not a filtered
copy (waveform correlation with the input ~0.1), so it can't be blended with the
original: a wet/dry mix comb-filters, and crossovers that kept the real voice
below 600 Hz-8 kHz either changed nothing or kept the wobble. Experimental: its
pitch and harmonics can sound synthetic.
Weights (~1.2 GB) come from Hugging Face on first use; CPU runs at ~10x real time.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

from .audio import SR

REPO = "sarulab-speech/sidon-v0.1"
# The demo uses 96 s chunks, but w2v-BERT's attention memory grows with the square
# of the chunk length: 96 s peaked at ~17 GB. Each chunk also sees CONTEXT_S of the
# audio before it (features dropped afterwards) so the joins don't start cold.
CHUNK_S = 20
CONTEXT_S = 2
FPS = 50  # feature frames per second (hop 160 at 16 kHz, stacked in pairs)


def available() -> bool:
    import importlib.util
    return importlib.util.find_spec("transformers") is not None


@lru_cache(maxsize=1)
def _load():
    import torch
    from huggingface_hub import hf_hub_download
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from transformers import SeamlessM4TFeatureExtractor
        pre = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    fe = torch.jit.load(hf_hub_download(REPO, "feature_extractor_cpu.pt"), map_location="cpu").eval()
    dec = torch.jit.load(hf_hub_download(REPO, "decoder_cpu.pt"), map_location="cpu").eval()
    return pre, fe, dec


_last: dict = {}  # one-entry cache: changing only the output level in the app doesn't re-run the model


def run(audio: np.ndarray) -> np.ndarray:
    """48 kHz mono in, 48 kHz mono out at the same RMS level, time-aligned with the input."""
    import hashlib
    key = hashlib.sha1(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()
    if _last.get("key") != key:
        _last.clear()
        _last.update(key=key, out=_synthesise(audio))
    from . import progress
    progress.fraction(1.0)
    return _last["out"].copy()


def _synthesise(audio: np.ndarray) -> np.ndarray:
    import torch
    import torchaudio.functional as AF
    from . import progress
    peak = float(np.abs(audio).max())
    if peak == 0:
        return audio.copy()
    pre, fe, dec = _load()
    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).view(1, -1)
    x = AF.highpass_biquad(x, SR, 50)
    x = AF.resample(x, SR, 16_000).view(-1)
    x = torch.nn.functional.pad(0.9 * x / x.abs().max(), (0, 24_000))
    step, ctx = 16_000 * CHUNK_S, 16_000 * CONTEXT_S
    starts = range(0, x.numel(), step)
    out, cache = [], None
    with torch.inference_mode():
        for i, s in enumerate(starts):
            c = min(ctx, s)  # whole seconds, so it maps to exactly c // 320 frames
            chunk = x[s - c: s + step]
            feats = pre(torch.nn.functional.pad(chunk, (160, 160)).numpy(), sampling_rate=16_000,
                        return_tensors="pt")["input_features"]
            f = fe(feats)["last_hidden_state"][:, c * FPS // 16_000:]
            if cache is not None:
                f = torch.cat([cache, f], dim=1)
            out.append(dec(f.transpose(1, 2)).view(-1)[:-960])
            cache = f[:, -1:]
            progress.fraction((i + 1) / len(starts))
    y = torch.cat(out).numpy()
    y = np.pad(y, (0, max(0, audio.size - y.size)))[: audio.size].astype(np.float32)
    # its output sits on a small DC offset, which in gated silence is a near-constant signal
    from scipy.signal import butter, sosfiltfilt
    y = sosfiltfilt(butter(2, 20, "high", fs=SR, output="sos"), y).astype(np.float32)
    y = np.roll(y, -_lag(audio, y))
    rms_in, rms_out = np.sqrt(np.mean(audio ** 2)), np.sqrt(np.mean(y ** 2))
    return y * (rms_in / rms_out) if rms_out > 0 else y


def _lag(x: np.ndarray, y: np.ndarray, max_ms: float = 50.0) -> int:
    """How many samples late y is (Sidon's output runs ~8 ms behind). Its phase is
    unrelated to the input, so this matches the 1-6 kHz envelopes (syllable timing).
    The envelopes are built at 16 kHz and compared at 2 kHz, and only lags within
    max_ms are tried: a full-rate FFT correlation of a long file took several GB."""
    from scipy.signal import butter, resample_poly, sosfiltfilt
    sr, down = 16_000, 8
    band = butter(4, [1000, 6000], "band", fs=sr, output="sos")
    smooth = butter(4, 200, "low", fs=sr, output="sos")

    def env(a):
        e = sosfiltfilt(smooth, np.abs(sosfiltfilt(band, resample_poly(a, 1, SR // sr))))[::down]
        return (e - e.mean()).astype(np.float32)

    ex, ey = env(x), env(y)
    rate = sr // down
    m = int(rate * max_ms / 1000)
    n = min(ex.size, ey.size) - 2 * m
    if n <= 0:
        return 0
    c =np.array([np.dot(ey[m + k: m + k + n], ex[m: m + n]) for k in range(-m, m + 1)])
    i = int(np.argmax(c))
    frac = 0.0
    if 0 < i < c.size - 1:  # parabolic peak, for finer than 0.5 ms
        d = c[i - 1] - 2 * c[i] + c[i + 1]
        frac = 0.5 * (c[i - 1] - c[i + 1]) / d if d else 0.0
    return int(round((i - m + frac) * SR / rate))

