"""Sidon: speech restoration that re-synthesises the voice as close-mic studio speech.

Two TorchScript models from sarulab-speech/sidon-v0.1 (MIT): w2v-BERT 2.0's first
layers fine-tuned to predict clean speech features from the recording, and a
vocoder that builds 48 kHz audio from them. It only hears up to 8 kHz (16 kHz
input) and generates everything above, so it replaces the rebuild step.

It runs last, on the finished voice. Its output is a new waveform, not a filtered
copy (waveform correlation with the input ~0.1), so a wet/dry blend would
comb-filter. Its pitch and harmonics are what sound synthetic, so by default
only its top end is used: the real voice below a crossover, Sidon above it.
Weights (~1.2 GB) come from Hugging Face on first use; CPU runs at ~10x real time.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

from .audio import SR

REPO = "sarulab-speech/sidon-v0.1"
CHUNK_S = 96  # the demo's chunk length; features carry over between chunks


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


# the real voice below these frequencies, Sidon above; 'strong' is all Sidon
CROSSOVER_HZ = {"subtle": 8000.0, "medium": 4000.0}
_last: dict = {}  # one-entry cache: moving the crossover in the app doesn't re-run the model


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
    chunks = x.split(16_000 * CHUNK_S)
    out, cache = [], None
    with torch.inference_mode():
        for i, chunk in enumerate(chunks):
            feats = pre(torch.nn.functional.pad(chunk, (160, 160)).numpy(), sampling_rate=16_000,
                        return_tensors="pt")["input_features"]
            f = fe(feats)["last_hidden_state"]
            if cache is not None:
                f = torch.cat([cache, f], dim=1)
            out.append(dec(f.transpose(1, 2)).view(-1)[:-960])
            cache = f[:, -1:]
            progress.fraction((i + 1) / len(chunks))
    y = torch.cat(out).numpy()
    y = np.pad(y, (0, max(0, audio.size - y.size)))[: audio.size].astype(np.float32)
    y = np.roll(y, -_lag(audio, y))
    rms_in, rms_out = np.sqrt(np.mean(audio ** 2)), np.sqrt(np.mean(y ** 2))
    return y * (rms_in / rms_out) if rms_out > 0 else y


def _lag(x: np.ndarray, y: np.ndarray, max_ms: float = 50.0) -> int:
    """How many samples late y is (Sidon's output runs ~8 ms behind). Its phase is
    unrelated to the input, so this matches the 1-6 kHz envelopes (syllable timing)."""
    from scipy.signal import butter, correlate, sosfiltfilt
    band = butter(4, [1000, 6000], "band", fs=SR, output="sos")
    smooth = butter(2, 200, "low", fs=SR, output="sos")

    def env(a):
        e = sosfiltfilt(smooth, np.abs(sosfiltfilt(band, a)))
        return e - e.mean()

    c = correlate(env(y), env(x), mode="full", method="fft")
    lags = np.arange(-x.size + 1, y.size)
    win = np.abs(lags) <= int(SR * max_ms / 1000)
    return int(lags[win][np.argmax(c[win])])


def split(voice: np.ndarray, studio: np.ndarray, above_hz: float) -> np.ndarray:
    """The real voice below above_hz, Sidon's above. Zero-phase Butterworth pair:
    power-complementary, so with Sidon's phase unrelated to the input the two
    halves sum flat across the crossover."""
    from scipy.signal import butter, sosfiltfilt
    lo = sosfiltfilt(butter(4, above_hz, "low", fs=SR, output="sos"), voice)
    hi = sosfiltfilt(butter(4, above_hz, "high", fs=SR, output="sos"), studio)
    return (lo + hi).astype(np.float32)
