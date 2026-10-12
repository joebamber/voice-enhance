"""Sidon: speech restoration that re-synthesises the voice as close-mic studio speech.

Two TorchScript models from sarulab-speech/sidon-v0.1 (MIT): w2v-BERT 2.0's first
layers fine-tuned to predict clean speech features from the recording, and a
vocoder that builds 48 kHz audio from them. It only hears up to 8 kHz (16 kHz
input) and generates everything above, so it replaces the rebuild step.

It runs last, on the finished voice. Its output is a new waveform, not a filtered
copy, so it can't be blended back with the input (that would comb-filter).
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


def run(audio: np.ndarray) -> np.ndarray:
    """48 kHz mono in, 48 kHz mono out at the same RMS level."""
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
    rms_in, rms_out = np.sqrt(np.mean(audio ** 2)), np.sqrt(np.mean(y ** 2))
    return y * (rms_in / rms_out) if rms_out > 0 else y
