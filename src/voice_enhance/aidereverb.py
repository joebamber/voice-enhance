"""AI dereverb: anvuew's dereverb_room, a BS-Roformer that removes room reverb from a voice.

It predicts a mask over the recording's own spectrogram, so the voice that comes
out is the recording's, not a re-synthesis: it stays in phase with the input and
can be blended with it (the strength setting). Trained on mono single-voice
recordings with simulated room reverb (pyroomacoustics), at 44.1 kHz.
Model code from ZFTurbo's Music-Source-Separation-Training (MIT, vendored in
roformer/); weights (~120 MB, GPL-3.0) come from Hugging Face on first use.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

from .audio import SR

REPO = "anvuew/dereverb_room"
CONFIG = "dereverb_room_anvuew.yaml"
WEIGHTS = "dereverb_room_anvuew_sdr_13.7432.ckpt"
FS = 44_100
CHUNK = 384_000  # samples per pass, the length it was trained on (~8.7 s)
FADE = CHUNK // 10  # neighbouring chunks overlap and crossfade over this much


def available() -> bool:
    import importlib.util
    return all(importlib.util.find_spec(m) is not None
               for m in ("beartype", "rotary_embedding_torch", "einops", "yaml"))


@lru_cache(maxsize=1)
def _load():
    import torch
    import yaml
    from huggingface_hub import hf_hub_download
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from .roformer.bs_roformer import BSRoformer
        with open(hf_hub_download(REPO, CONFIG)) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)["model"]
        cfg["use_torch_checkpoint"] = False  # a training-time memory saver
        model = BSRoformer(**cfg)
        model.load_state_dict(torch.load(hf_hub_download(REPO, WEIGHTS), map_location="cpu"))
    return model.eval()


def run(audio: np.ndarray) -> np.ndarray:
    """48 kHz mono in, the same with the room reverb removed (fully: blend for less)."""
    from . import accel
    if not audio.size or not np.any(audio):
        return audio.copy()
    return accel.attempt("AI dereverb", lambda dev: _run(audio, dev))


def _run(audio: np.ndarray, dev) -> np.ndarray:
    import torch
    import torchaudio.functional as AF
    from . import progress
    model = _load().to(dev)
    x = AF.resample(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)), SR, FS)
    n = x.numel()
    out, weight = torch.zeros(n), torch.zeros(n)
    ramp = torch.linspace(0.0, 1.0, FADE)
    step = CHUNK - FADE
    starts = list(range(0, max(1, n - FADE), step))
    with torch.inference_mode():
        for i, s in enumerate(starts):
            part = x[s: s + CHUNK]
            m = part.numel()
            y = model(torch.nn.functional.pad(part, (0, CHUNK - m)).view(1, 1, -1).to(dev)).reshape(-1)[:m].cpu()
            w = torch.ones(m)
            if s > 0:
                w[:FADE] = ramp[:m]
            if s + m < n:
                w[-FADE:] = ramp.flip(0)
            out[s: s + m] += y * w
            weight[s: s + m] += w
            progress.fraction((i + 1) / len(starts))
    y = AF.resample(out / weight.clamp(min=1e-6), FS, SR).numpy()
    return np.pad(y, (0, max(0, audio.size - y.size)))[: audio.size].astype(np.float32)
