"""Resemble Enhance: the open model closest to Adobe Podcast's approach.

Two parts, both trained on noisy, reverberant, band-limited speech:
- denoiser: a mask-based cleanup (like DeepFilterNet, but it also takes out room reverb)
- enhancer: a generative model (latent flow matching + UnivNet vocoder) that
  re-synthesises the voice as clean, full-band studio speech -- i.e. it
  *rebuilds* the audio rather than filtering it, which is what Adobe does.

resemble-enhance pins training-only packages (deepspeed, gradio) that don't
install on a Mac; they're never used for inference, so a tiny stand-in for
deepspeed is registered before import. Weights (~1 GB) come from Hugging Face
on first use.
"""

from __future__ import annotations

import os
import sys
import types
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

from .audio import SR

CACHE = Path(os.environ.get("VOICE_ENHANCE_CACHE", Path.home() / ".cache" / "voice-enhance"))
REPO = "ResembleAI/resemble-enhance"
MODEL_SR = 44_100


def available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("resemble_enhance") is not None
    except Exception:
        return False


def _stub_deepspeed() -> None:
    try:
        import deepspeed  # noqa: F401
        return
    except Exception:
        pass

    def mod(name: str, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class _Unused:
        def __init__(self, *a, **k):
            raise RuntimeError("deepspeed is only needed for training resemble-enhance")

    ds = mod("deepspeed", DeepSpeedConfig=_Unused, init_distributed=lambda *a, **k: None)
    ds.accelerator = mod("deepspeed.accelerator", get_accelerator=lambda: None)
    ds.runtime = mod("deepspeed.runtime")
    ds.runtime.engine = mod("deepspeed.runtime.engine", DeepSpeedEngine=_Unused)
    ds.runtime.utils = mod("deepspeed.runtime.utils", clip_grad_norm_=lambda *a, **k: None)


def device():
    import torch
    if os.environ.get("VOICE_ENHANCE_DEVICE", "").lower() == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _run_dir() -> Path:
    from huggingface_hub import snapshot_download
    local = CACHE / "resemble-enhance"
    snapshot_download(repo_id=REPO, local_dir=str(local), allow_patterns=["enhancer_stage2/*"])
    return local / "enhancer_stage2"


@lru_cache(maxsize=1)
def _load():
    import dataclasses
    import logging
    import torch
    _stub_deepspeed()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from resemble_enhance.enhancer.enhancer import Enhancer
        from resemble_enhance.enhancer.hparams import HParams
    logging.getLogger("resemble_enhance").setLevel(logging.ERROR)
    run_dir = _run_dir()
    hp = HParams.load(run_dir)
    # the denoiser and stage-1 weights are inside the stage-2 checkpoint
    hp = dataclasses.replace(hp, denoiser_run_dir=None, enhancer_stage1_run_dir=None)
    model = Enhancer(hp)
    state = torch.load(run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt", map_location="cpu")
    model.load_state_dict(state["module"])
    model.eval()
    dev = device()
    model.to(dev)
    return model, dev


def _progress_trange():
    """Make resemble's chunk loop drive our progress bar."""
    import resemble_enhance.inference as ri
    from . import progress

    def trange(*args, **kwargs):
        items = list(range(*args))
        for i, v in enumerate(items):
            yield v
            progress.fraction((i + 1) / max(1, len(items)))

    ri.trange = trange


def run(audio: np.ndarray, mode: str = "enhance", nfe: int = 64, denoise_first: float = 0.9,
        temperature: float = 0.5) -> np.ndarray:
    """mode 'enhance' (denoise + generative rebuild) or 'denoise' (cleanup only)."""
    import torch
    import torchaudio.functional as AF
    from resemble_enhance.inference import inference
    model, dev = _load()
    _progress_trange()
    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    x = AF.resample(x, SR, MODEL_SR)
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mode == "denoise":
            y, sr = inference(model=model.denoiser, dwav=x, sr=MODEL_SR, device=dev)
        else:
            model.configurate_(nfe=nfe, solver="midpoint", lambd=denoise_first, tau=temperature)
            y, sr = inference(model=model, dwav=x, sr=MODEL_SR, device=dev)
    y = AF.resample(y.float().cpu(), sr, SR).numpy()
    out = np.zeros(audio.size, dtype=np.float32)
    n = min(out.size, y.size)
    out[:n] = y[:n]
    return out
