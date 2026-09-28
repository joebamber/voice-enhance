"""Speech-cleanup backends. Each takes/returns mono float32 at 48 kHz, time-aligned
with the input and the same length, so outputs can be blended sample-for-sample."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings
from contextlib import contextmanager
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from .audio import SR, save_wav

CACHE = Path(os.environ.get("VOICE_ENHANCE_CACHE", Path.home() / ".cache" / "voice-enhance"))


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size >= n:
        return x[:n]
    return np.pad(x, (0, n - x.size))


def chunked(fn, audio: np.ndarray, chunk_s: float = 60.0, overlap_s: float = 1.0) -> np.ndarray:
    """Run fn over long audio in overlapping chunks, joined with linear crossfades.
    Keeps memory flat for hour-long recordings."""
    chunk, ov = int(chunk_s * SR), int(overlap_s * SR)
    if audio.size <= chunk:
        return _fit(fn(audio), audio.size)
    hop = chunk - ov
    out = np.zeros(audio.size, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, ov, dtype=np.float32)
    start = 0
    while start < audio.size:
        seg = audio[start:start + chunk]
        y = _fit(fn(seg), seg.size)
        if start > 0:
            k = min(ov, y.size)
            y[:k] *= fade_in[:k]
            out[start:start + k] *= 1.0 - fade_in[:k]
        out[start:start + y.size] += y
        if start + chunk >= audio.size:
            break
        start += hop
    return out


# --- DeepFilterNet 3 ------------------------------------------------------------

@contextmanager
def _quiet_stderr():
    """DeepFilterNet shells out to git and prints deprecation noise on load."""
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), fd)
        try:
            yield
        finally:
            os.dup2(saved, fd)
            os.close(saved)


@lru_cache(maxsize=1)
def _df():
    with warnings.catch_warnings(), _quiet_stderr():
        warnings.simplefilter("ignore")
        from loguru import logger
        logger.remove()  # DeepFilterNet logs chattily via loguru
        from df.enhance import init_df
        model, state, _ = init_df(log_level="ERROR")
    return model, state


def deepfilter(audio: np.ndarray, atten_lim_db: float | None = None) -> np.ndarray:
    """DeepFilterNet3. atten_lim_db caps how far any sound can be pushed down
    (None = unlimited). A low cap is a 'gentle' pass."""
    import torch

    model, state = _df()  # load first so its import noise is silenced
    from df.enhance import enhance

    def run(seg: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = enhance(model, state, torch.from_numpy(seg).unsqueeze(0), pad=True,
                        atten_lim_db=atten_lim_db)
        return y.squeeze(0).numpy()

    return chunked(run, audio)


# --- ClearerVoice-Studio (optional extra) ---------------------------------------

def clearvoice_available() -> bool:
    try:
        import clearvoice  # noqa: F401
        return True
    except Exception:
        return False


@contextmanager
def _in_dir(path: Path):
    # ClearVoice downloads weights into ./checkpoints relative to the CWD.
    path.mkdir(parents=True, exist_ok=True)
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@lru_cache(maxsize=2)
def _cv(task: str, model: str):
    from clearvoice import ClearVoice
    with _in_dir(CACHE / "clearvoice"):
        return ClearVoice(task=task, model_names=[model])


def _cv_run(task: str, model: str, audio: np.ndarray) -> np.ndarray:
    cv = _cv(task, model)

    def run(seg: np.ndarray) -> np.ndarray:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "in.wav"
            save_wav(seg, p)
            with _in_dir(CACHE / "clearvoice"):
                y = cv(input_path=str(p), online_write=False)
        if isinstance(y, dict):
            y = next(iter(y.values()))
        return np.asarray(y, dtype=np.float32).reshape(-1)

    return chunked(run, audio)


def clearvoice(audio: np.ndarray, super_resolution: bool = False) -> np.ndarray:
    if not clearvoice_available():
        raise SystemExit("ClearVoice isn't installed. Reinstall with: uv tool install '.[clearvoice]'")
    y = _cv_run("speech_enhancement", "MossFormer2_SE_48K", audio)
    if super_resolution:
        y = _cv_run("speech_super_resolution", "MossFormer2_SR_48K", y)
    return y


# --- Apple AUSoundIsolation (macOS) ----------------------------------------------

def apple_available() -> bool:
    return sys.platform == "darwin" and shutil.which("xcrun") is not None


def _apple_binary() -> Path:
    src = resources.files("voice_enhance").joinpath("apple/SoundIsolation.swift")
    binary = CACHE / f"soundisolation-{platform.machine()}"
    src_path = Path(str(src))
    if not binary.exists() or binary.stat().st_mtime < src_path.stat().st_mtime:
        CACHE.mkdir(parents=True, exist_ok=True)
        print("  compiling Apple SoundIsolation helper (one-off)...", file=sys.stderr)
        proc = subprocess.run(["xcrun", "swiftc", "-O", str(src_path), "-o", str(binary)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Swift compile failed:\n{proc.stderr}")
    return binary


def apple(audio: np.ndarray) -> np.ndarray:
    if not apple_available():
        raise SystemExit("The Apple backend needs macOS with Xcode command line tools (xcode-select --install).")
    import soundfile as sf
    binary = _apple_binary()
    with tempfile.TemporaryDirectory() as td:
        i, o = Path(td) / "in.wav", Path(td) / "out.wav"
        save_wav(audio, i)
        proc = subprocess.run([str(binary), str(i), str(o)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"AUSoundIsolation failed:\n{proc.stderr}")
        y, sr = sf.read(str(o), dtype="float32", always_2d=True)
    if sr != SR:
        raise RuntimeError(f"Unexpected sample rate from AUSoundIsolation: {sr}")
    return _fit(y.mean(axis=1), audio.size)


BACKENDS = {
    "deepfilter": lambda a: deepfilter(a),
    "clearvoice": lambda a: clearvoice(a),
    "clearvoice-sr": lambda a: clearvoice(a, super_resolution=True),
    "apple": lambda a: apple(a),
    "none": lambda a: a.copy(),
}


def available_backends() -> list[str]:
    names = ["none", "deepfilter"]
    if apple_available():
        names.append("apple")
    if clearvoice_available():
        names += ["clearvoice", "clearvoice-sr"]
    return names
