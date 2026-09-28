"""ffmpeg-based decode / encode / polish helpers."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48_000


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found. Install it with: brew install ffmpeg")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(cmd[:4])} ...):\n{proc.stderr[-2000:]}")
    return proc


def load(path: Path) -> np.ndarray:
    """Decode any audio/video file to mono float32 at 48 kHz."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn",
         "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Could not decode {path}:\n{proc.stderr.decode(errors='replace')[-2000:]}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError(f"No audio found in {path}")
    return audio


def save_wav(audio: np.ndarray, path: Path) -> None:
    sf.write(str(path), audio.astype(np.float32), SR, subtype="FLOAT")


# --- Polish chain ----------------------------------------------------------
# Roughly what a podcast engineer would do after cleanup: remove rumble, take
# out a little mud, add presence, tame sibilance, and level the dynamics.
# Loudness normalisation is done separately (two-pass loudnorm).
POLISH_FILTERS = [
    "highpass=f=80:poles=2",
    "equalizer=f=250:t=q:w=1.0:g=-2",     # mud / boxiness
    "equalizer=f=4500:t=q:w=1.2:g=2.5",   # presence / clarity
    "equalizer=f=11000:t=h:w=0.7:g=1.5",  # a touch of air (high shelf)
    "deesser=i=0.35:m=0.5:f=0.5",
    "acompressor=threshold=0.1:ratio=3:attack=8:release=160:knee=4",
]


def _loudnorm_measure(src: Path, pre_filters: list[str], lufs: float, tp: float, lra: float) -> dict:
    chain = ",".join(pre_filters + [f"loudnorm=I={lufs}:TP={tp}:LRA={lra}:print_format=json"])
    proc = _run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(src), "-af", chain, "-f", "null", "-"])
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr, re.S)
    if not m:
        raise RuntimeError("Could not parse loudnorm measurement output")
    return json.loads(m.group(0))


ENCODERS = {
    ".wav": ["-c:a", "pcm_s24le"],
    ".aif": ["-c:a", "pcm_s24be"],
    ".aiff": ["-c:a", "pcm_s24be"],
    ".flac": ["-c:a", "flac"],
    ".m4a": ["-c:a", "aac", "-b:a", "192k"],
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
}


def finish(src: Path, dst: Path, *, polish: bool, lufs: float, tp: float, lra: float = 11.0) -> dict:
    """Apply the polish chain + two-pass loudness normalisation and encode to dst.

    Returns the loudnorm measurement of the finished file's input to the final stage.
    """
    ext = dst.suffix.lower()
    if ext not in ENCODERS:
        raise SystemExit(f"Unsupported output format '{ext}'. Use one of: {', '.join(ENCODERS)}")
    pre = POLISH_FILTERS if polish else []
    meas = _loudnorm_measure(src, pre, lufs, tp, lra)
    loudnorm = (
        f"loudnorm=I={lufs}:TP={tp}:LRA={lra}"
        f":measured_I={meas['input_i']}:measured_TP={meas['input_tp']}"
        f":measured_LRA={meas['input_lra']}:measured_thresh={meas['input_thresh']}"
        f":offset={meas['target_offset']}:linear=true"
    )
    chain = ",".join(pre + [loudnorm])
    _run(["ffmpeg", "-nostdin", "-hide_banner", "-y", "-i", str(src), "-af", chain,
          "-ar", str(SR), "-ac", "1", *ENCODERS[ext], str(dst)])
    return meas


def measure(path: Path) -> dict:
    """Integrated loudness / true peak of a finished file (for reporting)."""
    proc = _run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-af",
                 "ebur128=peak=true", "-f", "null", "-"])
    tail = proc.stderr[proc.stderr.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", tail)
    p = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", tail)
    return {"lufs": float(i.group(1)) if i else float("nan"),
            "true_peak": float(p.group(1)) if p else float("nan")}
