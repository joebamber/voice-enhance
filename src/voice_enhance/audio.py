"""ffmpeg-based decode / encode / polish helpers."""

from __future__ import annotations

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
# Level-independent: the cleaned voice is first brought to a fixed working
# loudness so the compressor threshold means the same thing on every file.
WORK_LUFS = -20.0


def polish_filters(toned: bool) -> list[str]:
    return [
        # sub-rumble only when the tone curve is shaping the low end, otherwise a normal voice high-pass
        "highpass=f=50:poles=2" if toned else "highpass=f=70:poles=2",
        "deesser=i=0.3:m=0.5:f=0.5",
        # gentle 2:1 levelling, slow enough to keep the voice's natural movement
        "acompressor=threshold=-19dB:ratio=2:attack=20:release=250:knee=6:detection=rms",
    ]


ENCODERS = {
    ".wav": ["-c:a", "pcm_s24le"],
    ".aif": ["-c:a", "pcm_s24be"],
    ".aiff": ["-c:a", "pcm_s24be"],
    ".flac": ["-c:a", "flac"],
    ".m4a": ["-c:a", "aac", "-b:a", "192k"],
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
}


def _render(src: Path, dst: Path, filters: list[str], codec: list[str]) -> None:
    _run(["ffmpeg", "-nostdin", "-hide_banner", "-y", "-i", str(src), "-af", ",".join(filters),
          "-ar", str(SR), "-ac", "1", *codec, str(dst)])


def polish_stage(src: Path, dst: Path, *, polish: bool, toned: bool = False) -> None:
    """Bring to working loudness, then (optionally) high-pass, de-ess and level."""
    m1 = measure(src)["lufs"]
    if not np.isfinite(m1):
        raise RuntimeError("Audio is silent after cleanup")
    _render(src, dst, [f"volume={WORK_LUFS - m1:.2f}dB"] + (polish_filters(toned) if polish else []),
            ["-c:a", "pcm_f32le"])


def loudness_stage(src: Path, dst: Path, *, lufs: float, tp: float) -> None:
    """Normalise to `lufs` with a true-peak ceiling of `tp` dBTP.

    A static gain into an oversampled limiter, rather than ffmpeg's loudnorm,
    which silently switches to dynamic compression on anything with a wide
    range and flattens the delivery.
    """
    import tempfile
    ext = dst.suffix.lower()
    if ext not in ENCODERS:
        raise SystemExit(f"Unsupported output format '{ext}'. Use one of: {', '.join(ENCODERS)}")
    ceiling = 10 ** (tp / 20)

    def chain(gain: float) -> list[str]:
        return [f"volume={gain:.2f}dB", "aresample=192000",  # 4x oversampling ~ true peak
                f"alimiter=limit={ceiling:.4f}:attack=1:release=60:level=0", f"aresample={SR}"]

    gain = lufs - measure(src)["lufs"]
    with tempfile.TemporaryDirectory() as td:
        # The limiter shaves a little loudness off; measure and correct once.
        probe = Path(td) / "probe.wav"
        _render(src, probe, chain(gain), ["-c:a", "pcm_f32le"])
        err = lufs - measure(probe)["lufs"]
        if abs(err) > 0.15:
            gain += err
        _render(src, dst, chain(gain), ENCODERS[ext])


def measure(path: Path) -> dict:
    """Integrated loudness / true peak of a finished file (for reporting)."""
    proc = _run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-af",
                 "ebur128=peak=true", "-f", "null", "-"])
    tail = proc.stderr[proc.stderr.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", tail)
    p = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", tail)
    return {"lufs": float(i.group(1)) if i else float("nan"),
            "true_peak": float(p.group(1)) if p else float("nan")}
