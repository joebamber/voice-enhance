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


def _channels(path: Path) -> int:
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                           "stream=channels", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return max(1, int(proc.stdout.strip().splitlines()[0]))
    except (ValueError, IndexError):
        return 1


def load(path: Path) -> np.ndarray:
    """Decode any audio/video file to mono float32 at 48 kHz.

    Channels are averaged, so dual-mono recordings keep their exact level
    (ffmpeg's default downmix adds +3 dB and can push peaks over 0 dBFS).
    """
    n = _channels(path)
    mix = ["-af", "pan=mono|c0=" + "+".join(f"{1 / n:.6f}*c{i}" for i in range(n))] if n > 1 else []
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", *mix,
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


def _bounds(n: int, chunk: int, align: int = 1) -> list[int]:
    k = max(1, -(-n // chunk))
    return [round(i * n / k / align) * align for i in range(k)] + [n]


def spans(n: int, chunk: int, pad: int, align: int = 1) -> list[tuple[int, int]]:
    """The (a, b) ranges stitch() will ask its piece function for, in order."""
    bounds = _bounds(n, chunk, align)
    return [(max(0, s - pad), min(n, e + pad)) for s, e in zip(bounds, bounds[1:])]


def stitch(n: int, chunk: int, pad: int, fade: int, piece, align: int = 1) -> np.ndarray:
    """Run piece(a, b) -> rows a..b over [0, n) in equal pieces of at most `chunk` rows,
    each seeing `pad` more either side, and cross-fade them over `fade` at the joins.
    Piece boundaries are multiples of `align` (pass `pad` as one too)."""
    bounds = _bounds(n, chunk, align)
    k = len(bounds) - 1
    out = None
    for i, (s, e) in enumerate(zip(bounds, bounds[1:])):
        a, b = max(0, s - pad), min(n, e + pad)
        p = piece(a, b, (i + 1) / k)
        lo, hi = (s - fade // 2 if s > 0 else 0), (e + fade // 2 if e < n else n)
        w = np.ones(hi - lo, dtype=np.float32)
        if s > 0:
            w[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
        if e < n:
            w[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
        if out is None:
            out = np.zeros((n,) + p.shape[1:], dtype=np.float32)
        out[lo:hi] += w.reshape((-1,) + (1,) * (p.ndim - 1)) * p[lo - a: hi - a]
    return out


# --- Polish chain ----------------------------------------------------------
# Level-independent: the cleaned voice is first brought to a fixed working
# loudness so the compressor threshold means the same thing on every file.
WORK_LUFS = -20.0


def polish_filters() -> list[str]:
    return [
        "highpass=f=70:poles=2",
        # No de-esser: it took ~9 dB off every "s" and made speakers sound lispy.
        # No compression: dynamics are left to the editor, and compressing
        # lifts the room tails between words (measured: ~3.5 dB more reverb).
    ]


ENCODERS = {
    ".wav": ["-c:a", "pcm_s24le"],
    ".aif": ["-c:a", "pcm_s24be"],
    ".aiff": ["-c:a", "pcm_s24be"],
    ".flac": ["-c:a", "flac"],
    ".m4a": ["-c:a", "aac", "-b:a", "192k"],
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
}


def _input(path: Path) -> list[str]:
    """Name the format for our own WAVs: near-constant float data (Sidon's output in
    gated silence) can repeat byte 0x47 often enough that ffmpeg's probe takes the
    file for an MPEG-TS stream and ignores the WAV header."""
    return (["-f", "wav"] if path.suffix.lower() == ".wav" else []) + ["-i", str(path)]


def _render(src: Path, dst: Path, filters: list[str], codec: list[str]) -> None:
    _run(["ffmpeg", "-nostdin", "-hide_banner", "-y", *_input(src), "-af", ",".join(filters),
          "-ar", str(SR), "-ac", "1", *codec, str(dst)])


def polish_stage(src: Path, dst: Path, *, polish: bool) -> None:
    """Bring to working loudness, then (optionally) high-pass, de-ess and level."""
    m1 = measure(src)["lufs"]
    if not np.isfinite(m1):
        raise RuntimeError("Audio is silent after cleanup")
    _render(src, dst, [f"volume={WORK_LUFS - m1:.2f}dB"] + (polish_filters() if polish else []),
            ["-c:a", "pcm_f32le"])


def level_stage(src: Path, dst: Path, *, ref_lufs: float, headroom_db: float = -1.0) -> float:
    """Put the processed audio back at the source recording's level.

    No loudness normalisation, compression or limiting: a single static gain
    so the file drops into the edit at the same level as the raw recording.
    If that would push a sample above `headroom_db` dBFS, the whole file is
    turned down just enough instead, so it never clips. Returns the gain in dB.
    """
    import soundfile as sf
    ext = dst.suffix.lower()
    if ext not in ENCODERS:
        raise SystemExit(f"Unsupported output format '{ext}'. Use one of: {', '.join(ENCODERS)}")
    x, _ = sf.read(str(src), dtype="float32")
    gain = ref_lufs - measure(src)["lufs"] if np.isfinite(ref_lufs) else 0.0
    peak = float(np.abs(x).max()) or 1e-9
    peak_db = 20 * np.log10(peak)
    if peak_db + gain > headroom_db:
        gain = headroom_db - peak_db
    _render(src, dst, [f"volume={gain:.3f}dB"], ENCODERS[ext])
    return gain


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
    proc = _run(["ffmpeg", "-nostdin", "-hide_banner", *_input(path), "-af",
                 "ebur128=peak=true", "-f", "null", "-"])
    tail = proc.stderr[proc.stderr.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", tail)
    p = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", tail)
    return {"lufs": float(i.group(1)) if i else float("nan"),
            "true_peak": float(p.group(1)) if p else float("nan")}
