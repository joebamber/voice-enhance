"""Speech-cleanup backends. Each takes/returns mono float32 at 48 kHz, time-aligned
with the input and the same length, so outputs can be blended sample-for-sample."""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from contextlib import contextmanager
from functools import lru_cache
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
    from . import progress
    chunk, ov = int(chunk_s * SR), int(overlap_s * SR)
    if audio.size <= chunk:
        y = _fit(fn(audio), audio.size)
        progress.fraction(1.0)
        return y
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
        progress.fraction(min(1.0, (start + seg.size) / audio.size))
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


# DeepFilterNet's result depends on input level: on a quiet recording it
# leaves the background almost untouched (3-5 dB off) or, on a very quiet one,
# ducks the speech itself by 30 dB. Fed speech at this level it took 14-24 dB
# off the background on every test clip, with speech within 1 dB. Gains level
# off above -18.
DF_WORK_DB = -16.0
# How far below the file's speech level a chunk may sit and still be
# levelled up to DF_WORK_DB.
DF_LEVEL_RANGE_DB = 12.0


def speech_level_db(audio: np.ndarray) -> float:
    """Mean power of the louder 40% of 100 ms frames, dB re full scale."""
    f0 = SR // 10
    n = audio.size // f0
    if n < 5:
        return 20 * np.log10(np.sqrt((audio.astype(np.float64) ** 2).mean()) + 1e-9)
    lv = (audio[: n * f0].reshape(n, f0).astype(np.float64) ** 2).mean(1)
    return float(10 * np.log10(lv[lv >= np.percentile(lv, 60)].mean() + 1e-12))


def deepfilter(audio: np.ndarray, atten_lim_db: float | None = None) -> np.ndarray:
    """DeepFilterNet3. atten_lim_db caps how far any sound can be pushed down
    (None = unlimited). A low cap is a 'gentle' pass. Runs at a fixed working
    level (DF_WORK_DB) and hands back audio at the input's level."""
    import torch

    model, state = _df()  # load first so its import noise is silenced
    from df.enhance import enhance

    # Level is set per chunk, not once for the file: on a long recording the
    # speaker drifts (tires, leans back from the mic), and a chunk that reaches
    # the model 6-10 dB under the working level gets far less noise taken off,
    # so the background creeps back as the file goes on. The floor stops a
    # chunk that's mostly pause or room tone being boosted until the model
    # mistakes the noise for speech.
    floor_db = speech_level_db(audio) - DF_LEVEL_RANGE_DB

    def run(seg: np.ndarray) -> np.ndarray:
        gain = 10 ** ((DF_WORK_DB - max(speech_level_db(seg), floor_db)) / 20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = enhance(model, state, torch.from_numpy((seg * gain).astype(np.float32)).unsqueeze(0),
                        pad=True, atten_lim_db=atten_lim_db)
        return y.squeeze(0).numpy() / gain

    return chunked(run, audio).astype(np.float32)


# --- ClearerVoice-Studio: the rebuild ---------------------------------------

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


@contextmanager
def _silence():
    """Hide ClearVoice's per-call 'Running ...' line and its own tqdm bar
    (ours shows progress instead)."""
    import io
    from contextlib import redirect_stderr, redirect_stdout
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        yield


class _OutputToCPU:
    """Wraps ClearVoice's SR vocoder so its output comes back as float32 on the CPU.

    ClearVoice's sliding-window SR decoder writes each segment into a float64
    CPU tensor; with the model on Apple's GPU (MPS, no float64) that copy fails.
    Some torch/macOS pairings also refuse its long 1-D convolutions on MPS
    ("Output channels > 65536"); then the vocoder moves to the CPU for good.
    """

    def __init__(self, module):
        self._m = module
        self.expected = 0  # windows expected for the current file (for the progress bar)
        self.count = 0
        self.on_cpu = False

    def __call__(self, *a, **k):
        try:
            if self.on_cpu:
                a = [x.cpu() if hasattr(x, "cpu") else x for x in a]
            out = self._m(*a, **k)
        except NotImplementedError as e:
            if self.on_cpu or "MPS" not in str(e):
                raise
            self._m.to("cpu")
            self.on_cpu = True
            return self(*a, **k)
        out = out.float().cpu()
        self.count += 1
        if self.expected:
            from . import progress
            progress.fraction(self.count / self.expected)
        return out

    def __getattr__(self, name):
        return getattr(self._m, name)


def _force_cpu(cv) -> None:
    import torch
    for m in cv.models:
        m.device = torch.device("cpu")
        for sub in (list(m.model) if isinstance(m.model, (list, tuple, torch.nn.ModuleList)) else [m.model]):
            (sub._m if isinstance(sub, _OutputToCPU) else sub).to("cpu")


@lru_cache(maxsize=2)
def _cv(task: str, model: str):
    from clearvoice import ClearVoice
    import clearvoice.networks as cvn
    # ClearVoice wraps every call in its own tqdm bar; a second bar alongside ours
    # misbehaves on a real terminal, so make its loop a plain iterator.
    cvn.tqdm = lambda it, *a, **k: it
    with _in_dir(CACHE / "clearvoice"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = ClearVoice(task=task, model_names=[model])
    # Always use ClearVoice's own 4 s sliding window; one-pass decoding of 20 s
    # needs several GB of memory.
    for m in cv.models:
        m.args.one_time_decode_length = 4
        if task == "speech_super_resolution":
            # m.model is an nn.ModuleList [mossformer, vocoder], which won't hold a
            # plain wrapper, so swap in an ordinary list (decode only indexes it).
            parts = list(m.model)
            if len(parts) != 2:
                raise RuntimeError(f"Unexpected ClearVoice SR model layout ({len(parts)} parts)")
            m.model = [parts[0], _OutputToCPU(parts[1])]
    # Escape hatch if Apple's GPU path misbehaves: VOICE_ENHANCE_DEVICE=cpu
    if os.environ.get("VOICE_ENHANCE_DEVICE", "").lower() == "cpu":
        _force_cpu(cv)
    return cv


def _cv_run(task: str, model: str, audio: np.ndarray) -> np.ndarray:
    # One pass: ClearVoice's own 4 s sliding window keeps memory flat and
    # avoids the audible joins that crossfading separately generated pieces
    # produced (generated high band doesn't line up between pieces).
    cv = _cv(task, model)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "in.wav"
        save_wav(audio, p)
        with _in_dir(CACHE / "clearvoice"), warnings.catch_warnings(), _silence():
            warnings.simplefilter("ignore")
            y = cv(input_path=str(p), online_write=False)
    if isinstance(y, dict):
        y = next(iter(y.values()))
    return _fit(np.asarray(y, dtype=np.float32).reshape(-1), audio.size)


def _need_clearvoice() -> None:
    if not clearvoice_available():
        raise SystemExit("ClearerVoice isn't installed. Reinstall with:\n"
                         "  uv tool install --python 3.11 . --force --reinstall-package voice-enhance --overrides overrides.txt\n"
                         "or run without --rebuild")


_BAND_EDGE: dict = {}


def bandwidth_edge(audio: np.ndarray, drop_db: float = 20.0) -> float | None:
    """Where the recording's spectrum falls off a cliff (Hz), or None if it's full-band.

    A phone call stops dead around 3.4-4 kHz, Zoom/laptop audio around 7-8 kHz,
    a 22 kHz recording at 11 kHz. A good microphone just tapers off, with no
    cliff, so there's nothing to rebuild. (ClearVoice's own rule, "99% of the
    energy", puts the edge at ~8 kHz even on full-band speech and then replaces
    real sibilance with generated hiss.)
    """
    from scipy.signal import welch
    f0 = SR // 10
    n = audio.size // f0
    if n < 5:
        return None
    frames = audio[: n * f0].reshape(n, f0)
    lv = 10 * np.log10((frames ** 2).mean(1) + 1e-12)
    speech = frames[lv > np.percentile(lv, 50)].ravel()
    f, p = welch(speech, SR, nperseg=4096)
    level = 10 * np.log10(p + 1e-20)
    best = None
    for fc in f[(f >= 2500) & (f <= 0.93 * SR / 2)]:
        below = level[(f >= fc / 1.19) & (f < fc)].mean()
        above = level[(f > fc * 1.03) & (f <= fc * 1.25)].mean()
        drop = below - above
        if drop > drop_db and (best is None or drop > best[1]):
            best = (float(fc), float(drop))
    return best[0] if best else None


def _install_band_fix() -> None:
    """Make ClearVoice use our cutoff instead of re-detecting it per window
    (its per-window detection crashes on silence and wanders between windows)."""
    import clearvoice.utils.decode as dec
    from clearvoice.utils import bandwidth_sub as bs
    if getattr(dec.bandwidth_sub, "_voice_enhance", False):
        return

    def fixed_bandwidth_sub(low, high, fs=48000):
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        f_high = float(np.clip(_BAND_EDGE.get("hz", 0.95 * fs / 2), 1000.0, 0.95 * fs / 2))
        replaced = bs.replace_bandwidth(low, high, fs, 0.0, f_high)
        return bs.smooth_transition(replaced, low, fs)

    fixed_bandwidth_sub._voice_enhance = True
    dec.bandwidth_sub = fixed_bandwidth_sub


REBUILD_MAX_HZ = 12000.0  # recordings that reach this high aren't rebuilt


def rebuild(audio: np.ndarray, above_hz: float | None = None) -> tuple[np.ndarray, str]:
    """MossFormer2 48 kHz speech super-resolution, only where it's needed.

    Finds where the recording's bandwidth stops (or uses `above_hz`) and
    generates the missing band above it with a neural vocoder; everything
    below is kept as recorded. Full-band recordings are returned untouched.
    """
    from . import progress
    edge = above_hz or bandwidth_edge(audio)
    if edge is None:
        progress.fraction(1.0)
        return audio, "rebuild skipped (full bandwidth)"
    if above_hz is None and edge >= REBUILD_MAX_HZ:
        # e.g. the 16-17 kHz cutoff of an MP3/AAC export: nothing speech needs is missing
        progress.fraction(1.0)
        return audio, f"rebuild skipped (bandwidth to {_fmt_hz(edge)} is enough for speech)"
    _need_clearvoice()
    _install_band_fix()
    cv = _cv("speech_super_resolution", "MossFormer2_SR_48K")
    voc = cv.models[0].model[1]
    window, stride = 4 * SR, 3 * SR
    voc.count, voc.expected = 0, max(1, -(-(max(audio.size, window) - window) // stride) + 2)
    f_high = edge * 0.95 if above_hz is None else edge
    _BAND_EDGE["hz"] = f_high
    try:
        y = _cv_run("speech_super_resolution", "MossFormer2_SR_48K", audio)
    finally:
        _BAND_EDGE.clear()
        voc.expected = 0
    y = _cap_generated(y, audio, f_high)
    progress.fraction(1.0)
    return y, f"rebuilt above {_fmt_hz(edge)}"


def _cap_generated(y: np.ndarray, ref: np.ndarray, f_high: float, max_db: float = 6.0) -> np.ndarray:
    """Keep the generated part no louder than the real audio just below the cutoff.

    On windows with no speech (room tone, a quiet intro) the vocoder invents a
    loud hiss: 40-65 dB over the input, and ClearVoice's gentle crossover lets
    it leak well below the cutoff too. In natural speech the band above a
    cutoff sits within a few dB of the one below it (+12 dB on the strongest
    "s"), so at each moment the whole generated part is capped at `max_db`
    over the recording's [f_high/1.5, f_high) band.
    """
    from clearvoice.utils import bandwidth_sub as bs
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    from scipy.signal import istft, stft
    # y = kept + generated, with kept made exactly as in _install_band_fix
    kept = _fit(bs.lowpass_filter(np.asarray(ref, dtype=np.float64), SR, f_high), y.size)
    n = 1024
    f, _, G = stft(y - kept, SR, nperseg=n)
    _, _, R = stft(ref, SR, nperseg=n)
    gen = (np.abs(G) ** 2).sum(0)
    below = (np.abs(R[(f >= f_high / 1.5) & (f < f_high)]) ** 2).sum(0)
    m = min(gen.size, below.size)
    gain_db = np.minimum(0.0, 10 * np.log10((below[:m] * 10 ** (max_db / 10) + 1e-20) / (gen[:m] + 1e-20)))
    # ~50 ms smoothing, taking the lowest gain nearby so a hiss burst can't slip past
    gain_db = uniform_filter1d(minimum_filter1d(gain_db, 5), 5)
    G[:, :m] *= 10 ** (gain_db / 20)
    _, g = istft(G, SR, nperseg=n)
    return (kept + _fit(g, y.size)).astype(np.float32)


def _fmt_hz(hz: float) -> str:
    return f"{hz / 1000:.1f} kHz"


DF_SETTINGS = {"max_reduction_db": None}  # set from --max-reduction


BACKENDS = {
    "deepfilter": lambda a: deepfilter(a, atten_lim_db=DF_SETTINGS["max_reduction_db"]),
    "none": lambda a: a.copy(),
}


def available_backends() -> list[str]:
    return list(BACKENDS)
