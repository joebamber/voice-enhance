"""Sidon: speech restoration that re-synthesises the voice as close-mic studio speech.

Two TorchScript models from sarulab-speech/sidon-v0.1 (MIT): w2v-BERT 2.0's first
layers fine-tuned to predict clean speech features from the recording, and a
vocoder that builds 48 kHz audio from them. It only hears up to 8 kHz (16 kHz
input) and generates everything above, so it replaces the rebuild step.

It runs last, on the finished voice. Its output is a new waveform, not a filtered
copy (waveform correlation with the input ~0.1), so it can't be blended with the
original: a wet/dry mix comb-filters, and crossovers that kept the real voice
below 600 Hz-8 kHz either changed nothing or kept the wobble. Experimental: its
pitch and harmonics can sound synthetic. w2v-BERT features carry pitch only roughly, so
the vocoder snaps glides into steps (an auto-tune sound); follow_pitch() moves its
output back onto the recording's own pitch contour.
Weights (~1.2 GB) come from Hugging Face on first use; CPU runs at ~10x real time.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

from . import accel, silence
from .audio import SR, spans, stitch

REPO = "sarulab-speech/sidon-v0.1"
# The demo uses 96 s chunks, but w2v-BERT's attention memory grows with the square
# of the chunk length: 96 s peaked at ~17 GB. So the file is cut into equal pieces of
# at most CHUNK_S, each seeing CONTEXT_S of audio either side so the joins don't start
# cold, and their features are cross-faded over FEATURE_FADE_S. The vocoder then runs
# on the joined features in DECODE_S windows, cross-faded the same way: decoding each
# piece on its own left an audible jump at every join, and a short last piece (e.g. the
# final 4 s of a 44 s file) stumbled.
CHUNK_S = 20
CONTEXT_S = 2
FEATURE_FADE_S = 0.4
DECODE_S, DECODE_PAD_S, DECODE_FADE_S = 20, 1, 0.2
FPS = 50  # feature frames per second (hop 160 at 16 kHz, stacked in pairs)
HOP = 16_000 // FPS  # input samples (16 kHz) per frame
UP = SR // FPS  # output samples (48 kHz) per frame
# Lifting quiet speech: words more than LIFT_BELOW_DB under the loudest speech are raised
# (by up to the --lift-quiet amount) before Sidon and lowered again after it. Nothing
# more than LIFT_FLOOR_DB under the loudest speech is lifted: that's room tone.
LIFT_BELOW_DB, LIFT_FLOOR_DB = 6.0, 40.0


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


def run(audio: np.ndarray, follow: float = 0.0, lift_db: float = 0.0) -> np.ndarray:
    """48 kHz mono in, 48 kHz mono out at the same RMS level, time-aligned with the input.
    `follow` (0-1): how far to move its pitch onto the input's (see follow_pitch).
    `lift_db`: raise quiet words by up to this much for Sidon, which stumbles and sounds
    more robotic on them, then lower them back (see _lift_gain)."""
    import hashlib
    key = (hashlib.sha1(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest(), lift_db, silence.enabled, accel.enabled)
    if _last.get("key") != key:
        _last.clear()
        if lift_db > 0:
            g = _lift_gain(audio, lift_db)
            y = silence.skip(audio * g, _synthesise) / g
            rms_in, rms_out = np.sqrt(np.mean(audio ** 2)), np.sqrt(np.mean(y ** 2))
            y = (y * (rms_in / rms_out) if rms_out > 0 else y).astype(np.float32)
        else:
            y = silence.skip(audio, _synthesise)
        _last.update(key=key, out=y)
    out = _last["out"].copy()
    if follow > 0:
        out = follow_pitch(out, audio, follow)
    from . import progress
    progress.fraction(1.0)
    return out


def _lift_gain(audio: np.ndarray, max_db: float) -> np.ndarray:
    """Per-sample gain that raises speech quieter than LIFT_BELOW_DB under the loudest
    speech by up to `max_db`, eased off towards room-tone level, moving over ~100 ms."""
    from scipy.ndimage import uniform_filter1d
    fr = SR // 50  # 20 ms
    m = audio.size // fr
    if m < 10:
        return np.ones_like(audio)
    p = uniform_filter1d((audio[: m * fr].reshape(m, fr) ** 2).mean(1), 10)  # 200 ms
    lv = 10 * np.log10(p + 1e-12)
    top = np.percentile(lv, 95)
    gain = np.clip(top - LIFT_BELOW_DB - lv, 0, max_db)
    gain *= np.clip((lv - (top - LIFT_FLOOR_DB)) / 10, 0, 1)
    g = 10 ** (uniform_filter1d(gain, 5) / 20)
    return np.interp(np.arange(audio.size), (np.arange(m) + 0.5) * fr, g).astype(np.float32)


PITCH_STEP = 0.01  # s
PITCH_FLOOR, PITCH_CEILING = 60.0, 600.0  # Hz
MAX_SHIFT = 0.5  # octaves; trackers disagreeing by more than this is an octave error, not Sidon
FOLLOW_CHUNK_S, FOLLOW_PAD_S, FOLLOW_FADE_S = 30, 0.5, 0.05


def available_follow() -> bool:
    import importlib.util
    return importlib.util.find_spec("parselmouth") is not None


def follow_pitch(y: np.ndarray, ref: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Move Sidon's voice onto the recording's own pitch contour, keeping its new sound.

    Both are pitch-tracked (Praat) on the same 10 ms grid; wherever both are voiced
    and agree to within half an octave, Sidon's pitch is shifted `strength` of the
    way to the recording's with PSOLA (Praat's overlap-add), which re-spaces the
    pitch periods without re-synthesising anything. Elsewhere it's left alone.
    Works in 30 s pieces, in parallel, cross-faded together; PSOLA keeps timing, so they line up.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from itertools import repeat
    chunk, pad = int(FOLLOW_CHUNK_S * SR), int(FOLLOW_PAD_S * SR)
    todo = spans(y.size, chunk, pad)
    done = None
    if len(todo) > 1:
        # each piece is independent and Praat runs on one core: spread them over all of them
        # (the workers can't start when Python was fed its script on stdin: then it's one at a time)
        try:
            with ProcessPoolExecutor(min(len(todo), os.cpu_count() or 1)) as pool:
                done = list(pool.map(_repitch, (y[a:b] for a, b in todo), (ref[a:b] for a, b in todo),
                                     repeat(strength)))
        except BrokenProcessPool:
            pass
    if done is None:
        done = [_repitch(y[a:b], ref[a:b], strength) for a, b in todo]
    parts = dict(zip(todo, done))
    return stitch(y.size, chunk, pad, int(FOLLOW_FADE_S * SR), lambda a, b, _: parts[(a, b)])


def _repitch(y: np.ndarray, ref: np.ndarray, strength: float) -> np.ndarray:
    import parselmouth
    from parselmouth.praat import call
    from scipy.ndimage import median_filter
    snd = parselmouth.Sound(y.astype(np.float64), SR)
    own = snd.to_pitch_ac(PITCH_STEP, PITCH_FLOOR, 15, False, 0.03, 0.45, 0.01, 0.35, 0.14, PITCH_CEILING)
    want = parselmouth.Sound(ref.astype(np.float64), SR).to_pitch_ac(
        PITCH_STEP, PITCH_FLOOR, 15, False, 0.03, 0.45, 0.01, 0.35, 0.14, PITCH_CEILING)
    t, f0 = own.xs(), own.selected_array["frequency"]
    fr = np.interp(t, want.xs(), want.selected_array["frequency"])
    # interpolating across a voiced/unvoiced edge gives a made-up low value: need both neighbours voiced
    ref_voiced = np.interp(t, want.xs(), (want.selected_array["frequency"] > 0).astype(float)) == 1
    voiced = f0 > 0
    shift = np.zeros_like(f0)
    ok = voiced & ref_voiced
    shift[ok] = np.log2(fr[ok] / f0[ok])
    shift[np.abs(shift) > MAX_SHIFT] = 0.0
    shift = median_filter(shift, 3)  # a single odd frame is a tracking glitch
    if not voiced.any():
        return y.copy()
    manip = call(snd, "To Manipulation", PITCH_STEP, PITCH_FLOOR, PITCH_CEILING)
    tier = call("Create PitchTier", "follow", snd.xmin, snd.xmax)
    for ti, fi, si in zip(t[voiced], f0[voiced], shift[voiced]):
        call(tier, "Add point", float(ti), float(fi * 2 ** (strength * si)))
    call([manip, tier], "Replace pitch tier")
    z = call(manip, "Get resynthesis (overlap-add)").values[0].astype(np.float32)
    return np.pad(z, (0, max(0, y.size - z.size)))[: y.size]


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
    x = torch.nn.functional.pad(x, (0, -x.numel() % HOP))
    frames = x.numel() // HOP

    def features(dev):
        model = fe.to(dev)

        def piece(a, b, done):
            feats = pre(torch.nn.functional.pad(x[a * HOP: b * HOP], (160, 160)).numpy(), sampling_rate=16_000,
                        return_tensors="pt")["input_features"]
            f = model(feats.to(dev))["last_hidden_state"][0].cpu().numpy()[: b - a]
            progress.fraction(0.9 * done)
            return np.pad(f, ((0, b - a - len(f)), (0, 0)), mode="edge")

        return stitch(frames, CHUNK_S * FPS, CONTEXT_S * FPS, int(FEATURE_FADE_S * FPS), piece)

    def decode(dev):
        model = dec.to(dev)

        def piece(a, b, done):
            f = torch.from_numpy(np.ascontiguousarray(feats[a // UP: b // UP].T[None]))
            progress.fraction(0.9 + 0.1 * done)
            z = model(f.to(dev)).view(-1).cpu().numpy()  # 26 samples short of b - a, the same for every window
            return np.pad(z, (0, max(0, b - a - z.size)))[: b - a]

        return stitch(frames * UP, DECODE_S * SR, DECODE_PAD_S * SR, int(DECODE_FADE_S * SR), piece, align=UP)

    with torch.inference_mode():
        feats = accel.attempt("Studio mic features", features)
        y = accel.attempt("Studio mic vocoder", decode)
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

