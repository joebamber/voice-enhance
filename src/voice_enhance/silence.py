"""Skip long silences in the slow models (AI dereverb, Sidon).

They do as much work on a minute of dead air as on a minute of speech. Long
stretches far below the speech level are cut out, the rest is run through the
model as one stream (so short pieces don't each pay for a whole model window),
and the output is put back in place; the skipped stretches keep the stage's
input. Only long gaps are skipped, with PAD_S either side still processed, so
word edges and reverb tails get the model and every cut sits in quiet audio.
"""

from __future__ import annotations

import numpy as np

from .audio import SR

BELOW_DB = 35.0  # a 100 ms frame this far under the speech level counts as silent
MIN_GAP_S = 2.5  # only silences at least this long are skipped
PAD_S = 0.5  # processed either side of a skipped stretch
FADE_S = 0.05  # model output fades into the skipped stretch over this much

enabled = True  # --process-silence turns it off


def regions(audio: np.ndarray) -> list[tuple[int, int]]:
    """Sample ranges to skip: silences of at least MIN_GAP_S, less PAD_S at each end."""
    from .backends import speech_level_db
    f = SR // 10
    n = audio.size // f
    if n < 5:
        return []
    db = 10 * np.log10((audio[: n * f].reshape(n, f).astype(np.float64) ** 2).mean(1) + 1e-12)
    quiet = np.concatenate([[False], db < speech_level_db(audio) - BELOW_DB, [False]])
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    pad = int(PAD_S * 10)
    out = []
    for a, b in zip(edges[::2], edges[1::2]):  # quiet frames a..b
        if b - a >= MIN_GAP_S * 10:
            # a gap reaching the start or end of the file needs no pad on that side
            s = 0 if a == 0 else (a + pad) * f
            e = audio.size if b == n else (b - pad) * f
            out.append((s, e))
    return out


def skip(audio: np.ndarray, fn) -> np.ndarray:
    """fn(audio) -> same-length audio, run only on the parts that aren't long silence."""
    gaps = regions(audio) if enabled else []
    if not gaps:
        return fn(audio)
    keep, pos = [], 0
    for s, e in gaps:
        if s > pos:
            keep.append((pos, s))
        pos = e
    if pos < audio.size:
        keep.append((pos, audio.size))
    if not keep:
        return audio.copy()
    y = fn(np.concatenate([audio[a:b] for a, b in keep]))
    out = audio.astype(np.float32)  # a copy: the skipped stretches keep the input
    fade = int(FADE_S * SR)
    pos = 0
    for a, b in keep:
        p = y[pos: pos + b - a]
        pos += b - a
        w = np.ones(b - a, dtype=np.float32)
        k = min(fade, (b - a) // 2)
        if a > 0:
            w[:k] = np.linspace(0, 1, k, dtype=np.float32)
        if b < audio.size:
            w[b - a - k:] = np.linspace(1, 0, k, dtype=np.float32)
        out[a:b] = w * p + (1 - w) * audio[a:b]
    return out


def skipped_fraction(audio: np.ndarray) -> float:
    return sum(e - s for s, e in regions(audio)) / max(1, audio.size) if enabled else 0.0
