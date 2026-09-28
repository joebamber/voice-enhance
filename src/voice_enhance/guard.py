"""Laughter guard.

Speech enhancers (Adobe's included) are trained to keep *speech* and remove
everything else, so laughs, gasps, 'mmm's and sighs often get flattened.

The guard compares the original with the fully-cleaned signal frame by frame.
Where the original had a clear event well above the background noise floor
but the cleaner pushed it down hard, we crossfade to a *gently* cleaned version
(DeepFilterNet with capped attenuation) for that moment. Background noise
between phrases is never above the floor, so it stays removed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import SR

FRAME = SR // 100  # 10 ms


@dataclass
class GuardSettings:
    strength: float = 0.85     # 0 = off, 1 = fully restore flagged moments to the gentle pass
    event_db: float = 12.0     # how far above the noise floor something must be to count as an event
    suppressed_db: float = 10.0 # how much the full cleaner must have removed to count as 'flattened'
    strong_db: float = 15.0     # ...and at least one moment nearby must have been crushed this hard
    cluster_ms: float = 800.0   # laughter = repeated bursts; look for clusters within this window
    density: float = 0.2        # fraction of the window that must be flagged to count as a cluster
    bridge_ms: float = 600.0    # merge protected moments closer together than this (the gaps inside a laugh)
    long_ms: float = 250.0      # a single flagged stretch this long counts on its own (a sigh, a long 'haaa')
    hold_ms: float = 150.0     # keep the guard open a little after each flagged moment
    attack_ms: float = 15.0
    release_ms: float = 250.0


def _frame_db(x: np.ndarray) -> np.ndarray:
    n = x.size // FRAME
    f = x[: n * FRAME].reshape(n, FRAME)
    rms = np.sqrt(np.mean(f * f, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    # light smoothing (~30 ms)
    return np.convolve(db, np.ones(3) / 3, mode="same")


def _local_floor(db: np.ndarray, window_s: float = 8.0, pct: float = 10.0) -> np.ndarray:
    """Rolling background-noise estimate (10th percentile over ~8 s), so rooms or
    segments with more background noise don't get mistaken for 'events'."""
    hop, half = 100, int(window_s * 50)  # evaluate every 1 s over a +-4 s window
    centres = np.arange(0, db.size, hop)
    vals = np.array([np.percentile(db[max(0, c - half):c + half], pct) for c in centres])
    local = np.interp(np.arange(db.size), centres, vals)
    return np.maximum(local, np.percentile(db, pct))


def _runs(mask: np.ndarray):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))


def guard_mask(original: np.ndarray, cleaned: np.ndarray, s: GuardSettings) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Per-sample blend weight (0..strength) toward the gentle pass, plus the held regions (start, end) in seconds."""
    o, c = _frame_db(original), _frame_db(cleaned)
    floor = _local_floor(o)
    supp = o - c
    flagged = (o > floor + s.event_db) & (supp > s.suppressed_db)

    # Laughter is a *cluster* of bursts the cleaner crushed. Isolated blips
    # (breaths, plosives, a gap between words) are left to the cleaner.
    win = max(1, int(s.cluster_ms / 10))
    kernel = np.ones(win) / win
    dens = np.convolve(flagged.astype(np.float32), kernel, mode="same")
    strong = np.convolve((flagged & (supp > s.strong_db)).astype(np.float32), np.ones(win), mode="same") > 0
    core = flagged & (dens >= s.density) & strong
    long_f = int(s.long_ms / 10)
    for a, b in _runs(flagged):
        if b - a >= long_f:
            core[a:b] = True

    # bridge the small gaps inside a laugh, then hold a little afterwards
    hold_f, bridge_f = int(s.hold_ms / 10), int(s.bridge_ms / 10)
    runs = list(_runs(core))
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] < bridge_f:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    keep = np.zeros_like(core)
    for a, b in merged:
        keep[max(0, a - 2):min(b + hold_f, keep.size)] = True

    # attack / release envelope
    atk = 1 - np.exp(-10.0 / s.attack_ms)
    rel = 1 - np.exp(-10.0 / s.release_ms)
    env = np.zeros(keep.size, dtype=np.float32)
    level = 0.0
    for i, k in enumerate(keep):
        target = 1.0 if k else 0.0
        level += (target - level) * (atk if target > level else rel)
        env[i] = level

    t_frames = (np.arange(env.size) + 0.5) * FRAME
    per_sample = np.interp(np.arange(original.size), t_frames, env, left=float(env[0]) if env.size else 0.0,
                           right=float(env[-1]) if env.size else 0.0)
    regions = [(a * FRAME / SR, b * FRAME / SR) for a, b in _runs(keep)]
    return (per_sample * s.strength).astype(np.float32), regions


def apply_guard(original, cleaned, gentle, s: GuardSettings):
    m, regions = guard_mask(original, cleaned, s)
    return cleaned * (1 - m) + gentle * m, m, regions
