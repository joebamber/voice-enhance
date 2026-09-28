"""One progress bar per output file, split into weighted stages.

Weights are rough relative costs measured on an Apple Silicon Mac, so the
percentage and time-remaining estimate move sensibly: the rebuild dominates.
Inside the cleanup and rebuild stages the bar also advances chunk by chunk.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

from tqdm import tqdm

WEIGHTS = {"decode": 1, "cleanup": 6, "guard": 3, "rebuild": 80, "tone": 2, "polish": 4, "loudness": 4}
LABELS = {"decode": "reading", "cleanup": "cleaning up", "guard": "laughter guard", "rebuild": "rebuilding",
          "tone": "tone", "polish": "polishing", "loudness": "loudness"}

_active: "Progress | None" = None


class Progress:
    def __init__(self, label: str, stages: list[str]):
        self.label = label
        self.weights = {s: WEIGHTS[s] for s in stages}
        self.total = float(sum(self.weights.values()))
        self.base = 0.0
        self.current: str | None = None
        self.bar = tqdm(
            total=self.total, desc=f"{label} · {LABELS[stages[0]]:<15}" if stages else label, file=sys.stderr, leave=True, dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
            bar_format="  {desc} {percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
        )

    def _set(self, value: float) -> None:
        self.bar.n = min(value, self.total)
        self.bar.refresh()

    def stage(self, name: str) -> None:
        if self.current is not None:
            self.base += self.weights.get(self.current, 0)
        self.current = name
        self.bar.set_description_str(f"{self.label} · {LABELS[name]:<15}", refresh=False)
        self._set(self.base)

    def fraction(self, frac: float) -> None:
        """Progress within the current stage (0..1)."""
        if self.current is not None:
            self._set(self.base + self.weights.get(self.current, 0) * max(0.0, min(1.0, frac)))

    def close(self) -> None:
        self.bar.set_description_str(f"{self.label} · {'done':<15}", refresh=False)
        self._set(self.total)
        self.bar.close()


@contextmanager
def progress(label: str, stages: list[str]):
    global _active
    p = Progress(label, stages)
    prev, _active = _active, p
    try:
        yield p
        p.close()
    except BaseException:
        p.bar.close()
        raise
    finally:
        _active = prev


def stage(name: str) -> None:
    if _active is not None:
        _active.stage(name)


def fraction(frac: float) -> None:
    if _active is not None:
        _active.fraction(frac)


def write(msg: str) -> None:
    """Print without breaking an active bar."""
    tqdm.write(msg, file=sys.stderr)
