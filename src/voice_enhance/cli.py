"""voice-enhance: local, Adobe-Podcast-style cleanup for spoken audio."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import __version__
from .audio import SR, finish, load, measure, require_ffmpeg, save_wav
from .backends import BACKENDS, available_backends, deepfilter
from .guard import GuardSettings, apply_guard

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus",
              ".caf", ".mov", ".mp4", ".mkv", ".webm"}

GENTLE_DB = 8.0  # attenuation cap for the gentle pass the laughter guard falls back to


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def gather(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for i in inputs:
        p = Path(i).expanduser()
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS and "_enhanced" not in f.stem)
        elif p.exists():
            files.append(p)
        else:
            raise SystemExit(f"Not found: {p}")
    if not files:
        raise SystemExit("No audio files found.")
    return files


def ts(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:04.1f}"


def process(audio: np.ndarray, backend: str, *, mix: float, guard: GuardSettings | None,
            gentle_cache: dict, verbose: bool = False) -> tuple[np.ndarray, str]:
    t = time.time()
    cleaned = BACKENDS[backend](audio)
    note = f"{backend} {time.time() - t:.1f}s"
    if backend != "none" and guard and guard.strength > 0:
        if "g" not in gentle_cache:
            gentle_cache["g"] = deepfilter(audio, atten_lim_db=GENTLE_DB)
        cleaned, _, regions = apply_guard(audio, cleaned, gentle_cache["g"], guard)
        secs = sum(b - a for a, b in regions)
        note += f", laughter guard protected {len(regions)} moments ({secs:.1f}s)"
        if verbose and regions:
            note += "\n      at " + ", ".join(f"{ts(a)} ({b - a:.1f}s)" for a, b in regions)
    if mix < 1.0:
        cleaned = mix * cleaned + (1 - mix) * audio
    return cleaned, note


def render(audio: np.ndarray, dst: Path, args) -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "stage.wav"
        save_wav(audio, tmp)
        finish(tmp, dst, polish=not args.no_polish, lufs=args.lufs, tp=args.true_peak)
    m = measure(dst)
    log(f"    -> {dst.name}  ({m['lufs']:.1f} LUFS, peak {m['true_peak']:.1f} dBTP)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="voice-enhance",
        description="Clean up spoken audio locally: AI denoise, laughter guard, podcast polish, loudness normalisation.",
    )
    ap.add_argument("inputs", nargs="*", help="audio/video files or folders")
    ap.add_argument("-o", "--output", help="output file (single input) or folder")
    ap.add_argument("-b", "--backend", default="deepfilter", choices=list(BACKENDS),
                    help="cleanup engine (default: deepfilter)")
    ap.add_argument("--compare", action="store_true",
                    help="render every available backend (+ guard on/off) into <name>_compare/ for A/B listening")
    ap.add_argument("-f", "--format", default="wav", help="output format: wav, flac, m4a, mp3, aiff (default wav)")
    ap.add_argument("--lufs", type=float, default=-16.0, help="target loudness (default -16; use -19 for mono spec)")
    ap.add_argument("--true-peak", type=float, default=-1.5, help="true-peak ceiling in dBTP (default -1.5)")
    ap.add_argument("--mix", type=float, default=1.0, help="wet/dry mix 0-1 (default 1.0)")
    ap.add_argument("--laughter-guard", type=float, default=0.85, metavar="0-1",
                    help="how strongly to protect laughs and other non-speech voice sounds (default 0.85, 0 = off)")
    ap.add_argument("--no-polish", action="store_true", help="skip EQ / de-ess / compression (loudness still applied)")
    ap.add_argument("-v", "--verbose", action="store_true", help="list the timestamps the laughter guard protected")
    ap.add_argument("--list-backends", action="store_true", help="show which backends work on this machine")
    ap.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    if args.list_backends:
        print("\n".join(available_backends()))
        return

    if not args.inputs:
        ap.error("give at least one audio file or folder")
    require_ffmpeg()
    if not 0 <= args.mix <= 1 or not 0 <= args.laughter_guard <= 1:
        raise SystemExit("--mix and --laughter-guard must be between 0 and 1")
    fmt = "." + args.format.lower().lstrip(".")
    guard = GuardSettings(strength=args.laughter_guard)
    files = gather(args.inputs)

    out_arg = Path(args.output).expanduser() if args.output else None
    single_file_out = out_arg is not None and out_arg.suffix and len(files) == 1 and not args.compare

    for src in files:
        log(f"\n{src.name}")
        audio = load(src)
        log(f"  {audio.size / SR:.1f}s of audio")
        gentle: dict = {}

        if args.compare:
            base = out_arg if out_arg else src.parent
            folder = base / f"{src.stem}_compare"
            folder.mkdir(parents=True, exist_ok=True)
            # Original at matched loudness (no EQ) so the comparison is fair.
            log("  original (loudness-matched only)")
            with tempfile.TemporaryDirectory() as td:
                t = Path(td) / "o.wav"
                save_wav(audio, t)
                finish(t, folder / f"0_original{fmt}", polish=False, lufs=args.lufs, tp=args.true_peak)
            n = 1
            for b in available_backends():
                variants = ([("polish-only", guard)] if b == "none"
                            else [(f"{b}+guard", guard), (f"{b}_no-guard", None)])
                for label, g in variants:
                    log(f"  {label}")
                    y, note = process(audio, b, mix=args.mix, guard=g, gentle_cache=gentle,
                                      verbose=args.verbose)
                    log(f"    {note}")
                    render(y, folder / f"{n}_{label}{fmt}", args)
                    n += 1
            log(f"  compare set in {folder}")
            continue

        if single_file_out:
            dst = out_arg
        else:
            folder = out_arg if out_arg else src.parent
            folder.mkdir(parents=True, exist_ok=True)
            dst = folder / f"{src.stem}_enhanced{fmt}"
        y, note = process(audio, args.backend, mix=args.mix, guard=guard, gentle_cache=gentle,
                           verbose=args.verbose)
        log(f"  {note}")
        render(y, dst, args)


if __name__ == "__main__":
    main()
