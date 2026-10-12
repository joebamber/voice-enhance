"""voice-enhance: local, Adobe-Podcast-style cleanup for spoken audio."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import __version__
from . import progress as prog
from .audio import SR, level_stage, load, loudness_stage, measure, polish_stage, require_ffmpeg, save_wav
from .backends import BACKENDS, available_backends, clearvoice_available, deepfilter, rebuild
from .declip import declip
from .dereverb import dereverb
from .protect import protect
from .gate import gate
from .guard import GuardSettings, apply_guard

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus",
              ".caf", ".mov", ".mp4", ".mkv", ".webm"}

GENTLE_DB = 8.0  # attenuation cap for the gentle pass the laughter guard falls back to


def log(msg: str) -> None:
    prog.write(msg)


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


def memo(cache: dict | None, kind: str, key: tuple, fn):
    """Reuse the last result of an expensive step while its inputs are unchanged.

    Holds one result per kind, so moving a slider in the GUI only redoes the
    steps after the AI cleanup instead of the whole file.
    """
    if cache is None:
        return fn(), False
    hit = cache.get(kind)
    if hit is not None and hit[0] == key:
        return hit[1], True
    value = fn()
    cache[kind] = (key, value)
    return value, False


def cleanup_key(backend: str) -> tuple:
    from . import backends as _b
    return backend, _b.DF_SETTINGS["max_reduction_db"]


def process(audio: np.ndarray, backend: str, *, mix: float, guard: GuardSettings | None,
            cache: dict | None = None, verbose: bool = False, rebuild_hf: bool = False,
            dry: float = 1.0, rebuild_above: float | None = None,
            repair_clipping: bool = True, protect_db: float = 0.0,
            gate_db: float = 30.0) -> tuple[np.ndarray, str]:
    """`cache` (one dict per source file) keeps the slow steps between calls."""
    notes = []
    if repair_clipping:
        prog.stage("declip")
        (audio, n_fixed), _ = memo(cache, "declip", (), lambda: declip(audio))
        if n_fixed:
            notes.append(f"repaired {n_fixed} clipped peaks")
    t = time.time()
    prog.stage("cleanup")
    key = (repair_clipping, cleanup_key(backend))
    cleaned, cached = memo(cache, "cleanup", key, lambda: BACKENDS[backend](audio))
    notes.append(f"{backend} " + ("(reused)" if cached else f"{time.time() - t:.1f}s"))
    # DeepFilterNet stays sample-aligned with the input, so loud (already clean)
    # speech can come from the original.
    if protect_db > 0 and backend == "deepfilter":
        cleaned, frac = protect(audio, cleaned, clean_above_db=protect_db)
        notes.append(f"loud speech kept from the original {frac * 100:.0f}% of the time")
    note = ", ".join(notes)
    if backend != "none" and guard and guard.strength > 0:
        prog.stage("guard")
        gentle, _ = memo(cache, "gentle", (repair_clipping,), lambda: deepfilter(audio, atten_lim_db=GENTLE_DB))
        cleaned, _, regions = apply_guard(audio, cleaned, gentle, guard)
        secs = sum(b - a for a, b in regions)
        note += f", laughter guard protected {len(regions)} moments ({secs:.1f}s)"
        if verbose and regions:
            note += "\n    at " + ", ".join(f"{ts(a)} ({b - a:.1f}s)" for a, b in regions)
    if mix < 1.0:
        cleaned = mix * cleaned + (1 - mix) * audio
    if backend != "none" and gate_db > 0:
        # after the mix, so --mix can't bring the background voices back
        cleaned, closed = gate(cleaned, below_db=gate_db)
        note += f", gate closed {closed * 100:.0f}% of the time"
    if backend != "none" and dry > 0:
        prog.stage("dereverb")
        cleaned = dereverb(cleaned, dry)
    if rebuild_hf:
        # after the guard, so laughs get rebuilt too instead of flipping between
        # a rebuilt and an un-rebuilt version
        t = time.time()
        prog.stage("rebuild")
        key += (protect_db, guard and guard.strength, mix, gate_db, dry, rebuild_above)
        (cleaned, what), cached = memo(cache, "rebuild", key, lambda: rebuild(cleaned, rebuild_above))
        note += f", {what} (" + ("reused" if cached else f"{time.time() - t:.1f}s") + ")"
    return cleaned, note


def rebuild_wanted(args) -> bool:
    """Studio mic generates its own top end from the voice below 8 kHz, so a rebuild before it is wasted."""
    return args.rebuild and not args.studio


def stages_for(args, backend: str, guard: GuardSettings | None, rebuild_hf: bool, decode: bool) -> list[str]:
    st = ["decode"] if decode else []
    if args.declip:
        st.append("declip")
    st.append("cleanup")
    if backend != "none" and guard and guard.strength > 0:
        st.append("guard")
    if backend != "none" and args.dereverb > 0:
        st.append("dereverb")
    if rebuild_hf:
        st.append("rebuild")
    return st + ["polish"] + (["studio"] if args.studio else []) + ["loudness"]


def short(name: str, n: int = 28) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def render(audio: np.ndarray, dst: Path, args, ref_lufs: float) -> str:
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.wav", Path(td) / "b.wav"
        save_wav(audio, a)
        prog.stage("polish")
        polish_stage(a, b, polish=not args.no_polish)
        if args.studio:
            # last of all, on the finished voice; only the level is set after it
            from . import sidon
            prog.stage("studio")
            voice = load(b)
            studio = sidon.run(voice)
            fc = sidon.CROSSOVER_HZ.get(args.studio_strength)
            if fc:
                # never above where the recording stops, or the band in between would be empty
                from .backends import bandwidth_edge
                edge = bandwidth_edge(voice)
                studio = sidon.split(voice, studio, min(fc, 0.95 * edge) if edge else fc)
            save_wav(studio, b)
        prog.stage("loudness")
        if args.lufs is not None:
            loudness_stage(b, dst, lufs=args.lufs, tp=args.true_peak)
        else:
            level_stage(b, dst, ref_lufs=ref_lufs)
    m = measure(dst)
    return f"{dst.name}  ({m['lufs']:.1f} LUFS, peak {m['true_peak']:.1f} dBTP)"


def source_lufs(audio: np.ndarray) -> float:
    with tempfile.TemporaryDirectory() as td:
        t = Path(td) / "src.wav"
        save_wav(audio, t)
        return measure(t)["lufs"]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="voice-enhance",
        description="Clean up spoken audio locally: AI denoise, laughter guard, podcast polish, loudness normalisation.",
    )
    ap.add_argument("inputs", nargs="*", help="audio/video files or folders")
    ap.add_argument("-o", "--output", help="output file (single input) or folder")
    ap.add_argument("-b", "--backend", default="deepfilter", choices=list(BACKENDS),
                    help="noise removal: deepfilter (default: DeepFilterNet AI noise removal) or none")
    ap.add_argument("--no-rebuild", dest="rebuild", action="store_false",
                    help="skip the rebuild step (regenerating missing high frequencies)")
    ap.add_argument("--rebuild-above", type=float, default=None, metavar="HZ",
                    help="force the rebuild to regenerate everything above this frequency (default: detect where "
                         "the recording's bandwidth stops; recordings that reach 12 kHz aren't rebuilt)")
    ap.add_argument("--protect", type=float, default=0.0, metavar="dB",
                    help="keep speech that's this far above the room noise from the original, so the AI can't duck "
                         "it; try 22 if loud syllables dip in level (default 0 = off: the original's noise stays out "
                         "of the speech)")
    ap.add_argument("--gate", type=float, default=30.0, metavar="dB",
                    help="silence anything this far below the main speaker, e.g. faint voices in the background "
                         "that the AI keeps because they're speech (default 30; 0 = off, higher = keep quieter sounds)")
    ap.add_argument("--max-reduction", type=float, default=None, metavar="dB",
                    help="deepfilter: cap how far any sound is pushed down (e.g. 12). Leaves a little natural "
                         "room tone instead of dead silence, which avoids watery/warbly artefacts")
    ap.add_argument("--studio", action="store_true",
                    help="finish by re-synthesising the voice with Sidon, so it sounds like a close, high-quality "
                         "podcast mic. Runs after everything else (only the level is set after it); replaces the "
                         "rebuild. Use -b none to give it the voice without AI noise removal")
    ap.add_argument("--studio-strength", default="subtle", choices=["subtle", "medium", "strong"],
                    help="with --studio: 'subtle' (default) keeps the real voice below 8 kHz and uses Sidon only "
                         "above it, which avoids its synthetic sound; 'medium' crosses over at 4 kHz; 'strong' is "
                         "all Sidon")
    ap.add_argument("--declip", action="store_true",
                    help="repair clipped (flat-topped) peaks before cleanup (default off)")
    ap.add_argument("--no-declip", dest="declip", action="store_false", help=argparse.SUPPRESS)  # the old default
    ap.add_argument("--dereverb", type=float, default=0.0, metavar="0-1",
                    help="extra DSP tail suppression after the AI (default 0 = off; the AI handles reverb)")
    ap.add_argument("--compare", action="store_true",
                    help="render DeepFilterNet with and without the laughter guard "
                         "into <name>_compare/ for A/B listening")
    ap.add_argument("-f", "--format", default="wav", help="output format: wav, flac, m4a, mp3, aiff (default wav)")
    ap.add_argument("--lufs", type=float, default=None,
                    help="normalise to this loudness with a true-peak limiter (off by default: output stays at the "
                         "source recording's level, loudness is left to the editor)")
    ap.add_argument("--true-peak", type=float, default=-1.5, help="true-peak ceiling in dBTP when --lufs is used (default -1.5)")
    ap.add_argument("--mix", type=float, default=1.0, help="wet/dry mix 0-1 (default 1.0)")
    ap.add_argument("--laughter-guard", type=float, default=0.85, metavar="0-1",
                    help="how strongly to protect laughs and other non-speech voice sounds (default 0.85, 0 = off)")
    ap.add_argument("--no-polish", action="store_true", help="skip EQ / de-ess / compression (loudness still applied)")
    ap.add_argument("-v", "--verbose", action="store_true", help="list the timestamps the laughter guard protected")
    ap.add_argument("--list-backends", action="store_true", help="list the noise-removal options")
    ap.add_argument("--gui", action="store_true",
                    help="open the app in your browser: sliders, and an A/B player against the original")
    ap.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def configure(args) -> None:
    """Check the settings and apply the ones the engines read globally."""
    if not 0 <= args.mix <= 1 or not 0 <= args.laughter_guard <= 1:
        raise SystemExit("--mix and --laughter-guard must be between 0 and 1")
    from . import backends as _b
    _b.DF_SETTINGS["max_reduction_db"] = args.max_reduction
    from . import sidon
    if args.studio and not sidon.available():
        raise SystemExit("--studio needs the transformers package. Reinstall with:\n"
                         "  uv tool install --python 3.11 . --force --reinstall-package voice-enhance --overrides overrides.txt")


def main(argv: list[str] | None = None) -> None:
    ap = parser()
    args = ap.parse_args(argv)
    # Apple Silicon GPU: let any op PyTorch hasn't implemented on MPS fall back to CPU
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # ClearVoice reads audio through pydub, which runs ffmpeg without -nostdin.
    # With a terminal on stdin, ffmpeg can swallow keypresses or get suspended,
    # hanging the run. We never read stdin, so point it at /dev/null.
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            null = os.open(os.devnull, os.O_RDONLY)
            os.dup2(null, 0)
            os.close(null)
    except OSError:
        pass

    if args.list_backends:
        print("\n".join(available_backends()))
        return

    if args.gui:
        require_ffmpeg()
        from . import gui
        gui.serve(args.inputs[0] if args.inputs else None)
        return
    if not args.inputs:
        ap.error("give at least one audio file or folder")
    require_ffmpeg()
    if rebuild_wanted(args) and not clearvoice_available():
        raise SystemExit("The rebuild step needs ClearerVoice. Reinstall with:\n"
                         "  uv tool install --python 3.11 . --force --reinstall-package voice-enhance --overrides overrides.txt\n"
                         "or run with --no-rebuild")
    configure(args)
    fmt = "." + args.format.lower().lstrip(".")
    guard = GuardSettings(strength=args.laughter_guard)
    files = gather(args.inputs)

    out_arg = Path(args.output).expanduser() if args.output else None
    single_file_out = out_arg is not None and out_arg.suffix and len(files) == 1 and not args.compare

    for src in files:
        cache: dict = {}
        started = time.time()

        if args.compare:
            log(f"\n{src.name}")
            audio = load(src)
            base = out_arg if out_arg else src.parent
            folder = base / f"{src.stem}_compare"
            folder.mkdir(parents=True, exist_ok=True)
            # Original at matched loudness (no EQ) so the comparison is fair.
            ref = source_lufs(audio)
            log("  original" + (" (loudness-matched only)" if args.lufs is not None else ""))
            with tempfile.TemporaryDirectory() as td:
                t = Path(td) / "o.wav"
                save_wav(audio, t)
                if args.lufs is not None:
                    loudness_stage(t, folder / f"0_original{fmt}", lufs=args.lufs, tp=args.true_peak)
                else:
                    level_stage(t, folder / f"0_original{fmt}", ref_lufs=float("nan"))
            variants = [("deepfilter", "deepfilter", guard),  # the default
                        ("deepfilter_no-guard", "deepfilter", None)]
            for n, (label, b, g) in enumerate(variants, start=1):
                with prog.progress(short(f"{n}_{label}", 30),
                                   stages_for(args, b, g, rebuild_wanted(args), decode=False), backend=b):
                    y, note = process(audio, b, mix=args.mix, guard=g, cache=cache,
                                      verbose=args.verbose, rebuild_hf=rebuild_wanted(args), dry=args.dereverb,
                                      repair_clipping=args.declip, protect_db=args.protect, rebuild_above=args.rebuild_above,
                                      gate_db=args.gate)
                    result = render(y, folder / f"{n}_{label}{fmt}", args, ref)
                log(f"  -> {result}")
                log(f"     {note}")
            log(f"  compare set in {folder}")
            continue

        if single_file_out:
            dst = out_arg
        else:
            folder = out_arg if out_arg else src.parent
            folder.mkdir(parents=True, exist_ok=True)
            dst = folder / f"{src.stem}_enhanced{fmt}"
        log(f"\n{src.name}")
        with prog.progress(short(src.stem), stages_for(args, args.backend, guard, rebuild_wanted(args), decode=True),
                           backend=args.backend):
            prog.stage("decode")
            audio = load(src)
            y, note = process(audio, args.backend, mix=args.mix, guard=guard, cache=cache,
                              rebuild_hf=rebuild_wanted(args), verbose=args.verbose, dry=args.dereverb, repair_clipping=args.declip, protect_db=args.protect,
                              rebuild_above=args.rebuild_above, gate_db=args.gate)
            result = render(y, dst, args, source_lufs(audio))
        mins, secs = divmod(time.time() - started, 60)
        log(f"  -> {result}  [{audio.size / SR / 60:.1f} min audio in {int(mins)}m{int(secs):02d}s]")
        log(f"     {note}")


if __name__ == "__main__":
    main()
