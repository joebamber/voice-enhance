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
from .guard import GuardSettings, apply_guard
from .tone import PROFILES, apply_eq, correction, target_from, with_presence

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


def process(audio: np.ndarray, backend: str, *, mix: float, guard: GuardSettings | None,
            gentle_cache: dict, verbose: bool = False, rebuild_hf: bool = False,
            dry: float = 1.0, rebuild_above: float | None = None,
            repair_clipping: bool = True, protect_db: float = 18.0) -> tuple[np.ndarray, str]:
    notes = []
    if repair_clipping:
        prog.stage("declip")
        audio, n_fixed = declip(audio)
        if n_fixed:
            notes.append(f"repaired {n_fixed} clipped peaks")
    t = time.time()
    prog.stage("cleanup")
    cleaned = BACKENDS[backend](audio)
    notes.append(f"{backend} {time.time() - t:.1f}s")
    # Mask-based cleaners stay sample-aligned with the input, so loud (already
    # clean) speech can come from the original; generative ones can't be blended.
    if protect_db > 0 and backend in ("deepfilter", "clearvoice", "resemble-denoise"):
        cleaned, frac = protect(audio, cleaned, clean_above_db=protect_db)
        notes.append(f"loud speech kept from the original {frac * 100:.0f}% of the time")
    note = ", ".join(notes)
    if backend != "none" and guard and guard.strength > 0:
        prog.stage("guard")
        if "g" not in gentle_cache:
            gentle_cache["g"] = deepfilter(audio, atten_lim_db=GENTLE_DB)
        cleaned, _, regions = apply_guard(audio, cleaned, gentle_cache["g"], guard)
        secs = sum(b - a for a, b in regions)
        note += f", laughter guard protected {len(regions)} moments ({secs:.1f}s)"
        if verbose and regions:
            note += "\n    at " + ", ".join(f"{ts(a)} ({b - a:.1f}s)" for a, b in regions)
    if mix < 1.0:
        cleaned = mix * cleaned + (1 - mix) * audio
    if backend != "none" and dry > 0:
        prog.stage("dereverb")
        cleaned = dereverb(cleaned, dry)
    if rebuild_hf:
        # after the guard, so laughs get rebuilt too instead of flipping between
        # a rebuilt and an un-rebuilt version
        t = time.time()
        prog.stage("rebuild")
        cleaned, what = rebuild(cleaned, rebuild_above)
        note += f", {what} ({time.time() - t:.1f}s)"
    return cleaned, note


def is_toned(args, tone: str | None = None) -> bool:
    tone = tone or args.tone
    return not (args.no_polish or tone == "neutral" or args.tone_amount == 0)


def stages_for(args, backend: str, guard: GuardSettings | None, rebuild_hf: bool, decode: bool,
               tone: str | None = None) -> list[str]:
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
    if is_toned(args, tone):
        st.append("tone")
    return st + ["polish", "loudness"]


def short(name: str, n: int = 28) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def render(audio: np.ndarray, dst: Path, args, ref_lufs: float, tone: str | None = None) -> str:
    toned = is_toned(args, tone)
    target = args._targets[tone or args.tone] if toned else None
    if toned:
        prog.stage("tone")
        curve = correction(audio, target, args.tone_amount)
        audio = apply_eq(audio, curve)
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.wav", Path(td) / "b.wav"
        save_wav(audio, a)
        prog.stage("polish")
        polish_stage(a, b, polish=not args.no_polish, toned=toned)
        if toned:
            # Compression reacts to the extra low end and shifts the balance a
            # little, so measure the polished result and trim once more.
            polished = load(b)
            trim = correction(polished, target, args.tone_amount)
            trim = [(f, max(-6.0, min(6.0, g))) for f, g in trim]
            save_wav(apply_eq(polished, trim), b)
            if args.verbose:
                total = {f: g for f, g in curve}
                for f, g in trim:
                    total[f] += g
                log("    tone: " + " ".join(f"{f}:{g:+.0f}" for f, g in total.items() if abs(g) >= 1))
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="voice-enhance",
        description="Clean up spoken audio locally: AI denoise, laughter guard, podcast polish, loudness normalisation.",
    )
    ap.add_argument("inputs", nargs="*", help="audio/video files or folders")
    ap.add_argument("-o", "--output", help="output file (single input) or folder")
    ap.add_argument("-b", "--backend", default="deepfilter", choices=list(BACKENDS),
                    help="engine: deepfilter (default: natural-sounding AI noise removal), resemble (generative re-synthesis; removes more reverb but can sound robotic, and is slow), resemble-denoise, clearvoice, apple, none")
    ap.add_argument("--no-rebuild", dest="rebuild", action="store_false",
                    help="skip the rebuild step (regenerating missing high frequencies)")
    ap.add_argument("--rebuild-above", type=float, default=None, metavar="HZ",
                    help="force the rebuild to regenerate everything above this frequency (default: detect where "
                         "the recording's bandwidth stops; full-band recordings aren't rebuilt)")
    ap.add_argument("--protect", type=float, default=18.0, metavar="dB",
                    help="keep speech that's this far above the room noise from the original, so the AI can't duck "
                         "it (default 18; 0 = off, higher = let the AI process more)")
    ap.add_argument("--max-reduction", type=float, default=None, metavar="dB",
                    help="deepfilter: cap how far any sound is pushed down (e.g. 12). Leaves a little natural "
                         "room tone instead of dead silence, which avoids watery/warbly artefacts")
    ap.add_argument("--quality", type=int, default=64, metavar="STEPS",
                    help="resemble: generation steps (default 64; 32 is ~2x faster, slightly rougher; up to 128)")
    ap.add_argument("--temperature", type=float, default=0.5, metavar="0-1",
                    help="resemble: prior temperature (default 0.5; lower = steadier, higher = more natural variation)")
    ap.add_argument("--denoise-first", action="store_true",
                    help="resemble: run its denoiser before re-synthesis (stronger cleanup; can sound more processed)")
    ap.add_argument("--no-declip", dest="declip", action="store_false",
                    help="don't repair clipped (flat-topped) peaks before cleanup")
    ap.add_argument("--dereverb", type=float, default=0.0, metavar="0-1",
                    help="extra DSP tail suppression after the AI (default 0 = off; the AI handles reverb)")
    ap.add_argument("--compare", action="store_true",
                    help="render every backend with and without the laughter guard and the rebuild "
                         "into <name>_compare/ for A/B listening")
    ap.add_argument("-f", "--format", default="wav", help="output format: wav, flac, m4a, mp3, aiff (default wav)")
    ap.add_argument("--lufs", type=float, default=None,
                    help="normalise to this loudness with a true-peak limiter (off by default: output stays at the "
                         "source recording's level, loudness is left to the editor)")
    ap.add_argument("--true-peak", type=float, default=-1.5, help="true-peak ceiling in dBTP when --lufs is used (default -1.5)")
    ap.add_argument("--mix", type=float, default=1.0, help="wet/dry mix 0-1 (default 1.0)")
    ap.add_argument("--laughter-guard", type=float, default=0.85, metavar="0-1",
                    help="how strongly to protect laughs and other non-speech voice sounds (default 0.85, 0 = off)")
    ap.add_argument("--tone", default="warm",
                    help="tone target: 'warm' (default: EQ toward Adobe's measured tone, adds the body a thin mic lacks), 'neutral', "
                         "or a path to a reference recording whose sound you want to match")
    ap.add_argument("--tone-amount", type=float, default=1.0, metavar="0-1",
                    help="how far to move toward the tone target (default 1.0)")
    ap.add_argument("--presence", type=float, default=5.0, metavar="dB",
                    help="clarity: lift 3-8 kHz relative to the tone target (default +5; 0 = exactly the target)")
    ap.add_argument("--warmth", type=float, default=-4.0, metavar="dB",
                    help="low end relative to the tone target: lower = less boom (default -4; 0 = Adobe's full low end)")
    ap.add_argument("--no-polish", action="store_true", help="skip EQ / de-ess / compression (loudness still applied)")
    ap.add_argument("-v", "--verbose", action="store_true", help="list the timestamps the laughter guard protected")
    ap.add_argument("--list-backends", action="store_true", help="show which backends work on this machine")
    ap.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
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

    if not args.inputs:
        ap.error("give at least one audio file or folder")
    require_ffmpeg()
    if args.rebuild and not clearvoice_available():
        raise SystemExit("The rebuild step needs ClearerVoice. Reinstall with:\n"
                         "  uv tool install --python 3.11 . --force --reinstall-package voice-enhance\n"
                         "or run with --no-rebuild")
    if not 0 <= args.mix <= 1 or not 0 <= args.laughter_guard <= 1:
        raise SystemExit("--mix and --laughter-guard must be between 0 and 1")
    if not 0 <= args.tone_amount <= 1:
        raise SystemExit("--tone-amount must be between 0 and 1")
    if args.tone != "neutral":
        if args.tone not in PROFILES and not Path(args.tone).expanduser().exists():
            raise SystemExit(f"--tone must be warm, neutral, or an existing reference file (got {args.tone})")
        args._targets = {args.tone: with_presence(target_from(args.tone), args.presence, args.warmth)}
    else:
        args._targets = {}
    args._targets.setdefault("warm", with_presence(target_from("warm"), args.presence, args.warmth))
    from . import backends as _b
    _b.DF_SETTINGS["max_reduction_db"] = args.max_reduction
    from . import resemble
    if not 1 <= args.quality <= 128 or not 0 <= args.temperature <= 1:
        raise SystemExit("--quality must be 1-128 and --temperature 0-1")
    resemble.SETTINGS.update(nfe=args.quality, temperature=args.temperature,
                             denoise_first=0.9 if args.denoise_first else 0.1)
    fmt = "." + args.format.lower().lstrip(".")
    guard = GuardSettings(strength=args.laughter_guard)
    files = gather(args.inputs)

    out_arg = Path(args.output).expanduser() if args.output else None
    single_file_out = out_arg is not None and out_arg.suffix and len(files) == 1 and not args.compare

    for src in files:
        gentle: dict = {}
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
            avail = available_backends()
            variants = [("deepfilter_warm", "deepfilter", guard, "warm"),        # the default
                        ("deepfilter_neutral", "deepfilter", guard, "neutral"),
                        ("deepfilter_no-guard", "deepfilter", None, "warm")]
            if "resemble" in avail:
                variants.append(("resemble_warm", "resemble", guard, "warm"))
            for n, (label, b, g, tone) in enumerate(variants, start=1):
                with prog.progress(short(f"{n}_{label}", 30),
                                   stages_for(args, b, g, args.rebuild, decode=False, tone=tone), backend=b):
                    y, note = process(audio, b, mix=args.mix, guard=g, gentle_cache=gentle,
                                      verbose=args.verbose, rebuild_hf=args.rebuild, dry=args.dereverb,
                                      repair_clipping=args.declip, protect_db=args.protect, rebuild_above=args.rebuild_above)
                    result = render(y, folder / f"{n}_{label}{fmt}", args, ref, tone=tone)
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
        with prog.progress(short(src.stem), stages_for(args, args.backend, guard, args.rebuild, decode=True),
                           backend=args.backend):
            prog.stage("decode")
            audio = load(src)
            y, note = process(audio, args.backend, mix=args.mix, guard=guard, gentle_cache=gentle,
                              rebuild_hf=args.rebuild, verbose=args.verbose, dry=args.dereverb, repair_clipping=args.declip, protect_db=args.protect,
                              rebuild_above=args.rebuild_above)
            result = render(y, dst, args, source_lufs(audio))
        mins, secs = divmod(time.time() - started, 60)
        log(f"  -> {result}  [{audio.size / SR / 60:.1f} min audio in {int(mins)}m{int(secs):02d}s]")
        log(f"     {note}")


if __name__ == "__main__":
    main()
