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
from .protect import protect
from .dehiss import dehiss
from .gate import gate
from .guard import GuardSettings, apply_guard
from . import accel, silence

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
            rebuild_above: float | None = None,
            repair_clipping: bool = True, protect_db: float = 0.0,
            gate_db: float = 30.0, ai_dereverb: float = 0.0, dehiss_db: float = 0.0) -> tuple[np.ndarray, str]:
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
        # after the blend, so --blend can't bring the background voices back
        cleaned, closed = gate(cleaned, below_db=gate_db)
        note += f", gate closed {closed * 100:.0f}% of the time"
    if ai_dereverb > 0:
        # after the gate, so the whole voice is dereverbed (laughs and protected speech included);
        # it keeps the recording's phase, so the strength is a plain blend with the input
        from . import aidereverb
        t = time.time()
        prog.stage("aidereverb")
        ai_key = key + (protect_db, guard and guard.strength, mix, gate_db, silence.enabled, accel.enabled)
        skipped = silence.skipped_fraction(cleaned)
        wet, cached = memo(cache, "aidereverb", ai_key, lambda: silence.skip(cleaned, aidereverb.run))
        cleaned = cleaned + ai_dereverb * (wet - cleaned)
        note += ", AI dereverb (" + ("reused" if cached else f"{time.time() - t:.1f}s, {accel.where('AI dereverb')}") + ")"
        if skipped > 0:
            note += f", {skipped * 100:.0f}% skipped as silence"
    if dehiss_db > 0 and backend != "none":
        # after the AI steps, so changing it doesn't re-run them; the hiss is
        # measured in the original, where the gaps between words still have it
        prog.stage("dehiss")
        cleaned = dehiss(cleaned, audio, max_db=dehiss_db)
        note += f", de-hiss {dehiss_db:g} dB"
    if rebuild_hf:
        # after the guard, so laughs get rebuilt too instead of flipping between
        # a rebuilt and an un-rebuilt version
        t = time.time()
        prog.stage("rebuild")
        key += (protect_db, guard and guard.strength, mix, gate_db, ai_dereverb, dehiss_db, rebuild_above)
        (cleaned, what), cached = memo(cache, "rebuild", key, lambda: rebuild(cleaned, rebuild_above))
        note += f", {what} (" + ("reused" if cached else f"{time.time() - t:.1f}s") + ")"
    return cleaned, note


# studio models, and the chain: Sidon's studio sound, then StuPASE re-rendering it
USES_SIDON = ("sidon", "sidon+stupase")
USES_STUPASE = ("stupase", "sidon+stupase")


def rebuild_wanted(args) -> bool:
    """Sidon, and StuPASE with no crossover, make the whole top end from the voice below
    8 kHz, so a rebuild before them is wasted. With a crossover, the top end above it comes
    from the recording, so it's rebuilt as usual."""
    return args.rebuild and not (args.studio in USES_SIDON or (args.studio in USES_STUPASE and not args.stupase_crossover))


def stupase_rebuilds(args) -> bool:
    """With no crossover StuPASE stops at 8 kHz, so the rebuild runs after it instead of before."""
    return args.studio in USES_STUPASE and not args.stupase_crossover and args.rebuild and clearvoice_available()


def stages_for(args, backend: str, guard: GuardSettings | None, rebuild_hf: bool, decode: bool) -> list[str]:
    st = ["decode"] if decode else []
    if args.declip:
        st.append("declip")
    st.append("cleanup")
    if backend != "none" and guard and guard.strength > 0:
        st.append("guard")
    if args.ai_dereverb > 0:
        st.append("aidereverb")
    if backend != "none" and args.dehiss > 0:
        st.append("dehiss")
    if rebuild_hf:
        st.append("rebuild")
    st.append("polish")
    if args.studio:
        st.append("studio")
        if stupase_rebuilds(args):
            st.append("rebuild")  # StuPASE stops at 8 kHz
    if finishing(args):
        st.append("finish")
    return st + ["loudness"]


def finishing(args) -> bool:
    from . import finish
    return finish.active(args.level_speech, args.presence, args.low_shelf, args.top_cut)


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
            prog.stage("studio")
            voice = load(b)
            if args.studio in USES_SIDON:
                from . import sidon
                voice = sidon.run(voice, follow=args.follow_pitch, lift_db=args.lift_quiet)
            if args.studio in USES_STUPASE:
                # in the chain StuPASE follows Sidon's spectrum, so it keeps Sidon's sound but renders
                # the voice its own way (breathy sounds like an "h" come out more natural)
                from . import stupase
                voice = stupase.run(voice, args.stupase_crossover or 0, rebuild_top=stupase_rebuilds(args))
            save_wav(voice, b)
        if finishing(args):
            # after Studio mic, so it evens out and shapes the voice it made
            from . import finish
            prog.stage("finish")
            save_wav(finish.run(load(b), level_db=args.level_speech, presence_db=args.presence,
                                shelf_db=args.low_shelf, top_hz=args.top_cut), b)
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
    ap.add_argument("--rebuild", action="store_true",
                    help="regenerate missing high frequencies on band-limited recordings, e.g. phone or Zoom (default off)")
    ap.add_argument("--no-rebuild", dest="rebuild", action="store_false", help=argparse.SUPPRESS)  # the old default
    ap.add_argument("--rebuild-above", type=float, default=None, metavar="HZ",
                    help="force the rebuild to regenerate everything above this frequency (default: detect where "
                         "the recording's bandwidth stops; recordings that reach 12 kHz aren't rebuilt)")
    ap.add_argument("--protect", type=float, default=0.0, metavar="dB",
                    help="keep speech that's this far above the room noise from the original, so the AI can't duck "
                         "it; try 22 if loud syllables dip in level (default 0 = off: the original's noise stays out "
                         "of the speech)")
    ap.add_argument("--gate", type=float, default=0.0, metavar="dB",
                    help="silence anything this far below the main speaker, e.g. faint voices in the background "
                         "that the AI keeps because they're speech, e.g. 30 (default 0 = off; higher keeps quieter sounds)")
    ap.add_argument("--dehiss", type=float, default=0.0, metavar="dB",
                    help="turn the steady treble hiss down by up to this much while someone's speaking, where "
                         "DeepFilterNet lets it through, e.g. 12 (default 0 = off). Only above ~5 kHz; S's and T's "
                         "are louder than the hiss and pass")
    ap.add_argument("--max-reduction", type=float, default=None, metavar="dB",
                    help="deepfilter: cap how far any sound is pushed down (e.g. 12). Leaves a little natural "
                         "room tone instead of dead silence, which avoids watery/warbly artefacts")
    ap.add_argument("--studio", action="store_const", const="sidon", default="sidon",
                    help="experimental: finish by re-synthesising the voice with Sidon, so it sounds like a close, high-quality "
                         "podcast mic. Runs after everything else but the finishing steps; replaces the "
                         "rebuild. It can wobble or sound synthetic. Use -b none to give it the voice without AI "
                         "noise removal (default on)")
    ap.add_argument("--no-studio", dest="studio", action="store_const", const=None,
                    help="skip the Studio mic re-synthesis: keep the recording's own (cleaned) voice")
    ap.add_argument("--follow-pitch", type=float, default=1.0, metavar="0-1",
                    help="--studio: move Sidon's pitch back onto the recording's own, so glides don't snap into steps "
                         "(the auto-tune sound); below 1 goes part of the way (default 1; 0 = off)")
    ap.add_argument("--lift-quiet", type=float, default=0.0, metavar="dB",
                    help="--studio: raise quiet words by up to this much before Sidon, which stumbles and sounds more "
                         "robotic on them, and lower them back after it, e.g. 12 (default 0 = off)")
    ap.add_argument("--stupase", dest="studio", action="store_const", const="stupase",
                    help="experimental: like --studio, but with StuPASE, which re-synthesises the voice while following "
                         "the recording's own spectrum, to keep it sounding like the speaker. It works up to 8 kHz; "
                         "the recording supplies everything above --stupase-crossover")
    ap.add_argument("--sidon-stupase", dest="studio", action="store_const", const="sidon+stupase",
                    help="experimental: --studio, then StuPASE re-rendering Sidon's voice. StuPASE follows Sidon's "
                         "spectrum, so it keeps Sidon's studio sound but renders breathy sounds (an 'h') more naturally; "
                         "above --stupase-crossover, Sidon's own top end is kept. Takes as long as both")
    ap.add_argument("--stupase-crossover", type=float, default=5000.0, metavar="HZ",
                    help="--stupase: below this StuPASE makes the voice, above it the recording supplies it, since "
                         "StuPASE blurs sibilants (an 's' comes out like an 'f'). 0 = all StuPASE, with the rebuild "
                         "regenerating everything above 8 kHz (default 5000)")
    ap.add_argument("--level-speech", type=float, default=10.0, metavar="dB",
                    help="even out the speaking level, riding quieter speech up by up to this much and louder "
                         "speech down by up to 8 dB, towards its median; pauses aren't raised (default 10; 0 = off)")
    ap.add_argument("--presence", type=float, default=3.5, metavar="dB",
                    help="boost the presence band around 2.8 kHz by this much (default 3.5; 0 = off)")
    ap.add_argument("--low-shelf", type=float, default=3.0, metavar="dB",
                    help="turn everything below 120 Hz down by this much (default 3; 0 = off)")
    ap.add_argument("--top-cut", type=float, default=12000.0, metavar="HZ",
                    help="remove everything above this frequency with a steep filter, like Voice ReGen's 12 kHz "
                         "edge; Sidon's invented top end sits up there (default 12000; 0 = off)")
    ap.add_argument("--declip", action="store_true",
                    help="repair clipped (flat-topped) peaks before cleanup (default off)")
    ap.add_argument("--no-declip", dest="declip", action="store_false", help=argparse.SUPPRESS)  # the old default
    ap.add_argument("--ai-dereverb", type=float, default=1.0, metavar="0-1",
                    help="experimental: remove room reverb with an AI model (anvuew's dereverb_room), after the "
                         "cleanup. It keeps the recording's own voice (no re-synthesis); below 1 blends it with the "
                         "reverberant version (default 1; 0 = off)")
    ap.add_argument("--process-silence", action="store_true",
                    help="run AI dereverb and Studio mic over long silences too, instead of skipping them "
                         "(slower; for comparing)")
    ap.add_argument("--cpu", action="store_true",
                    help="run the AI models on the CPU only, not the Mac's GPU (slower; for comparing)")
    ap.add_argument("--compare", action="store_true",
                    help="render DeepFilterNet with and without the laughter guard "
                         "into <name>_compare/ for A/B listening")
    ap.add_argument("-f", "--format", default="wav", help="output format: wav, flac, m4a, mp3, aiff (default wav)")
    ap.add_argument("--lufs", type=float, default=None,
                    help="normalise to this loudness with a true-peak limiter (off by default: output stays at the "
                         "source recording's level, loudness is left to the editor)")
    ap.add_argument("--true-peak", type=float, default=-1.5, help="true-peak ceiling in dBTP when --lufs is used (default -1.5)")
    ap.add_argument("--blend", type=float, default=0.0, metavar="0-1",
                    help="blend this much of the original (noise included) back in, e.g. 0.2 (default 0 = off)")
    ap.add_argument("--mix", type=float, default=None, help=argparse.SUPPRESS)  # the old form: --mix 0.8 = --blend 0.2
    ap.add_argument("--laughter-guard", type=float, default=0.0, metavar="0-1",
                    help="how strongly to protect laughs and other non-speech voice sounds, e.g. 0.85 (default 0 = off)")
    ap.add_argument("--no-polish", action="store_true", help="skip EQ / de-ess / compression (loudness still applied)")
    ap.add_argument("-v", "--verbose", action="store_true", help="list the timestamps the laughter guard protected")
    ap.add_argument("--list-backends", action="store_true", help="list the noise-removal options")
    ap.add_argument("--gui", action="store_true",
                    help="open the app in your browser: sliders, and an A/B player against the original")
    ap.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def configure(args) -> None:
    """Check the settings and apply the ones the engines read globally."""
    if args.mix is not None:
        args.blend = 1 - args.mix
    silence.enabled = not args.process_silence
    accel.enabled = not args.cpu
    if not 0 <= args.blend <= 1 or not 0 <= args.laughter_guard <= 1 or not 0 <= args.ai_dereverb <= 1 \
            or not 0 <= args.follow_pitch <= 1 or args.lift_quiet < 0 \
            or min(args.level_speech, args.presence, args.low_shelf, args.top_cut) < 0:
        raise SystemExit("--blend, --laughter-guard, --ai-dereverb and --follow-pitch must be between 0 and 1, "
                         "and --lift-quiet, --level-speech, --presence, --low-shelf and --top-cut can't be negative")
    from . import backends as _b
    _b.DF_SETTINGS["max_reduction_db"] = args.max_reduction
    from . import aidereverb, sidon, stupase
    if args.ai_dereverb > 0 and not aidereverb.available():
        raise SystemExit("--ai-dereverb needs the beartype, rotary-embedding-torch and einops packages. Reinstall with:\n"
                         "  uv tool install --python 3.11 . --force --reinstall-package voice-enhance --overrides overrides.txt")
    if args.studio in USES_STUPASE and not stupase.available():
        raise SystemExit("--stupase needs the x-transformers and torchdiffeq packages. Reinstall with:\n"
                         "  uv tool install --python 3.11 . --force --reinstall-package voice-enhance --overrides overrides.txt")
    if args.studio in USES_SIDON and not sidon.available():
        raise SystemExit("--studio needs the transformers package. Reinstall with:\n"
                         "  uv tool install --python 3.11 . --force --reinstall-package voice-enhance --overrides overrides.txt")
    if args.studio in USES_SIDON and args.follow_pitch > 0 and not sidon.available_follow():
        raise SystemExit("--follow-pitch needs the praat-parselmouth package. Reinstall with:\n"
                         "  uv tool install --python 3.11 . --force --reinstall-package voice-enhance --overrides overrides.txt\n"
                         "or add --follow-pitch 0")


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
                         "or run without --rebuild")
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
            # the guard is off by default, so compare against it at its usual strength
            on = guard if guard.strength > 0 else GuardSettings(strength=0.85)
            variants = [("deepfilter", "deepfilter", on),
                        ("deepfilter_no-guard", "deepfilter", None)]
            for n, (label, b, g) in enumerate(variants, start=1):
                with prog.progress(short(f"{n}_{label}", 30),
                                   stages_for(args, b, g, rebuild_wanted(args), decode=False), backend=b):
                    y, note = process(audio, b, mix=1 - args.blend, guard=g, cache=cache,
                                      verbose=args.verbose, rebuild_hf=rebuild_wanted(args), ai_dereverb=args.ai_dereverb,
                                      repair_clipping=args.declip, protect_db=args.protect, rebuild_above=args.rebuild_above,
                                      gate_db=args.gate, dehiss_db=args.dehiss)
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
            y, note = process(audio, args.backend, mix=1 - args.blend, guard=guard, cache=cache,
                              rebuild_hf=rebuild_wanted(args), verbose=args.verbose, ai_dereverb=args.ai_dereverb, repair_clipping=args.declip, protect_db=args.protect,
                              rebuild_above=args.rebuild_above, gate_db=args.gate, dehiss_db=args.dehiss)
            result = render(y, dst, args, source_lufs(audio))
        mins, secs = divmod(time.time() - started, 60)
        log(f"  -> {result}  [{audio.size / SR / 60:.1f} min audio in {int(mins)}m{int(secs):02d}s]")
        log(f"     {note}")


if __name__ == "__main__":
    main()
