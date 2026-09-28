# voice-enhance

Local, Adobe-Podcast-style cleanup for spoken audio. Everything runs on your Mac; nothing is uploaded.

```
voice-enhance "Podcast 003.wav"
# -> Podcast 003_enhanced.wav  (-16 LUFS, -1.5 dBTP)
```

The pipeline:

1. **Decode.** Any audio/video ffmpeg can read, downmixed to mono at 48 kHz.
2. **AI cleanup.** [DeepFilterNet 3](https://github.com/Rikorose/DeepFilterNet) by default. It removes noise and room tone and runs about 25x faster than real time on CPU.
3. **Laughter guard.** See below.
4. **Tone match.** A linear-phase EQ, worked out for each file, that moves the voice toward a target sound. The default is `warm`, measured from Adobe's output (see below).
5. **Polish.** A de-esser and a gentle 2:1 compressor, run at a fixed working level so they behave the same on every file.
6. **Loudness.** A static gain into a 4x-oversampled limiter. The default is -16 LUFS with a -1.5 dBTP ceiling. This deliberately avoids ffmpeg's `loudnorm`, which quietly switches to heavy compression on dynamic speech.

## Laughter guard

Adobe Enhance Speech (and most speech enhancers) is trained to keep *speech* and remove everything else, so laughs, gasps and "mmm"s often get flattened.

The guard runs a second, *gentle* cleanup pass, capped at 8 dB of reduction. It then compares the original with the full cleanup frame by frame. Wherever the original had a **cluster of bursts well above the local background noise** that the full cleanup crushed, it crossfades to the gentle pass for that moment. Laughter is exactly that pattern: repeated "ha" bursts at speech level. Background noise between phrases never rises above the local floor, and isolated breaths or plosives don't form clusters, so the cleaner still removes those.

On the EV Café Peter Jones test file (Podcast 003, about 6:38), the full DeepFilterNet pass dropped the laugh by 20-40 dB. With the guard, it stays within a few dB of the original.

```
--laughter-guard 0.85   # default; 0 = off, 1 = fully gentle during laughs
-v                      # print the timestamps the guard protected, so you can spot-check them
```

## Tone: why Adobe sounds warmer

Measured on the Peter Jones episode, Adobe's output differs from a straight denoise mainly in two ways:

- **Much more low end.** Around +15 to +20 dB at 80-125 Hz, which is the "rich", close-mic sound.
- **Softer presence.** 5-9 dB less at 2-6 kHz, plus a gentler top end.

`--tone warm` (the default) stores Adobe's long-term speech spectrum as a target. For each file it measures what the voice actually has and EQs the difference, so a thin mic gets more help than a full one. Boosts are capped, and nothing below 70 Hz is ever boosted. After compression it re-measures and trims once more. On the test episode the result lands within about 3 dB of Adobe across the spectrum.

```
--tone warm               # default, Adobe-like
--tone neutral            # no tonal shaping, just a 70 Hz high-pass
--tone ref.wav            # match any recording whose sound you like
--tone-amount 0.6         # go 60% of the way
--presence 5              # default: lift 3-8 kHz by 5 dB relative to the target
--presence 0              # exactly Adobe's balance (warmest, least clear)
```

Adobe's balance by itself sounds slightly dull without its regenerated detail, so by default `--presence` gives back the 3-8 kHz clarity region while keeping the warm low end.

EQ can't copy everything Adobe does. Its model *regenerates* the voice, which adds harmonic density that no EQ can create. For that, see **Rebuilding thin audio** below. The `warm` target was also measured on two male voices, so for very different voices a `--tone ref.wav` from a show you like is the better choice.

## Rebuilding thin audio

```
voice-enhance zoom-call.wav --rebuild
```

`--rebuild` runs ClearerVoice's MossFormer2 super-resolution model after cleanup and the laughter guard. It finds where the recording's bandwidth runs out (about 4 kHz for a phone, about 8 kHz for a laptop or Zoom, about 16 kHz for a low-bitrate MP3) and generates the missing high band with a neural vocoder. Everything below that cutoff is left as recorded. It runs after the guard, so laughs get rebuilt along with the speech.

- It rebuilds the *top* end (clarity, air, crisp consonants). It doesn't invent low end; the `warm` tone handles that.
- It's generative, so listen for artefacts, especially on laughter and sibilance.
- It's much slower than the cleanup. It uses the Apple Silicon GPU where it can. If the GPU path misbehaves, `VOICE_ENHANCE_DEVICE=cpu voice-enhance ...` forces the CPU.
- It needs the optional extra: `uv tool install --python 3.11 '.[clearvoice]' --force`.

To try everything at once:

```
./scripts/try-rebuild.sh
```

This installs the extra and renders `--compare` sets for `out/laugh-test.wav`, plus a phone-quality version (`out/thin/phone.wav`, cut off at about 4 kHz) and a Zoom/laptop-quality version (`out/thin/laptop-zoom.wav`, 16 kHz at 32 kbps MP3). The results land in `out/rebuild-test/`.

## Install (macOS)

```
brew install ffmpeg uv
cd voice-enhance
uv tool install --python 3.11 .          # puts `voice-enhance` on your PATH
# optional ClearerVoice models:
uv tool install --python 3.11 '.[clearvoice]' --force
```

The DeepFilterNet weights download from GitHub on first run.

## Usage

```
voice-enhance file.wav                        # -> file_enhanced.wav next to it
voice-enhance *.wav -o ~/Desktop/clean -f m4a # batch, AAC out
voice-enhance ep.wav --lufs -19               # mono/speech spec (-19 LUFS)
voice-enhance ep.wav --mix 0.9                # leave 10% of the original in
voice-enhance ep.wav --no-polish              # cleanup + loudness only
voice-enhance ep.wav --compare                # A/B folder of every backend, guard on/off
voice-enhance --list-backends
```

`--compare` writes `ep_compare/` with the loudness-matched original, a polish-only version, and each backend with and without the laughter guard. All files are at the same loudness, so the A/B comparison is fair.

## Backends

| `-b` | What | Notes |
|---|---|---|
| `deepfilter` (default) | DeepFilterNet 3 | Fast, natural, doesn't invent audio. Tested. |
| `apple` | macOS AUSoundIsolation (the engine behind Voice Isolation in FCP/Logic); left out of `--compare` | Compiles a small Swift helper on first use (needs Xcode CLT). **Not yet tested on a Mac.** |
| `clearvoice` | ClearerVoice-Studio MossFormer2 SE 48 kHz | Optional extra; weights from Hugging Face. Code path tested, real weights not yet. |
| `none` | polish + loudness only | |

## Tuning

The polish chain lives in `polish_filters()` in `src/voice_enhance/audio.py` (plain ffmpeg filter syntax). The tone targets and boost caps are in `src/voice_enhance/tone.py`. The guard thresholds are in `GuardSettings` in `src/voice_enhance/guard.py`.
