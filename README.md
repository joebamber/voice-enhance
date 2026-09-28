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
4. **Polish.** An 80 Hz high-pass, a small cut at 250 Hz (mud), presence at 4.5 kHz, a touch of air, a de-esser and a gentle 3:1 compressor.
5. **Loudness.** Two-pass EBU R128 normalisation. The default target is -16 LUFS with a -1.5 dBTP ceiling.

## Laughter guard

Adobe Enhance Speech (and most speech enhancers) is trained to keep *speech* and remove everything else, so laughs, gasps and "mmm"s often get flattened.

The guard runs a second, *gentle* cleanup pass, capped at 8 dB of reduction. It then compares the original with the full cleanup frame by frame. Wherever the original had a **cluster of bursts well above the local background noise** that the full cleanup crushed, it crossfades to the gentle pass for that moment. Laughter is exactly that pattern: repeated "ha" bursts at speech level. Background noise between phrases never rises above the local floor, and isolated breaths or plosives don't form clusters, so the cleaner still removes those.

On the EV Café Peter Jones test file (Podcast 003, about 6:38), the full DeepFilterNet pass dropped the laugh by 20-40 dB. With the guard, it stays within a few dB of the original.

```
--laughter-guard 0.85   # default; 0 = off, 1 = fully gentle during laughs
-v                      # print the timestamps the guard protected, so you can spot-check them
```

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
| `apple` | macOS AUSoundIsolation (the engine behind Voice Isolation in FCP/Logic) | Compiles a small Swift helper on first use (needs Xcode CLT). **Not yet tested on a Mac.** |
| `clearvoice` | ClearerVoice-Studio MossFormer2 SE 48 kHz | Optional extra; weights from Hugging Face. **Not yet tested.** |
| `clearvoice-sr` | as above + MossFormer2 super-resolution | The closest to Adobe's "rebuild a thin mic" effect. **Not yet tested.** |
| `none` | polish + loudness only | |

## Tuning

The polish chain lives in `POLISH_FILTERS` in `src/voice_enhance/audio.py` (plain ffmpeg filter syntax). The guard thresholds are in `GuardSettings` in `src/voice_enhance/guard.py`.
