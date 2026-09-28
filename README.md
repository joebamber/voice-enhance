# voice-enhance

Local, Adobe-Podcast-style speech cleanup. It runs entirely on your Mac.

## Install

### In homebrew
```
brew install ffmpeg uv
```

### In the root folder
```
uv tool install --python 3.11 '.[clearvoice]' --force --reinstall-package voice-enhance
```

## Commands

| Command | What it does |
|---|---|
| `voice-enhance ep.wav` | Clean up, warm tone, polish and normalise to -16 LUFS. Writes `ep_enhanced.wav` next to the input. |
| `voice-enhance ep.wav --rebuild` | Also regenerates missing high frequencies (for phone, Zoom or laptop audio). |
| `voice-enhance ep.wav --compare` | Renders every option into `ep_compare/`, all at the same loudness, for A/B listening. |
| `voice-enhance *.wav -o ~/Desktop/clean` | Batch-processes files into a folder. |
| `-f m4a` / `mp3` / `flac` / `aiff` | Sets the output format (default: 24-bit WAV). |
| `--lufs -19` | Sets the loudness target (default -16). |
| `--true-peak -1` | Sets the peak ceiling in dBTP (default -1.5). |
| `-b clearvoice` / `-b apple` / `-b none` | Picks the cleanup engine (default: DeepFilterNet). |
| `--laughter-guard 0.5` | Sets how much laughter to protect from the cleanup (0 = off, default 0.85). |
| `-v` | Lists the timestamps where the laughter guard kicked in. |
| `--warmth -8` | Sets the low end: lower is less boomy (default -4, 0 = Adobe's full low end). |
| `--presence 8` | Sets clarity: lifts 3-8 kHz (default +5). |
| `--tone neutral` | Turns off tone matching. |
| `--tone ref.wav` | Matches the tone of a recording you like. |
| `--mix 0.9` | Leaves 10% of the original signal in. |
| `--no-polish` | Cleanup and loudness only: no EQ, de-essing or compression. |
| `--list-backends` | Shows which cleanup engines are installed. |
| `VOICE_ENHANCE_DEVICE=cpu voice-enhance ...` | Runs the ClearerVoice models on the CPU instead of the Apple Silicon GPU. |
| `./scripts/try-rebuild.sh` | Installs, then runs `--compare` on the clips in `out/` (not in the repo). |
