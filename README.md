# voice-enhance

Local, Adobe-Podcast-style speech cleanup. It runs entirely on your Mac.

## Install

### In homebrew
```
brew install ffmpeg uv
```

### In the root folder
```
uv tool install --python 3.11 . --force --reinstall-package voice-enhance --overrides overrides.txt
```

## Commands

| Command | What it does |
|---|---|
| `voice-enhance ep.wav` | Runs Resemble Enhance, a generative AI model that removes noise and reverb and re-synthesises the voice as clean studio speech (Adobe Podcast's approach). It then applies a warm EQ matched to Adobe's measured tone, so thin mics get their body back. It also repairs clipped peaks and protects laughter. There's no compression or loudness processing: the output stays at the source level. Writes `ep_enhanced.wav` next to the input. Drag a file into Terminal to paste its path. |
| `--no-rebuild` | Skips the rebuild. |
| `--rebuild-above 8000` | Forces the rebuild to regenerate everything above that frequency, even on full-band audio. |
| `--dereverb 0.5` | Adds DSP reverb-tail suppression on top of the AI (default off). |
| `--quality 32` | Sets Resemble's generation steps (default 64). 32 is about twice as fast; 128 is slower and smoother. |
| `--temperature 0.3` | Sets Resemble's variation (default 0.5). Lower is steadier; higher sounds more natural but less consistent. |
| `--denoise-first` | Makes Resemble run its denoiser before re-synthesis: stronger cleanup, but can sound more processed. |
| `--no-declip` | Doesn't repair clipped peaks. |
| `voice-enhance ep.wav --compare` | Renders every engine and setting into `ep_compare/`, all at the source level, for A/B listening. |
| `voice-enhance *.wav -o ~/Desktop/clean` | Batch-processes files into a folder. |
| `-f m4a` / `mp3` / `flac` / `aiff` | Sets the output format (default: 24-bit WAV). |
| `--lufs -16` | Opt-in loudness normalisation plus a true-peak limiter (off by default). |
| `--true-peak -1` | Sets the limiter ceiling when `--lufs` is used (default -1.5 dBTP). |
| `-b resemble-denoise` / `-b deepfilter` / `-b clearvoice` / `-b none` | Picks a different engine: resemble-denoise (the AI cleanup without the re-synthesis), deepfilter (fast noise removal only), clearvoice, or none. |
| `--laughter-guard 0.5` | Sets how much laughter to protect from the cleanup (0 = off, default 0.85). |
| `-v` | Lists the timestamps where the laughter guard kicked in. |
| `--warmth -8` (with `--tone warm`) | Sets the low end: lower is less boomy (default -4, 0 = Adobe's full low end). |
| `--presence 8` (with `--tone warm`) | Sets clarity: lifts 3-8 kHz (default +5). |
| `--tone neutral` | Skips the warm EQ and keeps the AI's own tone. |
| `--tone ref.wav` | Matches the tone of a recording you like. |
| `--mix 0.9` | Leaves 10% of the original signal in. |
| `--no-polish` | Cleanup and loudness only: no EQ, de-essing or compression. |
| `--list-backends` | Shows which cleanup engines are installed. |
| `VOICE_ENHANCE_DEVICE=cpu voice-enhance ...` | Runs the ClearerVoice models on the CPU instead of the Apple Silicon GPU. |
| `./scripts/try-rebuild.sh` | Installs, then runs `--compare` on the clips in `out/` (not in the repo). |
