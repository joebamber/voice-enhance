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

| Command | What it does | Default |
|---|---|---|
| `voice-enhance ep.wav` | Repairs clipped peaks, removes noise with DeepFilterNet (natural-sounding AI cleanup), protects laughter, rebuilds missing high frequencies if the recording is band-limited (e.g. phone or Zoom), and applies a warm EQ matched to Adobe's measured tone. There's no compression or loudness processing: the output stays at the source level. Writes `ep_enhanced.wav` next to the input. Drag a file into Terminal to paste its path. | |
| `--no-rebuild` | Skips the rebuild. | Rebuild on |
| `--rebuild-above 8000` | Forces the rebuild to regenerate everything above that frequency, even on full-band audio. | Auto-detected |
| `--dereverb 0.5` | Adds DSP reverb-tail suppression on top of the AI. | 0 (off) |
| `--quality 32` | With `-b resemble`: sets Resemble's generation steps. 32 is about twice as fast; 128 is slower and smoother. | 64 |
| `--temperature 0.3` | With `-b resemble`: sets Resemble's variation. Lower is steadier; higher sounds more natural but less consistent. | 0.5 |
| `--denoise-first` | With `-b resemble`: makes Resemble run its denoiser before re-synthesis: stronger cleanup, but can sound more processed. | Off |
| `--no-declip` | Doesn't repair clipped peaks. | Declip on |
| `--protect 0` | Sets speech protection (0 turns it off). Speech this far above the room noise comes from the original, so the AI can't duck loud syllables. Higher values let the AI process more. | 22 dB |
| `--max-reduction 12` | Caps how far DeepFilterNet pushes any sound down (in dB), leaving a little natural room tone instead of dead silence. | No cap |
| `voice-enhance ep.wav --compare` | Renders every engine and setting into `ep_compare/`, all at the source level, for A/B listening. | |
| `voice-enhance *.wav -o ~/Desktop/clean` | Batch-processes files into a folder. | Next to the input |
| `-f m4a` / `mp3` / `flac` / `aiff` | Sets the output format. | 24-bit WAV |
| `--lufs -16` | Opt-in loudness normalisation plus a true-peak limiter. | Off |
| `--true-peak -1` | Sets the limiter ceiling when `--lufs` is used. | -1.5 dBTP |
| `-b resemble` | Uses Resemble Enhance instead: generative re-synthesis that removes more room echo, but it's slow and can sound robotic. `-b clearvoice` / `-b none` are also available. | `deepfilter` |
| `--laughter-guard 0.5` | Sets how much laughter to protect from the cleanup (0 = off). | 0.85 |
| `-v` | Lists the timestamps where the laughter guard kicked in. | Off |
| `--warmth -8` (with `--tone warm`) | Sets the low end: lower is less boomy (0 = Adobe's full low end). | -4 dB |
| `--presence 8` (with `--tone warm`) | Sets clarity: lifts 3-8 kHz. | +5 dB |
| `--tone neutral` | Skips the warm EQ and keeps the AI's own tone. | `warm` |
| `--tone ref.wav` | Matches the tone of a recording you like. | `warm` |
| `--mix 0.9` | Leaves 10% of the original signal in. | 1.0 |
| `--no-polish` | Cleanup and loudness only: no EQ, de-essing or compression. | Polish on |
| `--list-backends` | Shows which cleanup engines are installed. | |
| `VOICE_ENHANCE_DEVICE=cpu voice-enhance ...` | Runs the ClearerVoice models on the CPU instead of the Apple Silicon GPU. | Apple Silicon GPU |
| `./scripts/try-rebuild.sh` | Installs, then runs `--compare` on the clips in `out/` (not in the repo). | |

## Tuning the background noise

By default DeepFilterNet removes as much noise as it can, down to near silence in the pauses. These dial it up or down:

| To get… | Use | How it works |
|---|---|---|
| **Less** noise removal, natural room tone in the pauses | `--max-reduction 12` | Caps how far any sound is pushed down, in dB. Lower values leave more room tone. Also avoids watery or warbly artefacts on hard material. |
| **Less** noise removal, everywhere | `--mix 0.8` | Blends 20% of the original (noise included) back in. Lower values bring back more of the noise. |
| **More** noise removal under loud speech | `--protect 30` (or `--protect 0`) | Speech more than this many dB above the room noise comes from the original, noise and all, because there the voice masks it. Raising it hands more of the speech to the AI; 0 hands it all over, but the AI may then duck loud syllables. Lower (e.g. `--protect 15`) keeps more of the original voice. |
| **More** room echo removed | `--dereverb 0.5` | Adds DSP reverb-tail suppression after the AI (0–1). |
| **Much more** cleanup (echo and noise) | `-b resemble --denoise-first` | Generative re-synthesis. Strongest, but slow and can sound robotic. |
| No noise removal at all | `-b none` | Keeps declip, rebuild and EQ only. |

The laughter guard (`--laughter-guard`) also lets a little noise through around laughs, so lower it if you hear noise come and go there.

`--compare` only compares engines, so to hear these settings side by side, render a few variants of a short clip:

```
voice-enhance clip.wav -o clip_mr12.wav --max-reduction 12
voice-enhance clip.wav -o clip_mix80.wav --mix 0.8
```
