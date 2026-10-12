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

## The app

```
voice-enhance --gui
```

Opens the app in your browser. It runs on your Mac like the command line. Drop a recording onto the page (or click **Open…**), and it's cleaned up with the default settings.

- **A/B:** press **B** (or click **Original** / **Enhanced**) to switch between the original and the cleaned-up version at the same spot while it plays. **Space** plays and pauses, the arrow keys jump 5 s, and clicking the waveform seeks.
- **The waveform** is drawn in dB, so you can see the noise between words and what the gate removes.
- **Sliders** re-render when you let go. Only the settings marked *re-runs AI* (AI noise removal, max reduction, declip) redo the AI cleanup. The others reuse it and update in about a second (the rebuild re-runs too, on band-limited recordings). With **Studio mic** on, every change also re-runs Sidon at the end (about 6 minutes per hour of audio). Double-click a slider to reset it.
- **Save…** writes the version you're hearing in the format you pick, next to the original by default.
- **The command line** at the bottom gives the same settings as a `voice-enhance` command, for batch-processing a folder.

`voice-enhance --gui ep.wav` opens a file straight away. Keep the Terminal window open while you use the app, and press Ctrl+C there to quit.

## Commands

| Command | What it does | Default |
|---|---|---|
| `voice-enhance ep.wav` | Removes noise with DeepFilterNet (natural-sounding AI cleanup), protects laughter, rebuilds missing high frequencies if the recording is band-limited below 12 kHz (e.g. phone or Zoom), and silences anything far quieter than the speaker. The recording keeps its own tone (no EQ). There's no compression or loudness processing: the output stays at the source level. Writes `ep_enhanced.wav` next to the input. Drag a file into Terminal to paste its path. | |
| `--no-rebuild` | Skips the rebuild. | Rebuild on |
| `--rebuild-above 8000` | Forces the rebuild to regenerate everything above that frequency, even on full-band audio. | Auto-detected |
| `--dereverb 0.5` | Adds DSP reverb-tail suppression on top of the AI. | 0 (off) |
| `--studio` | Finishes by re-synthesising the voice with [Sidon](https://huggingface.co/sarulab-speech/sidon-v0.1) so it sounds like a close, high-quality podcast mic. It's the last step: everything else (cleanup, gate, polish) runs first, and only the output level is set after it. It hears the voice up to 8 kHz and generates the rest, so it replaces the rebuild. About 6 minutes per hour of audio. Add `-b none` to give it the voice without AI noise removal. | Off |
| `--declip` | Repairs clipped (flat-topped) peaks before cleanup. | Off |
| `--protect 22` | Turns on speech protection: speech this far above the room noise comes from the original, noise and all, so the AI can't duck loud syllables. Use it if loud words dip in level. Higher values let the AI process more. | 0 (off) |
| `--max-reduction 12` | Caps how far DeepFilterNet pushes any sound down (in dB), leaving a little natural room tone instead of dead silence. | No cap |
| `--gate 40` | Sets the background gate: anything this many dB below the main speaker is silenced, e.g. faint voices in the background, which the AI keeps because they're speech. Higher keeps quieter sounds; 0 turns it off. | 30 dB |
| `voice-enhance ep.wav --compare` | Renders DeepFilterNet with and without the laughter guard into `ep_compare/`, all at the source level, for A/B listening. | |
| `voice-enhance *.wav -o ~/Desktop/clean` | Batch-processes files into a folder. | Next to the input |
| `-f m4a` / `mp3` / `flac` / `aiff` | Sets the output format. | 24-bit WAV |
| `--lufs -16` | Opt-in loudness normalisation plus a true-peak limiter. | Off |
| `--true-peak -1` | Sets the limiter ceiling when `--lufs` is used. | -1.5 dBTP |
| `-b none` | Skips the AI noise removal (and the gate and laughter guard that go with it). | `deepfilter` |
| `--laughter-guard 0.5` | Sets how much laughter to protect from the cleanup (0 = off). | 0.85 |
| `-v` | Lists the timestamps where the laughter guard kicked in. | Off |
| `--mix 0.9` | Leaves 10% of the original signal in. | 1.0 |
| `--no-polish` | Cleanup and loudness only: no EQ, de-essing or compression. | Polish on |
| `--list-backends` | Lists the noise-removal options (`deepfilter`, `none`). | |
| `VOICE_ENHANCE_DEVICE=cpu voice-enhance ...` | Runs the ClearerVoice models on the CPU instead of the Apple Silicon GPU. | Apple Silicon GPU |
| `./scripts/try-rebuild.sh` | Installs, then runs `--compare` on the clips in `out/` (not in the repo). | |

## Tuning the background noise

By default DeepFilterNet removes as much noise as it can, and the background gate then silences whatever is left more than 30 dB below the speaker (faint background voices, which the AI keeps because they're speech). The pauses end up silent. These dial it up or down:

| To get… | Use | How it works |
|---|---|---|
| **Less** noise removal, natural room tone in the pauses | `--max-reduction 12 --gate 0` | Caps how far any sound is pushed down, in dB. Lower values leave more room tone. Also avoids watery or warbly artefacts on hard material. Without `--gate 0` the gate silences that room tone again. |
| Keep quiet sounds far below the speaker (distant voices, soft breaths) | `--gate 40` (or `--gate 0`) | The gate silences anything more than this many dB below the main speaker. Raise it to keep quieter sounds; 0 turns it off. |
| **Less** noise removal, everywhere | `--mix 0.8` | Blends 20% of the original (noise included) back in. Lower values bring back more of the noise. |
| Loud words dipping in level | `--protect 22` | Off by default, so the AI cleans all of the speech. If it ducks loud syllables, this takes speech more than 22 dB above the room noise from the original instead, noise and all (there the voice mostly masks it). Lower (e.g. `--protect 15`) keeps more of the original voice. |
| **More** room echo removed | `--dereverb 0.5` | Adds DSP reverb-tail suppression after the AI (0–1). |
| No noise removal at all | `-b none` | Keeps declip (if on), rebuild and EQ only. |

The laughter guard (`--laughter-guard`) also lets a little noise through around laughs, so lower it if you hear noise come and go there.

`--compare` only compares the laughter guard, so to hear these settings side by side, render a few variants of a short clip:

```
voice-enhance clip.wav -o clip_mr12.wav --max-reduction 12 --gate 0
voice-enhance clip.wav -o clip_mix80.wav --mix 0.8
```
