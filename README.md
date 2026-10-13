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

Opens the app in your browser. It runs on your Mac like the command line. Drop a recording onto the page (or click **Open…**) to load it, choose your settings, then press **Process** (⌘↩).

- **A/B:** press **B** (or click **Original** / **Enhanced**) to switch between the original and the cleaned-up version at the same spot while it plays. **Space** plays and pauses, the arrow keys jump 5 s, and clicking the waveform seeks.
- **The waveform** is drawn in dB, so you can see the noise between words and what the gate removes.
- **Settings** only take effect when you press **Process**, so you can change several first; until then, Enhanced plays the previous version. **Cancel** (or Esc) stops processing and keeps the last finished version playable.
- **Range** next to Process can be set to a 15, 30 or 60 s sample starting at the playhead, to try settings quickly on a long file. Enhanced is silent outside the sample, Process lights up again when the playhead leaves it, and Save writes just the sample (its name ends with e.g. `sample1m23s`). Only the settings marked *re-runs AI* (AI noise removal, max reduction, declip) redo the AI cleanup. The others reuse it and process in about a second (the rebuild re-runs too, on band-limited recordings). With **Studio mic** on, every change also re-runs its model at the end (StuPASE: about 12 minutes per hour of audio; Sidon: about 6). Moving the StuPASE crossover doesn't re-run it. Double-click a slider to reset it.
- **Save…** writes the version you're hearing in the format you pick, next to the original by default. The suggested name ends with the settings that differ from the defaults, e.g. `ep_enhanced [noise-none aidereverb0.8].wav`.
- **The command line** at the bottom gives the same settings as a `voice-enhance` command, for batch-processing a folder.

`voice-enhance --gui ep.wav` opens a file straight away. Keep the Terminal window open while you use the app, and press Ctrl+C there to quit.

## Commands

| Command | What it does | Default |
|---|---|---|
| `voice-enhance ep.wav` | Removes noise with DeepFilterNet (natural-sounding AI cleanup), then room reverb with AI dereverb. The recording keeps its own tone (no EQ). There's no compression or loudness processing: the output stays at the source level. Writes `ep_enhanced.wav` next to the input. Drag a file into Terminal to paste its path. | |
| `--rebuild` | Rebuilds missing high frequencies if the recording is band-limited below 12 kHz (e.g. phone or Zoom). | Off |
| `--rebuild-above 8000` | Forces the rebuild to regenerate everything above that frequency, even on full-band audio. | Auto-detected |
| `--ai-dereverb 1` | *Experimental.* Removes room reverb with an AI model ([anvuew's dereverb_room](https://huggingface.co/anvuew/dereverb_room), a BS-Roformer trained on single voices with simulated room reverb), after the cleanup and gate. It masks the reverb out of the recording rather than re-synthesising the voice, so the voice stays your own; values below 1 blend some room back; `--ai-dereverb 0` turns it off. In the app, changing the strength doesn't re-run it. The weights (~120 MB) are GPL-3.0. | 1 (on) |
| `--studio` | *Experimental.* Finishes by re-synthesising the voice with [Sidon](https://huggingface.co/sarulab-speech/sidon-v0.1) so it sounds like a close, high-quality podcast mic. It's the last step: everything else (cleanup, gate, polish) runs first, and only the output level is set after it. It hears the voice up to 8 kHz and generates the rest, so it replaces the rebuild. It can wobble or sound synthetic, and it can't be layered with the original (its waveform is new, so the two sound doubled). About 6 minutes per hour of audio. Add `-b none` to give it the voice without AI noise removal. | Off |
| `--stupase` | *Experimental.* Like `--studio`, but with [StuPASE](https://huggingface.co/cisco-ai/stupase) (Cisco, Apache 2.0), which is built to keep the speaker's own voice: it re-synthesises the voice while following the recording's own spectrum. It works at 16 kHz and blurs sibilants (an S comes out like an F), so by default it only makes the voice below 5 kHz and the recording supplies everything above: S, T and breath. Like Sidon, it can't be layered full-band with the original. About 12 minutes per hour of audio. | Off |
| `--stupase-crossover 4000` | Where `--stupase` hands over to the recording. 0 = all StuPASE, with the rebuild regenerating everything above 8 kHz. | 5000 Hz |
| `--declip` | Repairs clipped (flat-topped) peaks before cleanup. | Off |
| `--protect 22` | Turns on speech protection: speech this far above the room noise comes from the original, noise and all, so the AI can't duck loud syllables. Use it if loud words dip in level. Higher values let the AI process more. | 0 (off) |
| `--max-reduction 12` | Caps how far DeepFilterNet pushes any sound down (in dB), leaving a little natural room tone instead of dead silence. | No cap |
| `--gate 40` | Sets the background gate: anything this many dB below the main speaker is silenced, e.g. faint voices in the background, which the AI keeps because they're speech. Higher keeps quieter sounds; 0 turns it off. | 0 (off) |
| `--dehiss 12` | Turns the recording's steady treble hiss down by up to this many dB. DeepFilterNet lets it through above ~5 kHz while someone's speaking, to keep the S's and T's. The hiss is measured between words in the original; only treble no louder than it is turned down, so sibilants pass (they lose a dB or few at the very top). Plain signal processing, no AI. 0 turns it off. | 0 (off) |
| `voice-enhance ep.wav --compare` | Renders DeepFilterNet with and without the laughter guard into `ep_compare/`, all at the source level, for A/B listening. | |
| `voice-enhance *.wav -o ~/Desktop/clean` | Batch-processes files into a folder. | Next to the input |
| `-f m4a` / `mp3` / `flac` / `aiff` | Sets the output format. | 24-bit WAV |
| `--lufs -16` | Opt-in loudness normalisation plus a true-peak limiter. | Off |
| `--true-peak -1` | Sets the limiter ceiling when `--lufs` is used. | -1.5 dBTP |
| `-b none` | Skips the AI noise removal (and the gate and laughter guard that go with it). | `deepfilter` |
| `--laughter-guard 0.85` | Protects laughs from the cleanup: higher protects more (0 = off). | 0 (off) |
| `-v` | Lists the timestamps where the laughter guard kicked in. | Off |
| `--blend 0.1` | Blends 10% of the original signal (noise included) back in. | 0 (off) |
| `--no-polish` | Cleanup and loudness only: no EQ, de-essing or compression. | Polish on |
| `--list-backends` | Lists the noise-removal options (`deepfilter`, `none`). | |
| `VOICE_ENHANCE_DEVICE=cpu voice-enhance ...` | Runs the ClearerVoice models on the CPU instead of the Apple Silicon GPU. | Apple Silicon GPU |
| `./scripts/try-rebuild.sh` | Installs, then runs `--compare` on the clips in `out/` (not in the repo). | |

## Tuning the background noise

By default DeepFilterNet removes as much noise as it can, and AI dereverb removes the room's echo. The background gate (`--gate 30`) can also silence whatever is left more than 30 dB below the speaker (faint background voices, which the AI keeps because they're speech). These dial it up or down:

| To get… | Use | How it works |
|---|---|---|
| **Less** noise removal, natural room tone in the pauses | `--max-reduction 12` | Caps how far any sound is pushed down, in dB. Lower values leave more room tone. Also avoids watery or warbly artefacts on hard material. Leave the gate off, or it silences that room tone again. |
| Silence faint voices in the background | `--gate 30` | The gate silences anything more than this many dB below the main speaker. Raise it to keep quieter sounds; 0 (the default) turns it off. |
| Hiss you hear while someone's speaking | `--dehiss 12` | Turns down treble (above ~5 kHz) that's no louder than the recording's hiss, by up to 12 dB. Higher removes more but can dull the S's. |
| **Less** noise removal, everywhere | `--blend 0.2` | Blends 20% of the original (noise included) back in. Higher values bring back more of the noise. |
| Loud words dipping in level | `--protect 22` | Off by default, so the AI cleans all of the speech. If it ducks loud syllables, this takes speech more than 22 dB above the room noise from the original instead, noise and all (there the voice mostly masks it). Lower (e.g. `--protect 15`) keeps more of the original voice. |
| Some room echo back | `--ai-dereverb 0.7` | On at 1 by default. Lower values leave a little room in; 0 turns it off. |
| No noise removal at all | `-b none` | Keeps declip, rebuild (if on), AI dereverb and EQ. |

The laughter guard (`--laughter-guard`) also lets a little noise through around laughs, so lower it if you hear noise come and go there.

`--compare` only compares the laughter guard, so to hear these settings side by side, render a few variants of a short clip:

```
voice-enhance clip.wav -o clip_mr12.wav --max-reduction 12
voice-enhance clip.wav -o clip_blend20.wav --blend 0.2
```
