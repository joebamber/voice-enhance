#!/usr/bin/env bash
# One-shot: install voice-enhance and render A/B
# sets for the laugh clip plus two deliberately thin versions of it.
#
#   ./scripts/try-rebuild.sh
#
# First run downloads the model weights from Hugging Face (a few hundred MB,
# cached in ~/.cache/voice-enhance/clearvoice).
set -euo pipefail
cd "$(dirname "$0")/.."

for tool in ffmpeg uv; do
  if ! command -v "$tool" >/dev/null; then
    echo "Installing $tool with Homebrew..."
    brew install "$tool"
  fi
done

echo "==> Installing voice-enhance (with ClearerVoice)"
uv tool install --python 3.11 . --force --reinstall-package voice-enhance --overrides overrides.txt
export PATH="$HOME/.local/bin:$PATH"

echo "==> Backends available:"
voice-enhance --list-backends

OUT=out/rebuild-test
mkdir -p "$OUT"
for f in out/laugh-test.wav out/thin/phone.wav out/thin/laptop-zoom.wav; do
  if [[ -f "$f" ]]; then
    echo
    echo "==> $f"
    time voice-enhance "$f" --compare -o "$OUT"
  else
    echo "skipping $f (not found)"
  fi
done

echo
echo "Done. Listen in: $(pwd)/$OUT"
open "$OUT" 2>/dev/null || true
