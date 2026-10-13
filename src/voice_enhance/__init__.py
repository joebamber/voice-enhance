"""Local, Adobe-Podcast-style speech enhancement."""

__version__ = "0.10.0"

import os as _os

# Metal-unsupported operations run on the CPU instead of failing (see accel.py);
# torch only reads this when it's first imported.
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
