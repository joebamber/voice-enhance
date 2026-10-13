"""Run the slow models on the Mac's GPU (Metal / MPS) when there is one.

Each model is tried on the GPU first. If that fails (an operation Metal doesn't
support, out of memory, a TorchScript file with the CPU baked in), the step is
re-run on the CPU and that model stays on the CPU for the rest of the session.
Operations Metal lacks fall back to the CPU one by one where torch can manage it
(PYTORCH_ENABLE_MPS_FALLBACK, set in __init__ before torch is imported).
"""

from __future__ import annotations

enabled = True  # --cpu turns it off
failed: dict[str, str] = {}  # model -> why it's on the CPU


def device():
    import torch
    return torch.device("mps" if enabled and torch.backends.mps.is_available() else "cpu")


def attempt(name: str, fn):
    """fn(device), on the GPU if possible, else on the CPU."""
    import torch
    dev = device()
    if dev.type != "cpu" and name not in failed:
        try:
            return fn(dev)
        except (RuntimeError, NotImplementedError) as e:
            failed[name] = (str(e).strip().splitlines() or [type(e).__name__])[0][:200]
            torch.mps.empty_cache()
            from . import progress
            progress.write(f"{name}: GPU failed, using the CPU ({failed[name]})")
    return fn(torch.device("cpu"))


def where(name: str) -> str:
    return "CPU" if device().type == "cpu" or name in failed else "GPU"
