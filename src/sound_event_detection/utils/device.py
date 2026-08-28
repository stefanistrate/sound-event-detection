"""Torch device selection shared by the CLIs and the serving entry points."""

import os

import torch

DEVICE_ENV_VAR = "SED_DEVICE"


def default_device() -> str:
    """Pick the best available torch device on this machine.

    Prefers CUDA, then Apple Silicon's Metal backend (MPS), then CPU. The MPS
    wheels ship with the ``cpu`` dependency group, so a plain ``uv sync`` on a
    Mac already has GPU support -- only the CUDA wheels need ``--group gpu``.

    Returns
    -------
    str
        ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: str | None = None) -> str:
    """Resolve an explicit device, else ``SED_DEVICE``, else `default_device`.

    Parameters
    ----------
    device : str | None
        Explicit device string. When ``None``, the ``SED_DEVICE`` environment
        variable is consulted before falling back to auto-detection.

    Returns
    -------
    str
        The device to place the model on.
    """
    if device is not None:
        return device
    return os.environ.get(DEVICE_ENV_VAR) or default_device()
