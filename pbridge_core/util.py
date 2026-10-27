"""Small helpers."""

from __future__ import annotations
import torch


def unsqueeze_xdim(z: torch.Tensor, xdim) -> torch.Tensor:
    """Append trailing singleton dims so `z` broadcasts against a tensor whose
    non-batch shape is `xdim` (e.g. z: (B,), xdim: (C,H,W) -> (B,1,1,1))."""
    bc_dim = (Ellipsis,) + (None,) * len(xdim)
    return z[bc_dim]
