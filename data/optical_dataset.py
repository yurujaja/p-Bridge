"""Natural-image dataset for p-Bridge synthetic training.

Same pools as lambda-/gamma-Bridge (BSDS500 train+val, optionally DIV2K HR),
loaded as intensity fields in [eps, 1]. The intensity floor is not cosmetic:
Assumption 2 requires x0 in Omega = (0, inf) for p >= 1, and mu -> 0 sends
CV -> inf for every p < 2 (§13.2, "Intensity range"). We keep the lambda-Bridge
convention eps = 0.05 for all p so that the cross-p comparison is not
contaminated by a different clipping rule.

`OpticalTrainDataset` yields clean crops only; the training loop draws fresh
corruption every step. `PairedEvalDataset` yields deterministic (x0, Z_obs)
pairs at a *given* (p, nu_obs), which is what makes the E0 cross-inference
matrix possible: the same clean crops and the same seed, corrupted under a
different exponent.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .download_bsds500 import ensure_bsds500

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
DEFAULT_EPS = 0.05


def _load_image(path: Path, channels: int) -> np.ndarray:
    if channels == 1:
        return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if channels == 3:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    raise ValueError(f"unsupported channels: {channels}")


def _random_crop(img: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    H, W = img.shape[:2]
    if H < size or W < size:
        ph, pw = max(0, size - H), max(0, size - W)
        pad = ((0, ph), (0, pw)) if img.ndim == 2 else ((0, ph), (0, pw), (0, 0))
        img = np.pad(img, pad, mode="edge")
        H, W = img.shape[:2]
    y = rng.randint(0, H - size)
    x = rng.randint(0, W - size)
    return img[y:y + size, x:x + size]


def _prep(img: np.ndarray, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """(H,W) or (H,W,C) in [0,1] -> (C,H,W) in [eps, 1]."""
    img = np.clip(img, 0.0, 1.0)
    img = eps + (1.0 - eps) * img
    if img.ndim == 2:
        return torch.from_numpy(img).float().unsqueeze(0)
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()


def _collect_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS]


def build_pool(extra_roots: Optional[list] = None, split_seed: int = 0,
               split: str = "trainval") -> list[Path]:
    """Deterministic training pool: BSDS500 train+val (+ extra roots)."""
    root = ensure_bsds500()
    if split == "trainval":
        pool = sorted(list((root / "train").glob("*.jpg")) + list((root / "val").glob("*.jpg")))
    else:
        pool = sorted((root / split).glob("*.jpg"))
    if extra_roots:
        for r in extra_roots:
            rp = Path(r)
            if rp.exists():
                pool.extend(sorted(_collect_images(rp)))
    rng = random.Random(split_seed)
    rng.shuffle(pool)
    return pool


class OpticalTrainDataset(Dataset):
    """Random clean crops; corruption happens in the training loop."""

    def __init__(self, crop_size: int = 256, split_seed: int = 0, length: int = 4000,
                 extra_roots: Optional[list] = None, eps: float = DEFAULT_EPS,
                 channels: int = 1):
        self.pool = build_pool(extra_roots, split_seed)
        self.crop_size = int(crop_size)
        self.length = int(length)
        self.eps = float(eps)
        self.channels = int(channels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        rng = random.Random((idx + 1) * 2654435761 % 2**32)
        img = _load_image(rng.choice(self.pool), self.channels)
        return {"x0": _prep(_random_crop(img, self.crop_size, rng), self.eps)}


class PairedEvalDataset(Dataset):
    """Held-out BSDS500 test crops with deterministic Tweedie corruption.

    The corruption exponent `p` and index `nu_obs` are explicit arguments and
    need not match the model under test — that mismatch *is* experiment E0.

    Returns:
        x0:     (C,H,W) clean intensity in [eps, 1]
        Z_obs:  (C,H,W) additive observation at index nu_obs
        x_obs:  (C,H,W) = Z_obs / nu_obs, the intensity-domain view
        latent: (C,H,W) latent jump count N of Z_obs for 1<p<2, else empty.
                Needed to thin Z_obs downward (Prop. 18 — the simulator owns N,
                so both layers stay exact); the cross-severity protocol in
                eval/run_eval.py uses it to put the conditioning channel back at
                the severity the model trained on. Zero-size when p is not in
                (1,2), so a DataLoader still collates it.
    """

    def __init__(self, crop_size: int = 256, p: float = 1.0, nu_obs: float = 10.0,
                 phi: float = 1.0, num_samples: int = 64, seed: int = 42,
                 eps: float = DEFAULT_EPS, channels: int = 1,
                 split: str = "test"):
        # `split` exists so an inference-time choice can be *selected* somewhere
        # the reported tables never look. "test" is the reported set and stays
        # the default; "val" and "train" are inside the training pool
        # (build_pool takes BSDS500 train+val), which makes them a conservative
        # place to tune — the model has already seen those images, so a setting
        # that still helps on unseen ones is a real gain rather than a fit.
        from pbridge_core.tweedie import Tweedie   # local import: avoids cycle
        root = ensure_bsds500()
        self.split = str(split)
        self.paths = sorted((root / self.split).glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(f"no images in BSDS500 split {self.split!r} under {root}")
        self.crop_size = int(crop_size)
        self.fam = Tweedie(p=float(p), phi=float(phi))
        self.p = float(p)
        self.nu_obs = float(nu_obs)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.eps = float(eps)
        self.channels = int(channels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        rng = random.Random(self.seed + idx)
        img = _load_image(self.paths[idx % len(self.paths)], self.channels)
        x0 = _prep(_random_crop(img, self.crop_size, rng), self.eps)
        # Deterministic draw: the same crop and the same seed across every p, so
        # the E0 matrix differs only in the corruption law, not in the content.
        state = torch.random.get_rng_state()
        torch.manual_seed(self.seed + idx * 7919)
        nu = torch.full_like(x0, self.nu_obs)
        # Draws exactly as sample_Z did (one poisson then one gamma, same
        # arguments), so Z_obs stays bit-identical to every number published
        # before the latent was exposed — verified at p in {1, 1.5, 2}.
        Z_obs, latent = self.fam.sample_Z_with_latent(nu, x0)
        torch.random.set_rng_state(state)
        return {"x0": x0, "Z_obs": Z_obs, "x_obs": Z_obs / self.nu_obs,
                "latent": latent if latent is not None else torch.zeros(0)}


def dataset_mean_intensity(extra_roots: Optional[list] = None,
                           eps: float = DEFAULT_EPS, channels: int = 1,
                           max_images: int | None = None) -> float:
    """mu_ref of §13.1: the dataset mean intensity, computed over full images
    (not crops) so it is a property of the pool and not of a sampling seed.

    This number must be held constant across every run and reported in every
    table caption; `scripts/calibrate.py` caches it to calibration.json.
    """
    pool = build_pool(extra_roots)
    if max_images is not None:
        pool = pool[:max_images]
    total, count = 0.0, 0
    for path in pool:
        img = np.clip(_load_image(path, channels), 0.0, 1.0)
        img = eps + (1.0 - eps) * img
        total += float(img.sum())
        count += img.size
    return total / max(count, 1)
