"""Metrics for the p-Bridge.

Full-reference: PSNR and SSIM on intensities. Family-specific diagnostics:
the Pearson residual of Eq. (16) (mean 0, variance 1 under a calibrated
predictor, for every p — Prop. 26), the ratio mean (the shrinkage statistic
gamma-Bridge reports), and the empirical zero-atom mass (Remark 37).

Intensity stratification (§13.2) is not optional here: Eq. (22) equalises CV
only at mu_ref, and the exponents differ away from it — which is the effect
under study. Averaging over the whole image hides exactly the signal E0 is
meant to detect, so every table is accompanied by its darkest-decile version.
"""

from __future__ import annotations

import math

import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

# Fixed data range: intensities live in [eps, 1] by construction, so a shared
# range keeps PSNR comparable across images, exponents and severities.
DATA_RANGE = 1.0


def psnr(x_ref: np.ndarray, x_hat: np.ndarray, data_range: float = DATA_RANGE) -> float:
    mse = float(np.mean((x_ref - x_hat) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def ssim(x_ref: np.ndarray, x_hat: np.ndarray, data_range: float = DATA_RANGE) -> float:
    kwargs = {"data_range": data_range}
    if x_ref.ndim == 3:
        kwargs["channel_axis"] = 0 if x_ref.shape[0] in (1, 3) else -1
    return float(ssim_fn(x_ref, x_hat, **kwargs))


def masked_psnr(x_ref: np.ndarray, x_hat: np.ndarray, mask: np.ndarray,
                data_range: float = DATA_RANGE) -> float:
    """PSNR over a pixel subset. Used for the darkest-decile column."""
    if mask.sum() == 0:
        return float("nan")
    mse = float(np.mean((x_ref[mask] - x_hat[mask]) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def decile_thresholds(x0_all: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Intensity decile edges computed once over the whole eval set, so the same
    pixel population defines "dark" in every cell of the E0 matrix."""
    return np.quantile(x0_all.ravel(), np.linspace(0.0, 1.0, n_bins + 1))


def psnr_by_decile(x_ref: np.ndarray, x_hat: np.ndarray,
                   edges: np.ndarray) -> list[float]:
    """PSNR stratified by the clean intensity of each pixel (§13.2)."""
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (x_ref >= lo) & (x_ref < hi if i < len(edges) - 2 else x_ref <= hi)
        out.append(masked_psnr(x_ref, x_hat, mask))
    return out


def log_rmse(x_ref: np.ndarray, x_hat: np.ndarray, floor: float = 1e-6) -> float:
    """RMSE of log(x_hat) - log(x0): a scale-free error, i.e. a relative error.

    This is the statistic the dark-decile prediction actually speaks to.
    Eq. (21) is a claim about CV(mu) = sqrt(phi) mu^(p/2-1) nu^(-1/2), which is
    *relative*: for p < 2 the noise is proportionally worse in dark regions.
    Absolute PSNR cannot see this — dark pixels have small values, so their
    squared error is small whatever the model does, and a dark-decile PSNR comes
    out flattering rather than diagnostic. Reporting both keeps §13.3's PSNR
    tables while testing the prediction with an instrument that can detect it.
    Lower is better.
    """
    a = np.log(np.maximum(x_ref, floor))
    b = np.log(np.maximum(x_hat, floor))
    return float(np.sqrt(np.mean((a - b) ** 2)))


def masked_log_rmse(x_ref: np.ndarray, x_hat: np.ndarray, mask: np.ndarray,
                    floor: float = 1e-6) -> float:
    if mask.sum() == 0:
        return float("nan")
    return log_rmse(x_ref[mask], x_hat[mask], floor)


def log_rmse_by_decile(x_ref: np.ndarray, x_hat: np.ndarray,
                       edges: np.ndarray) -> list[float]:
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (x_ref >= lo) & (x_ref < hi if i < len(edges) - 2 else x_ref <= hi)
        out.append(masked_log_rmse(x_ref, x_hat, mask))
    return out


def pearson_residual_stats(fam, x_obs: np.ndarray, x_hat: np.ndarray,
                           nu_obs: float) -> dict:
    """eps = (x_obs - x_hat)/sqrt(V(x_hat)/nu_obs), Eq. (16).

    Reported for every p: a calibrated predictor gives mean 0 and variance 1
    regardless of the local intensity (Prop. 26). At p=1 this is the Fano-1
    statistic and at p=2 the ratio-variance statistic of the source papers.
    """
    xh = np.maximum(x_hat, 1e-6)
    V = np.full_like(xh, fam.phi) if fam.p == 0.0 else fam.phi * xh ** fam.p
    eps = (x_obs - xh) / np.sqrt(V / nu_obs)
    return {"res_mean": float(eps.mean()), "res_var": float(eps.var(ddof=1))}


def ratio_mean(x_ref: np.ndarray, x_hat: np.ndarray) -> float:
    """mean(x_hat / x0) — the systematic shrinkage statistic. Cor. 24 predicts a
    mean-consistent loss pushes this towards 1.

    NOT comparable to gamma-Bridge's published 0.947: that repo's `ratio_stats`
    reports mean(x_obs / x_hat), which is a different statistic and lands on the
    other side of 1. Nor are the two reciprocals — E[1/X] != 1/E[X] and these
    pixelwise ratios are heavy-tailed. Measured on one p=2 severe checkpoint:
    mean(x_hat/x0) = 1.0296 while mean(x_obs/x_hat) = 0.9918 on the same
    predictions. Convert before comparing across the two repos.
    """
    return float((np.maximum(x_hat, 1e-6) / np.maximum(x_ref, 1e-6)).mean())


def zero_fraction(Z: np.ndarray) -> float:
    """Empirical Pr[Z = 0] — compare against exp(-nu*lambda) for 1<p<2."""
    return float((Z == 0).mean())
