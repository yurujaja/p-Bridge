"""Tweedie / additive-EDM primitives for the p-Bridge.

Everything the exponent `p` touches lives in this file. Per the unified notes,
`p` enters in exactly three places (Remark 11, §12):

    1. the law of the forward state / stochastic increment  -> `sample_Z`
    2. the loss                                             -> `deviance`
    3. the thinning kernel used to build training couples    -> `thin`

The deterministic reverse (Thm 10) is p-independent and therefore lives in
`diffusion.py`, not here.

Conventions (§13.1, §13.2). The additive variable is Z ~ ED*(theta0, nu) with

    E[Z] = nu * mu,        Var(Z) = nu * phi * mu^p,
    x    = Z / nu,         Var(x)  = phi * mu^p / nu,

so the coefficient of variation is CV(mu) = sqrt(phi) * mu^(p/2-1) * nu^(-1/2).
We fix phi = 1 throughout and equalise severity across p by solving (22) for
nu_obs; see `nu_obs_for_severity`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

# p values with a directly samplable thinning kernel (Observation 17).
CLOSED_FORM_P = (0.0, 1.0, 2.0)

_TINY = 1e-12


def _as_tensor(v, ref: Tensor) -> Tensor:
    if torch.is_tensor(v):
        return v.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(v, device=ref.device, dtype=ref.dtype)


# --------------------------------------------------------------------------- #
# Severity calibration (§13.1) — the single easiest place to make an
# unpublishable table, so it is a function and not a magic number.
# --------------------------------------------------------------------------- #

def nu_obs_for_severity(p: float, eta: float, mu_ref: float, phi: float = 1.0) -> float:
    """Observation index meeting target severity eta = CV(mu_ref), Eq. (22).

        nu_obs(p) = phi * eta^-2 * mu_ref^(p-2)

    phi carries p-dependent units, so it is *not* meaningful to hold nu fixed
    across p; this is what makes cross-p tables comparable.
    """
    return float(phi) * float(eta) ** -2.0 * float(mu_ref) ** (float(p) - 2.0)


def cv_at(p: float, nu: float, mu: float, phi: float = 1.0) -> float:
    """CV(mu) = sqrt(phi) mu^(p/2-1) nu^(-1/2). Inverse of the above."""
    return math.sqrt(phi) * mu ** (p / 2.0 - 1.0) * nu ** -0.5


# --------------------------------------------------------------------------- #
# The family
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Tweedie:
    """Additive Tweedie EDM with power-variance exponent `p` and dispersion `phi`.

    Supported: p = 0 (Gaussian), p = 1 (Poisson), 1 < p < 2 (compound
    Poisson-Gamma), p = 2 (Gamma). p in (0, 1) does not exist (Remark 34) and
    p > 2 has no direct sampler for the thinning kernel (Conjecture 21).
    """

    p: float
    phi: float = 1.0

    def __post_init__(self):
        p = float(self.p)
        if 0.0 < p < 1.0:
            raise ValueError(
                f"no Tweedie member exists for 0 < p < 1 (got p={p}); the stable "
                "index would exceed 2 — see Remark 34."
            )
        if p < 0.0 or p > 2.0:
            raise NotImplementedError(
                f"p={p} outside the implemented range [0, 2]; p>2 has no directly "
                "samplable thinning kernel (Conjecture 21)."
            )
        if p == 1.0 and float(self.phi) != 1.0:
            # A Poisson count has Var = E; an over-dispersed p=1 law is not in
            # the additive family. Severity is set through nu, not phi (§13.1).
            raise ValueError(f"p=1 requires phi=1, got phi={self.phi}")

    # --------------------------------------------------------------- structure

    @property
    def is_gaussian(self) -> bool:
        return self.p == 0.0

    @property
    def is_poisson(self) -> bool:
        return self.p == 1.0

    @property
    def is_gamma(self) -> bool:
        return self.p == 2.0

    @property
    def is_compound(self) -> bool:
        return 1.0 < self.p < 2.0

    @property
    def alpha(self) -> float:
        """Gamma jump shape, Eq. (11): alpha = (2-p)/(p-1). Only for 1<p<2.

        Limits (Remark 19): p->2- => alpha->0; p->1+ => alpha->inf; p=3/2 => 1.
        """
        if not self.is_compound:
            raise ValueError(f"alpha is defined only for 1<p<2, got p={self.p}")
        return (2.0 - self.p) / (self.p - 1.0)

    @property
    def excess_noise_factor(self) -> float:
        """F^2 = 1 + 1/alpha, so p = 2 - 1/F^2 (Eq. 12) — the datasheet reading."""
        return 1.0 + 1.0 / self.alpha

    def V(self, mu: Tensor | float):
        """Variance function V(mu) = phi * mu^p (phi * 1 at p=0)."""
        if self.is_gaussian:
            if torch.is_tensor(mu):
                return torch.full_like(mu, self.phi)
            return self.phi
        if torch.is_tensor(mu):
            return self.phi * mu.clamp_min(_TINY) ** self.p
        return self.phi * max(float(mu), _TINY) ** self.p

    # --------------------------------------------------------------- forward

    def sample_Z(self, nu: Tensor, mu: Tensor, generator=None) -> Tensor:
        """Draw Z ~ ED*(theta0(mu), nu), i.e. E[Z]=nu*mu, Var(Z)=nu*phi*mu^p.

        `nu` and `mu` broadcast together; the result has their broadcast shape.
        Used both for the forward marginal (§13.2) and, with nu := delta_t, for
        the stochastic reverse increment (Eq. 7) — they are the same object.
        """
        mu = mu.clamp_min(_TINY)
        nu = _as_tensor(nu, mu)
        nu, mu = torch.broadcast_tensors(nu, mu)
        nu = nu.clamp_min(0.0)

        if self.is_poisson:
            return torch.poisson(nu * mu, generator=generator)

        if self.is_gaussian:
            noise = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype,
                                generator=generator)
            return nu * mu + torch.sqrt(nu * self.phi) * noise

        if self.is_gamma:
            # Gamma(shape=k, scale=s) has E=ks, Var=ks^2; matching E[Z]=nu*mu and
            # Var(Z)=nu*phi*mu^2 gives k = nu/phi, s = phi*mu. At phi=1 this is
            # Gamma(nu, rate=1/mu), exactly Table 1.
            return _gamma_sample(nu / self.phi, scale=self.phi * mu)

        # 1 < p < 2: compound Poisson-Gamma, Eqs. (23)-(24).
        lam, alpha, beta = self.compound_params(mu)
        N = torch.poisson(nu * lam, generator=generator)
        # Sum of N iid Gamma(alpha, beta) is Gamma(N*alpha, beta); N=0 -> Z=0
        # exactly, which is the zero atom of Remark 37.
        Z = _gamma_sample(N * alpha, scale=beta)
        return torch.where(N > 0, Z, torch.zeros_like(Z))

    def compound_params(self, mu: Tensor):
        """(lambda, alpha, beta) of Eq. (24) for the 1<p<2 latent augmentation."""
        p, phi = self.p, self.phi
        mu = mu.clamp_min(_TINY)
        lam = mu ** (2.0 - p) / (phi * (2.0 - p))
        alpha = (2.0 - p) / (p - 1.0)
        beta = phi * (p - 1.0) * mu ** (p - 1.0)
        return lam, alpha, beta

    def zero_atom_mass(self, nu: Tensor | float, mu: Tensor | float):
        """Pr[Z = 0] = exp(-nu*lambda) for 1<p<2, else 0 (Remark 37)."""
        if not self.is_compound:
            return torch.zeros_like(mu) if torch.is_tensor(mu) else 0.0
        mu_t = mu if torch.is_tensor(mu) else torch.as_tensor(float(mu))
        lam, _, _ = self.compound_params(mu_t)
        nu_t = _as_tensor(nu, lam)
        return torch.exp(-nu_t * lam)

    # --------------------------------------------------------------- thinning

    def thin(self, Z_hi: Tensor, nu_hi: Tensor, nu_lo: Tensor,
             latent: Tensor | None = None, generator=None):
        """Downward (measurement-removing) transition Z_lo | Z_hi.

        Implements Observation 17 for p in {0,1,2} and Proposition 18 for
        1<p<2, where `latent` carries the latent jump count N of Z_hi and the
        returned latent count is that of Z_lo (the simulator owns N, so both
        layers stay exact — Remark 20).

        Returns (Z_lo, latent_lo); latent_lo is None unless 1<p<2.
        """
        nu_hi = _as_tensor(nu_hi, Z_hi)
        nu_lo = _as_tensor(nu_lo, Z_hi)
        ratio = (nu_lo / nu_hi.clamp_min(_TINY)).clamp(0.0, 1.0)   # alpha_t

        if self.is_poisson:
            # Binomial(Z_hi, alpha_t): unit retention Bern(alpha_t).
            return _binomial_sample(Z_hi, ratio.expand_as(Z_hi)), None

        if self.is_gamma:
            # Z_lo / Z_hi ~ Beta(nu_lo/phi, (nu_hi-nu_lo)/phi), independent of
            # Z_hi (Lukacs); the shape index of the additive Gamma is nu/phi.
            a = (nu_lo / self.phi).expand_as(Z_hi)
            b = ((nu_hi - nu_lo) / self.phi).clamp_min(_TINY).expand_as(Z_hi)
            return Z_hi * _beta_sample(a, b), None

        if self.is_gaussian:
            # N(alpha_t * Z_hi, phi * nu_lo * (nu_hi - nu_lo) / nu_hi).
            var = self.phi * nu_lo * (nu_hi - nu_lo).clamp_min(0.0) / nu_hi.clamp_min(_TINY)
            noise = torch.randn(Z_hi.shape, device=Z_hi.device, dtype=Z_hi.dtype,
                                generator=generator)
            return ratio * Z_hi + torch.sqrt(var.clamp_min(0.0)) * noise, None

        # 1 < p < 2: two closed-form layers (Prop. 18).
        if latent is None:
            raise ValueError("1<p<2 thinning needs the latent jump count of Z_hi")
        alpha = self.alpha
        n_lo = _binomial_sample(latent, ratio.expand_as(latent))
        a = (n_lo * alpha).clamp_min(_TINY)
        b = ((latent - n_lo) * alpha).clamp_min(_TINY)
        frac = _beta_sample(a, b)
        # Degenerate ends: no jumps retained -> 0; all retained -> Z_hi.
        frac = torch.where(n_lo <= 0, torch.zeros_like(frac), frac)
        frac = torch.where(n_lo >= latent, torch.ones_like(frac), frac)
        return Z_hi * frac, n_lo

    def sample_Z_with_latent(self, nu: Tensor, mu: Tensor, generator=None):
        """As `sample_Z`, additionally returning the latent jump count for
        1<p<2 (needed to seed `thin`). Returns (Z, latent_or_None)."""
        if not self.is_compound:
            return self.sample_Z(nu, mu, generator=generator), None
        mu = mu.clamp_min(_TINY)
        nu = _as_tensor(nu, mu)
        nu, mu = torch.broadcast_tensors(nu, mu)
        lam, alpha, beta = self.compound_params(mu)
        N = torch.poisson(nu.clamp_min(0.0) * lam, generator=generator)
        Z = _gamma_sample(N * alpha, scale=beta)
        Z = torch.where(N > 0, Z, torch.zeros_like(Z))
        return Z, N

    # --------------------------------------------------------------- loss

    def deviance(self, y: Tensor, mu: Tensor) -> Tensor:
        """Unit deviance d_p(y, mu), Eq. (15). Pointwise >= 0, zero iff y == mu,
        and (Thm 23) minimised in mu exactly at E[y] — which is the estimand the
        deterministic reverse tracks (Cor. 24).

        Guarded at y = 0 with the limiting form (Remark 37). For p >= 1 the
        target y = x0 is bounded below by the intensity floor anyway, so the
        guard matters only for diagnostics evaluated at observations.
        """
        p = self.p

        if p == 0.0:
            # No positivity guard here: Omega = R at p=0 and the identity head
            # emits negative mu legitimately. Clamping mu would freeze the
            # gradient wherever mu <= 0, and the UNet's zero-init output conv
            # starts every pixel at exactly mu = 0 -- i.e. below the clamp. The
            # p=0 cell was untrainable from step 0 until this branch was lifted
            # above the guard.
            return (y - mu) ** 2

        mu = mu.clamp_min(_TINY)
        y_pos = y.clamp_min(_TINY)

        if p == 1.0:
            # 2[ y log(y/mu) - (y - mu) ];  y log y -> 0 as y -> 0.
            ylog = torch.where(y > 0, y * (torch.log(y_pos) - torch.log(mu)),
                               torch.zeros_like(y))
            return 2.0 * (ylog - (y - mu))

        if p == 2.0:
            # 2[ y/mu - log(y/mu) - 1 ] — Itakura-Saito; diverges as y -> 0,
            # so the intensity floor of Assumption 2 is load-bearing here.
            return 2.0 * (y_pos / mu - (torch.log(y_pos) - torch.log(mu)) - 1.0)

        # general 1 < p < 2 (also valid for p > 2)
        t1 = y_pos ** (2.0 - p) / ((1.0 - p) * (2.0 - p))
        t1 = torch.where(y > 0, t1, torch.zeros_like(t1))   # y^(2-p) -> 0
        t2 = y * mu ** (1.0 - p) / (1.0 - p)
        t3 = mu ** (2.0 - p) / (2.0 - p)
        return 2.0 * (t1 - t2 + t3)

    def pearson_residual(self, x_obs: Tensor, mu: Tensor, nu_obs: float | Tensor) -> Tensor:
        """eps = (x_obs - mu) / sqrt(V(mu)/nu_obs), Eq. (16).

        Under mu == x0 this is pixelwise zero-mean unit-variance irrespective of
        the local intensity (Prop. 26), hence safe to pool across pixels. The
        p=2 and p=1 specialisations are gamma-Bridge Eq. (12) and the Fano-1
        term of lambda-Bridge Eq. (12) respectively (Remark 27).
        """
        nu = _as_tensor(nu_obs, mu)
        if not self.is_gaussian:
            # V(mu) = phi * mu^p needs mu > 0; at p=0 it is the constant phi, so
            # the guard is both unnecessary and harmful -- it would bias eps and
            # cut the moment term's gradient for negative mu (see deviance).
            mu = mu.clamp_min(1e-6)
        return (x_obs - mu) / torch.sqrt(self.V(mu) / nu)


# --------------------------------------------------------------------------- #
# Samplers: thin wrappers that survive degenerate shape parameters.
# --------------------------------------------------------------------------- #

def _gamma_sample(concentration: Tensor, scale: Tensor) -> Tensor:
    """Gamma(concentration, scale) elementwise; concentration == 0 yields 0.

    torch._standard_gamma is undefined at concentration 0, so we clamp and mask.
    It takes no `generator`; reproducibility is handled by seeding at the call
    site (as the eval dataset does).
    """
    conc = concentration.clamp_min(_TINY)
    g = torch._standard_gamma(conc)
    out = g * scale
    return torch.where(concentration > 0, out, torch.zeros_like(out))


def _beta_sample(a: Tensor, b: Tensor) -> Tensor:
    """Beta(a, b) elementwise via the Gamma ratio (avoids Dirichlet overhead)."""
    ga = torch._standard_gamma(a.clamp_min(_TINY))
    gb = torch._standard_gamma(b.clamp_min(_TINY))
    s = (ga + gb).clamp_min(_TINY)
    return (ga / s).clamp(0.0, 1.0)


def _binomial_sample(total_count: Tensor, probs: Tensor) -> Tensor:
    """Binomial(n, p) elementwise with float-valued integer n."""
    n = total_count.clamp_min(0.0)
    p = probs.to(n.dtype).clamp(0.0, 1.0)
    return torch.distributions.Binomial(total_count=n, probs=p).sample()
