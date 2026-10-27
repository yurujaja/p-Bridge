"""PBridge — the nu-indexed additive-EDM diffusion bridge for any Tweedie p.

Generalises `poisson-bridge/pbridge_core/diffusion.py` (p=1) to the whole
power-variance axis. The structure is deliberately unchanged so that the p=1
instance reproduces lambda-Bridge exactly; everything that depends on p is
delegated to `tweedie.Tweedie`.

State. We carry the *additive* variable Z_t ~ ED*(theta0, nu(t)) and expose the
intensity view x_t = Z_t / nu(t), which is what the network sees (§13.5, item 1).
At p=1, Z is the integer photon count and x = n/D, i.e. the two coordinate
systems of lambda-Bridge.

Schedule (Assumption 3). nu(0) = nu_max (clean end), nu(T-1) = nu_obs
(observation end), strictly monotone. `nu_obs` is *not* a free knob across p:
it is fixed by the severity calibration of Eq. (22) — see
`tweedie.nu_obs_for_severity` and `scripts/calibrate.py`.

The reverse:
    stochastic     Z_{t-1} = Z_t + Y,  Y ~ ED*(theta_hat0, delta_t)     (Eq. 7)
    deterministic  Z_{t-1} = Z_t + delta_t * x0_hat                     (Eq. 8)
The second is p-independent (Theorem 10) — in the intensity domain it is
x_{t-1} = alpha_t x_t + (1 - alpha_t) x0_hat with alpha_t = nu(t)/nu(t-1),
literally the same equation as in both source papers.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from tqdm import tqdm

from .tweedie import Tweedie
from .util import unsqueeze_xdim


def nu_schedule(t: torch.Tensor, nu_obs: float, nu_max: float,
                mode: str = "log") -> torch.Tensor:
    """nu(t) with nu(0) = nu_max and nu(1) = nu_obs, t in [0, 1].

    'log' is log-linear interpolation (geometric halving cadence, the
    lambda-/gamma-Bridge default); 'linear' is the ablation.
    """
    if mode == "log":
        log_hi, log_lo = math.log(float(nu_max)), math.log(float(nu_obs))
        return torch.exp((1.0 - t) * log_hi + t * log_lo)
    if mode == "linear":
        return (1.0 - t) * float(nu_max) + t * float(nu_obs)
    raise ValueError(f"unknown schedule mode: {mode}")


class PBridge:
    """Discretised nu-indexed bridge for a Tweedie exponent p.

    Args:
        num_timesteps: number of discrete steps T.
        p:             power-variance exponent (0, or [1, 2]).
        nu_obs:        observation-end index; set by Eq. (22), not by hand.
        nu_max:        clean-end index. Ratio nu_max/nu_obs is the dynamic range
                       and should be held fixed across p so that the clean-end
                       CV is also comparable.
        phi:           dispersion; fixed to 1 by the calibration convention.
        schedule:      'log' (default) or 'linear'.
    """

    def __init__(
        self,
        num_timesteps: int,
        p: float,
        nu_obs: float,
        nu_max: float,
        phi: float = 1.0,
        schedule: str = "log",
        device: str | torch.device = "cpu",
    ):
        assert nu_max > nu_obs, f"nu_max ({nu_max}) must exceed nu_obs ({nu_obs})"
        self.T = int(num_timesteps)
        self.p = float(p)
        self.phi = float(phi)
        self.nu_obs = float(nu_obs)
        self.nu_max = float(nu_max)
        self.schedule = schedule
        self.device = torch.device(device)
        self.fam = Tweedie(p=self.p, phi=self.phi)

        steps = torch.arange(self.T, dtype=torch.float32)
        t = steps / max(self.T - 1, 1)
        nu = nu_schedule(t, nu_obs=self.nu_obs, nu_max=self.nu_max, mode=schedule)

        self.t = t.to(self.device)              # (T,)
        self.nu = nu.to(self.device)            # (T,)
        self.log_nu = torch.log(self.nu)        # (T,) — network conditioning

    # ------------------------------------------------------------------ convenience

    def nu_at(self, step) -> torch.Tensor:
        if not torch.is_tensor(step):
            step = torch.as_tensor(step, device=self.device, dtype=torch.long)
        return self.nu[step.to(self.device).long()]

    def t_at(self, step) -> torch.Tensor:
        if not torch.is_tensor(step):
            step = torch.as_tensor(step, device=self.device, dtype=torch.long)
        return self.t[step.to(self.device).long()]

    def cv_at(self, step, mu: float) -> float:
        """Coefficient of variation of x_step at intensity mu — the quantity the
        severity calibration equalises across p."""
        nu = float(self.nu_at(torch.as_tensor(int(step))))
        return math.sqrt(self.phi) * mu ** (self.p / 2.0 - 1.0) * nu ** -0.5

    # ------------------------------------------------------------------ forward

    def q_sample(self, step: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Forward marginal Z_t ~ ED*(theta0(x0), nu(t)) (Prop. 7)."""
        assert step.ndim == 1 and step.shape[0] == x0.shape[0], (step.shape, x0.shape)
        nu_b = unsqueeze_xdim(self.nu_at(step).to(x0.dtype), x0.shape[1:]).to(x0.device)
        return self.fam.sample_Z(nu_b, x0)

    def q_sample_thinning_chain(self, step: torch.Tensor, x0: torch.Tensor):
        """Generate (Z_t, Z_obs) lying on the SAME downward thinning chain.

            Z_max ~ ED*(theta0, nu_max)
            Z_t   | Z_max ~ thin(nu_max -> nu(t))
            Z_obs | Z_t   ~ thin(nu(t) -> nu_obs)

        This is the coupling the reverse sampler actually realises (Thm 13), so
        the network sees the observation on the same statistical footing at
        training time as at inference time. For 1<p<2 the latent jump count is
        carried along so that both layers of Prop. 18 stay exact (Remark 20).

        Returns (Z_t, Z_obs, nu_step) with nu_step of shape (B,).
        """
        assert step.ndim == 1 and step.shape[0] == x0.shape[0]
        xdim = x0.shape[1:]
        nu_max_b = torch.full_like(x0, self.nu_max)
        Z_max, latent = self.fam.sample_Z_with_latent(nu_max_b, x0)

        nu_step = self.nu_at(step).to(x0.dtype)                      # (B,)
        nu_step_b = unsqueeze_xdim(nu_step, xdim).to(x0.device).expand_as(x0)
        Z_t, latent_t = self.fam.thin(Z_max, nu_max_b, nu_step_b, latent=latent)

        nu_obs_b = torch.full_like(x0, self.nu_obs)
        # nu(t) can equal nu_obs at the terminal step; thin() clamps the ratio to 1.
        Z_obs, _ = self.fam.thin(Z_t, nu_step_b, nu_obs_b, latent=latent_t)
        return Z_t, Z_obs, nu_step

    # ------------------------------------------------------------------ reverse

    def p_posterior(
        self,
        nprev: int,
        n: int,
        Z_state: torch.Tensor,
        x0_hat: torch.Tensor,
        ot_ode: bool = False,
        naive: bool = False,
    ) -> torch.Tensor:
        """One reverse step from step `n` (low nu) to step `nprev` (high nu).

        * ot_ode=True  -> deterministic, Z_prev = Z + delta_t * x0_hat (Eq. 8).
                          p-independent by Theorem 10.
        * ot_ode=False -> stochastic superposition, Z_prev = Z + Y with
                          Y ~ ED*(theta_hat0, delta_t) (Eq. 7). This is the one
                          place in the sampler where p appears.
        * naive=True   -> ablation: discard Z and synthesise nu_prev * x0_hat.
        """
        assert nprev < n, f"{nprev=}, {n=}"
        B = x0_hat.shape[0]
        device, dtype = x0_hat.device, x0_hat.dtype
        xdim = Z_state.shape[1:]
        nu_n = self.nu_at(torch.full((B,), n, device=device, dtype=torch.long)).to(dtype)
        nu_prev = self.nu_at(torch.full((B,), nprev, device=device, dtype=torch.long)).to(dtype)

        if naive:
            nu_prev_b = unsqueeze_xdim(nu_prev, xdim).to(device)
            return (nu_prev_b * x0_hat).clamp_min(0.0)

        dnu = (nu_prev - nu_n).clamp_min(0.0)
        dnu_b = unsqueeze_xdim(dnu, xdim).to(device).expand_as(Z_state)
        if ot_ode:
            return Z_state + dnu_b * x0_hat
        return Z_state + self.fam.sample_Z(dnu_b, x0_hat)

    # ------------------------------------------------------------------ sampling

    def ddpm_sampling(
        self,
        steps: list[int],
        pred_x0_fn: Callable[[torch.Tensor, int], torch.Tensor],
        Z_obs: torch.Tensor,
        ot_ode: bool = False,
        log_steps: list[int] | None = None,
        verbose: bool = True,
        naive_posterior: bool = False,
    ):
        """Reverse chain from `steps[-1]` (observation end) to `steps[0]`.

        Returns (Z_states, pred_x0s), each (B, K, C, H, W) with index 0 at the
        stopping step; `pred_x0s[:, 0]` is the exported denoised intensity.
        """
        assert len(steps) >= 2, "need at least start + end step"
        assert steps[0] >= 0 and steps[-1] < self.T
        assert steps == sorted(set(steps))

        Z = Z_obs.detach().to(self.device).float()
        Z_states, pred_x0s = [], []
        log_steps = log_steps or steps
        assert log_steps[0] == steps[0], (log_steps[0], steps[0])

        rev = steps[::-1]
        pairs = list(zip(rev[1:], rev[:-1]))     # (prev, n)
        it = tqdm(pairs, desc=f"p-Bridge sampling (p={self.p})",
                  total=len(pairs)) if verbose else pairs
        for prev_step, step in it:
            x0_hat = pred_x0_fn(Z, step)
            Z = self.p_posterior(prev_step, step, Z, x0_hat,
                                 ot_ode=ot_ode, naive=naive_posterior)
            if prev_step in log_steps:
                pred_x0s.append(x0_hat.detach().cpu())
                Z_states.append(Z.detach().cpu())

        def stack_bwd(z):
            return torch.flip(torch.stack(z, dim=1), dims=(1,))

        return stack_bwd(Z_states), stack_bwd(pred_x0s)
