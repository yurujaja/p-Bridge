"""Sanity checks demanded by Remark 36 — run these before any training.

"A ten-line unit test here prevents a month of debugging a mis-specified
simulator." Every check below is a statement from the notes, not a regression
test against our own output.

Run:  python -m tests.test_tweedie      (or pytest tests/test_tweedie.py)
"""

from __future__ import annotations

import math

import torch

from pbridge_core.tweedie import Tweedie, nu_obs_for_severity, cv_at
from pbridge_core.diffusion import PBridge

P_GRID = [0.0, 1.0, 1.25, 1.5, 1.75, 2.0]
N = 400_000
TOL = 0.02          # relative tolerance on Monte-Carlo moments


def _report(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    assert ok, f"{name} failed: {detail}"


def test_forward_moments():
    """Prop. 7 / Eqs. (23)-(24): E[Z] = nu*mu, Var(Z) = nu*phi*mu^p."""
    print("forward moments  E[Z]=nu*mu, Var(Z)=nu*phi*mu^p")
    torch.manual_seed(0)
    for p in P_GRID:
        for mu_val, nu_val in [(0.05, 3.0), (0.5, 10.0), (1.0, 100.0)]:
            fam = Tweedie(p=p)
            mu = torch.full((N,), mu_val, dtype=torch.float64)
            nu = torch.full((N,), nu_val, dtype=torch.float64)
            Z = fam.sample_Z(nu, mu)
            m, v = Z.mean().item(), Z.var().item()
            m_ref = nu_val * mu_val
            v_ref = nu_val * mu_val ** p
            ok = (abs(m - m_ref) / m_ref < TOL) and (abs(v - v_ref) / v_ref < 4 * TOL)
            _report(f"p={p} mu={mu_val} nu={nu_val}", ok,
                    f"mean {m:.4f} vs {m_ref:.4f}, var {v:.4f} vs {v_ref:.4f}")


def test_alpha_limits():
    """Remark 19: p->2- => alpha->0; p->1+ => alpha->inf; p=3/2 => alpha=1."""
    print("compound jump shape alpha = (2-p)/(p-1)")
    _report("p=1.5 -> alpha=1", abs(Tweedie(p=1.5).alpha - 1.0) < 1e-12)
    _report("p->2- -> alpha->0", Tweedie(p=1.999).alpha < 1e-2,
            f"alpha={Tweedie(p=1.999).alpha:.4g}")
    _report("p->1+ -> alpha->inf", Tweedie(p=1.001).alpha > 1e2,
            f"alpha={Tweedie(p=1.001).alpha:.4g}")
    # Eq. (12): p = 2 - 1/F^2.
    for p in (1.25, 1.5, 1.75):
        F2 = Tweedie(p=p).excess_noise_factor
        _report(f"p={p} recovered from F^2", abs((2.0 - 1.0 / F2) - p) < 1e-12,
                f"F^2={F2:.4f}")


def test_nonexistence_region():
    """Remark 34: no Tweedie member exists for 0 < p < 1."""
    print("non-existence of 0 < p < 1")
    for p in (0.3, 0.5, 0.9):
        try:
            Tweedie(p=p)
            _report(f"p={p} rejected", False, "constructor accepted it")
        except ValueError:
            _report(f"p={p} rejected", True)


def test_zero_atom():
    """Remark 37: Pr[Z=0] = exp(-nu*lambda) for 1<p<2, and 0 elsewhere."""
    print("zero atom  Pr[Z=0] = exp(-nu*lambda)")
    torch.manual_seed(1)
    for p in (1.25, 1.5, 1.75):
        fam = Tweedie(p=p)
        mu = torch.full((N,), 0.05, dtype=torch.float64)   # darkest allowed pixel
        nu = torch.full((N,), 2.0, dtype=torch.float64)
        Z = fam.sample_Z(nu, mu)
        emp = (Z == 0).double().mean().item()
        theo = float(fam.zero_atom_mass(2.0, torch.tensor(0.05, dtype=torch.float64)))
        _report(f"p={p}", abs(emp - theo) < 0.01, f"empirical {emp:.4f} vs {theo:.4f}")
    for p in (1.0, 2.0):
        _report(f"p={p} has no atom",
                float(Tweedie(p=p).zero_atom_mass(2.0, torch.tensor(0.5))) == 0.0)


def test_thinning_moments():
    """Obs. 17 / Prop. 18: thinning nu_hi -> nu_lo reproduces the nu_lo marginal."""
    print("thinning kernels land on the right marginal")
    torch.manual_seed(2)
    nu_hi, nu_lo, mu_val = 50.0, 7.0, 0.4
    for p in P_GRID:
        fam = Tweedie(p=p)
        mu = torch.full((N,), mu_val, dtype=torch.float64)
        Z_hi, latent = fam.sample_Z_with_latent(
            torch.full((N,), nu_hi, dtype=torch.float64), mu)
        Z_lo, _ = fam.thin(Z_hi,
                           torch.full((N,), nu_hi, dtype=torch.float64),
                           torch.full((N,), nu_lo, dtype=torch.float64),
                           latent=latent)
        m, v = Z_lo.mean().item(), Z_lo.var().item()
        m_ref, v_ref = nu_lo * mu_val, nu_lo * mu_val ** p
        ok = (abs(m - m_ref) / m_ref < TOL) and (abs(v - v_ref) / v_ref < 4 * TOL)
        _report(f"p={p}", ok, f"mean {m:.4f} vs {m_ref:.4f}, var {v:.4f} vs {v_ref:.4f}")
        if p >= 1.0:
            # Subordinator: thinning removes measurement, never adds it.
            _report(f"p={p} nested (Z_lo <= Z_hi)", bool((Z_lo <= Z_hi + 1e-9).all()))


def test_deviance_minimiser():
    """Thm 23 / Cor. 24: argmin_mu E[d_p(Y, mu)] = E[Y], for every p."""
    print("deviance is minimised exactly at the mean")
    torch.manual_seed(3)
    for p in P_GRID:
        fam = Tweedie(p=p)
        # A deliberately skewed Y so mean != median; l1 would pick the median.
        Y = torch.distributions.LogNormal(0.0, 0.6).sample((200_000,)).double() * 0.3
        target = Y.mean()
        grid = target * torch.linspace(0.6, 1.6, 401).double()
        losses = torch.stack([fam.deviance(Y, torch.full_like(Y, float(g))).mean()
                              for g in grid])
        mu_star = grid[losses.argmin()]
        rel = abs(float(mu_star) - float(target)) / float(target)
        _report(f"p={p}", rel < 0.01,
                f"argmin {float(mu_star):.4f} vs mean {float(target):.4f} "
                f"(median {float(Y.median()):.4f})")


def test_deviance_properties():
    """d_p >= 0 with equality iff y == mu; no NaN at the zero atom."""
    print("deviance positivity, zero, and y=0 guard")
    y = torch.linspace(0.01, 2.0, 500).double()
    for p in P_GRID:
        fam = Tweedie(p=p)
        d_self = fam.deviance(y, y.clone())
        _report(f"p={p} d(y,y)=0", float(d_self.abs().max()) < 1e-8,
                f"max {float(d_self.abs().max()):.2e}")
        d_off = fam.deviance(y, y * 1.5)
        _report(f"p={p} d>=0", bool((d_off >= -1e-9).all()))
        if p != 2.0:   # d_2(0, mu) = +inf by construction (Itakura-Saito)
            z = torch.zeros(10).double()
            d0 = fam.deviance(z, torch.full((10,), 0.5, dtype=torch.float64))
            _report(f"p={p} finite at y=0", bool(torch.isfinite(d0).all()),
                    f"d={float(d0[0]):.4f}")


def test_severity_calibration():
    """Eq. (22): nu_obs(p) makes CV(mu_ref) == eta for every p."""
    print("severity calibration CV(mu_ref) == eta")
    mu_ref = 0.48
    for eta in (0.25, 0.5, 1.0):
        for p in P_GRID:
            nu = nu_obs_for_severity(p, eta, mu_ref)
            got = cv_at(p, nu, mu_ref)
            _report(f"p={p} eta={eta}", abs(got - eta) < 1e-10,
                    f"nu_obs={nu:.4f} -> CV={got:.6f}")


def test_empirical_cv_matches():
    """The calibration is not just algebra: draw and measure."""
    print("empirical CV at mu_ref matches eta")
    torch.manual_seed(4)
    mu_ref, eta = 0.48, 0.5
    for p in P_GRID:
        nu = nu_obs_for_severity(p, eta, mu_ref)
        fam = Tweedie(p=p)
        mu = torch.full((N,), mu_ref, dtype=torch.float64)
        x = fam.sample_Z(torch.full((N,), nu, dtype=torch.float64), mu) / nu
        cv = float(x.std() / x.mean())
        _report(f"p={p}", abs(cv - eta) / eta < 0.03, f"CV={cv:.4f} vs eta={eta}")


def test_deterministic_reverse_is_p_independent():
    """Thm 10: alpha_t = nu(t)/nu(t-1) involves neither kappa nor V, hence not p.

    Two bridges with the same schedule shape must produce bit-identical
    deterministic steps in the intensity domain.
    """
    print("deterministic reverse is independent of p")
    torch.manual_seed(5)
    x_t = torch.rand(16, 1, 8, 8).double() + 0.1
    x0_hat = torch.rand(16, 1, 8, 8).double() + 0.1
    outs = []
    for p in P_GRID:
        br = PBridge(num_timesteps=20, p=p, nu_obs=2.0, nu_max=2.0e4, phi=1.0)
        t, tprev = 10, 7
        nu_t, nu_prev = float(br.nu[t]), float(br.nu[tprev])
        Z = nu_t * x_t
        Z_prev = br.p_posterior(tprev, t, Z, x0_hat, ot_ode=True)
        outs.append(Z_prev / nu_prev)
        # and it equals the textbook mixing form
        a = nu_t / nu_prev
        mix = a * x_t + (1.0 - a) * x0_hat
        _report(f"p={p} matches alpha*x_t+(1-alpha)*x0",
                float((outs[-1] - mix).abs().max()) < 1e-10)
    spread = max(float((o - outs[0]).abs().max()) for o in outs)
    _report("identical across p", spread < 1e-12, f"max spread {spread:.2e}")


def test_bridge_chain_marginal():
    """Thm 8 + Thm 13: the training couple's Z_t has the nu(t) marginal."""
    print("thinning-chain training couple has the right marginals")
    torch.manual_seed(6)
    mu_ref = 0.48
    for p in P_GRID:
        nu_obs = nu_obs_for_severity(p, 0.5, mu_ref)
        br = PBridge(num_timesteps=100, p=p, nu_obs=nu_obs, nu_max=1.0e4 * nu_obs)
        x0 = torch.full((2048, 1, 16, 16), mu_ref)
        step = torch.full((2048,), 60, dtype=torch.long)
        Z_t, Z_obs, nu_step = br.q_sample_thinning_chain(step, x0)
        nu_t = float(nu_step[0])
        for name, Z, nu in (("Z_t", Z_t, nu_t), ("Z_obs", Z_obs, nu_obs)):
            m, v = Z.mean().item(), Z.var().item()
            m_ref, v_ref = nu * mu_ref, nu * mu_ref ** p
            ok = (abs(m - m_ref) / m_ref < 0.03) and (abs(v - v_ref) / v_ref < 0.08)
            _report(f"p={p} {name}", ok,
                    f"mean {m:.4f} vs {m_ref:.4f}, var {v:.5f} vs {v_ref:.5f}")


def main():
    tests = [
        test_forward_moments,
        test_alpha_limits,
        test_nonexistence_region,
        test_zero_atom,
        test_thinning_moments,
        test_deviance_minimiser,
        test_deviance_properties,
        test_severity_calibration,
        test_empirical_cv_matches,
        test_deterministic_reverse_is_p_independent,
        test_bridge_chain_marginal,
    ]
    for fn in tests:
        print(f"\n=== {fn.__name__} ===")
        fn()
    print("\nall sanity checks passed.")


if __name__ == "__main__":
    main()
