"""Compute and cache the severity calibration of §13.1.

    mu_ref      = dataset mean intensity (held constant across ALL runs)
    nu_obs(p)   = eta^-2 * mu_ref^(p-2)                         Eq. (22)
    nu_max(p)   = R * nu_obs(p),  R the dynamic range, fixed across p so that
                  the clean-end CV is comparable too.

Writes calibration.json at the repo root and prints the table that every
experiment caption must quote (eta, mu_ref, nu_obs(p)).

Usage:
    python -m scripts.calibrate --extra_roots data/div2k/DIV2K_train_HR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from data.optical_dataset import dataset_mean_intensity          # noqa: E402
from pbridge_core.tweedie import Tweedie, nu_obs_for_severity, cv_at   # noqa: E402

# Severity ladder. eta = CV(mu_ref); at p=1, nu_obs*mu_ref = eta^-2 is the mean
# event count per pixel at mu_ref, so eta=1 is exactly the "<~1 event per pixel"
# sparse-photon regime that §13.3 calls `severe`.
SEVERITIES = {"mild": 0.25, "moderate": 0.5, "severe": 1.0}
P_GRID = [1.0, 1.5, 2.0]
DYNAMIC_RANGE = 1.0e4        # nu_max / nu_obs; matches lambda-Bridge D_max/D_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra_roots", type=str, nargs="*",
                    default=["data/div2k/DIV2K_train_HR"])
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--dynamic_range", type=float, default=DYNAMIC_RANGE)
    ap.add_argument("--out", type=str, default=str(_ROOT / "calibration.json"))
    args = ap.parse_args()

    print("[calibrate] scanning pool for mu_ref (full images, not crops) ...", flush=True)
    mu_ref = dataset_mean_intensity(extra_roots=args.extra_roots,
                                    channels=args.channels)
    print(f"[calibrate] mu_ref = {mu_ref:.6f}", flush=True)

    table = {}
    print(f"\n{'p':>6} {'severity':>10} {'eta':>6} {'nu_obs':>10} {'nu_max':>12} "
          f"{'CV(mu_ref)':>11} {'events/px':>10} {'P[Z=0]@mu_ref':>14}")
    for p in P_GRID:
        fam = Tweedie(p=p)
        for name, eta in SEVERITIES.items():
            nu_obs = nu_obs_for_severity(p, eta, mu_ref)
            nu_max = args.dynamic_range * nu_obs
            # Mean latent jump count per pixel at mu_ref: the physical severity
            # reading. p=1 -> photons; 1<p<2 -> compound-Poisson events.
            if fam.is_compound:
                lam = mu_ref ** (2 - p) / (2 - p)
                events = nu_obs * lam
                p0 = float(__import__("math").exp(-events))
            elif fam.is_poisson:
                events = nu_obs * mu_ref
                p0 = float(__import__("math").exp(-events))
            else:
                events = float("inf")     # Gamma: infinite jump activity
                p0 = 0.0
            table[f"p{p}_{name}"] = {
                "p": p, "severity": name, "eta": eta,
                "nu_obs": nu_obs, "nu_max": nu_max,
                "cv_mu_ref": cv_at(p, nu_obs, mu_ref),
                "events_per_px_at_mu_ref": events,
                "zero_atom_at_mu_ref": p0,
            }
            print(f"{p:>6} {name:>10} {eta:>6} {nu_obs:>10.4f} {nu_max:>12.2f} "
                  f"{cv_at(p, nu_obs, mu_ref):>11.4f} {events:>10.3f} {p0:>14.4f}")

    out = {
        "mu_ref": mu_ref,
        "phi": 1.0,
        "eps_intensity_floor": 0.05,
        "dynamic_range": args.dynamic_range,
        "severities": SEVERITIES,
        "p_grid": P_GRID,
        "channels": args.channels,
        "extra_roots": args.extra_roots,
        "table": table,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[calibrate] wrote {args.out}")
    print("[calibrate] quote mu_ref, eta and nu_obs(p) in every table caption (§13.9).")


if __name__ == "__main__":
    main()
