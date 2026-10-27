"""Evaluate a p-Bridge checkpoint, optionally on data corrupted under a
*different* exponent — which is the whole of experiment E0.

Cross-p protocol. A checkpoint trained at p_model carries its own calibrated
nu_obs(p_model); the evaluation data is generated at p_data with nu_obs(p_data).
Both observations are handed to the network in the intensity domain
x_obs = Z_obs / nu_obs, and by Eq. (22) both have CV(mu_ref) = eta. So the two
differ only in the *shape* of the noise, never in its amount — which is exactly
the quantity E0 is asking about. Concretely we seed the model's reverse chain
with Z_start = nu_obs(p_model) * x_obs(data); the model then runs its own
schedule, its own alpha_t, its own conditioning.

Usage:
    python -m eval.run_eval --ckpt checkpoints/e0/p1.0_severe/ckpt_0015000.pt \
        --nfe 1 5 25 --eval_p 1.0 1.5 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from pbridge_core.diffusion import PBridge
from pbridge_core.network import build_default_unet
from pbridge_core.tweedie import Tweedie, nu_obs_for_severity
from data.optical_dataset import PairedEvalDataset
from eval.metrics import (psnr, ssim, masked_psnr, decile_thresholds,
                          psnr_by_decile, pearson_residual_stats, ratio_mean,
                          zero_fraction, log_rmse, masked_log_rmse,
                          log_rmse_by_decile)


def load_model(ckpt_path: Path, device, prefer_ema: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    channels = int(a.get("channels", 1))
    # The shape is read back out of the checkpoint, not assumed. Every cell
    # trained before 2026-09-17 predates --channel_mult/--attention_resolutions
    # and has neither key, so the fallbacks are exactly the values those runs
    # were built with: a wrong-but-loadable shape would not raise, it would
    # score nonsense.
    G = build_default_unet(
        image_size=a["image_size"],
        in_channels=channels * (2 if a.get("cond_x1") else 1),
        out_channels=channels,
        model_channels=a["model_channels"],
        channel_mult=tuple(a.get("channel_mult") or (1, 2, 4)),
        attention_resolutions=tuple(a.get("attention_resolutions") or ()),
        cond_p=bool(a.get("cond_p")),
    ).to(device)
    state = ckpt.get("G_ema") if (prefer_ema and ckpt.get("G_ema")) else ckpt["G"]
    G.load_state_dict(state, strict=False)
    G.eval()
    bridge = PBridge(num_timesteps=a["T"], p=a["p"], nu_obs=a["nu_obs"],
                     nu_max=a["nu_max"], phi=a.get("phi", 1.0),
                     schedule=a.get("schedule", "log"), device=device)
    return G, bridge, a


def make_pred_fn(G, bridge, args, cond=None, p_override: float | None = None,
                 x0_range: tuple[float, float] | None = None):
    """Wrap the network as x0_hat = head(G(Z/nu, t, log nu, p, cond)).

    `x0_range` is Omega, the interval the clean intensity is known to occupy
    ([eps, 1] with eps = 0.05, data/optical_dataset.py). Clamping x0_hat to it
    *inside* the chain is Assumption 2 rather than a heuristic, and the position
    matters: x0_hat is fed back through `p_posterior` to form the next state, so
    a prediction outside Omega does not merely score badly, it compounds. The
    pre-existing `out.clamp(-tau, tau)` bounds the pre-exponential output only,
    which still permits x0_hat up to exp(5) = 148x the physical ceiling.

    Measured under p-mismatch (p_model=1 on p_data=2, severe): without this the
    exp head reaches 15.4x the ceiling by the final reverse step and a minority
    of images blow up, giving a seed SD of 1.17 dB at NFE=25; with it the SD is
    0.035 and every non-diverging cell moves by <= 0.001 dB. Pass None to
    reproduce the numbers published before 2026-09-12.
    """
    tau = float(args.get("tau_logx0", 5.0))
    direct = bool(args.get("direct_pred"))
    head = str(args.get("head", "exp"))
    identity = (float(args["p"]) == 0.0)
    use_p = bool(args.get("cond_p"))
    p_val = float(args["p"] if p_override is None else p_override)

    def fn(Z_state, step_int: int):
        B = Z_state.shape[0]
        step = torch.full((B,), step_int, device=Z_state.device, dtype=torch.long)
        log_nu = bridge.log_nu[step_int].expand(B)
        x_input = Z_state / bridge.nu[step_int]
        p_vec = torch.full((B,), p_val, device=Z_state.device) if use_p else None
        out = G(x_input, step.float(), log_nu, p=p_vec, cond=cond)
        if head == "logres":
            # gamma-Bridge's multiplicative anchor, train.py predict_x0. The
            # Omega clamp below still applies: it bounds x0_hat, not the head.
            x0_hat = x_input.clamp_min(1e-4) * torch.exp(out.clamp(-tau, tau))
        elif identity:
            x0_hat = out
        elif direct:
            x0_hat = torch.relu(out)
        else:
            x0_hat = torch.exp(out.clamp(-tau, tau))
        if x0_range is not None:
            x0_hat = x0_hat.clamp(*x0_range)
        return x0_hat

    return fn


def make_step_list(T: int, nfe: int, target_step: int = 0,
                   entry_step: int | None = None, skew: float = 1.0) -> list[int]:
    """`nfe` network evaluations means nfe+1 schedule points.

    `entry_step` is where the reverse chain starts (its noisy end). It is T-1
    for matched severity and lower for cross-severity, where the data enters the
    bridge part-way along — see `entry_step_for_eta`.

    `skew` reshapes *where* those points sit between the two fixed endpoints:
    u -> u**skew on the unit interval before mapping onto [target_step, top].
    skew=1 is the original uniform-in-index placement and is bit-identical to
    the previous `np.linspace`, so it stays the default and every published
    number is reproduced by it. The schedule is log-spaced in nu, so uniform in
    index means uniform in log nu; skew > 1 clusters the evaluations towards the
    clean end, skew < 1 towards the noisy entry.

    This is a free knob at NFE=5, which is the operating point everything is
    reported at, so it must be *selected on data the reported tables never see*
    — see scripts/nfe_schedule_sweep.py. Choosing it on the evaluation set would
    be tuning on the test set for a quantity worth a few tenths of a dB.
    """
    top = (T - 1) if entry_step is None else int(entry_step)
    if skew == 1.0:
        idx = np.linspace(target_step, top, num=nfe + 1)
    else:
        u = np.linspace(0.0, 1.0, num=nfe + 1) ** float(skew)
        idx = target_step + (top - target_step) * u
    idx = idx.round().astype(int)
    steps = sorted(set(int(i) for i in idx))
    if len(steps) < 2:
        steps = [target_step, top]
    return steps


def entry_step_for_eta(bridge, p_model: float, eval_eta: float, mu_ref: float,
                       phi: float = 1.0) -> tuple[int, float]:
    """Where on the model's own schedule does the data's noise level sit?

    Cross-severity protocol. nu is the bridge's only noise coordinate: at a
    given (p, nu) the conditional law of the state is fixed, and the severity
    label only chooses where the chain *starts*. Eq. (22) gives CV(mu_ref) = eta
    at nu_obs(p, eta), so data of severity eval_eta belongs at the step whose
    nu equals nu_obs(p_model, eval_eta) — a point on the model's schedule, hence
    p_model rather than eval_p.

    Because nu_obs falls as eta rises, a severe-trained bridge spans
    [nu_obs(severe), nu_max] which *contains* the milder starting points: at
    p=1.5 severe covers [1.4667, 14666.9] while moderate starts at 5.8667 (85%
    of the way along its log-nu range) and mild at 23.4670 (70%). The reverse
    does not hold — severe's start lies below mild's range entirely — so this
    generalises from severe upwards and not back down. Entering above the
    model's clean end is refused; entering below its noisy end is clamped and
    flagged, because that is the direction expected to fail and it must be
    labelled rather than silently reported.

    Returns (entry_step, nu_target, extrapolating).
    """
    nu_target = float(nu_obs_for_severity(p_model, eval_eta, mu_ref, phi))
    if nu_target > bridge.nu_max:
        raise SystemExit(
            f"eval_eta={eval_eta} needs nu={nu_target:.4f}, above this model's clean "
            f"end nu_max={bridge.nu_max:.1f}: the data is cleaner than anything the "
            f"bridge represents. Train at that severity instead.")
    extrapolating = nu_target < float(bridge.nu_obs)
    # The schedule is log-spaced, so nearest-in-log-nu is nearest-in-step.
    k = int(torch.argmin((bridge.log_nu - float(np.log(nu_target))).abs()).item())
    return k, nu_target, extrapolating


@torch.no_grad()
def evaluate_checkpoint(ckpt_path: Path, eval_p: float, eta: float, mu_ref: float,
                        nfe: int, device, num_samples: int = 64,
                        ot_ode: bool = True, phi: float = 1.0, seed: int = 42,
                        range_clamp: bool = True,
                        p_cond: float | None = None,
                        keep_images: int = 0,
                        step_skew: float = 1.0,
                        split: str = "test",
                        naive_posterior: bool | None = None):
    """Restore `num_samples` crops corrupted at (eval_p, eta) with the model in
    `ckpt_path`. Returns the aggregate metric dict for one cell of the E0 matrix.

    `range_clamp` confines x0_hat to Omega inside the reverse chain; see
    `make_pred_fn`. It is on by default because leaving it off lets a mismatched
    model diverge, which shows up as effect size rather than as the artefact it
    is. Set False only to reproduce pre-2026-09-12 numbers.

    `eta` is the severity of the *evaluation data*. When it differs from the
    severity the checkpoint was trained at, the chain enters the bridge part-way
    along instead of at its noisy end — see `entry_step_for_eta`.

    `naive_posterior` overrides the checkpoint's own setting, which is how the
    Eq. (18) row of the ablation table is obtained without a retrain: the model
    is trained with the exact reverse coupling and only *sampled* with the naive
    one, which isolates the coupling at inference instead of confounding it with
    what the network learned. None keeps the checkpoint's training-time value.
    """
    G, bridge, a = load_model(ckpt_path, device)
    nu_obs_data = nu_obs_for_severity(eval_p, eta, mu_ref, phi)
    fam_data = Tweedie(p=eval_p, phi=phi)

    # A p-conditional checkpoint (E2) has no single exponent of its own: --p only
    # set its in-loop eval reference. Evaluating it means (a) telling the network
    # an exponent and (b) running *that* exponent's bridge, since one
    # BridgeBundle per drawn p -- with nu_obs(p) from Eq. (22) -- is the pairing
    # it trained on. Leaving either at the checkpoint's --p evaluates a
    # conditional model exactly as if it were a fixed p=1.5 specialist, which
    # would show the conditioning buying nothing for purely mechanical reasons.
    #
    # `p_cond` is the exponent the network is TOLD, which need not be the data's:
    # sweeping it against eval_p gives the conditional model its own mismatch
    # matrix, and that is the control proving the conditioning is not inert. It
    # defaults to eval_p (the oracle-p setting).
    conditional = bool(a.get("cond_p"))
    if conditional:
        p_cond = float(eval_p if p_cond is None else p_cond)
        dyn = float(a["nu_max"]) / float(a["nu_obs"])      # held fixed across p
        nu_obs_model = nu_obs_for_severity(p_cond, float(a["eta"]), mu_ref, phi)
        bridge = PBridge(num_timesteps=a["T"], p=p_cond, nu_obs=nu_obs_model,
                         nu_max=dyn * nu_obs_model, phi=a.get("phi", 1.0),
                         schedule=a.get("schedule", "log"), device=device)
    else:
        p_cond = None

    ds = PairedEvalDataset(crop_size=a["image_size"], p=eval_p, nu_obs=nu_obs_data,
                           phi=phi, num_samples=num_samples, seed=seed,
                           channels=int(a.get("channels", 1)), split=split)

    model_eta = float(a["eta"])
    cross_severity = abs(eta - model_eta) > 1e-9
    if cross_severity:
        # bridge.p, not a["p"]: for a conditional checkpoint the bridge was just
        # rebuilt at p_cond, and the entry step has to be located on *that*
        # schedule. Passing a["p"]=1.5 would aim at nu_obs(1.5, eta) while
        # reading off the p_cond bridge -- the wrong step, silently.
        entry_step, nu_entry, extrapolating = entry_step_for_eta(
            bridge, float(bridge.p), eta, mu_ref, phi)
    else:
        entry_step, nu_entry, extrapolating = bridge.T - 1, float(bridge.nu_obs), False
    steps = make_step_list(bridge.T, nfe, entry_step=entry_step, skew=step_skew)

    # The conditioning channel is NOT a point on the bridge. Along training it is
    # pinned at nu_obs while the state sweeps nu(t) upward, so the net only ever
    # sees "state at least as clean as cond". Handing it the transferred (cleaner)
    # observation puts that channel out of distribution and the model collapses --
    # measured -7.24 dB at eta 1.0->0.25 before this was fixed. lambda-Bridge's own
    # smart-start (poisson-bridge/sample.py) keeps cond at D_obs for exactly this
    # reason. So thin the observation back down to the severity the model trained
    # at, which is always possible downward and is the bridge's own forward map.
    nu_cond = float(nu_obs_for_severity(eval_p, model_eta, mu_ref, phi))
    thin_cond = cross_severity and not extrapolating
    use_cond = bool(a.get("cond_x1"))
    # Omega is read off the dataset rather than hardcoded so the two can never
    # drift apart: `_prep` maps images to exactly [eps, 1].
    x0_range = (float(ds.eps), 1.0) if range_clamp else None
    naive = (bool(a.get("naive_posterior")) if naive_posterior is None
             else bool(naive_posterior))

    x0_list, hat_list, obs_list, Zobs_list = [], [], [], []
    for i in range(len(ds)):
        s = ds[i]
        x0 = s["x0"].to(device).unsqueeze(0)
        x_obs = s["x_obs"].to(device).unsqueeze(0)
        # Re-index the observation onto the model's own nu scale at the entry
        # step; the intensity view (and hence its CV) is unchanged. For matched
        # severity this is nu_obs and entry_step is T-1, i.e. the original path.
        Z_start = x_obs * bridge.nu[entry_step]
        if thin_cond:
            lat = s["latent"]
            Z_c, _ = fam_data.thin(
                s["Z_obs"].to(device).unsqueeze(0),
                torch.tensor(nu_obs_data, device=device),
                torch.tensor(nu_cond, device=device),
                latent=(lat.to(device).unsqueeze(0) if lat.numel() else None))
            x_cond = Z_c / nu_cond
        else:
            x_cond = x_obs
        fn = make_pred_fn(G, bridge, a, cond=x_cond if use_cond else None,
                          p_override=p_cond, x0_range=x0_range)
        _, pred_x0s = bridge.ddpm_sampling(steps=steps, pred_x0_fn=fn, Z_obs=Z_start,
                                           ot_ode=ot_ode, verbose=False,
                                           naive_posterior=naive)
        x0_list.append(x0[0].cpu().numpy())
        hat_list.append(pred_x0s[:, 0][0].clamp(0.0, 1.5).cpu().numpy())
        obs_list.append(x_obs[0].cpu().numpy())
        Zobs_list.append(s["Z_obs"].numpy())

    x0_all = np.stack(x0_list)
    hat_all = np.stack(hat_list)
    obs_all = np.stack(obs_list)
    Z_all = np.stack(Zobs_list)
    edges = decile_thresholds(x0_all)
    dark = x0_all < edges[1]

    out = {
        "ckpt": str(ckpt_path),
        # For a conditional model p_model is the exponent it was *told*, not the
        # checkpoint's --p (which only fixed its in-loop eval reference).
        "p_model": float(p_cond if conditional else a["p"]),
        "p_data": float(eval_p),
        "ckpt_p_arg": float(a["p"]),
        "eta": eta, "mu_ref": mu_ref,
        "nu_obs_model": float(bridge.nu_obs), "nu_obs_data": float(nu_obs_data),
        "nfe": nfe, "ot_ode": ot_ode, "num_samples": len(ds),
        # Written unconditionally: a cell run under a non-default step placement
        # or a non-default image split must never be readable as a stock one.
        "step_skew": float(step_skew), "steps": list(steps), "split": ds.split,
        # Cross-severity bookkeeping, always written so a transferred cell can
        # never be read as a matched one.
        "eta_model": model_eta, "eta_data": float(eta),
        "cross_severity": bool(cross_severity),
        "entry_step": int(entry_step), "nu_entry": float(nu_entry),
        "steps_used": len(steps) - 1,
        "extrapolating": bool(extrapolating),
        "cond_thinned_to_nu": float(nu_cond) if thin_cond else None,
        # E2 bookkeeping: which p the network was told, and whether the bridge was
        # rebuilt to match. Recorded always, so a conditional row is never read as
        # a specialist row.
        "conditional": bool(conditional), "p_cond": p_cond,
        "x0_range": list(x0_range) if x0_range else None,
        # Written unconditionally for the same reason as step_skew: a cell
        # sampled with the naive coupling must never be readable as a stock one.
        "naive_posterior": bool(naive),
        # Optional passthrough for the qualitative figure. The arrays are built
        # here anyway; returning the first few costs nothing and means the figure
        # cannot drift from the numbers by re-running the sampler separately.
        **({"images": {"x0": x0_all[:keep_images].copy(),
                       "obs": obs_all[:keep_images].copy(),
                       "hat": hat_all[:keep_images].copy()}} if keep_images else {}),
        "psnr": psnr(x0_all, hat_all),
        "psnr_dark_decile": masked_psnr(x0_all, hat_all, dark),
        "psnr_by_decile": psnr_by_decile(x0_all, hat_all, edges),
        # Scale-free companion: the instrument Eq. (21)'s dark-region claim
        # is actually about. Lower is better.
        "log_rmse": log_rmse(x0_all, hat_all),
        "log_rmse_dark_decile": masked_log_rmse(x0_all, hat_all, dark),
        "log_rmse_by_decile": log_rmse_by_decile(x0_all, hat_all, edges),
        "psnr_noisy": psnr(x0_all, np.clip(obs_all, 0.0, 1.5)),
        "psnr_noisy_dark_decile": masked_psnr(x0_all, np.clip(obs_all, 0.0, 1.5), dark),
        "ssim": float(np.mean([ssim(a_, b_) for a_, b_ in zip(x0_list, hat_list)])),
        "ratio_mean": ratio_mean(x0_all, hat_all),
        "zero_frac_obs": zero_fraction(Z_all),
        "zero_frac_theory": float(np.mean(
            fam_data.zero_atom_mass(nu_obs_data, torch.from_numpy(x0_all)).numpy()))
        if fam_data.is_compound else 0.0,
    }
    # Computed under fam_data -- the DATA's true family and nu_obs, not the
    # model's. That is deliberate and the comment here used to say the opposite.
    # Under the true family a calibrated predictor gives res_var = 1 whatever p it
    # was trained at (Prop. 26), so deviation from 1 is a goodness-of-fit test
    # that does not inherit the model's own possibly-wrong assumptions. This is
    # what makes res_var usable as an *independent* control on conclusions drawn
    # from log-RMSE, which it was used for on 2026-09-13 (EXPERIMENTS.md §2⑤).
    # Switching this to the model's family would silently turn it into a
    # self-consistency check and break that use.
    out.update(pearson_residual_stats(fam_data, obs_all, hat_all, nu_obs_data))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--eval_p", type=float, nargs="+", default=None,
                    help="corruption exponents to test on (default: the model's own)")
    ap.add_argument("--eta", type=float, default=None,
                    help="severity OF THE DATA (default: the model's own). Setting "
                         "it to something else is the cross-severity test: the chain "
                         "enters the bridge at the step whose nu matches, rather than "
                         "at the noisy end. Works upwards (severe model -> milder "
                         "data) only; the reverse is flagged as extrapolating.")
    ap.add_argument("--nfe", type=int, nargs="+", default=[1, 5, 25])
    ap.add_argument("--num_samples", type=int, default=64)
    ap.add_argument("--stochastic", action="store_true",
                    help="stochastic reverse (Eq. 7) instead of the deterministic default")
    ap.add_argument("--no_range_clamp", action="store_true",
                    help="do not confine x0_hat to Omega inside the chain; reproduces "
                         "the pre-2026-09-12 numbers, including the divergence at "
                         "p_model=1 on p_data=2")
    ap.add_argument("--naive_posterior", action="store_true",
                    help="ablation of Eq. (18): discard the state and synthesise "
                         "nu_prev*x0_hat at each reverse step. Overrides the "
                         "checkpoint's training-time setting, so the model is "
                         "trained with the exact coupling and only sampled "
                         "without it — the coupling isolated at inference.")
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)["args"]
    eval_p = args.eval_p or [float(ck["p"])]
    eta = args.eta if args.eta is not None else float(ck["eta"])
    mu_ref = float(ck["mu_ref"])

    rows = []
    for pd in eval_p:
        for nfe in args.nfe:
            r = evaluate_checkpoint(Path(args.ckpt), pd, eta, mu_ref, nfe, device,
                                    num_samples=args.num_samples,
                                    ot_ode=not args.stochastic,
                                    range_clamp=not args.no_range_clamp,
                                    naive_posterior=args.naive_posterior or None)
            rows.append(r)
            tag = ""
            if r["cross_severity"]:
                tag = (f"  [transfer eta {r['eta_model']}->{r['eta_data']}, "
                       f"enter t={r['entry_step']}/{ck['T'] - 1}, "
                       f"{r['steps_used']} steps"
                       + (", EXTRAPOLATING" if r["extrapolating"] else "") + "]")
            print(f"p_model={r['p_model']:<5} p_data={r['p_data']:<5} eta={eta:<5} "
                  f"NFE={nfe:<3} PSNR={r['psnr']:6.3f}  dark={r['psnr_dark_decile']:6.3f}  "
                  f"SSIM={r['ssim']:.4f}  res_var={r['res_var']:.3f}  "
                  f"ratio={r['ratio_mean']:.4f}{tag}", flush=True)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rows, indent=2))
        print(f"[save] {args.out_json}")


if __name__ == "__main__":
    main()
