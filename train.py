from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from pbridge_core.diffusion import PBridge
from pbridge_core.network import build_default_unet, count_params
from pbridge_core.tweedie import nu_obs_for_severity
from data.optical_dataset import OpticalTrainDataset, PairedEvalDataset


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="checkpoints/pbridge")
    # ---- the p axis and the severity calibration ----
    p.add_argument("--p", type=float, required=True,
                   help="Tweedie power-variance exponent: 0, 1, (1,2) or 2")
    p.add_argument("--eta", type=float, default=None,
                   help="target severity CV(mu_ref); sets nu_obs via Eq. (22)")
    p.add_argument("--nu_obs", type=float, default=None,
                   help="override the calibrated nu_obs (discouraged: breaks "
                        "cross-p comparability)")
    p.add_argument("--mu_ref", type=float, default=None,
                   help="reference intensity; default reads calibration.json")
    p.add_argument("--dynamic_range", type=float, default=None,
                   help="nu_max / nu_obs; default reads calibration.json (1e4)")
    p.add_argument("--phi", type=float, default=1.0)
    p.add_argument("--calibration", type=str, default="calibration.json")
    # ---- bridge / data ----
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--schedule", type=str, default="log", choices=["log", "linear"])
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--channels", type=int, default=1, choices=[1, 3])
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--dataset_length", type=int, default=16000)
    p.add_argument("--extra_roots", type=str, nargs="*", default=None,
                   help="image roots appended to the BSDS500 train+val pool; "
                        "the paper's runs add data/div2k/DIV2K_train_HR")
    p.add_argument("--num_workers", type=int, default=4)
    # ---- model ----
    p.add_argument("--model_channels", type=int, default=64)
    p.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 4],
                   help="channels per resolution level, as multiples of "
                        "--model_channels; length sets the depth")
    p.add_argument("--attention_resolutions", type=int, nargs="*", default=[],
                   help="downsample rates (1,2,4,8,...) at which to insert "
                        "attention; NOT pixel resolutions")
    p.add_argument("--cond_x1", action="store_true",
                   help="concatenate x_obs as a second input channel (recommended)")
    p.add_argument("--cond_p", action="store_true",
                   help="feed p to the conditioning vector (E2 generalist; a "
                        "specialist run does not need it)")
    # ---- E2: mixed-p training. Either flag turns a specialist run into a
    # generalist run; without one of them --cond_p only tells the net a constant.
    p.add_argument("--p_range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="E2: draw p ~ U[LO, HI] per micro-batch instead of using "
                        "--p. Quantised by --p_quantum so the per-p bridges can be "
                        "cached. Implies --cond_p.")
    p.add_argument("--p_choices", type=float, nargs="+", default=None,
                   help="E2: draw p uniformly from this explicit set instead of "
                        "--p_range. Implies --cond_p.")
    p.add_argument("--p_quantum", type=float, default=0.01,
                   help="quantisation of --p_range; one cached bridge per level")
    p.add_argument("--no_p_balance", action="store_true",
                   help="E2: do NOT divide the deviance by mu_ref^(2-p). Only for "
                        "an ablation — without it p=2 gets 2.03x the gradient "
                        "weight of p=1 and the generalist tilts towards p=2.")
    p.add_argument("--head", choices=["exp", "residual", "logres"], default="exp",
                   help="'residual' predicts the Eq. (16) Pearson residual and "
                        "inverts it for x0_hat; 'logres' is gamma-Bridge's "
                        "multiplicative anchor x_t * exp(G), sound only at p=2 "
                        "(Remark 33); see predict_x0")
    p.add_argument("--nu_weight_alpha", type=float, default=1.0,
                   help="exponent on the deviance weight (nu/nu_obs)**alpha, "
                        "renormalised to preserve the mean. 1 = unchanged, 0 = flat")
    p.add_argument("--nu_weight_cap", type=float, default=None,
                   help="clip the deviance weight nu(t)/nu_obs at this value. "
                        "Uncapped it spans [1, 1e4], so the clean end of the "
                        "schedule carries four orders of magnitude more gradient "
                        "than the noisy end. 1.0 makes the weighting flat.")
    p.add_argument("--t_sample", choices=["uniform", "sampler"], default="uniform",
                   help="'sampler' draws t only from the steps an NFE-step "
                        "reverse chain actually visits. Uniform over T=100 spends "
                        "~95%% of the gradient on states inference never sees.")
    p.add_argument("--t_sample_nfe", type=int, default=3,
                   help="the NFE whose step set --t_sample sampler draws from")
    p.add_argument("--tau_logx0", type=float, default=5.0,
                   help="clip on the log-intensity head (§9: tau = 5)")
    p.add_argument("--direct_pred", action="store_true",
                   help="ablation: ReLU head instead of exp(clip(G))")
    # ---- loss ----
    p.add_argument("--lambda_rec", type=float, default=10.0)
    p.add_argument("--lambda_mom", type=float, default=1.0)
    p.add_argument("--lambda_cons", type=float, default=5.0)
    p.add_argument("--nu_weight_raw", action="store_true",
                   help="weight the deviance by raw nu(t) rather than nu(t)/nu_obs; "
                        "reproduces Eq. (14) literally but is not comparable across runs")
    p.add_argument("--naive_posterior", action="store_true",
                   help="ablation: synthesise nu' * x0_hat in the consistency loss")
    # ---- optimisation ----
    p.add_argument("--lr_G", type=float, default=1e-4)
    p.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant",
                   help="cosine decays lr_G -> lr_min over total_steps after warmup")
    p.add_argument("--lr_min", type=float, default=1e-7,
                   help="floor of the cosine decay; ignored when constant")
    p.add_argument("--lr_warmup", type=int, default=0,
                   help="linear warmup steps from 0 to lr_G; 0 disables")
    p.add_argument("--betas", type=float, nargs=2, default=(0.5, 0.999))
    p.add_argument("--total_steps", type=int, default=15_000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--min_step", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--find_unused", action="store_true")
    # ---- logging ----
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def resolve_calibration(args):
    """Fill mu_ref / nu_obs / nu_max from calibration.json + Eq. (22).

    Mutates `args` so the resolved numbers land in the checkpoint, which is what
    the eval scripts read back — no run can end up without its calibration.
    """
    cal_path = _ROOT / args.calibration
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    if args.mu_ref is None:
        args.mu_ref = cal.get("mu_ref")
        if args.mu_ref is None:
            raise SystemExit(
                f"no mu_ref: run `python -m scripts.calibrate` first, or pass --mu_ref"
            )
    if args.dynamic_range is None:
        args.dynamic_range = cal.get("dynamic_range", 1.0e4)
    if args.nu_obs is None:
        if args.eta is None:
            raise SystemExit("pass --eta (recommended) or --nu_obs")
        args.nu_obs = nu_obs_for_severity(args.p, args.eta, args.mu_ref, args.phi)
    elif args.eta is None:
        args.eta = math.sqrt(args.phi) * args.mu_ref ** (args.p / 2 - 1) * args.nu_obs ** -0.5
    args.nu_max = args.dynamic_range * args.nu_obs
    return args


class EMAShadow:
    def __init__(self, module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.enabled = decay > 0.0
        if self.enabled:
            self.shadow = {n: p.detach().clone() for n, p in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module: torch.nn.Module):
        if not self.enabled:
            return
        for n, p in module.state_dict().items():
            if p.dtype.is_floating_point and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[n] = p.detach().clone()

    def state_dict(self):
        return dict(self.shadow) if self.enabled else {}


def ddp_setup():
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank, True
    return 0, 1, 0, False


def is_main(rank: int) -> bool:
    return rank == 0


# ------------------------------------------------------------------ heads

def predict_x0(G, x_input, step, log_nu, p_vec, cond, tau, direct_pred, identity_head,
               head: str = "exp", fam=None, nu_step=None, eps: float = 1e-4):
    """x0_hat from the raw network output.

    identity_head is required at p=0 where Omega = R; for p >= 1 the log head
    enforces x0_hat > 0 with no clipping heuristic (Prop. 30).

    `head="residual"` is the Tweedie analogue of epsilon-prediction. Instead of
    regressing x0 through an exponential, the network predicts the standardised
    residual of Eq. (16),

        r = (x_t/nu_t - x0) / sqrt(V(x0)/nu_t),

    which is pixelwise zero-mean and unit-variance under the data's own family,
    and x0_hat is recovered by inverting it. Two reasons this is worth trying.
    Direct x0 regression is badly conditioned at the noisy end of the schedule --
    the target does not shrink as the state gets noisier, which is why diffusion
    models predict a normalised residual almost universally -- and the
    exponential head is a relative-error parameterisation being scored by PSNR,
    an absolute-error metric dominated by bright pixels.

    The scale uses V(x_input) rather than V(x0_hat), which would be implicit.
    That is a bias, not an approximation of convenience: at high nu the two
    agree, and at low nu the state is the only thing available at inference.

    Positivity is enforced by a floor rather than by the head, so Omega is no
    longer guaranteed by construction. The Omega clamp already in
    eval/run_eval.py was measured to fire almost never and to cost 0.000 dB at
    the operating point, so the floor is a formality -- but it IS a weakening of
    Prop. 30 and must not be quietly dropped from the write-up.

    `head="logres"` is gamma-Bridge's head transplanted: x0_hat = x_t * exp(G),
    multiplicative, log-symmetric, positive by construction. It is NOT the
    "residual" head above, which is linear in G and kept positive by a floor.
    The anchor is x_input = Z_t/nu_t, whose mean is x0, so G regresses a pure
    log-correction and the output level is carried by the state instead of being
    learned.

    Remark 33 of unified-bridge_v2 licenses it at p=2 only: CV(x_t) =
    x0^(p/2-1)/sqrt(nu) is signal-independent iff p = 2, so below 2 the anchor
    itself is unreliable in dark regions. The absolute head was adopted for
    uniformity along p and the trade was never measured AT p=2, where both heads
    are sound. That is what this option exists to measure, after the whole-scene
    SAR round found the absolute head does not merely drift in level but
    collapses (r_bar = 0.02, a near-constant output) where gamma-Bridge holds at
    0.67-0.88 on the same pixels. train_setup refuses it away from p=2 rather
    than trusting the caller.
    """
    out = G(x_input, step.float(), log_nu, p=p_vec, cond=cond)
    if head == "logres":
        return x_input.clamp_min(eps) * torch.exp(out.clamp(-tau, tau))
    if head == "residual":
        assert fam is not None and nu_step is not None
        scale = torch.sqrt(fam.V(x_input.clamp_min(eps)) / nu_step.view(-1, 1, 1, 1))
        return (x_input - out * scale).clamp_min(eps)
    if identity_head:
        return out
    if direct_pred:
        return torch.relu(out)
    return torch.exp(out.clamp(-tau, tau))


# ------------------------------------------------------------------ losses

def pow_norm(bridge, nu_scale: float, alpha: float, min_step: int) -> float:
    """Constant keeping mean_t[w] fixed when the weight exponent changes.

    Computed over exactly the t the trainer samples, so alpha=1 returns 1.0 and
    the run is bit-identical to one that never knew about this flag.
    """
    if alpha == 1.0:
        return 1.0
    t = torch.arange(min_step, bridge.T, device=bridge.nu.device)
    w = (bridge.nu_at(t).double() / nu_scale)
    return float(w.mean() / (w ** alpha).mean())


def bregman_loss(fam, x0, x0_hat, nu_step, nu_scale: float = 1.0,
                 weight_cap: float | None = None,
                 alpha: float = 1.0, pow_c: float = 1.0) -> torch.Tensor:
    """(nu(t)/nu_scale) * B_psi(x0, x0_hat), Eq. (14) up to a per-run constant.

    Minimised in x0_hat exactly at E[x0 | x_t, x_obs] (Thm 23, Cor. 24) — the
    same estimand the deterministic reverse tracks, for every p.

    `nu_scale` defaults to nu_obs. Eq. (14) prescribes the *relative* weighting
    across t, and dividing by a constant leaves that untouched; what it fixes is
    a cross-run confound. nu carries p-dependent units, so nu_max differs by 2x
    between p=1 and p=2 at the same severity (21512 vs 10000) and by 16x between
    mild and severe. Left unnormalised, the rec:mom ratio — and, under
    grad_clip, the effective step size — would differ between cells of the E0
    matrix for reasons that have nothing to do with the exponent. At the
    lambda-Bridge flagship config (nu_obs = D_obs = 1) this is the identity, so
    the p=1 instance stays numerically compatible with the published run.
    """
    nu_b = (nu_step / nu_scale).view(-1, 1, 1, 1).to(x0.dtype)
    if alpha != 1.0:
        nu_b = pow_c * nu_b.clamp_min(1e-12) ** alpha
    if weight_cap is not None:
        nu_b = nu_b.clamp(max=float(weight_cap))
    return (nu_b * 0.5 * fam.deviance(x0, x0_hat)).mean()


def moment_loss(fam, x_obs, x0_hat, nu_obs) -> torch.Tensor:
    """Pearson residual moment matching, Eqs. (16)-(17). At p=2 this is the
    gamma-Bridge ratio-variance term and at p=1 the Fano-1 term (Remark 27)."""
    eps = fam.pearson_residual(x_obs, x0_hat, nu_obs)
    return eps.mean().pow(2) + (eps.var(unbiased=False) - 1.0).pow(2)


@torch.no_grad()
def eval_endpoint(model, bridge, eval_ds, device, args, chunk=4):
    """1-step endpoint eval: predict x0 straight from the observation.

    Cheap in-loop proxy only; the reported numbers come from eval/run_eval.py
    at NFE in {1, 5, 25} (§13.9).
    """
    model.eval()
    x0s = torch.stack([eval_ds[i]["x0"] for i in range(len(eval_ds))]).to(device)
    x_obss = torch.stack([eval_ds[i]["x_obs"] for i in range(len(eval_ds))]).to(device)
    identity_head = (args.p == 0.0)
    parts = []
    for lo in range(0, x_obss.shape[0], chunk):
        xo = x_obss[lo:lo + chunk]
        B = xo.shape[0]
        step = torch.full((B,), bridge.T - 1, device=device, dtype=torch.long)
        log_nu = torch.log(bridge.nu_at(step))
        p_vec = torch.full((B,), args.p, device=device) if args.cond_p else None
        parts.append(predict_x0(model, xo, step, log_nu, p_vec,
                                xo if args.cond_x1 else None,
                                args.tau_logx0, args.direct_pred, identity_head,
                                head=args.head, fam=bridge.fam,
                                nu_step=bridge.nu_at(step)))
    x0_hat = torch.cat(parts, dim=0)
    mse = ((x0s - x0_hat.clamp(0.0, 1.5)) ** 2).mean().item()
    eps = bridge.fam.pearson_residual(x_obss, x0_hat, bridge.nu_obs)
    model.train()
    return {
        "eval_psnr": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
        "eval_res_mean": float(eps.mean()),
        "eval_res_var": float(eps.var(unbiased=False)),
        "eval_ratio_mean": float((x0_hat.clamp_min(1e-6) / x0s).mean()),
    }


class BridgeBundle:
    """Everything in the training step that depends on p, resolved together.

    p does not enter alone: nu_obs is set from p by Eq. (22) (the calibration
    convention — nu is *never* held fixed across p), nu_max follows from the
    dynamic range, and nu_scale is the deviance weight nu(t)/nu_obs. Mixing p
    within a run means mixing all four, so they are built and cached as a unit.
    Getting this wrong is silent: the run trains, and only the cross-p table
    comes out meaningless.

    Second normalisation, needed only here. nu_scale=nu_obs already makes the
    deviance identical across *severities* at fixed t (verified: 60.5424 at t=0
    for eta in {0.25, 0.5, 1.0}). It does not equalise across *p*: on identical
    predictions the deviance runs 60.54 at p=1 to 122.75 at p=2, a flat 2.03x at
    every t. A specialist never notices — one run, one p. A mixed-p run would
    hand p=2 twice the gradient weight of p=1 and then look as though the
    generalist simply prefers p=2. Small-delta expansion gives the culprit:
    B_psi ~ x0^(2-p) delta^2 / 2phi, i.e. an intensity moment, so dividing by
    mu_ref^(2-p) flattens it to 1.085x (E[x0^(2-p)] gives 1.063x but is batch
    noise; mu_ref is pinned constant by the calibration convention).
    """

    __slots__ = ("p", "bridge", "fam", "nu_obs", "nu_scale", "nu_pow_norm")

    def __init__(self, p: float, args, device):
        self.p = float(p)
        self.nu_obs = nu_obs_for_severity(self.p, args.eta, args.mu_ref, args.phi)
        dyn = float(args.dynamic_range)
        self.bridge = PBridge(num_timesteps=args.T, p=self.p, nu_obs=self.nu_obs,
                              nu_max=dyn * self.nu_obs,
                              phi=args.phi, schedule=args.schedule, device=device)
        self.fam = self.bridge.fam
        self.nu_scale = 1.0 if args.nu_weight_raw else float(self.nu_obs)
        if not args.no_p_balance:
            self.nu_scale *= float(args.mu_ref) ** (2.0 - self.p)
        self.nu_pow_norm = pow_norm(self.bridge, self.nu_scale,
                                    args.nu_weight_alpha, args.min_step)


def make_p_sampler(args, device):
    """-> (draw() -> BridgeBundle, describe) or None for a specialist run."""
    if args.p_range is None and args.p_choices is None:
        return None, ""

    if args.p_choices is not None:
        levels = sorted({round(float(v), 6) for v in args.p_choices})
        desc = f"p in {levels} (discrete)"
    else:
        lo, hi = sorted(args.p_range)
        q = float(args.p_quantum)
        n = int(round((hi - lo) / q))
        levels = [round(lo + k * q, 6) for k in range(n + 1)]
        desc = f"p ~ U[{lo}, {hi}] quantised at {q} ({len(levels)} levels)"

    # p=0 cannot be mixed in: identity_head is an architectural choice made once
    # at construction (Omega = R there, log head elsewhere), not a per-batch one.
    bad = [v for v in levels if v != 0.0 and not (1.0 <= v <= 2.0)] + \
          [v for v in levels if v == 0.0]
    if bad:
        shown = sorted(set(bad))
        tail = f" ... ({len(shown)} values)" if len(shown) > 6 else ""
        raise SystemExit(f"--p_range/--p_choices must lie in [1, 2]; got {shown[:6]}{tail}"
                         + (" -- p=0 needs its own run: identity_head is fixed at "
                            "build time" if 0.0 in bad else ""))

    # Index the python list directly: a float32 round-trip through a tensor
    # perturbs the level (1.95 -> 1.9500000476837158) and would miss the cache.
    bundles = [BridgeBundle(v, args, device) for v in levels]

    def draw() -> BridgeBundle:
        return bundles[int(torch.randint(len(bundles), (1,)).item())]

    return draw, desc


def main():
    args = resolve_calibration(get_args())
    if (args.p_range is not None or args.p_choices is not None) and not args.cond_p:
        # Otherwise the net sees a constant p while the data's p moves under it,
        # which trains a blurred average and looks like a working generalist.
        args.cond_p = True
    if args.head == "logres":
        # Remark 33: the multiplicative anchor is sound only where CV(x_t) is
        # signal-independent, i.e. exactly at p=2. A mixed-p run would use it at
        # levels the notes forbid, so refuse both that and a scalar p != 2.
        levels = ([args.p] if args.p_range is None and args.p_choices is None
                  else (list(args.p_choices) if args.p_choices is not None
                        else list(args.p_range)))
        if any(float(v) != 2.0 for v in levels):
            raise SystemExit("--head logres requires p = 2 exactly (Remark 33: the "
                             f"x_t anchor is unreliable for p < 2); got {levels}")
    rank, world_size, local_rank, is_ddp = ddp_setup()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    writer = None
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir = _ROOT / "logs" / out_dir.name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        print(f"[env] rank={rank}/{world_size} device={device} ddp={is_ddp}", flush=True)
        print(f"[calib] p={args.p}  eta={args.eta:.4f}  mu_ref={args.mu_ref:.6f}  "
              f"nu_obs={args.nu_obs:.6f}  nu_max={args.nu_max:.2f}  phi={args.phi}",
              flush=True)

    # ------------------ bridge ------------------
    bridge = PBridge(num_timesteps=args.T, p=args.p, nu_obs=args.nu_obs,
                     nu_max=args.nu_max, phi=args.phi, schedule=args.schedule,
                     device=device)
    fam = bridge.fam
    identity_head = (args.p == 0.0)
    if is_main(rank):
        print(f"[bridge] CV(mu_ref) at t=T-1: {bridge.cv_at(args.T-1, args.mu_ref):.4f} "
              f"| at t=0: {bridge.cv_at(0, args.mu_ref):.6f}", flush=True)
        if fam.is_compound:
            print(f"[bridge] compound: alpha={fam.alpha:.4f}  F^2={fam.excess_noise_factor:.4f}  "
                  f"P[Z_obs=0 | mu_ref]={float(fam.zero_atom_mass(args.nu_obs, torch.tensor(args.mu_ref))):.4f}",
                  flush=True)

    # ------------------ data ------------------
    train_ds = OpticalTrainDataset(crop_size=args.image_size, length=args.dataset_length,
                                   extra_roots=args.extra_roots, channels=args.channels)
    if is_ddp:
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    else:
        sampler = None
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    eval_ds = PairedEvalDataset(crop_size=args.image_size, p=args.p, nu_obs=args.nu_obs,
                                phi=args.phi, num_samples=32, channels=args.channels)
    if is_main(rank):
        print(f"[data] pool={len(train_ds.pool)} imgs  eval={len(eval_ds)}", flush=True)

    # ------------------ network ------------------
    in_ch = args.channels * (2 if args.cond_x1 else 1)
    G_core = build_default_unet(image_size=args.image_size, in_channels=in_ch,
                                out_channels=args.channels,
                                model_channels=args.model_channels,
                                channel_mult=tuple(args.channel_mult),
                                attention_resolutions=tuple(args.attention_resolutions),
                                cond_p=args.cond_p).to(device)
    if is_main(rank):
        print(f"[net] G params = {count_params(G_core)/1e6:.2f}M", flush=True)

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        G_core.load_state_dict(ck.get("G_ema") or ck["G"], strict=False)
        if is_main(rank):
            print(f"[resume] loaded {args.resume}", flush=True)
        if is_ddp:
            for prm in G_core.parameters():
                dist.broadcast(prm.data, src=0)

    G = DDP(G_core, device_ids=[local_rank], find_unused_parameters=args.find_unused,
            broadcast_buffers=False) if is_ddp else G_core
    opt_G = torch.optim.Adam(G_core.parameters(), lr=args.lr_G, betas=tuple(args.betas))

    # `None` means "never touch the param groups", so a constant-lr run takes
    # exactly the code path every earlier cell took. global_step is 0-based here:
    # it is incremented after opt_G.step(), so the first update sees 0.
    if args.lr_schedule == "constant" and args.lr_warmup <= 0:
        lr_at = None
    else:
        def lr_at(gs: int) -> float:
            if args.lr_warmup > 0 and gs < args.lr_warmup:
                return args.lr_G * (gs + 1) / args.lr_warmup
            if args.lr_schedule != "cosine":
                return args.lr_G
            span = max(1, args.total_steps - args.lr_warmup)
            prog = min(1.0, (gs - args.lr_warmup) / span)
            return args.lr_min + 0.5 * (args.lr_G - args.lr_min) * (1.0 + math.cos(math.pi * prog))
    if is_main(rank):
        print(f"[opt] lr={args.lr_G:g} schedule={args.lr_schedule} "
              f"warmup={args.lr_warmup} lr_min={args.lr_min:g}", flush=True)
    ema = EMAShadow(G_core, decay=args.ema_decay)
    G_eval = build_default_unet(image_size=args.image_size, in_channels=in_ch,
                                out_channels=args.channels,
                                model_channels=args.model_channels,
                                channel_mult=tuple(args.channel_mult),
                                attention_resolutions=tuple(args.attention_resolutions),
                                cond_p=args.cond_p).to(device) if ema.enabled else None

    # ------------------ train ------------------
    accum = max(1, int(args.grad_accum))
    tau = float(args.tau_logx0)
    # Dimensionless deviance weight nu(t)/nu_obs in [1, R], identical across
    # every cell of the E0 matrix (see bregman_loss).
    nu_scale = 1.0 if args.nu_weight_raw else float(args.nu_obs)
    # Specialist-run counterpart of BridgeBundle.nu_pow_norm; unused on the
    # mixed-p path, where each bundle carries its own.
    spec_pow_c = pow_norm(bridge, nu_scale, args.nu_weight_alpha, args.min_step)
    # E2: when mixing p, every micro-batch swaps in its own bundle and the
    # specialist-run values above go unused. Drawing per micro-batch rather than
    # per sample is deliberate — one bridge per forward, no intra-batch grouping,
    # and with --grad_accum k each optimiser step still averages k values of p.
    draw_bundle, p_desc = make_p_sampler(args, device)
    if draw_bundle is not None and is_main(rank):
        print(f"[E2] mixed-p training: {p_desc}; {accum} draw(s) per rank x "
              f"{world_size} rank(s) = {accum * world_size} values of p per "
              f"optimiser step", flush=True)
        print(f"[E2] deviance p-balance: "
              + ("OFF (ablation — p=2 gets 2.03x the weight of p=1)"
                 if args.no_p_balance else "mu_ref^(2-p), flattens 2.03x -> 1.09x"),
              flush=True)
        print(f"[E2] in-loop eval is at the single reference p={args.p} "
              f"(eval_ds and its bridge are built there); the cross-p table comes "
              f"from eval/run_eval.py", flush=True)
        if accum == 1:
            print("[E2] warning: --grad_accum 1 means one p per step; "
                  "k>=4 gives a much less noisy gradient across the axis", flush=True)
    global_step, epoch, t0 = 0, 0, time.time()
    G.train()

    def _new_iter():
        nonlocal epoch
        if is_ddp and sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        return iter(train_dl)

    dl_iter = _new_iter()

    def _next_batch():
        nonlocal dl_iter
        try:
            return next(dl_iter)
        except StopIteration:
            dl_iter = _new_iter()
            return next(dl_iter)

    while global_step < args.total_steps:
        opt_G.zero_grad(set_to_none=True)
        acc = {"G": 0.0, "rec": 0.0, "mom": 0.0, "cons": 0.0}

        for micro in range(accum):
            batch = _next_batch()
            x0 = batch["x0"].to(device, non_blocking=True)
            B = x0.size(0)

            if draw_bundle is None:
                b_bridge, b_fam, b_nu_obs, b_nu_scale, b_p = \
                    bridge, fam, args.nu_obs, nu_scale, args.p
                b_pow_c = spec_pow_c
            else:
                bundle = draw_bundle()
                b_bridge, b_fam, b_nu_obs, b_nu_scale, b_p = (
                    bundle.bridge, bundle.fam, bundle.nu_obs,
                    bundle.nu_scale, bundle.p)
                b_pow_c = bundle.nu_pow_norm

            if args.t_sample == "sampler":
                # The steps a reverse chain of --t_sample_nfe evaluations lands
                # on, endpoints included. Uniform over them, so the gradient
                # goes where inference goes.
                cand = torch.tensor(
                    sorted(set(np.linspace(args.min_step, b_bridge.T - 1,
                                           args.t_sample_nfe + 1).round().astype(int))),
                    device=device, dtype=torch.long)
                step = cand[torch.randint(len(cand), (B,), device=device)]
            else:
                step = torch.randint(args.min_step, b_bridge.T, (B,), device=device)

            # Training couple on a single downward thinning chain (Thm 13), so
            # the observation the net sees at t is the terminal state of the very
            # chain whose intermediate state is x_t.
            Z_t, Z_obs, nu_step = b_bridge.q_sample_thinning_chain(step, x0)
            log_nu = torch.log(nu_step.to(x0.dtype))
            x_input = Z_t / nu_step.view(-1, 1, 1, 1).to(x0.dtype)
            nu_obs_b = (b_nu_obs.view(-1, 1, 1, 1) if torch.is_tensor(b_nu_obs)
                        else b_nu_obs)
            x_obs = Z_obs / nu_obs_b
            cond = x_obs if args.cond_x1 else None
            p_vec = torch.full((B,), b_p, device=device) if args.cond_p else None

            is_last = (micro == accum - 1)
            ctx = G.no_sync() if (is_ddp and not is_last) else contextlib.nullcontext()
            with ctx:
                x0_hat = predict_x0(G, x_input, step, log_nu, p_vec, cond, tau,
                                    args.direct_pred, identity_head,
                                    head=args.head, fam=b_fam, nu_step=nu_step)
                loss_rec = bregman_loss(b_fam, x0, x0_hat, nu_step, b_nu_scale, args.nu_weight_cap,
                              args.nu_weight_alpha, b_pow_c) * args.lambda_rec
                loss_mom = moment_loss(b_fam, x_obs, x0_hat, nu_obs_b) * args.lambda_mom

                if args.lambda_cons > 0:
                    # Advance one deterministic step towards the clean end (Eq. 18);
                    # stop-gradient so the loss cannot collapse into a pull towards
                    # the first prediction.
                    step_prime = (torch.rand(B, device=device) * step.float()).long()
                    step_prime = step_prime.clamp_max(step - 1)
                    nu_prime = b_bridge.nu_at(step_prime).to(x0.dtype)
                    dnu = (nu_prime - nu_step).clamp_min(0.0).view(-1, 1, 1, 1)
                    with torch.no_grad():
                        if args.naive_posterior:
                            Z_prime = nu_prime.view(-1, 1, 1, 1) * x0_hat.detach()
                        else:
                            Z_prime = Z_t + dnu * x0_hat.detach()
                    x_input_prime = Z_prime / nu_prime.view(-1, 1, 1, 1)
                    x0_hat_prime = predict_x0(G, x_input_prime, step_prime,
                                              torch.log(nu_prime), p_vec, cond, tau,
                                              args.direct_pred, identity_head,
                                              head=args.head, fam=b_fam,
                                              nu_step=nu_prime)
                    loss_cons = bregman_loss(b_fam, x0, x0_hat_prime, nu_prime, b_nu_scale,
                                  args.nu_weight_cap, args.nu_weight_alpha, b_pow_c) * args.lambda_cons
                else:
                    loss_cons = torch.zeros((), device=device)

                loss_G = loss_rec + loss_mom + loss_cons
                (loss_G / accum).backward()

            acc["G"] += loss_G.item()
            acc["rec"] += loss_rec.item()
            acc["mom"] += loss_mom.item()
            acc["cons"] += loss_cons.item()

        if lr_at is not None:
            lr_now = lr_at(global_step)
            for g in opt_G.param_groups:
                g["lr"] = lr_now

        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(G_core.parameters(), args.grad_clip)
        opt_G.step()
        ema.update(G_core)
        for k in acc:
            acc[k] /= accum
        global_step += 1

        if is_main(rank) and (global_step % args.log_every == 0 or global_step == 1):
            it_s = global_step / max(time.time() - t0, 1e-6)
            cur_lr = opt_G.param_groups[0]["lr"]
            print(f"step {global_step:>6}  it/s {it_s:5.2f}  ws={world_size}  "
                  f"G {acc['G']:.3f}  rec {acc['rec']:.3f}  mom {acc['mom']:.3f}  "
                  f"cons {acc['cons']:.3f}  lr {cur_lr:.3e}", flush=True)
            writer.add_scalar("opt/lr", cur_lr, global_step)
            for k, v in acc.items():
                writer.add_scalar(f"loss/{k}", v, global_step)

        if is_main(rank) and global_step % args.eval_every == 0:
            if ema.enabled:
                G_eval.load_state_dict(ema.shadow, strict=False)
                m = eval_endpoint(G_eval, bridge, eval_ds, device, args)
            else:
                m = eval_endpoint(G_core, bridge, eval_ds, device, args)
            print("  [eval-ema] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
            for k, v in m.items():
                writer.add_scalar(f"eval/{k}", v, global_step)

        if is_main(rank) and (global_step % args.ckpt_every == 0
                              or global_step == args.total_steps):
            path = out_dir / f"ckpt_{global_step:07d}.pt"
            torch.save({"step": global_step, "G": G_core.state_dict(),
                        "G_ema": ema.state_dict(), "args": vars(args)}, path)
            print(f"  [ckpt] saved {path}", flush=True)

    if is_main(rank):
        writer.close()
        print("done.", flush=True)
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
