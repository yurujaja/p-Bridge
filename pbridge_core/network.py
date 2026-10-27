"""PBridgeUNet — guided-diffusion UNet with a log-nu conditioning branch and an
optional p conditioning branch fused into the time embedding.

Signature: `(x, timesteps, log_nu, p=None, cond=None) -> raw predictor output`.
The head is applied by the caller.

Head (§9, §13.5 item 3). For all p >= 1 we use the *direct* log-intensity head
x0_hat = exp(clip(G, +-tau)), which enforces x0_hat in Omega = (0, inf) by
construction. We deliberately do not offer gamma-Bridge's log-*residual* head:
CV(x_t) = x0^(p/2-1)/sqrt(nu) is signal-independent only at p = 2, so an anchor
on x_t is unreliable in dark regions for p < 2 (Remark 31). At p = 0 an identity
head is required instead, since Omega = R.

p conditioning is off by default: an E0 specialist sees a single exponent, and
adding a constant embedding would only change its parameter count relative to
the lambda-Bridge architecture it must be comparable to. It exists here so that
E2 (the p-conditioned generalist) is a flag flip rather than a port.
"""

from __future__ import annotations

import math

import torch as th
import torch.nn as nn

from .gd.unet import UNetModel
from .gd.nn import timestep_embedding, linear


def scalar_embedding(v: th.Tensor, dim: int) -> th.Tensor:
    """Sinusoidal embedding of a real scalar; range-agnostic, so one set of
    frequencies covers log nu across the whole severity ladder."""
    half = dim // 2
    freqs = th.exp(
        -math.log(10000.0)
        * th.arange(start=0, end=half, dtype=th.float32, device=v.device)
        / max(half - 1, 1)
    )
    args = v.float()[:, None] * freqs[None]
    emb = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = th.nn.functional.pad(emb, (0, 1))
    return emb


class PBridgeUNet(UNetModel):
    """UNet + log-nu (and optionally p) conditioning fused into the time embedding."""

    def __init__(self, *args, cond_p: bool = False, **kwargs):
        kwargs.setdefault("num_classes", None)
        super().__init__(*args, **kwargs)
        time_embed_dim = self.model_channels * 4
        self.log_nu_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.cond_p = bool(cond_p)
        if self.cond_p:
            # Same embedding treatment as log nu (§13.5 item 2).
            self.p_embed = nn.Sequential(
                linear(self.model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

    def forward(self, x: th.Tensor, timesteps: th.Tensor, log_nu: th.Tensor,
                p: th.Tensor | None = None, cond: th.Tensor | None = None) -> th.Tensor:
        assert timesteps.ndim == 1 and timesteps.shape[0] == x.shape[0]
        assert log_nu.ndim == 1 and log_nu.shape[0] == x.shape[0], \
            f"log_nu shape {tuple(log_nu.shape)} vs batch {x.shape[0]}"

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb = emb + self.log_nu_embed(scalar_embedding(log_nu, self.model_channels))
        if self.cond_p:
            assert p is not None, "network was built with cond_p=True but p was not passed"
            emb = emb + self.p_embed(scalar_embedding(p, self.model_channels))

        if cond is not None:
            assert cond.shape == x.shape, f"cond {cond.shape} vs x {x.shape}"
            x_in = th.cat([x, cond], dim=1)
        else:
            x_in = x

        hs = []
        h = x_in.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h.type(x.dtype))


def build_default_unet(
    image_size: int = 256,
    in_channels: int = 1,
    out_channels: int = 1,
    model_channels: int = 64,
    channel_mult=(1, 2, 4),
    num_res_blocks: int = 2,
    attention_resolutions=(),
    use_scale_shift_norm: bool = True,
    dropout: float = 0.0,
    cond_p: bool = False,
) -> PBridgeUNet:
    """Same capacity as the lambda-Bridge backbone, so E0 numbers are directly
    comparable to the published p=1 results."""
    return PBridgeUNet(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        channel_mult=channel_mult,
        dropout=dropout,
        use_scale_shift_norm=use_scale_shift_norm,
        num_classes=None,
        cond_p=cond_p,
    )


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
