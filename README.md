# p-Bridge — minimal core

Reference implementation of the diffusion bridge indexed by a power-variance
exponent `p` and a cumulative measurement weight `ν`, for *The Noise Plane: One
Diffusion Bridge Across Exponent and Dispersion*.

A single family of bridges covers the Tweedie exponents `p ∈ [1, 2]` — Poisson
shot noise at `p = 1`, gamma/speckle at `p = 2`, compound Poisson-gamma in
between — with a `p`-independent deterministic reverse step and a loss that is
the Bregman divergence of the corresponding unit deviance.

## Scope of this release

This is the **minimal core**: the model, the forward simulator, the training
loop and the evaluation harness for the synthetic optical experiments. It is
the code path every other result is built on, stripped of the project's
experiment-management layer.

What is here:

- the Tweedie simulator, thinning kernels and severity calibration
- the bridge: schedule, training couple, deterministic and stochastic reverse
- training, one exponent per run or `p` drawn per batch (the conditional model)
- cross-`p` evaluation (PSNR/SSIM, per-decile, Pearson-residual diagnostics)

What is **not** here: the real-sensor pipelines (fluorescence microscopy and
SAR), the baseline implementations and their wrappers, the figure and table
generators, and the SLURM job scripts. No trained checkpoints are distributed —
every command below trains from scratch. The released code therefore reproduces
the synthetic protocol and ablations, not the real-data sections or the
baseline comparison tables.

## Install

```bash
pip install -r requirements.txt
```

PyTorch ≥ 2.0 with CUDA. Training below assumes multi-GPU via `torchrun`;
`python train.py ...` works single-process with the global batch adjusted.

## Data

BSDS500 downloads itself on first use (~70 MB, into `data/bsds500/`):

```bash
python -m data.download_bsds500
```

The training pool in the paper is BSDS500 train+val **plus** DIV2K train HR
(1100 images total). DIV2K is not auto-downloaded — fetch `DIV2K_train_HR` from
<https://data.vision.ee.ethz.ch/cvl/DIV2K/> and unpack it to
`data/div2k/DIV2K_train_HR/`, then pass `--extra_roots data/div2k/DIV2K_train_HR`.

Evaluation uses the BSDS500 **test** split only, so it needs no DIV2K.

## Before anything else

```bash
python -m pytest tests/ -q          # 11 checks of the simulator against the theory
python -m scripts.calibrate --extra_roots data/div2k/DIV2K_train_HR
```

`tests/test_tweedie.py` checks statements, not past output: forward moments
`E[Z] = νμ` and `Var(Z) = νφμ^p` for every `p`, the `α` limits, non-existence on
`0 < p < 1`, the zero atom `e^{-νλ}`, thinning kernels landing on the right
marginal, the deviance minimiser sitting at the mean (not the median), the
severity calibration, and bit-identical deterministic reverse steps across `p`.

`scripts.calibrate` writes `calibration.json`. A pre-computed one for the
paper's pool is already in the repo, so this step is only needed if you change
the data.

## Severity calibration — read this before reading any number

`ν` carries `p`-dependent units, so **it is meaningless to hold `ν` fixed across
`p`**. Every run instead fixes a severity `η := CV(μ_ref)` and solves

```
nu_obs(p) = eta^-2 * mu_ref^(p-2)        phi = 1
nu_max(p) = R * nu_obs(p)                R = 1e4, fixed across p
```

with `μ_ref = 0.464864`, the mean intensity of the BSDS500+DIV2K pool after the
`[0.05, 1]` offset. Resolved values:

| p | η=0.25 (mild) | η=0.5 (moderate) | η=1.0 (severe) |
| --- | --- | --- | --- |
| 1.0 | ν_obs = 34.42 | 8.60 | 2.15 |
| 1.5 | 23.47 | 5.87 | 1.47 |
| 2.0 | 16.00 | 4.00 | 1.00 |

At `p=1, η=1` this is exactly one photon per pixel at `μ_ref` — the definition of
*severe*. Quote `η`, `μ_ref` and `ν_obs(p)` with any result. Because the
calibration equalises CV only *at* `μ_ref`, the exponents still differ away from
it, which is the effect under study; that is why the evaluator also reports
`psnr_dark_decile` and `psnr_by_decile`.

## Train

The model reported throughout the synthetic sections is the **conditional**
one: a single checkpoint spanning the whole axis, with `p` drawn per
micro-batch from `U[1, 2]` on a 0.05 grid and fed to the network alongside
`log ν`. Four GPUs × micro-batch 2 × accum 3 = global batch 24.

```bash
torchrun --nproc_per_node=4 train.py \
    --p 1.5 --p_range 1.0 2.0 --p_quantum 0.05 \
    --eta 1.0 \
    --T 100 --schedule log \
    --image_size 256 --model_channels 64 --cond_x1 \
    --nu_weight_alpha 0.0 \
    --lambda_rec 10 --lambda_mom 1 --lambda_cons 5 \
    --batch_size 2 --grad_accum 3 --total_steps 50000 \
    --lr_G 1e-4 --lr_schedule cosine --lr_warmup 1000 --lr_min 1e-7 \
    --ema_decay 0.999 --grad_clip 1.0 \
    --dataset_length 16000 --num_workers 4 \
    --extra_roots data/div2k/DIV2K_train_HR \
    --seed 0 \
    --out_dir checkpoints/cond_severe
```

Three things in that command are not the argparse defaults and matter:

- **`--nu_weight_alpha 0.0`.** The deviance weight `ν(t)/ν_obs` spans `[1, 1e4]`
  uncapped, so the clean end of the schedule carries four orders of magnitude
  more gradient than the noisy end — and the noisy end is the state a one-step
  evaluation is scored at. `alpha` compresses that dynamic range while
  preserving the mean weight, so the `10/1/5` loss balance is untouched.
  `alpha=1` is the identity and is the argparse default; the reported model is
  flat. Nothing in the theory is at stake: the per-`t` minimiser is the
  posterior mean under every weighting.
- **`--p 1.5` is not the training exponent** when `--p_range` is given. It only
  fixes the reference point for the cheap in-loop eval. `--p_range` (or
  `--p_choices`) implies `--cond_p`.
- **the global batch, not the GPU count, is the invariant.** Keep
  `micro × accum × ngpu = 24` when the layout changes. The conditional run also
  needs enough `p`-draws per optimiser step: `p` is drawn per micro-batch per
  rank, so `ngpu × accum ≥ 8`.

Severity is severe-only by measurement: one severe-trained checkpoint covers
the whole noise range and beats per-severity specialists, so milder cells are
obtained by entering the same bridge part-way along rather than by training.

A fixed-exponent specialist is the same command with `--p <value>` and no
`--p_range`.

## Evaluate

```bash
python -m eval.run_eval \
    --ckpt checkpoints/cond_severe/ckpt_0050000.pt \
    --nfe 1 5 25 --eval_p 1.0 1.5 2.0 \
    --out_json results/eval.json
```

Results are reported at **NFE=5** throughout; NFE is not monotone and the
evaluator prints the whole sweep so this is visible rather than assumed.

**Cross-`p` protocol.** A checkpoint trained at `p_model` carries its own
calibrated `ν_obs(p_model)`; the evaluation data is generated at `p_data` with
`ν_obs(p_data)`. Both are handed to the network in the intensity domain
`x = Z/ν`, and by the calibration both have `CV(μ_ref) = η`. The chain is seeded
with `Z_start = ν_obs(p_model) · x_obs(data)` and then runs the model's own
schedule and conditioning. So the two differ **only in the shape of the noise,
never in its amount** — which is the quantity the mismatch matrix asks about.

## Layout

```
pbridge_core/tweedie.py     everything p touches: V(mu), d_p, forward sampler,
                            thinning kernels, severity calibration
pbridge_core/diffusion.py   PBridge — schedule, training couple, reverse chain
pbridge_core/network.py     UNet + log-nu (and optional p) conditioning
pbridge_core/gd/            OpenAI guided-diffusion UNet, vendored
train.py                    training loop
eval/run_eval.py            evaluate a checkpoint, optionally under a DIFFERENT p
eval/metrics.py             PSNR/SSIM, per-decile, Pearson-residual diagnostics
data/optical_dataset.py     BSDS500(+extra roots) pool, paired eval set
scripts/calibrate.py        mu_ref and the nu_obs(p) table -> calibration.json
tests/test_tweedie.py       run before any training
```

## Two deliberate deviations from the derivation

1. **`B_ψ` instead of `d_p` in the loss.** `train.bregman_loss` uses
   `B_ψ = d_p/2`. The equivalence theorem is indifferent to the positive factor,
   and this makes the `p=1` instance numerically identical to the published
   `p=1` bridge, so the `(10, 1, 5)` weights carry over unchanged.

2. **The deviance weight is `ν(t)/ν_obs`, not `ν(t)`.** The derivation
   prescribes the weighting *across* `t`, which dividing by a per-run constant
   leaves untouched. What it fixes is a cross-run confound: `ν_max` differs 2×
   between `p=1` and `p=2` at equal severity and 16× between mild and severe, so
   the raw weighting would make the rec:mom ratio — and, under `grad_clip`, the
   effective step size — differ between cells for reasons unrelated to the
   exponent. `--nu_weight_raw` restores the literal form.

## License

MIT, see [LICENSE](LICENSE).

`pbridge_core/gd/` is vendored from
[OpenAI guided-diffusion](https://github.com/openai/guided-diffusion) and stays
under its own MIT license, reproduced in
[`pbridge_core/gd/LICENSE_GUIDED_DIFFUSION`](pbridge_core/gd/LICENSE_GUIDED_DIFFUSION).

BSDS500 and DIV2K are distributed by their respective owners under their own
terms; this repository downloads or references them but redistributes neither.
