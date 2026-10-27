"""No-op logger stub — guided_diffusion.fp16_util references logger.logkv_mean/log
in the MixedPrecisionTrainer path we don't use. Kept minimal to avoid pulling
in the full guided_diffusion logger with its heavy deps."""


def logkv_mean(key, val):  # pragma: no cover
    pass


def log(msg):  # pragma: no cover
    pass


def logkv(key, val):  # pragma: no cover
    pass
