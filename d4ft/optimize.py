import optax

from d4ft.config import OptimizerConfig


def get_optimizer(cfg: OptimizerConfig) -> optax.GradientTransformation:
  if cfg.lr_decay == "piecewise":
    lr = optax.piecewise_constant_schedule(
      init_value=cfg.lr,
      boundaries_and_scales={
        int(cfg.epochs * 0.5): 0.5,
        int(cfg.epochs * 0.75): 0.5
      }
    )
  elif cfg.lr_decay == "exponential":
    lr = optax.exponential_decay(
      cfg.lr, cfg.scheduler_step, cfg.scheduler_gamma, staircase=True
    )
  else:
    lr = cfg.lr

  optimizer_kwargs = {"learning_rate": lr}

  if cfg.optimizer == "sgd":
    optimizer = optax.sgd(**optimizer_kwargs)
  elif cfg.optimizer == "adam":
    optimizer = optax.adam(**optimizer_kwargs)
  else:
    raise NotImplementedError

  return optimizer
