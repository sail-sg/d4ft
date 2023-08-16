# Copyright 2023 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from d4ft.config import GDConfig
from d4ft.types import TrainingState


def get_optimizer(
  cfg: GDConfig,
  params: hk.Params,
  rng_key: jax.Array,
) -> Dict[str, Tuple[optax.GradientTransformation, TrainingState]]:
  opt_states = dict()
  if cfg.meta_opt == "none":  # load schedule
    if cfg.lr_decay == "piecewise":
      lr = optax.piecewise_constant_schedule(
        init_value=cfg.lr,
        boundaries_and_scales={
          int(cfg.epochs * 0.5): 0.5,
          int(cfg.epochs * 0.75): 0.25,
          int(cfg.epochs * 0.825): 0.125,
        }
      )
    elif cfg.lr_decay == "cosine":
      lr = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1.0,
        warmup_steps=50,
        decay_steps=cfg.epochs - 50,
        end_value=0.0,
      )
    else:
      lr = cfg.lr
    optimizer = getattr(optax, cfg.optimizer)(learning_rate=lr)

  else:  # meta learns the learning rate
    init_lr = jnp.array(cfg.lr)
    meta_lr = jnp.array(cfg.meta_lr)
    optimizer = optax.inject_hyperparams(getattr(optax, cfg.optimizer))(
      learning_rate=init_lr
    )
    meta_opt = getattr(optax, cfg.meta_opt)(learning_rate=meta_lr)
    meta_params = -np.log(1. / init_lr - 1)
    meta_opt_state = meta_opt.init(meta_params)
    meta_state = TrainingState(meta_params, meta_opt_state, rng_key)
    opt_states["meta"] = (meta_opt, meta_state)

  opt_state = optimizer.init(params)
  state = TrainingState(params, opt_state, rng_key)

  opt_states["main"] = (optimizer, state)

  return opt_states
