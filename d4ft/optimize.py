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

import optax

from d4ft.config import OptimizerConfig


def get_optimizer(cfg: OptimizerConfig) -> optax.GradientTransformation:
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

  if cfg.optimizer == "sgd":
    optimizer = optax.sgd(learning_rate=lr)
  elif cfg.optimizer == "adam":
    optimizer = optax.adam(learning_rate=lr)
  elif cfg.optimizer == "adamw":
    optimizer = optax.chain(
      optax.clip(1.0),
      optax.adamw(learning_rate=lr),
    )
  else:
    raise NotImplementedError

  return optimizer
