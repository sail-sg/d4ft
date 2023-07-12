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
"""Solve DFT with gradient descent"""

from typing import Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from absl import logging
from d4ft.config import DirectMinimizationConfig, OptimizerConfig
from d4ft.logger import RunLogger
from d4ft.optimize import get_optimizer
from d4ft.types import (
  Hamiltonian, HamiltonianHKFactory, Trajectory, Transition
)


def sgd(
  direct_min_cfg: DirectMinimizationConfig, optim_cfg: OptimizerConfig,
  H_factory: HamiltonianHKFactory, key: jax.random.KeyArray
) -> Tuple[float, Trajectory, Hamiltonian]:

  H_transformed = hk.without_apply_rng(hk.multi_transform(H_factory))
  params = H_transformed.init(key)

  H = Hamiltonian(*H_transformed.apply)

  # init optimizer
  optimizer = get_optimizer(optim_cfg)
  opt_state = optimizer.init(params)

  @jax.jit
  def update(params: chex.ArrayTree, opt_state: chex.ArrayTree) -> Tuple:
    """update parameter, and accumulate gradients"""
    val_and_grads_fn = jax.value_and_grad(H.energy_fn, has_aux=True)
    (_, aux), grad = val_and_grads_fn(params)
    energies, mo_grads = aux
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, energies, mo_grads

  # GD loop
  e_total = 0.
  traj = []
  prev_e_total = 0.
  converged = False
  logger = RunLogger()
  for step in range(optim_cfg.epochs):
    new_params, opt_state, energies, mo_grads = update(params, opt_state)

    logger.log_step(energies, step)
    segment_df = logger.get_segment_summary()
    e_total = segment_df.e_total.mean()

    mo_coeff = H.mo_coeff_fn(params, apply_spin_mask=False)
    t = Transition(mo_coeff, energies, mo_grads)
    traj.append(t)

    params = new_params

    if step < 100:  # don't check for convergence
      continue

    # check convergence
    e_total_std = jnp.stack([t.energies.e_total for t in traj[-50:]]).std()
    logging.info(f"e_total std: {e_total_std}")
    if e_total_std < direct_min_cfg.converge_threshold:
      converged = True
      break

    e_total = energies.e_total
    if jnp.abs(prev_e_total - e_total) < direct_min_cfg.converge_threshold:
      converged = True
      break
    else:
      prev_e_total = e_total

  logging.info(f"Converged: {converged}")

  return e_total, traj, H
