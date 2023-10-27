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

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from absl import logging

from d4ft.config import GDConfig
from d4ft.logger import RunLogger
from d4ft.optimize import get_optimizer
from d4ft.types import Hamiltonian, TrainingState, Trajectory, Transition


def scipy_opt(
  solver_cfg: GDConfig, H: Hamiltonian, params: hk.Params,
  key: jax.random.KeyArray
) -> float:
  energy_fn_jit = jax.jit(lambda mo_coeff: H.energy_fn(mo_coeff, key)[0])
  import jaxopt
  solver = jaxopt.BFGS(fun=energy_fn_jit, maxiter=500)
  res = solver.run(params)
  return res


def sgd(
  solver_cfg: GDConfig, H: Hamiltonian, params: hk.Params,
  key: jax.random.KeyArray
) -> Tuple[RunLogger, Trajectory]:

  @jax.jit
  def update(state: TrainingState) -> Tuple:
    """update parameter, and accumulate gradients"""
    rng_key, next_rng_key = jax.random.split(state.rng_key)
    val_and_grads_fn = jax.value_and_grad(H.energy_fn, has_aux=True)
    (loss, aux), grad = val_and_grads_fn(state.params, rng_key)
    energies, mo_grads = aux
    updates, opt_state = optimizer.update(grad, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    return loss, TrainingState(
      params, opt_state, next_rng_key
    ), energies, mo_grads

  @jax.jit
  def meta_loss(meta_params: hk.Params, state: TrainingState):
    opt_state = state.opt_state
    opt_state.hyperparams["learning_rate"] = jax.nn.sigmoid(meta_params)
    state = TrainingState(state.params, opt_state, state.rng_key)

    # for _ in range(10):
    _, new_state, energies, mo_grads = update(state)
    # state = new_state

    loss = H.energy_fn(new_state.params, new_state.rng_key)[0]

    return loss, (new_state, energies, mo_grads)

  @jax.jit
  def meta_step(state: TrainingState, meta_state: TrainingState):
    grad, aux = jax.grad(meta_loss, has_aux=True)(meta_state.params, state)
    new_state, energies, mo_grads = aux
    meta_updates, meta_opt_state = meta_opt.update(grad, meta_state.opt_state)
    new_meta_params = optax.apply_updates(meta_state.params, meta_updates)
    return TrainingState(
      new_meta_params, meta_opt_state, meta_state.rng_key
    ), new_state, energies, mo_grads

  # init state
  opt_states = get_optimizer(solver_cfg, params, key)
  optimizer, state = opt_states["main"]
  if solver_cfg.meta_opt != "none":
    meta_opt, meta_state = opt_states["meta"]

  # GD loop
  traj = []
  converged = False
  logger = RunLogger()
  e_total_std = 0.
  for step in range(solver_cfg.epochs):

    if solver_cfg.meta_opt == "none":
      loss, new_state, energies, mo_grads = update(state)
      logging.info(f"{loss=}")
    else:
      meta_state, new_state, energies, mo_grads = meta_step(state, meta_state)
      logging.info(f"cur lr: {jax.nn.sigmoid(meta_state.params):.4f}")

    logger.log_step(energies, step, e_total_std)
    logger.get_segment_summary()

    mo_coeff = H.mo_coeff_fn(state.params, state.rng_key, apply_spin_mask=False)
    t = Transition(mo_coeff, energies, mo_grads)

    traj.append(t)

    state = new_state

    if step < solver_cfg.hist_len:  # don't check for convergence
      continue

    # check convergence
    e_total_std = jnp.stack(
      [t.energies.e_total for t in traj[-solver_cfg.hist_len:]]
    ).std()
    if e_total_std < solver_cfg.converge_threshold:
      converged = True
      break

  logging.info(f"Converged: {converged}")

  return logger, traj
