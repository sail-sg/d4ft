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

from typing import NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging

from d4ft.config import GDConfig
from d4ft.logger import RunLogger
from d4ft.optimize import get_optimizer
from d4ft.types import Hamiltonian, HamiltonianHKFactory, Trajectory, Transition


def scipy_opt(
  gd_cfg: GDConfig, H_factory: HamiltonianHKFactory, key: jax.random.KeyArray
) -> float:

  H_transformed = hk.without_apply_rng(hk.multi_transform(H_factory))
  init_params = H_transformed.init(key)

  H = Hamiltonian(*H_transformed.apply)

  energy_fn_jit = jax.jit(lambda mo_coeff: H.energy_fn(mo_coeff)[0])

  import jaxopt
  solver = jaxopt.BFGS(fun=energy_fn_jit, maxiter=500)
  res = solver.run(init_params)
  return res


class TrainingState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState
  rng_key: jax.Array


def sgd(
  gd_cfg: GDConfig, H_factory: HamiltonianHKFactory, key: jax.random.KeyArray
) -> Tuple[RunLogger, Trajectory, Hamiltonian]:

  # H_transformed = hk.without_apply_rng(hk.multi_transform(H_factory))
  H_transformed = hk.multi_transform(H_factory)
  H = Hamiltonian(*H_transformed.apply)

  @jax.jit
  def update(state: TrainingState) -> Tuple:
    """update parameter, and accumulate gradients"""
    rng_key, next_rng_key = jax.random.split(state.rng_key)
    val_and_grads_fn = jax.value_and_grad(H.energy_fn, has_aux=True)
    (_, aux), grad = val_and_grads_fn(state.params, rng_key)
    energies, mo_grads = aux
    updates, opt_state = optimizer.update(grad, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    return TrainingState(params, opt_state, next_rng_key), energies, mo_grads

  @jax.jit
  def meta_loss(
    meta_params: hk.Params,
    params: hk.Params,
    opt_state: optax.OptState,
    rng_key: jax.Array,
  ):
    opt_state.hyperparams["learning_rate"] = jax.nn.sigmoid(meta_params)
    state = TrainingState(params, opt_state, rng_key)

    # for _ in range(10):
    new_state, energies, mo_grads = update(state)
    # state = new_state

    loss = H.energy_fn(new_state.params, new_state.rng_key)[0]

    return loss, (new_state, energies, mo_grads)

  @jax.jit
  def meta_step(state: TrainingState, meta_params, meta_state):
    grad, (new_state, energies, mo_grads) = jax.grad(
      meta_loss, has_aux=True
    )(meta_params, state.params, state.opt_state, state.rng_key)
    meta_updates, meta_state = meta_opt.update(grad, meta_state)
    new_meta_params = optax.apply_updates(meta_params, meta_updates)

    return new_meta_params, meta_state, new_state, energies, mo_grads

  # init state
  params = H_transformed.init(key)

  if gd_cfg.meta_opt == "none":
    optimizer = get_optimizer(gd_cfg)
  else:
    init_lr = jnp.array(gd_cfg.lr)
    meta_lr = jnp.array(gd_cfg.meta_lr)
    optimizer = optax.inject_hyperparams(getattr(optax, gd_cfg.optimizer))(
      learning_rate=init_lr
    )
    meta_opt = getattr(optax, gd_cfg.meta_opt)(learning_rate=meta_lr)
    meta_params = -np.log(1. / init_lr - 1)
    meta_state = meta_opt.init(meta_params)

  opt_state = optimizer.init(params)
  state = TrainingState(params, opt_state, key)

  # GD loop
  traj = []
  converged = False
  logger = RunLogger()
  # mo_grad_norm_fn = jax.jit(jax.vmap(partial(jnp.linalg.norm, ord=2), 0, 0))
  for step in range(gd_cfg.epochs):

    if gd_cfg.meta_opt == "none":
      new_state, energies, mo_grads = update(state)
    else:
      meta_params, meta_state, new_state, energies, mo_grads = meta_step(
        state, meta_params, meta_state
      )
      logging.info(f"cur lr: {jax.nn.sigmoid(meta_params):.4f}")

    logger.log_step(energies, step)
    logger.get_segment_summary()

    # logging.info(mo_grads[-1])
    # breakpoint()

    mo_coeff = H.mo_coeff_fn(state.params, state.rng_key, apply_spin_mask=False)
    t = Transition(mo_coeff, energies, mo_grads)
    traj.append(t)

    # params = new_params
    state = new_state

    if step < gd_cfg.hist_len:  # don't check for convergence
      continue

    # gradient norm for each spin
    # mo_grad_norm = mo_grad_norm_fn(sum(mo_grads))
    # logging.info(f"mo grad norm: {mo_grad_norm.mean()}")

    # check convergence
    e_total_std = jnp.stack(
      [t.energies.e_total for t in traj[-gd_cfg.hist_len:]]
    ).std()
    logging.info(f"e_total std: {e_total_std}")
    if e_total_std < gd_cfg.converge_threshold:
      converged = True
      break

    # if not improving, stop early
    # recent_e_min = logger.data_df[-gd_cfg.hist_len:].e_total.min()
    # if recent_e_min > logger.data_df.e_total.min():
    #   break

  logging.info(f"Converged: {converged}")

  return logger, traj, H
