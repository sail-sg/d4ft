"""Solve DFT with gradient descent"""

from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from absl import logging

from d4ft.config import DFTConfig, OptimizerConfig
from d4ft.logger import RunLogger
from d4ft.optimize import get_optimizer
from d4ft.types import (
  Hamiltonian, HamiltonianHKFactory, Trajectory, Transition
)


def sgd_solver(
  dft_cfg: DFTConfig, optim_cfg: OptimizerConfig, H_fac: HamiltonianHKFactory,
  key: jax.random.KeyArray
) -> Tuple[float, Trajectory, Hamiltonian]:

  H_transformed = hk.without_apply_rng(hk.multi_transform(H_fac))
  params = H_transformed.init(key)
  H = Hamiltonian(*H_transformed.apply)

  # init optimizer
  optimizer = get_optimizer(optim_cfg)
  opt_state = optimizer.init(params)

  @jax.jit
  def update(params, opt_state):
    """update parameter, and accumulate gradients"""
    val_and_grads_fn = jax.value_and_grad(H.energy_fn, has_aux=True)
    (_, aux), grad = val_and_grads_fn(params)
    energies, mo_grads = aux
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, energies, mo_grads

  # GD loop
  traj = []
  prev_e_total = 0.
  converged = False
  logger = RunLogger()
  for step in range(optim_cfg.epochs):
    new_params, opt_state, energies, mo_grads = update(params, opt_state)

    logger.log_step(energies, step)
    segment_df = logger.get_segment_summary()
    e_total = segment_df.e_total.mean()

    mo_coeff = H.mo_coeff_fn(params)
    t = Transition(mo_coeff, energies, mo_grads)
    traj.append(t)

    params = new_params

    # check convergence
    e_total = energies.e_total
    if jnp.abs(prev_e_total - e_total) < dft_cfg.converge_threshold:
      converged = True
      break
    else:
      prev_e_total = e_total

  logging.info(f"Converged: {converged}")

  return e_total, traj, H
