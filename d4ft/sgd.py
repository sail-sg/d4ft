# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import optax
from absl import logging

from d4ft.energy import calc_energy, get_intor, prescreen
from d4ft.integral.quadrature.utils import make_quadrature_points_batches
from d4ft.logger import RunLogger
from d4ft.molecule import Molecule


def sgd(
  mol: Molecule,
  epoch: int,
  lr: float,
  seed: int = 137,
  converge_threshold: float = 1e-3,
  batch_size: int = 1000,
  optimizer: str = "sgd",
  pre_cal: bool = False,
  lr_decay: bool = False,
  os_scheme: str = "none",
):
  """Run the main training loop."""
  params = mol._init_param(seed)

  if batch_size > mol.grids.shape[0]:
    batch_size = mol.grids.shape[0]

  optimizer_kwargs = {"learning_rate": lr}
  if lr_decay:
    total_steps = mol.grids.shape[0] / batch_size * epoch
    scheduler = optax.piecewise_constant_schedule(
      init_value=lr,
      boundaries_and_scales={
        int(total_steps * 0.5): 0.5,
        int(total_steps * 0.75): 0.5
      }
    )
    optimizer_kwargs["learning_rate"] = scheduler

  if optimizer == 'sgd':
    optimizer = optax.sgd(**optimizer_kwargs)
  elif optimizer == 'adam':
    optimizer = optax.adam(**optimizer_kwargs)
  else:
    raise NotImplementedError

  opt_state = optimizer.init(params)

  # prescreen
  intor_kwargs = dict()
  use_os = os_scheme != "none"
  if use_os and not pre_cal:
    idx_count = prescreen(mol)

  intors = get_intor(
    mol,
    batch_size,
    seed,
    pre_cal,
    mol.xc,
    os_scheme,
    **intor_kwargs,
  )

  @jax.jit
  def update(params, opt_state, batch1, batch2, idx_count):
    loss = lambda params: calc_energy(
      intors, mol.nuclei, params, batch1, batch2, idx_count
    )
    (_, energies), grad = jax.value_and_grad(loss, has_aux=True)(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, energies

  batches = make_quadrature_points_batches(
    mol.grids, mol.weights, batch_size, epochs=epoch, num_copies=2, seed=seed
  )
  num_batches = mol.grids.shape[0] // batch_size

  logging.info(f"Starting... Random Seed: {seed}, Batch size: {batch_size}")
  logging.info(f"Batch size: {batch_size}, Number of batches: {num_batches}")
  logging.info(f"Total grid points: {len(mol.grids)}")

  stochastic_os = os_scheme in ["uniform", "is"]

  # if stochastic_os:
  #   batch_size_os = 2000

  #   sampled_idx = make_4c_batches_sampled(
  #     schwartz_bound,
  #     num_batches,
  #     batch_size_os,
  #     epoch,
  #     seed,
  #     imp_sampling=(os_scheme == "is"),
  #   )

  #   sampled_idx = make_4c_batches(len(idx_count), batch_size_os, epoch, seed)

  #   batches = zip(batches, sampled_idx)

  prev_e_total = 0.
  converged = False

  # use quadrature batch
  logger = RunLogger()

  for i, batch in enumerate(batches):
    # TODO: sample unique abcd
    if stochastic_os:
      (batch1, batch2), sampled_idx = batch
      idx_count_i = idx_count[sampled_idx]
    else:
      batch1, batch2 = batch
      idx_count_i = None
      if use_os and not pre_cal:
        idx_count_i = idx_count

    # SGD step
    # idx_count_i = prescreen(mol) # compute each step
    params, opt_state, energies = update(
      params, opt_state, batch1, batch2, idx_count_i
    )
    logger.log_step(energies, i)

    if stochastic_os:
      logger.log_ewm("e_hartree", i)

    # if i % 10 == 0:
    #   logging.info(f"EVAL")
    #   params, opt_state, energies = update(
    #     params, opt_state, batch1, batch2, idx_count
    #   )
    #   logger.log_step(energies, i)

    # log at the end of each epoch
    if i % num_batches == 0:
      segment_df = logger.get_segment_summary()
      e_total = segment_df.e_total.mean()

      # check convergence
      if jnp.abs(prev_e_total - e_total) < converge_threshold:
        converged = True
        break
      else:
        prev_e_total = e_total

  logging.info(f"Converged: {converged}")
  logger.log_summary()

  return e_total, params, logger
