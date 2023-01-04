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

import time

import jax
import jax.numpy as jnp
import optax
from absl import logging

from d4ft.energy import calc_energy, precal_ao_matrix
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

  e_kwargs = {}
  if pre_cal:
    logging.info('Preparing for integration...')
    start = time.time()
    ao_kin_mat, ao_ext_mat = precal_ao_matrix(mol, batch_size, seed)
    logging.info(f"Pre-calculation finished. Time: {(time.time()-start):.3f}")
    e_kwargs = dict(
      ao_kin_mat=ao_kin_mat,
      ao_ext_mat=ao_ext_mat,
      nocc=mol.nocc,
    )

  @jax.jit
  def update(params, opt_state, batch1, batch2):

    def loss(params):
      return calc_energy(
        orbitals=lambda r: mol.mo(params, r) * mol.nocc,
        nuclei=mol.nuclei,
        batch1=batch1,
        batch2=batch2,
        mo_params=params,
        pre_cal=pre_cal,
        **e_kwargs
      )

    (e_total, energies), grad = jax.value_and_grad(loss, has_aux=True)(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, energies

  logging.info(f"Starting... Random Seed: {seed}, Batch size: {batch_size}")
  logging.info(f"Batch size: {batch_size}")
  logging.info(f"Total grid points: {len(mol.grids)}")

  batches = make_quadrature_points_batches(
    mol.grids, mol.weights, batch_size, epochs=epoch, num_copies=2, seed=seed
  )

  num_batches = mol.grids.shape[0] // batch_size

  prev_e_total = 0.
  converged = False

  logger = RunLogger()
  for i, (batch1, batch2) in enumerate(batches):
    # SGD step
    params, opt_state, energies = update(params, opt_state, batch1, batch2)
    logger.log_step(energies, i)

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
