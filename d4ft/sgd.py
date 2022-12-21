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

from d4ft.ao_int import external_integral, kinetic_integral
from d4ft.energy import energy_gs, energy_gs_with_precal
from d4ft.functions import decov
from d4ft.molecule import Molecule
from d4ft.sampler import batch_sampler


def sgd(
  mol: Molecule,
  epoch: int,
  lr: float,
  seed: int = 123,
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

  if batch_size > mol.grids.shape[0]:
    batch_size = mol.grids.shape[0]

  @jax.jit
  def sampler(seed):
    return batch_sampler(mol.grids, mol.weights, batch_size, seed=seed)

  dataset1 = sampler(seed)

  if pre_cal:
    logging.info('Preparing for integration...')
    start = time.time()
    overlap_decov = decov(mol.ao.overlap())

    @jax.jit
    def _ao_kin_mat_fun(batch):
      g, w = batch
      _ao_kin_mat = kinetic_integral(mol.ao, g, w)
      _ao_kin_mat = overlap_decov @ _ao_kin_mat @ overlap_decov.T

      return _ao_kin_mat

    @jax.jit
    def _ao_ext_mat_fun(batch):
      g, w = batch
      _ao_ext_mat = external_integral(mol.ao, mol.nuclei, g, w)
      _ao_ext_mat = overlap_decov @ _ao_ext_mat @ overlap_decov.T

      return _ao_ext_mat

    _ao_kin_mat = jnp.zeros([mol.nao, mol.nao])
    _ao_ext_mat = jnp.zeros([mol.nao, mol.nao])

    for batch in dataset1:

      _ao_kin_mat += _ao_kin_mat_fun(batch)
      _ao_ext_mat += _ao_ext_mat_fun(batch)

    _ao_kin_mat /= len(list(dataset1))
    _ao_ext_mat /= len(list(dataset1))
    logging.info(f"Pre-calculation finished. Time: {(time.time()-start):.3f}")

  @jax.jit
  def update(params, opt_state, batch1, batch2):

    def loss(params):

      def mo(r):
        return mol.mo(params, r) * mol.nocc

      if pre_cal:
        return energy_gs_with_precal(
          mo, mol.nuclei, params, _ao_kin_mat, _ao_ext_mat, mol.nocc, batch1,
          batch2
        )
      else:
        return energy_gs(mo, mol.nuclei, batch1, batch2)

    (e_total, e_splits), grad = jax.value_and_grad(loss, has_aux=True)(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, (e_total, *e_splits)

  logging.info(f"Starting... Random Seed: {seed}, Batch size: {batch_size}")

  prev_loss = 0.
  start_time = time.time()
  e_train = []
  converged = False

  logging.info(f"Batch size: {batch_size}")
  logging.info(f"Total grid points: {len(mol.grids)}")

  for i in range(epoch):
    iter_time = 0
    Es_batch = []

    batchs1 = sampler(seed + i)
    batchs2 = sampler(seed + i + 1)

    for batch1, batch2 in zip(batchs1, batchs2):
      _time = time.time()
      params, opt_state, Es = update(params, opt_state, batch1, batch2)
      iter_time += time.time() - _time
      Es_batch.append(Es)

    # retrieve all energy terms
    e_total, e_kin, e_ext, e_xc, e_hartree, e_nuc = jnp.mean(
      jnp.array(Es_batch), axis=0
    )
    # track total energy for convergence check
    e_train.append(e_total)

    if (i + 1) % 1 == 0:
      logging.info(
        f"Iter: {i+1}/{epoch}. \
          Ground State Energy: {e_total:.3f}. \
          Time: {iter_time:.3f}"
      )

    if jnp.abs(prev_loss - e_total) < converge_threshold:
      converged = True
      break
    else:
      prev_loss = e_total

  logging.info(
    f"Converged: {converged}. \n"
    f"Total epochs run: {i+1}. \n"
    f"Training Time: {(time.time() - start_time):.3f}s. \n"
  )
  logging.info("Energy:")
  logging.info(f" Ground State: {e_total}")
  logging.info(f" Kinetic: {e_kin}")
  logging.info(f" External: {e_ext}")
  logging.info(f" Exchange-Correlation: {e_xc}")
  logging.info(f" Hartree: {e_hartree}")
  logging.info(f" Nucleus Repulsion: {e_nuc}")

  return e_total, params
