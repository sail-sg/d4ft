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
from absl import logging

from d4ft.energy import calc_energy, calc_hamiltonian, get_intor, precal_scf
from d4ft.integral.quadrature.utils import make_quadrature_points_batches
from d4ft.logger import RunLogger
from d4ft.molecule import Molecule


def scf(
  mol: Molecule,
  epoch: int,
  seed=123,
  momentum=0.5,
  converge_threshold: float = 1e-3,
  stochastic: bool = True,
  pre_cal: bool = False,
  batch_size: int = 10000,
  os_scheme: str = "none",
):
  params = mol._init_param(seed)
  mo_params, _ = params
  diag_one = jnp.eye(mol.mo.nmo)
  if not mol.restricted_mo:
    diag_one = jnp.repeat(diag_one[None], 2, 0)

  if pre_cal:
    logging.info('Preparing for integration...')
    start = time.time()
    precal_h = precal_scf(mol, batch_size, seed)
    logging.info(f"Pre-calculation finished. Time: {(time.time()-start):.3f}")
  else:
    precal_h = None

  @jax.jit
  def update(mo_params, fock):
    _, mo_params = jnp.linalg.eigh(fock)
    transpose_axis = (1, 0) if mol.restricted_mo else (0, 2, 1)
    mo_params = jnp.transpose(mo_params, transpose_axis)
    return mo_params, fock

  @jax.jit
  def calc_fock_jit(mo_params, batch1, batch2):
    fock = calc_fock(
      ao=lambda r: mol.mo((diag_one, None), r),
      mo_old=(lambda r: mol.mo((mo_params, None), r) * mol.nocc),
      nuclei=mol.nuclei,
      batch1=batch1,
      batch2=batch2,
      precal_h=precal_h,
    )
    return fock

  intors = get_intor(mol, batch_size, seed, pre_cal, mol.xc, os_scheme)

  @jax.jit
  def calc_energy_jit(mo_params, batch1, batch2):
    return calc_energy(
      intors, mol.nuclei, (mo_params, None), batch1, batch2, None
    )

  # the main loop.
  logging.info(" Starting...SCF loop")
  fock = jnp.eye(mol.nao)

  prev_e_total = 0.
  converged = False

  if stochastic:
    batches = make_quadrature_points_batches(
      mol.grids,
      mol.weights,
      batch_size,
      epochs=epoch * 2,  # separate batch for energy
      num_copies=2,
      seed=seed
    )
    num_batches = mol.grids.shape[0] // batch_size
  else:  # go through all quadrature points at once
    batch = (mol.grids, mol.weights)
    batches = ((batch, batch) for _ in range(epoch * 2))
    num_batches = 1

  logger = RunLogger()
  t = 0
  for _ in range(epoch):
    # update the fock matrix
    new_focks = []
    for _ in range(num_batches):
      batch1, batch2 = next(batches)
      new_fock = calc_fock_jit(mo_params, batch1, batch2)
      new_focks.append(new_fock)
    new_fock = jnp.mean(jnp.array(new_focks), axis=0)
    if mol.restricted_mo:
      # NOTE: for restricted case, up and down spin fock is the same
      new_fock = new_fock[0]
    fock = (1 - momentum) * new_fock + momentum * fock

    # update mo
    mo_params, fock = update(mo_params, fock)

    # evalute energies
    for _ in range(num_batches):
      batch1, batch2 = next(batches)
      e_total, energies = calc_energy_jit(mo_params, batch1, batch2)
      logger.log_step(energies, t)
      t += 1

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
