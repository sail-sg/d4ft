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
"""
energy integrands and integration.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from d4ft.functions import decov, distmat, set_diag_zero, wave2density
from d4ft.integral.quadrature import (
  electron_repulsion_integral, electron_repulsion_integral_scf,
  kinetic_integral, nuclear_attraction_integral
)
from d4ft.integral.quadrature.utils import (
  make_quadrature_points_batches, quadrature_integral
)
from d4ft.molecule import Molecule

from collections import namedtuple

Energies = namedtuple(
  "Energies", ["e_total", "e_kin", "e_ext", "e_xc", "e_hartree", "e_nuc"]
)


def integrand_exc_lda(mo: Callable):
  r"""LDA with spin.
  https://www.chem.fsu.edu/~deprince/programming_projects/lda/
  Local spin-density approximation
    E_\sigma = 2^(1/3) C \int \rho_\sigma^(4/3) dr
    where C = (3/4)(3/\pi)^(1/3)
  Args:
    mo (Callable): a [3] -> [2, N] function, where N is the number of molecular
    orbitals. mo only takes one argment, which is the coordinate.
  Returns:
  """
  C = -3 / 4 * (3 / jnp.pi)**(1 / 3)
  const = jnp.power(2., 1 / 3) * C
  return lambda x: const * jnp.sum(wave2density(mo, keep_spin=True)(x)**(4 / 3))


def integrand_vxc_lda(ao: Callable, mo_old):
  """
  v_xc = -(3/pi n(r))^(1/3)
  Return:
    [2, N, N] array
  """
  density = wave2density(mo_old)

  def g(n):
    return -(3 / jnp.pi * n)**(1 / 3)

  return lambda r: g(density(r)) * jax.vmap(jnp.outer)(ao(r), ao(r))


def e_nuclear(nuclei):
  """
    Potential energy between atomic nuclears.
  """
  nuclei_loc = nuclei['loc']
  nuclei_charge = nuclei['charge']
  dist_nuc = distmat(nuclei_loc)
  charge_outer = jnp.outer(nuclei_charge, nuclei_charge)
  charge_outer = set_diag_zero(charge_outer)
  return jnp.sum(charge_outer / (dist_nuc + 1e-15)) / 2


def precal_ao_matrix_to_qrmo(params, mat, nocc):
  """Given a whitened AO integral matrix
  <phi_i|O|phi_j> where phi_i are AOs and O is some
  observable, compute the orthogonal MO integral matrix.

  QR decomposition is used to ensure orthogonality

  Args:
    params: a tuple of (mo_params, ao_params)
  """
  mo_params, _ = params
  mo_params = jnp.expand_dims(mo_params, 0)
  mo_params = jnp.repeat(mo_params, 2, 0)  # shape: [2, N, N]

  def transform_by_qrmo_coeff(param, nocc):
    orthogonal, _ = jnp.linalg.qr(param)
    orthogonal *= jnp.expand_dims(nocc, axis=0)
    return jnp.sum(jnp.diagonal(orthogonal.T @ mat @ orthogonal))

  return jnp.sum(jax.vmap(transform_by_qrmo_coeff)(mo_params, nocc))


def precal_ao_matrix(mol: Molecule, batch_size: int, seed: int):
  overlap_decov = decov(mol.ao.overlap())

  batches = make_quadrature_points_batches(
    mol.grids, mol.weights, batch_size, epochs=1, num_copies=1, seed=seed
  )

  @jax.jit
  def ao_kin_mat_fun(batch):
    kinetic = kinetic_integral(mol.ao, batch, use_jac=False, keepdims=True)
    return overlap_decov @ kinetic @ overlap_decov.T

  @jax.jit
  def ao_ext_mat_fun(batch):
    ext = nuclear_attraction_integral(
      mol.ao, mol.nuclei["loc"], mol.nuclei["charge"], batch, keepdims=True
    )
    return overlap_decov @ ext @ overlap_decov.T

  ao_kin_mat = jnp.zeros([mol.nao, mol.nao])
  ao_ext_mat = jnp.zeros([mol.nao, mol.nao])

  for batch in batches:
    ao_kin_mat += ao_kin_mat_fun(batch)
    ao_ext_mat += ao_ext_mat_fun(batch)

  # TODO: check whether this is needed
  # num_batches = mol.grids.shape[0] // batch_size
  # ao_kin_mat /= num_batches
  # ao_ext_mat /= num_batches

  return ao_kin_mat, ao_ext_mat


def precal_scf(mol: Molecule, batch_size: int, seed: int):
  """
  TODO: benchmark with minibatch_vmap
  """
  batches = make_quadrature_points_batches(
    mol.grids, mol.weights, batch_size, epochs=1, num_copies=1, seed=seed
  )

  diag_one = jnp.ones([2, mol.mo.nmo])
  diag_one = jax.vmap(jnp.diag)(diag_one)

  # NOTE: these ao will have the spin axis
  def ao(r):
    return mol.mo((diag_one, None), r)

  @jax.jit
  def kin_fun(batch):
    return kinetic_integral(ao, batch, use_jac=False, keepdims=True)

  @jax.jit
  def ext_fun(batch):
    return nuclear_attraction_integral(
      ao, mol.nuclei["loc"], mol.nuclei["charge"], batch, keepdims=True
    )

  kin = jnp.zeros([2, mol.nmo, mol.nmo])
  ext = jnp.zeros([2, mol.nmo, mol.nmo])

  for batch in batches:
    kin += kin_fun(batch)
    ext += ext_fun(batch)

  return kin, ext


def calc_fock(
  ao: Callable,
  mo_old: Callable,
  nuclei,
  batch1,
  batch2=None,
  precal_h=None,
):
  """Calculate Fock matrix for SCF."""
  if batch2 is None:
    batch2 = batch1
  if precal_h is None:
    kin = kinetic_integral(ao, batch1, use_jac=False, keepdims=True)
    ext = nuclear_attraction_integral(
      ao, nuclei["loc"], nuclei["charge"], batch1, keepdims=True
    )
  else:
    kin, ext = precal_h
  hartree = electron_repulsion_integral_scf(ao, mo_old, batch1, batch2)
  vxc = quadrature_integral(integrand_vxc_lda(ao, mo_old), batch1)
  return kin + ext + hartree + vxc


def calc_energy(
  orbitals: Callable,
  nuclei: dict,
  batch1=None,
  batch2=None,
  mo_params=None,
  xc='lda',
  pre_cal: bool = False,
  **kwargs
):
  """
  TODO: write the reason for having two separate batch
  """
  if batch2 is None:
    batch2 = batch1

  if pre_cal:
    e_kin = precal_ao_matrix_to_qrmo(
      mo_params, kwargs["ao_kin_mat"], kwargs["nocc"]
    )
    e_ext = precal_ao_matrix_to_qrmo(
      mo_params, kwargs["ao_ext_mat"], kwargs["nocc"]
    )
  else:
    e_kin = kinetic_integral(orbitals, batch1, use_jac=True)
    e_ext = nuclear_attraction_integral(
      orbitals, nuclei["loc"], nuclei["charge"], batch1
    )

  e_hartree = electron_repulsion_integral(
    orbitals, batch1=batch1, batch2=batch2
  )

  if xc == 'lda':
    e_xc = quadrature_integral(integrand_exc_lda(orbitals), batch1)
  else:
    raise NotImplementedError

  e_nuc = e_nuclear(nuclei)
  e_total = e_kin + e_ext + e_xc + e_hartree + e_nuc

  return e_total, Energies(e_total, e_kin, e_ext, e_xc, e_hartree, e_nuc)
