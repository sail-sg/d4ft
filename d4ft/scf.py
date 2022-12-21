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
from d4ft.energy import energy_gs, wave2density, integrand_kinetic
from typing import Callable
from absl import logging


def hamil_kinetic(ao: Callable, batch):
  r"""
  \int \phi_i \nabla^2 \phi_j dx
  Args:
    ao (Callable):
    batch: a tuple of (grids, weights)
  Return:
    [2, N, N] Array.
  """
  return integrate_s(integrand_kinetic(ao, True), batch)


def hamil_external(ao: Callable, nuclei, batch):
  r"""
  \int \phi_i \nabla^2 \phi_j dx
  Args:
    mo (Callable): a [3] -> [2, N] function
    batch: a tuple (grids, weights)
  Return:
    [2, N, N] Array
  """
  nuclei_loc = nuclei['loc']
  nuclei_charge = nuclei['charge']

  def v(r):
    return -jnp.sum(
      nuclei_charge / jnp.sqrt(jnp.sum((r - nuclei_loc)**2, axis=1) + 1e-16)
    )

  def m(r):
    return jax.vmap(jnp.outer)(ao(r), ao(r))

  return integrate_s(lambda r: v(r) * m(r), batch)


def hamil_hartree(ao: Callable, mo_old, batch):
  density = wave2density(mo_old)

  def g(r):
    r"""
      g(r) = \int n(r')/|r-r'| dr'
    """

    def v(x):
      return density(x) / jnp.clip(
        jnp.linalg.norm(x - r), a_min=1e-8
      ) * jnp.any(x != r)

    return integrate_s(v, batch)

  def m(r):
    return jax.vmap(jnp.outer)(ao(r), ao(r))

  return integrate_s(lambda r: g(r) * m(r), batch)


def hamil_lda(ao: Callable, mo_old, batch):
  """
  v_xc = -(3/pi n(r))^(1/3)
  Return:
    [2, N, N] array
  """
  density = wave2density(mo_old)

  def g(n):
    return -(3 / jnp.pi * n)**(1 / 3)

  def m(r):
    return jax.vmap(jnp.outer)(ao(r), ao(r))

  return integrate_s(lambda r: g(density(r)) * m(r), batch)


def get_fock(ao: Callable, mo_old: Callable, nuclei, batch):
  return hamil_kinetic(ao, batch) + \
      hamil_external(ao, nuclei, batch) + \
      hamil_hartree(ao, mo_old, batch) + \
      hamil_lda(ao, mo_old, batch)


def integrate_s(integrand: Callable, batch):
  '''
  Integrate a [3] -> [2, N, ...] function.
  '''
  g, w = batch

  def v(r, w):
    return integrand(r) * w

  @jax.jit
  def f(g, w):
    return jnp.sum(jax.vmap(v)(g, w), axis=0)

  return f(g, w)


def scf(mol, epoch, seed=123, momentum=0.5):
  batch = (mol.grids, mol.weights)
  params = mol._init_param(seed)
  mo_params, _ = params
  _diag_one_ = jnp.ones([2, mol.mo.nmo])
  _diag_one_ = jax.vmap(jnp.diag)(_diag_one_)

  @jax.jit
  def update(mo_params, fock):

    def ao(r):
      return mol.mo((_diag_one_, None), r)

    def mo_old(r):
      return mol.mo((mo_params, None), r) * mol.nocc

    fock = get_fock(ao, mo_old, mol.nuclei, batch)
    fock = momentum * fock + (1 - momentum) * fock
    _, mo_params = jnp.linalg.eigh(fock)
    mo_params = jnp.transpose(mo_params, (0, 2, 1))

    def mo(r):
      return mol.mo((mo_params, None), r) * mol.nocc

    return mo_params, fock, energy_gs(mo, mol.nuclei, batch, batch)

  # the main loop.
  logging.info(" Starting...SCF loop")
  fock = jnp.eye(mol.nao)

  for i in range(epoch):
    mo_params, fock, Es = update(mo_params, fock)

    e_total, e_splits = Es
    e_kin, e_ext, e_xc, e_hartree, e_nuc = e_splits

    logging.info(f" Iter: {i+1}/{epoch}")
    logging.info(f" Ground State: {e_total}")
    logging.info(f" Kinetic: {e_kin}")
    logging.info(f" External: {e_ext}")
    logging.info(f" Exchange-Correlation: {e_xc}")
    logging.info(f" Hartree: {e_hartree}")
    logging.info(f" Nucleus Repulsion: {e_nuc}")
