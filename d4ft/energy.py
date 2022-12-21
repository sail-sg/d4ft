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
from d4ft.functions import set_diag_zero, distmat, wave2density
from d4ft.ao_int import _energy_precal


def integrand_kinetic(mo: Callable, keep_dim=False):
  r"""
  the kinetic intergrand:  - \psi(r) \nabla psi(r) /2
  Args:
    mo: a [3] -> [2, N, N] function, where N is the number of molecular
    orbitals. mo only takes one argment, which is the coordinate.
  Return:
    a [3] -> [1] function. If keep_dim is True, will return a
    [3] -> [2, N, N] function.
  """

  def f(r):
    hessian_diag = jnp.diagonal(jax.hessian(mo)(r), 0, 2, 3)
    return jnp.sum(hessian_diag, axis=2)

  if keep_dim:
    return lambda r: -jax.vmap(jnp.outer)(mo(r), f(r)) / 2

  return lambda r: -jnp.sum(f(r) * mo(r)) / 2


def integrand_kinetic_alt(mo: Callable):
  r"""
  the kinetic intergrand:  - \psi(r) \nabla psi(r) /2,
  this function calculates: jacobian(\psi(r)) * jacobian(\psi(r)) /2
  Args:
    mo: a [3] -> [2, N] function, where N is the number of molecular orbitals.
    mo only takes one argment, which is the coordinate.
  Return:
    a [3] -> [1] function.
  """

  def f(r):
    return jnp.sum(jax.jacobian(mo)(r)**2)

  return lambda r: f(r) / 2


def integrand_external(mo: Callable, nuclei, keep_dim=False):
  r"""
  the external intergrand: 1 / (r - R) * \psi^2.
  If keep_dim, return a function [3] -> [2, N], where each element reads
    \phi_i^2 /|r-R|
  """

  nuclei_loc = nuclei['loc']
  nuclei_charge = nuclei['charge']

  def v(r):
    return jnp.sum(
      nuclei_charge / jnp.sqrt(jnp.sum((r - nuclei_loc)**2, axis=1) + 1e-20)
    )

  if keep_dim:

    def o(r):
      return -v(r) * mo(r)**2
  else:

    def o(r):
      return -jnp.sum(v(r) * mo(r)**2)

  return o


def integrand_hartree(mo: Callable):
  r"""
  Return n(x)n(y)/|x-y|
  Args:
    mo (Callable): a [3] -> [2, N] function, where N is the number of molecular
    orbitals. mo only takes one argment, which is the coordinate.
  Return:
    a function: [3] x [3] -> [1]
  """

  def v(x, y):
    return wave2density(mo)(x) * wave2density(mo)(y) \
     * jnp.where(jnp.all(x == y), 2e-9, 1/jnp.sqrt(
      jnp.sum((x - y)**2))) / 2

  return v


def integrand_hartree_stochastic(mo: Callable, key, **kwargs):
  r"""
  Return n(x)n(y)/|x-y|
  Args:
    mo (Callable): a [3] -> [2, N] function, where N is the number of molecular
    orbitals. mo only takes one argment, which is the coordinate.
  Return:
    a function: [3] x [3] -> [1]
  """
  N = 16

  def v(x, y):
    return wave2density(mo)(x) * wave2density(mo)(y) \
     * jnp.where(jnp.all(x == y), 1e-10, 1/jnp.sqrt(
      jnp.sum((x - y)**2 + 1e-20))) / 2

  eps = jax.random.normal(key, [N, 3]) * 1e-8

  def w(x, y):

    def _v(z):
      return v(x, z)

    output = jnp.mean(jax.vmap(lambda e: v(x, y + e))(eps))

    def second_order(e):
      e = jnp.expand_dims(e, 1)
      return jnp.squeeze(e.T @ jax.hessian(_v)(y) @ e)

    output -= 1 / 2 * jnp.mean(jax.vmap(second_order)(eps))
    return output

  return w


def intor_hartree(mo, batch1, batch2, key, **kwargs):
  # integrand = integrand_hartree_stochastic(mo, key)
  g1, w1 = batch1
  keys = jax.random.split(key, g1.shape[0])

  if batch2:
    g2, w2 = batch2
  else:
    g2, w2 = batch1

  @jax.jit
  def f(g1, w1, g2, w2):
    w_mat = jax.vmap(
      lambda x: jax.
      vmap(lambda y, key: integrand_hartree_stochastic(mo, key)
           (x, y))(g2, keys)
    )(
      g1
    )
    # w_mat = jnp.where(w_mat>1e3, 0, w_mat)
    w1 = jnp.expand_dims(w1, 1)
    w2 = jnp.expand_dims(w2, 1)
    return jnp.squeeze(w1.T @ w_mat @ w2)

  return f(g1, w1, g2, w2)


def hartree_correction(mo: Callable, batch, v0=0, **kwargs):
  g, w = batch

  def v(r):
    return v0 * wave2density(mo)(r)**2

  @jax.jit
  def correct(x):
    return jnp.sum(jax.vmap(v)(x))

  return correct(g)


def integrand_x_lda(mo: Callable, keep_spin=False):
  """
  Local density approximation
  Args:
      mo (callable):
      keep_spin (bool, optional): If true, will return a array of shape [2].
        Defaults to False.

  Returns:
      Callable: integrand of lda.
  """
  const = -3 / 4 * (3 / jnp.pi)**(1 / 3)
  return lambda x: const * wave2density(mo, keep_spin=keep_spin)(x)**(4 / 3)


def integrand_x_lsda(mo: Callable):
  r"""
  https://www.chem.fsu.edu/~deprince/programming_projects/lda/
  Local spin-density approximation
    E_\sigma = 2^(1/3) C \int \rho_\sigma^(4/3) dr
    where C = (3/4)(3/\pi)^(1/3)
  Args:
    mo (Callable): a [3] -> [2, N] function, where N is the number of molecular
    orbitals. mo only takes one argment, which is the coordinate.
  Returns:
  """
  const = -3 / 4 * (3 / jnp.pi)**(1 / 3)

  def v(x):
    return jnp.power(2., 1 / 3) * const * jnp.sum(
      wave2density(mo, keep_spin=True)(x)**(4 / 3)
    )

  return v


def integrate(integrand: Callable, *coords_and_weights):
  """Numerically integrate the integrand.
  Args:
    integrand: a multivariable function.
    coords_and_weights: the points that the function will be evaluated
      and the weights of these points.
  Returns:
    A scalar value as the result of the integral.
  """
  # break down the coordinates and the weights
  coords = [coord for coord, _ in coords_and_weights]
  weights = [weight for _, weight in coords_and_weights]
  # vmap the integrand
  num = len(coords_and_weights)
  f = integrand
  for i in range(num):
    in_axes = (None,) * (num - i - 1) + (0,) + (None,) * (i)
    f = jax.vmap(f, in_axes=in_axes)
  out = f(*coords)
  # weighted sum
  for weight in reversed(weights):
    out = jnp.dot(out, weight)

  # # adjust for Hartree:
  # batch_size = len(coords[0])
  # out * (batch_size - (len(coords) == 2))
  return out


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


def energy_gs(
  mo: Callable, nuclei: dict, batch1, batch2=None, xc='lda', **kwargs
):
  """
  TODO: write the reason for having two separate batch
  """
  if batch2 is None:
    batch2 = batch1

  e_kin = integrate(integrand_kinetic_alt(mo), batch1)
  e_ext = integrate(integrand_external(mo, nuclei), batch1)
  e_hartree = integrate(integrand_hartree(mo, **kwargs), batch1, batch2)

  if xc == 'lda':
    e_xc = integrate(integrand_x_lsda(mo), batch1)
  else:
    raise NotImplementedError

  e_nuc = e_nuclear(nuclei)
  e_total = e_kin + e_ext + e_xc + e_hartree + e_nuc

  return e_total, (e_kin, e_ext, e_xc, e_hartree, e_nuc)


def energy_gs_with_precal(
  mo: Callable,
  nuclei: dict,
  params,
  _ao_kin_mat,
  _ao_ext_mat,
  nocc,
  batch1,
  batch2=None
):
  """
    calculate ground state energy with pre-calculated ao integrations.
  """
  e_kin = _energy_precal(params, _ao_kin_mat, nocc)
  e_ext = _energy_precal(params, _ao_ext_mat, nocc)
  e_hartree = integrate(integrand_hartree(mo), batch1, batch2)
  e_xc = integrate(integrand_x_lda(mo), batch1)
  e_nuc = e_nuclear(nuclei)
  e_total = e_kin + e_ext + e_xc + e_hartree + e_nuc

  return e_total, (e_kin, e_ext, e_xc, e_hartree, e_nuc)


if __name__ == '__main__':

  import d4ft
  from d4ft.geometries import c20_geometry
  mol = d4ft.Molecule(c20_geometry, spin=0, level=1, basis='6-31g')

  params = mol._init_param()

  def mo(r):
    return mol.mo(params, r)

  print(energy_gs(mo, mol.nuclei, mol.grids, mol.weights)[0])
