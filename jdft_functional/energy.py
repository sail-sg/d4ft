"""
energy integrands and integration.
"""

from typing import Callable
import jax
import jax.numpy as jnp
from jdft.functions import set_diag_zero, distmat
from ao_int import _energy_precal


def wave2density(mo: Callable, nocc=1., keep_spin=False):
  """
  Transform the wave function into density function.
  Args:
    mo: a [3] -> [2, N] function, where N is the number of molecular orbitals.
        mo only takes one argment, which is the coordinate.
    keep_spin: if True will return a 1D array with two elements indicating
    the density of each spin
  Return:
    density function: [3] -> float or 1D array.
  """

  if keep_spin:
    return lambda r: jnp.sum((mo(r) * nocc)**2, axis=1)
  else:
    return lambda r: jnp.sum((mo(r) * nocc)**2)


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
    return jnp.sum(jax.jacobian(mo)(r) * jax.jacobian(mo)(r))

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
      nuclei_charge / jnp.sqrt(jnp.sum((r - nuclei_loc)**2, axis=1) + 1e-10)
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
    mo: a [3] -> [2, N] function, where N is the number of molecular orbitals.
    mo only takes one argment, which is the coordinate.
  Return:
    a function: [3] x [3] -> [1]
  """

  def v(x, y):
    return wave2density(mo)(x) * wave2density(mo)(y) / jnp.sqrt(
      jnp.sum((x - y)**2) + 1e-10
    ) * jnp.where(jnp.all(x == y), 0, 1) / 2

  return v


def integrand_xc_lda(mo, keep_spin=False):
  const = -3 / 4 * (3 / jnp.pi)**(1 / 3)
  return lambda x: const * wave2density(mo, keep_spin=keep_spin)(x)**(4 / 3)


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


def energy_gs(mo: Callable, nuclei: dict, batch1, batch2):
  e_kin = integrate(integrand_kinetic_alt(mo), batch1)
  e_ext = integrate(integrand_external(mo, nuclei), batch1)
  e_hartree = integrate(integrand_hartree(mo), batch1, batch2)
  e_xc = integrate(integrand_xc_lda(mo), batch1)
  e_nuc = e_nuclear(nuclei)
  e_total = e_kin + e_ext + e_xc + e_hartree + e_nuc

  return e_total, (e_kin, e_ext, e_xc, e_hartree, e_nuc)


def _energy_gs(
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
  e_xc = integrate(integrand_xc_lda(mo), batch1)
  e_nuc = e_nuclear(nuclei)
  e_total = e_kin + e_ext + e_xc + e_hartree + e_nuc

  return e_total, (e_kin, e_ext, e_xc, e_hartree, e_nuc)


if __name__ == '__main__':

  import jdft
  from jdft.geometries import c20_geometry
  mol = jdft.molecule(c20_geometry, spin=0, level=1, basis='6-31g')

  params = mol._init_param()

  def mo(r):
    return lambda r: mol.mo(params, r)

  print(energy_gs(mo, mol.nuclei, mol.grids, mol.weights)[0])
