"""
energy integrands and integration.
"""

from typing import Callable
import jax
import jax.numpy as jnp
from jdft.functions import set_diag_zero, distmat


def wave2density(mo: Callable, keep_spin=False):
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
    return lambda r: jnp.sum(mo(r)**2, axis=1)
  else:
    return lambda r: jnp.sum(mo(r)**2)


def integrand_kinetic(mo: Callable):
  r"""
  the kinetic intergrand:  - \psi(r) \nabla psi(r) /2
  Args:
    mo: a [3] -> [2, N] function, where N is the number of molecular orbitals.
    mo only takes one argment, which is the coordinate.
  Return:
    a [3] -> [1] function.
  """

  def f(r):
    hessian_diag = jnp.diagonal(jax.hessian(mo)(r), 0, 2, 3)
    return jnp.sum(hessian_diag, axis=2)

  return lambda r: -jnp.sum(f(r) * mo(r)) / 2


def integrand_external(mo: Callable, nuclei):
  r"""
  the external intergrand: 1 / (r - R) * \psi^2
  """

  nuclei_loc = nuclei['loc']
  nuclei_charge = nuclei['charge']

  def v(r):
    return jnp.sum(nuclei_charge / jnp.linalg.norm(r - nuclei_loc, axis=1))

  return lambda r: -jnp.sum(v(r) * mo(r)**2)


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


def integrand_xc_lda(mo):
  const = -3 / 4 * (3 / jnp.pi)**(1 / 3)
  return lambda x: const * wave2density(mo)(x)**(4 / 3)


def integrate_single(integrand: Callable, batch):
  g, w = batch

  @jax.jit
  def f(g, w):
    return jnp.sum(jax.vmap(integrand)(g) * w)

  return f(g, w)


def integrate_double(integrand: Callable, batch1, batch2=None):
  r"""
  \int v(x, y) dx dy
  """
  g1, w1 = batch1
  if batch2 is None:
    g2, w2 = batch1
  else:
    g2, w2 = batch2

  @jax.jit
  def f(g1, w1, g2, w2):
    w_mat = jax.vmap(lambda x: jax.vmap(lambda y: integrand(x, y))(g2))(g1)
    # w_mat = jnp.where(w_mat>1e3, 0, w_mat)
    w1 = jnp.expand_dims(w1, 1)
    w2 = jnp.expand_dims(w2, 1)
    return jnp.squeeze(w1.T @ w_mat @ w2)

  return f(g1, w1, g2, w2)


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


def energy_gs(mo: Callable, nuclei: dict, batch1, batch2=None):
  e_kin = integrate_single(integrand_kinetic(mo), batch1)
  e_ext = integrate_single(integrand_external(mo, nuclei), batch1)
  e_hartree = integrate_double(integrand_hartree(mo), batch1, batch2)
  e_xc = integrate_single(integrand_xc_lda(mo), batch1)
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
