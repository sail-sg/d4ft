'''
energy integrads.
'''

import jax
import jax.numpy as jnp
from jdft.functions import set_diag_zero, distmat


def wave2density(mo, keep_spin=False):
  '''
  Transform the wave function into density function.
  Args:
    mo: a [3] -> [2, N] function, where N is the number of molecular orbitals.
        mo only takes one argment, which is the coordinate.
    keep_spin: if True will return a 1D array with two elements indicating the density of each spin
  Return [2]
    density: [3] -> float or 1D array.
  '''

  if keep_spin:
    return lambda r: jnp.sum(mo(r)**2, axis=1)
  else:
    return lambda r: jnp.sum(mo(r)**2)


def integrand_kinetic(mo):
  '''
  the kinetic intergrand:  - \psi(r) \nabla psi(r) /2
  Args:
    mo: a [3] -> [2, N] function, where N is the number of molecular orbitals.
    mo only takes one argment, which is the coordinate.
  Return:
    a [3] -> [1] function.
  '''

  def f(r):
    hessian_diag = jnp.diagonal(jax.hessian(mo)(r), 0, 2, 3)
    return jnp.sum(hessian_diag, axis=2)

  return lambda r: -jnp.sum(f(r) * mo(r)) / 2


def integrand_external(mo, nuclei):
  '''
  the external intergrand: 1 / (r - R) * \psi^2
  '''

  nuclei_loc = nuclei['loc']
  nuclei_charge = nuclei['charge']

  def v(r):
    return jnp.sum(nuclei_charge / jnp.linalg.norm(r - nuclei_loc, axis=1))

  return lambda r: -jnp.sum(v(r) * mo(r)**2)


def integrand_hartree(mo):
  '''
  Return n(x)n(y)/|x-y|
  Args:
    mo: a [3] -> [2, N] function, where N is the number of molecular orbitals.
    mo only takes one argment, which is the coordinate.
  Return:
    a function: [3] x [3] -> [1]
  '''

  return lambda x, y: wave2density(mo)(x) * wave2density(mo)(y) / jnp.sqrt(
    jnp.sum((x - y)**2) + 1e-10
  ) / 2


def integrand_xc_lda(mo):
  const = -3 / 4 * (3 / jnp.pi)**(1 / 3)
  return lambda x: const * wave2density(mo)(x)**(4 / 3)


def integrate_single(integrand, grids, weights):

  @jax.jit
  def f(g, w):
    return jnp.sum(jax.vmap(integrand)(g) * w)

  return f(grids, weights)


def integrate_double(integrand, grids, weights):
  '''
  \int v(x, y) dx dy
  '''

  @jax.jit
  def f(g, w):
    w_mat = jax.vmap(lambda x: jax.vmap(lambda y: integrand(x, y))(g))(g)
    w_mat = set_diag_zero(w_mat)
    w = jnp.expand_dims(w, 1)
    return jnp.squeeze(w.T @ w_mat @ w)

  return f(grids, weights)


def e_nuclear(nuclei):
  """Potential energy between atomic nuclears."""
  nuclei_loc = nuclei['loc']
  nuclei_charge = nuclei['charge']
  dist_nuc = distmat(nuclei_loc)
  charge_outer = jnp.outer(nuclei_charge, nuclei_charge)
  charge_outer = set_diag_zero(charge_outer)
  return jnp.sum(charge_outer / (dist_nuc + 1e-15)) / 2


def energy_gs(mo, nuclei, grids, weights):
  e_kin = integrate_single(integrand_kinetic(mo), grids, weights)
  e_ext = integrate_single(integrand_external(mo, nuclei), grids, weights)
  e_hartree = integrate_double(integrand_hartree(mo), grids, weights)
  e_xc = integrate_single(integrand_xc_lda(mo), grids, weights)
  e_nuc = e_nuclear(nuclei)

  return e_kin + e_ext + e_xc + e_hartree + e_nuc, (
    e_kin, e_ext, e_xc, e_hartree, e_nuc
  )


if __name__ == '__main__':

  import jdft
  from jdft.geometries import c20_geometry
  mol = jdft.molecule(c20_geometry, spin=0, level=1, basis='6-31g')

  params = mol._init_param()
  mo = lambda r: mol.mo(params, r)

  print(energy_gs(mo, mol.nuclei, mol.grids, mol.weights)[0])
