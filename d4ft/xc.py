"""Calculate the xc functional with numerical integration"""
from typing import Callable

import jax.numpy as jnp

from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.quadrature.utils import quadrature_integral, wave2density
from d4ft.system.mol import Mol
from d4ft.types import MoCoeffFlat


# TODO: integrate jax_xc
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


# TODO: integrate jax_xc
def get_xc_intor(mol: Mol, cgto: CGTO, xc_type: str = "lda") -> Callable:
  """only support quadrature now"""

  def xc_intor(mo_coeff):
    mo_coeff = mo_coeff.reshape(2, cgto.nao, cgto.nao)
    orbitals = lambda r: mo_coeff @ cgto.eval(r)
    batch = (mol.grids, mol.weights)

    if xc_type == 'lda':
      return quadrature_integral(integrand_exc_lda(orbitals), batch)
    else:
      raise NotImplementedError

  return xc_intor
