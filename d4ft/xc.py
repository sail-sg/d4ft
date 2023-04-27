"""Calculate the xc functional with numerical integration"""
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pyscf
from pyscf.dft import gen_grid

from d4ft.types import Array


def eval_ao(r: Array, mol: pyscf.gto.mole.Mole):
  """Evaluate N-body atomic orbitals at location r.

  Args:
        r: (3) coordinate.

  Returns:
    (N,) ao output
  """
  atom_coords = mol.atom_coords()
  output = []
  for idx in np.arange(len(mol.elements)):
    element = mol.elements[idx]
    coord = atom_coords[idx]
    for i in mol._basis[element]:
      prm_array = jnp.array(i[1:])
      exponents = prm_array[:, 0]
      coeffs = prm_array[:, 1]

      if i[0] == 0:  # s-orbitals
        output.append(
          jnp.sum(
            coeffs * jnp.exp(-exponents * jnp.sum((r - coord)**2)) *
            (2 * exponents / jnp.pi)**(3 / 4)
          )
        )

      elif i[0] == 1:  # p-orbitals
        output += [
          (r[j] - coord[j]) * jnp.sum(
            coeffs * jnp.exp(-exponents * jnp.sum((r - coord)**2)) *
            (2 * exponents / jnp.pi)**(3 / 4) * (4 * exponents)**0.5
          ) for j in np.arange(3)
        ]

  return jnp.array(output)


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


def quadrature_integral(integrand: Callable, *coords_and_weights) -> jnp.array:
  """Integrate a multivariate function with quadrature.

  - The integral can be computed with a randomly sampled batch of
  quadrature points, as described in the D4FT paper.

  Args:
    integrand: a multivariable function, with kwargs keepdims
    keepdims: whether to output the outer product matrix as explained above
    coords_and_weights: the quadrature points that the function will
      be evaluated and the weights of these points.

  Returns:
    A scalar value as the result of the integral.
  """
  # break down the coordinates and the weights
  coords = [coord for coord, _ in coords_and_weights]
  weights = [weight for _, weight in coords_and_weights]
  # vmap the integrand
  num_dims = len(coords_and_weights)
  fn = integrand
  for i in range(num_dims):
    in_axes = (None,) * (num_dims - i - 1) + (0,) + (None,) * (i)
    fn = jax.vmap(fn, in_axes=in_axes)
  out = fn(*coords)

  if len(weights) == 1:
    return jnp.sum((out.T * weights[0]).T, axis=0)

  # multi-dimensional quadrature
  for weight in reversed(weights):
    out = jnp.dot(out, weight)

  return out


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


def get_xc_intor(
  mol: pyscf.gto.mole.Mole,
  xc_type: str = "lda",
  quad_level: int = 1,  # TODO: change this
) -> Callable:
  """only support quadrature now"""

  g = gen_grid.Grids(mol)
  g.level = quad_level
  g.build()
  grids = jnp.array(g.coords)
  weights = jnp.array(g.weights)

  def xc_intor(mo_coeff):
    mo_coeff = mo_coeff.reshape(2, mol.nao, mol.nao)
    orbitals = lambda r: mo_coeff @ eval_ao(r, mol)
    batch = (grids, weights)

    if xc_type == 'lda':
      return quadrature_integral(integrand_exc_lda(orbitals), batch)
    else:
      raise NotImplementedError

  return xc_intor
