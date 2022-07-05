"""Primitive functions."""
import jax.numpy as jnp


def gauss_basis_func(r, zeta):
  """Gaussian basis function.

  input r: (3)-dimension
  zeta: scalar parameter
  """
  return jnp.exp(-zeta * jnp.sum(r**2))


def gaussian_inner_product(zeta1, zeta2):
  r"""Gaussian inner product.

  input r: (3)-dimension
  zeta: scalar parameter
  return
  \int_R exp(-zeta x^2) dx = (\pi / zeta)**0.5
  \int...\int_R exp(- \sum zetai xi^2 dx^i = \prod (\pi /zetai) **0.5
  """
  return jnp.pi / jnp.sqrt(zeta1 * zeta2)
