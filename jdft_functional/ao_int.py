import jax
import jax.numpy as jnp
from typing import Callable


def _ao_kin_int(ao: Callable, grids, weights):
  r"""
    \int phi(x) \nobla phi(x)^T dx
    Args:
      ao: R^3 -> R^N
      grids: [N, 3]
      weights: [N, ]
    Return:
      Array: [N, N]
  """

  def fun(r):  # R^3 -> R^N
    hessian_diag = jnp.diagonal(jax.hessian(ao)(r), 0, 1, 2)  # [N, 3]
    return -jnp.sum(hessian_diag, axis=1) / 2  # [N]

  def outer_f(r, w):
    return jnp.outer(ao(r), fun(r)) * w

  @jax.jit
  def reduce_outer_sum(x, w):
    return jnp.sum(jax.vmap(outer_f)(x, w), axis=0)

  return reduce_outer_sum(grids, weights)


def _ao_overlap_int(ao: Callable, grids, weights):
  r"""
    \int phi(x) \nobla phi(x)^T dx
    Args:
      ao: R^3 -> R^N
      grids: [N, 3]
      weights: [N, ]
    Return:
      Array: [N, N]
  """

  def outer_f(r, w):
    return jnp.outer(ao(r), ao(r)) * w

  @jax.jit
  def reduce_outer_sum(x, w):
    return jnp.sum(jax.vmap(outer_f)(x, w), axis=0)

  return reduce_outer_sum(grids, weights)


def _ao_ext_int(ao: Callable, nuclei: dict, grids, weights):
  r"""
    \int e_R phi^i(x) phi(x)^T / |x-R| dx
    Args:
      ao: R^3 -> R^N
      nuclei: a dict
      grids: [N, 3]
      weights: [N, ]

    Return:
      Array: [N, N]
  """
  nuclei_loc = nuclei['loc']
  nuclei_charge = nuclei['charge']

  def fun(r):
    return ao(r) * jnp.sum(
      nuclei_charge / jnp.linalg.norm(r - nuclei_loc, axis=1)
    )

  def outer_f(r, w):
    return jnp.outer(fun(r), ao(r)) * w

  @jax.jit
  def reduce_outer_sum(x, w):
    return jnp.sum(jax.vmap(outer_f)(x, w), axis=0)

  return -reduce_outer_sum(grids, weights)


def _energy_precal(params, _ao_kin_mat, nocc):
  mo_params, _ = params
  mo_params = jnp.expand_dims(mo_params, 0)
  mo_params = jnp.repeat(mo_params, 2, 0)  # shape: [2, N, N]

  def f(param, nocc):
    orthogonal, _ = jnp.linalg.qr(param)
    orthogonal *= jnp.expand_dims(nocc, axis=0)
    return jnp.sum(jnp.diagonal(orthogonal.T @ _ao_kin_mat @ orthogonal))

  return jnp.sum(jax.vmap(f)(mo_params, nocc))


# def _ao_overlap_int(ao, grids, weights):
#   r"""

#   """
