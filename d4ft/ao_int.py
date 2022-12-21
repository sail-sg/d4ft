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
from typing import Callable


def _int_matrix_quadrature(f: Callable, g: Callable, grids, weights):
  r"""
  Computes the integral matrix <f_i|g_j> with quadrature, where f_i and g_j
  are functions of the electron position.

  - Since the integral is computed with quadrature, i.e.

  .. math::
  <f_i|g_j>= \sum_k w_k \dot (f_i(r_k)g_j(r_k))

  the integral matrix can be computed as the outer product of the vector
  F(r)=(f_1(r), ..., f_N(r)) and G(r)=(g_1(r), ..., g_N(r)), i.e.

  .. math::
  (<f_i|g_j>)= \sum_k w_k \dot (F(r_k)G(r_k)^T)

  - The integral is computed with a randomly sampled batch of quadrature
  points, as described in the D4FT paper.

    Args:
      f: R^3 -> R^N
      g: R^3 -> R^N
      grids: [N, 3]
      weights: [N, ]
    Return:
      integral matrix: [N, N]
  """

  # compute the integral matrix for one quadrature point
  def integral_matrix(r, w):
    return jnp.outer(f(r), g(r)) * w

  # compute the sum over the sampled batch of quadrature point
  # as ammortized integral
  @jax.jit
  def reduce_outer_sum(x, w):
    return jnp.sum(jax.vmap(integral_matrix)(x, w), axis=0)

  return reduce_outer_sum(grids, weights)


def kinetic_integral(ao: Callable, grids, weights):
  r"""
  Computes the kinetic integral <phi_i| -1/2\nabla |phi_j>
  """

  def laplacian(r):  # R^3 -> R^N
    hessian_diag = jnp.diagonal(jax.hessian(ao)(r), 0, 1, 2)  # [N, 3]
    return -jnp.sum(hessian_diag, axis=1) / 2  # [N]

  return _int_matrix_quadrature(ao, laplacian, grids, weights)


def overlap_integral(ao: Callable, grids, weights):
  r"""
  Computes the overlap integral <phi_i|phi_j>
  """
  return _int_matrix_quadrature(ao, ao, grids, weights)


def external_integral(ao: Callable, nuclei: dict, grids, weights):
  r"""
  Computes the external potential integral <phi_i| Z/|r-R| |phi_j>
  """

  def nucleus_repulsion(r):
    return -ao(r) * jnp.sum(
      nuclei["charge"] / jnp.linalg.norm(r - nuclei["loc"], axis=1)
    )

  return _int_matrix_quadrature(ao, nucleus_repulsion, grids, weights)


def _energy_precal(params, _ao_kin_mat, nocc):
  mo_params, _ = params
  mo_params = jnp.expand_dims(mo_params, 0)
  mo_params = jnp.repeat(mo_params, 2, 0)  # shape: [2, N, N]

  def f(param, nocc):
    orthogonal, _ = jnp.linalg.qr(param)
    orthogonal *= jnp.expand_dims(nocc, axis=0)
    return jnp.sum(jnp.diagonal(orthogonal.T @ _ao_kin_mat @ orthogonal))

  return jnp.sum(jax.vmap(f)(mo_params, nocc))
