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
"""Kinetic integral using real space quadrature."""

from typing import Callable

import jax
import jax.numpy as jnp

from .utils import get_integrand, quadrature_integral


def integrand_kinetic(orbitals: Callable, keepdims: bool = False) -> Callable:
  r"""
  kinetic integrand

  .. math::
  <phi_i| -1/2\nabla |phi_j>

  Args:
    orbitals: a [3] -> [2, N] function, where the input is the 3D
  electron coordinate, N is the number of orbitals, and 2 is for each spin.
    keep_dim: whether to compute outer product matrix

  Return:
    the kinetic integrand, which is a [3] -> [1] function.
  If keepdims is True, will return a [3] -> [2, N, N] function.
  """

  def laplacian(r: jnp.array) -> jnp.array:
    hess_phi_r = jax.hessian(orbitals)(r)
    num_batch_dims = len(hess_phi_r.shape) - 2

    # skip the batch dimensions
    hessian_diag = jnp.diagonal(
      hess_phi_r, offset=0, axis1=num_batch_dims, axis2=num_batch_dims + 1
    )
    return -0.5 * jnp.sum(hessian_diag, axis=num_batch_dims)

  return get_integrand(orbitals, laplacian, keepdims)


def integrand_kinetic_jac(
  orbitals: Callable, keepdims: bool = False
) -> Callable:
  r"""
  Equivalent expression compute the kinetic integrand via
  integration by parts.

  .. math::
  1/2 * (\nabla\psi(r))**2

  Args:
    orbitals: a [3] -> [2, N] function, where the input is the 3D
  electron coordinate, N is the number of orbitals, and 2 is for each spin.
  Return:
    a [3] -> [1] function.

  TODO: write the derivation
  """

  jac_sqr = get_integrand(
    jax.jacobian(orbitals), jax.jacobian(orbitals), keepdims
  )
  return lambda r: 0.5 * jac_sqr(r)


def kinetic_integral(
  orbitals: Callable, batch, keepdims=False, use_jac=True
) -> jnp.array:
  return quadrature_integral(
    integrand_kinetic_jac(orbitals, keepdims)
    if use_jac else integrand_kinetic(orbitals, keepdims),
    batch,
  )
