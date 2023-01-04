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
"""Nuclear attraction integral using real space quadrature."""

from typing import Callable

import jax
import jax.numpy as jnp

from .utils import quadrature_integral, get_integrand


def integrand_nuclear_attraction(
  orbitals: Callable,
  nuclei_loc: jax.Array,
  nuclei_charge: float,
  keepdims=False,
) -> Callable:
  r"""
  The nuclear_attraction intergrand: - 1 / (r - R) * \psi^2.

  Args:
    orbitals: a [3] -> [2, N] function, where the input is the 3D
  electron coordinate, N is the number of orbitals, and 2 is for each spin.
    keep_dim: If keepdims, return a function [3] -> [2, N], where each element
  reads \phi_i^2 /|r-R|

  Return:
    the nuclear attraction integrand
  """

  def external_potential(r):
    EPS = jnp.finfo(r.dtype).eps
    return -jnp.sum(
      nuclei_charge / jnp.sqrt(jnp.sum((r - nuclei_loc)**2, axis=1) + EPS)
    ) * orbitals(r)

  return get_integrand(orbitals, external_potential, keepdims)


def nuclear_attraction_integral(
  orbitals: Callable,
  nuclei_loc,
  nuclei_charge,
  batch,
  keepdims=False,
) -> jax.Array:
  return quadrature_integral(
    integrand_nuclear_attraction(
      orbitals,
      nuclei_loc=nuclei_loc,
      nuclei_charge=nuclei_charge,
      keepdims=keepdims,
    ),
    batch,
  )
