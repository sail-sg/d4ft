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
"""Electron repulsion integral using real space quadrature."""

from typing import Callable

import jax
import jax.numpy as jnp

from .utils import quadrature_integral, wave2density, get_integrand


def coulomb(x, y):
  """
  TODO: handles the case where x==y
  TODO: rationale for the magic number 2e-9
  """
  # EPS = 2e-9
  EPS = jnp.finfo(x.dtype).eps
  return jnp.where(
    jnp.all(x == y),
    EPS,
    1 / jnp.sqrt(jnp.sum((x - y)**2)),
  )


def integrand_hartree(orbitals: Callable, **kwargs):
  r"""
  Return 1/2 n(x)n(y)/|x-y|

  TODO: add two batch correction coefficent

  Args:
    orbitals: a [3] -> [2, N] function, where N is the number of molecular
    orbitals. mo only takes one argment, which is the coordinate.
  Return:
    a function: [3] x [3] -> [1]
  """
  return lambda x, y: (
    wave2density(orbitals)(x) * wave2density(orbitals)(y) * 0.5 * coulomb(x, y)
  )


def electron_repulsion_integral_scf(
  orbitals: Callable, orbitals_old: Callable, batch1, batch2
) -> jax.Array:
  r"""Returns g(x)=\int n_old(y)/|x-y| dy

  The hartree term in the SCF Hamiltonian is
  \int g(x)n(x) dx

  Args:
    orbitals_old: used in scf
  """
  g = lambda x: quadrature_integral(
    lambda y: wave2density(orbitals_old)(y) * coulomb(x, y), batch1
  ) * orbitals(x)
  integrand = get_integrand(g, orbitals, keepdims=True)
  return quadrature_integral(integrand, batch2)


def electron_repulsion_integral(
  orbitals: Callable, batch1, batch2
) -> jax.Array:
  return quadrature_integral(integrand_hartree(orbitals), batch1, batch2)
