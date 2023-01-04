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
"""Overlap integral using real space quadrature."""

from typing import Callable

import jax
import jax.numpy as jnp

from .utils import quadrature_integral


def integrand_overlap(orbitals: Callable, keepdims: bool = True) -> Callable:
  if keepdims:
    return lambda r: -0.5 * jnp.outer(orbitals(r), orbitals(r))
  return lambda r: -0.5 * jnp.sum(orbitals(r) * orbitals(r))


def overlap_integral(orbitals: Callable, batch, keepdims=False) -> jax.Array:
  return quadrature_integral(integrand_overlap(orbitals, keepdims), batch)
