# Copyright 2023 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Calculate the xc functional with numerical integration"""
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.quadrature.utils import quadrature_integral, wave2density
from d4ft.types import MoCoeffFlat, QuadGridsNWeights


def get_xc_intor(
  grids_and_weights: QuadGridsNWeights,
  cgto: CGTO,
  xc_functional: Callable,
  polarized: bool = False,
) -> Callable:
  """Returns a function that calculates Exc from MO coefficients.

  Args:
    polarized: whether to use the polarized setting in XC functional
  """

  def xc_intor(mo_coeff: MoCoeffFlat) -> Float[Array, ""]:
    mo_coeff = mo_coeff.reshape(2, cgto.nao, cgto.nao)
    orbitals = lambda r: mo_coeff @ cgto.eval(r)
    density = wave2density(orbitals, polarized)
    xc_func = xc_functional(density, polarized)
    return jnp.sum(
      quadrature_integral(lambda r: density(r) * xc_func(r), grids_and_weights)
    )

  return xc_intor


def integrand_vxc_lda(ao: Callable, mo_old):
  """
  v_xc = -(3/pi n(r))^(1/3)
  Return:
    [2, N, N] array

  Example:
  vxc = quadrature_integral(integrand_vxc_lda(ao, mo_old), batch1)
  """
  density = wave2density(mo_old)

  def g(n):
    return -(3 / jnp.pi * n)**(1 / 3)

  return lambda r: g(density(r)) * jax.vmap(jnp.outer)(ao(r), ao(r))
