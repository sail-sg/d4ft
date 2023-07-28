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

import einops
import jax.numpy as jnp
import jax_xc
from jaxtyping import Array, Float

from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.quadrature.utils import quadrature_integral, wave2density
from d4ft.types import Fock, MoCoeff, QuadGridsNWeights


def get_xc_functional(xc_type: str, polarized: bool) -> Callable:
  """Returns (a linear combination of) xc functional.

  Args:
    xc_type: Name of the xc functional to use. To mix two XC functional, use the
      syntax a*xc_name_1+b*xc_name_2 where a, b are numbers.
  """
  if "+" in xc_type:
    weights, xc_names = zip(*(map(lambda x: x.split("*"), xc_type.split("+"))))
    weights = list(map(float, weights))
    xc_funcs = [getattr(jax_xc, xc_name)(polarized) for xc_name in xc_names]

    def xc_func(density: Callable, r: Float[Array, "3"]) -> float:
      ret = 0.
      for w, xc_func in zip(weights, xc_funcs):
        ret += w * xc_func(density, r)
      return ret

  else:
    xc_func = getattr(jax_xc, xc_type)(polarized)

  return xc_func


def get_xc_intor(
  grids_and_weights: QuadGridsNWeights,
  cgto: CGTO,
  xc_func: Callable,
  polarized: bool = False,
) -> Callable:
  """Returns a function that calculates Exc from MO coefficients.

  Args:
    polarized: whether to use the polarized setting in XC functional
  """

  def xc_intor(mo_coeff: MoCoeff) -> Float[Array, ""]:
    mo = lambda r: mo_coeff @ cgto.eval(r)
    density = wave2density(mo, polarized)
    return jnp.sum(
      quadrature_integral(
        lambda r: density(r) * xc_func(density, r), grids_and_weights
      )
    )

  return xc_intor


def get_lda_vxc_integrand(ao: Callable, density: Callable) -> Callable:

  def lda_vxc_integrand(r) -> Float[Array, "nao nao"]:
    """
    v_xc = -(3/pi n(r))^(1/3)
    Return:
      [2, N, N] array
    """
    ao_vals = ao(r)
    ao_outer = jnp.outer(ao_vals, ao_vals)
    ao_spin = jnp.stack([ao_outer, ao_outer])
    n = einops.rearrange(density(r), "spin -> spin 1 1")
    return -(3 / jnp.pi * n)**(1 / 3) * ao_spin

  return lda_vxc_integrand


def get_lda_vxc(
  grids_and_weights: QuadGridsNWeights,
  cgto: CGTO,
  polarized: bool = False,
) -> Callable:

  def vxc_fn(mo_coeff: MoCoeff) -> Fock:
    mo = lambda r: mo_coeff @ cgto.eval(r)
    density = wave2density(mo, polarized)
    vxc = quadrature_integral(
      get_lda_vxc_integrand(cgto.eval, density), grids_and_weights
    )
    return vxc

  return vxc_fn
