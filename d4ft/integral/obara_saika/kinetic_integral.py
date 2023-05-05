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
"""Kinetic integral using obara saika."""

from typing import Optional

import jax.numpy as jnp
from jax import lax

from d4ft.integral.gto.cgto import PrimitiveGaussian
from d4ft.integral.obara_saika import angular_stats, terms
from d4ft.types import AngularStats


def kinetic_integral(
  a: PrimitiveGaussian,
  b: PrimitiveGaussian,
  static_args: Optional[AngularStats] = None,
  use_horizontal: bool = False
):
  r"""kinetic integral using obara saika.

  TODO: add horizontal recursion

  Recursion formula for KIN (eqn. A12):
      [a,b]  = pa[i]*[a-1,b  ]
  + 1/(2*zeta) Na[i]*[a-2,b  ]
  + 1/(2*zeta) Nb[i]*[a-1,b-1]
  + 2*xi { (a,b) - 1/(2*za)(a-2,b)}

  where the overlap integral is denoted as (a,b).

  Args:
    a: A tuple of (na, center, exponent),
       `na.dtype == int`, `na.shape == (3)`.
       `center.dtype == float`, `center.shape == (3)`.
       `exponent.dtype == float`, `exponent.shape == ()`.
    b: Similar to a
    static_args: computed from `angular_static_args(na, nb)`,
        here `na, nb` need to be constant numpy array,
        which is not traced by `jax.jit`.
        More often, we will `vmap` this function to compute integrals
        for all the GTOs in the batch, in this case, the `static_args` needs
        to be computed from batched `na, nb`.

  Returns:
    integral: The two center kinetic integral,
  """
  (na, ra, za), (nb, rb, zb) = a, b
  zeta, _, pa, pb, ab, xi = terms.compute_common_terms(a, b)
  s = static_args or angular_stats.angular_static_args(na, nb)

  def vertical_0_b(i, T_0_0, O_0_0, max_b):
    """compute (0|T|0:max_b[i]), up to max_b[i]."""

    def compute_0_b(carry, bm1):
      T_0_bm2, T_0_bm1, O_0_bm2, O_0_bm1 = carry
      O_0_b = pb[i] * O_0_bm1 + 1 / 2 / zeta * bm1 * O_0_bm2
      T_0_b = (
        pb[i] * T_0_bm1 + 1 / 2 / zeta * bm1 * T_0_bm2 + 2 * xi * O_0_b -
        xi / zb * bm1 * O_0_bm2
      )
      return (T_0_bm1, T_0_b, O_0_bm1, O_0_b), (T_0_bm1, O_0_bm1)

    init = (0., T_0_0, 0., O_0_0)
    _, (T_0, O_0) = lax.scan(compute_0_b, init, jnp.arange(max_b[i] + 1))
    return T_0, O_0

  def vertical_a(i, T_0, O_0, max_a):
    """compute (0:max_a[i]|T|b), up to max_a[i]."""

    def compute_a(carry, am1):
      """Ref eqn.A12"""
      T_am2, T_am1, O_am2, O_am1 = carry
      O_a = pa[i] * O_am1 + 1 / 2 / zeta * am1 * O_am2
      O_a = O_a.at[1:].add(
        1 / 2 / zeta * jnp.arange(1, O_am1.shape[0]) * O_am1[:-1]
      )
      T_a = (
        pa[i] * T_am1 + 1 / 2 / zeta * am1 * T_am2 + 2 * xi * O_a -
        xi / za * am1 * O_am2
      )
      T_a = T_a.at[1:].add(
        1 / 2 / zeta * jnp.arange(1, O_am1.shape[0]) * T_am1[:-1]
      )
      return (T_am1, T_a, O_am1, O_a), (T_am1, O_am1)

    init = (jnp.zeros_like(T_0), T_0, jnp.zeros_like(O_0), O_0)
    _, (T, O) = lax.scan(compute_a, init, jnp.arange(max_a[i] + 1))
    return T[na[i], nb[i]], O[na[i], nb[i]]

  prefactor = terms.s_overlap(ra, rb, za, zb)
  T_0_0 = xi * (3 - 2 * xi * jnp.dot(ab, ab))
  O_0_0 = 1.

  for i in range(3):
    T_0, O_0 = vertical_0_b(i, T_0_0, O_0_0, s.max_b)
    T_0_0, O_0_0 = vertical_a(i, T_0, O_0, s.max_a)

  return prefactor * T_0_0
