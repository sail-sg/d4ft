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

import jax.numpy as jnp
from jax import lax
from . import utils
from .utils import s_overlap


def kinetic_integral(a, b, static_args=None):
  r"""kinetic integral using obara saika.

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
    integral: The two center nuclear attraction integral,
        with `R` denoting `nuclear_center`, `\int a(r1)(-nabla^2)b(r1) dr1`.
  """
  na, ra, za = a
  nb, rb, zb = b
  assert ra.shape == (3,), "do not pass batch data for this function, use vmap."
  if static_args is None:
    static_args = utils.angular_static_args(na, nb)
  s = static_args
  zeta = za + zb
  xi = za * zb / zeta
  rp = (za * ra + zb * rb) / zeta
  ab = ra - rb
  pa = rp - ra
  pb = rp - rb

  prefactor = s_overlap(ra, rb, za, zb)
  T_0_0 = xi * (3 - 2 * xi * jnp.dot(ab, ab))
  O_0_0 = 1.

  def vertical_0_b(i, T_0_0, O_0_0, max_b):

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

    def compute_a(carry, am1):
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
    return T[-1, -1], O[-1, -1]

  for i in range(3):
    T_0, O_0 = vertical_0_b(i, T_0_0, O_0_0, s.max_b)
    T_0_0, O_0_0 = vertical_a(i, T_0, O_0, s.max_a)

  return prefactor * T_0_0
