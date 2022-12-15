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
"""Overlap integral using obara saika."""

import jax.numpy as jnp
from jax import lax
from . import utils
from .utils import s_overlap, comb


def overlap_integral(a, b, static_args=None, vh=False):
  r"""Overlap integral using obara saika.

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
    vh: If True, compute with vertical + horizontal recursion.
        Otherwise, compute with vertical vertical recursion.

  Returns:
    integral: The two center overlap integral.
  """
  (na, ra, za), (nb, rb, zb) = a, b
  assert ra.shape == (3,), "do not pass batch data for this function, use vmap."
  if static_args is None:
    static_args = utils.angular_static_args(na, nb)
  s = static_args
  zeta = za + zb
  rp = (za * ra + zb * rb) / zeta
  pa = rp - ra
  pb = rp - rb
  ab = ra - rb

  prefactor = s_overlap(ra, rb, za, zb)

  def vertical_0_b(i, O_0_0, max_b):
    """compute (0|0:max_b), up to max_b."""

    def compute_0_b(carry, bm1):
      O_0_bm2, O_0_bm1 = carry
      O_0_b = pb[i] * O_0_bm1 + 1 / 2 / zeta * bm1 * O_0_bm2
      return (O_0_bm1, O_0_b), O_0_bm1

    init = (0., O_0_0)
    _, O_0 = lax.scan(compute_0_b, init, jnp.arange(max_b[i] + 1))
    return O_0

  def vertical_a(i, O_0, max_a):
    """compute (0:max_a|b), up to max_a."""

    def compute_a(carry, am1):
      O_am2, O_am1 = carry
      O_a = pa[i] * O_am1 + 1 / 2 / zeta * am1 * O_am2
      O_a = O_a.at[1:].add(
        1 / 2 / zeta * jnp.arange(1, O_a.shape[0]) * O_am1[:-1]
      )
      return (O_am1, O_a), O_am1

    init = (jnp.zeros_like(O_0), O_0)
    _, O = lax.scan(compute_a, init, jnp.arange(max_a[i] + 1))
    return O

  def horizontal(i, O_0, min_b):
    j = jnp.arange(min_b[i], O_0.shape[0])
    w = lax.select(
      jnp.logical_and(j >= nb[i], j <= na[i] + nb[i]),
      on_true=comb[na[i], j - nb[i]] * (-ab[i])**(na[i] - j + nb[i]),
      on_false=jnp.zeros_like(j, dtype=float)
    )
    return jnp.dot(w, O_0[min_b[i]:])

  O_0_0 = jnp.array(1.)
  if vh:
    for i in range(3):
      O_0 = vertical_0_b(i, O_0_0, s.max_ab)
      O_0_0 = horizontal(i, O_0, s.min_b)
  else:
    for i in range(3):
      O_0 = vertical_0_b(i, O_0_0, s.max_b)
      O = vertical_a(i, O_0, s.max_a)
      O_0_0 = O[na[i], nb[i]]

  return prefactor * O_0_0
