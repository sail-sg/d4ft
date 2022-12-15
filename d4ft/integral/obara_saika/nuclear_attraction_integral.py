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
"""Nuclear attraction integral using obara saika."""

import jax
import jax.numpy as jnp
from jax import lax
from . import utils
from .utils import F, comb


def nuclear_attraction_integral(
  nuclear_center, a, b, static_args=None, vh=False
):
  r"""Nuclear attraction integral using obara saika.

  Args:
    nuclear_center: (3,) array of nuclear center coordinates
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
        with `R` denoting `nuclear_center`, `\int a(r1)(1/|r1-R|)b(r1) dr1`.
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
  C = nuclear_center
  pc = rp - C
  pa = rp - ra
  pb = rp - rb
  U = zeta * jnp.dot(pc, pc)
  prefactor = 2 * (jnp.pi / zeta) * jnp.exp(-xi * jnp.dot(ab, ab))
  Ms = [s.max_xyz + 1, s.max_yz + 1, s.max_z + 1]
  M = Ms[0]
  A_0_0 = jax.vmap(F, in_axes=(0, None))(jnp.arange(M, dtype=float), U)

  def vertical_0_b(i, A_0_0, max_b):
    """Vertical recursion.
    The tensor we deal with here has the shape (C.shape[0], na+nb+1, M)
    """

    def compute_0_b(carry, bm1):
      x = carry
      A_0_bm1 = x[1]
      k = jnp.array([  # [2, 2]
        [
          bm1 / 2 / zeta,  # (m, b-2)
          -bm1 / 2 / zeta,  # (m+1, b-2)
        ],
        [
          pb[i],  # (m, b-1)
          -pc[i],  # (m+1, b-1)
        ],
      ])
      A_0_b = lax.conv_general_dilated(
        x[None],
        k[None],
        [1],
        padding=[(0, 1)],
        dimension_numbers=('NCH', 'OIH', 'NCH'),
      )[0, 0]
      return jnp.stack([A_0_bm1, A_0_b], axis=0), A_0_bm1

    init = jnp.stack([jnp.zeros_like(A_0_0), A_0_0], axis=0)
    _, A_0 = lax.scan(compute_0_b, init, jnp.arange(0, max_b[i] + 1))
    return A_0

  def vertical_a(i, A_0, max_a):

    def compute_a(carry, am1):
      x = carry
      A_am2, A_am1 = x
      k = jnp.array([  # [2, 1, 2]
        [[
          am1 / 2 / zeta,  # (m, b, a-2)
          -am1 / 2 / zeta,  # (m+1, b, a-2)
        ]],
        [[
          pa[i],  # (m, b, a-1)
          -pc[i],  # (m+1, b, a-1)
        ]],
      ])
      A_a = lax.conv_general_dilated(
        x[None],
        k[None],
        [1, 1],
        padding=[(0, 0), (0, 1)],
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
      )[0, 0]
      A_am1_m_mp1 = lax.conv_general_dilated(
        x[1][None, None],  # [Batch, 1, nb, M]
        jnp.array([[[[1., -1.]]]]),
        [1, 1],
        padding=[(1, 0), (0, 1)],
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
      )[0, 0]
      A_a = A_a + (
        jnp.arange(0, A_a.shape[0], dtype=float)[:, None] * A_am1_m_mp1[:-1] /
        2 / zeta
      )
      return jnp.stack([A_am1, A_a], axis=0), A_am1

    init = jnp.stack([jnp.zeros_like(A_0), A_0], axis=0)
    _, A = lax.scan(compute_a, init, jnp.arange(0, max_a[i] + 1))
    return A

  def horizontal(i, A_0, min_b):
    j = jnp.arange(min_b[i], A_0.shape[0])
    w = lax.select(
      jnp.logical_and(j >= nb[i], j <= na[i] + nb[i]),
      on_true=comb[na[i], j - nb[i]] * (-ab[i])**(na[i] - j + nb[i]),
      on_false=jnp.zeros_like(j, dtype=float),
    )
    return jnp.einsum("a,am->m", w, A_0[min_b[i]:, :])

  if vh:
    for i in range(3):
      A_0_0 = A_0_0[:Ms[i]]
      A_0 = vertical_0_b(i, A_0_0, s.max_ab)
      A_0_0 = horizontal(i, A_0, s.min_b)
  else:
    for i in range(3):
      A_0_0 = A_0_0[:Ms[i]]
      A_0 = vertical_0_b(i, A_0_0, s.max_b)
      A = vertical_a(i, A_0, s.max_a)
      A_0_0 = A[na[i], nb[i]]

  return -prefactor * A_0_0[0]
