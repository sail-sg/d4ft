# Copyright 2023 Garena Online Private Limited
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

from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from d4ft.integral.gto.cgto import PrimitiveGaussian
from d4ft.integral.obara_saika import angular_stats, boys, terms, utils
from d4ft.types import AngularStats


def nuclear_attraction_integral(
  nuclear_center,
  a: PrimitiveGaussian,
  b: PrimitiveGaussian,
  static_args: Optional[AngularStats] = None,
  use_horizontal: bool = False
):
  r"""Nuclear attraction integral using obara saika.

  Recursion formula for NUC (eqn. A19):
      [a,b]^m = pa[i]*[a-1,b  ]^m - pc[i]*[a-1,b  ]^m+1
  + 1/(2*zeta) Na[i]*{[a-2,b  ]^m -       [a-2,b  ]^m+1}
  + 1/(2*zeta) Nb[i]*{[a-1,b-1]^m -       [a-1,b-1]^m+1 }

  The computation strategy is as follows:
  0. compute [0,0]^m for m\in[0,M], where M is the maximum angular
  momentum of x,y,z axis over the batch of GTOs. Store the results into
  the tensor A_0_0 of shape (M,).
  1. vertical recursion 0_b: start from [0,0]^m, compute [0,b]^m
  for all m\in[0,M], b\in[0,B], where B is the maximum angular momentum for
  GTOs in b. Due to symmetry this is the same as computing [a,0]^m.
  Since b=0, the recursion formula becomes
      [a,0]^m = pa[i]*[a-1,0  ]^m - pc[i]*[a-1,0  ]^m+1
  + 1/(2*zeta) Na[i]*{[a-2,0  ]^m -       [a-2,0  ]^m+1}
  Now use the symmetry we get the [0,b]^m recursion rule
      [0,b]^m = pb[i]*[0  ,b-1]^m - pc[i]*[0  ,b-1]^m+1
  + 1/(2*zeta) Nb[i]*{[0  ,b-2]^m -       [0  ,b-2]^m+1}
  2a. vertical recursion a: start from [0,b]^m, compute [a,b]^m
  for all m\in[0,M], b\in[0,B], a\in[0,A]. Uses the full recursion
  formula A19.
  2b. horizontal recursion:

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
  (na, _, _), (nb, _, _) = a, b
  zeta, rp, pa, pb, ab, xi = terms.compute_common_terms(a, b)
  s = static_args or angular_stats.angular_static_args(na, nb)

  C = nuclear_center
  pc = rp - C
  U = zeta * jnp.dot(pc, pc)  # Eqn.A21

  Ms = [s.max_xyz + 1, s.max_yz + 1, s.max_z + 1]
  M = Ms[0]

  def vertical_0_b(i, A_0_0, max_b):
    """Vertical recursion.
    The tensor we deal with here has the shape (C.shape[0], na+nb+1, M)
    """

    def compute_0_b(carry, bm1):
      """
      carry: (A_0_bm2, A_0_bm1), where A_0_bm2=[0,b-2]^m,
        I_0_cm1=[0,b-1]^m are tensor of shape (M,)
      bm1: the current index of b to be computed
      """
      x = carry
      A_0_bm2, A_0_bm1 = x

      A_mp1 = -pc[i] * A_0_bm1 - (bm1 / 2 / zeta) * A_0_bm2
      A_0_b = pb[i] * A_0_bm1 + (bm1 / 2 / zeta) * A_0_bm2
      A_0_b = A_0_b.at[:-1].add(A_mp1[1:])

      return jnp.stack([A_0_bm1, A_0_b], axis=0), A_0_bm1

    init = jnp.stack([jnp.zeros_like(A_0_0), A_0_0], axis=0)
    _, A_0 = utils.py_scan(compute_0_b, init, jnp.arange(0, max_b[i] + 1))
    return A_0

  def vertical_a(i, A_0, max_a):

    def compute_a(carry, am1):
      """
      carry: (A_am2, A_am1), where A_am2=[a-2,b]^m,
        A_am1=[a-1,b]^m are tensor of shape (B, M)
      am1: the current index of a to be computed
      """
      x = carry
      A_am2, A_am1 = x

      A_mp1 = -pc[i] * A_am1 - (am1 / 2 / zeta) * A_am2
      A_a = pa[i] * A_am1 + (am1 / 2 / zeta) * A_am2
      A_a = A_a.at[:, :-1].add(A_mp1[:, 1:])
      A_am1_m_mp1 = A_am1.at[:, :-1].add(-A_am1[:, 1:])
      b_prefac = jnp.arange(1, A_a.shape[0], dtype=float)[:, None] / 2 / zeta
      A_a = A_a.at[1:].add(b_prefac * A_am1_m_mp1[:-1])

      return jnp.stack([A_am1, A_a], axis=0), A_am1

    init = jnp.stack([jnp.zeros_like(A_0), A_0], axis=0)
    _, A = utils.py_scan(compute_a, init, jnp.arange(0, max_a[i] + 1))
    return A

  def horizontal(i, A_0, min_b):
    j = jnp.arange(min_b[i], A_0.shape[0])
    w = lax.select(
      jnp.logical_and(j >= nb[i], j <= na[i] + nb[i]),
      on_true=utils.comb[na[i], j - nb[i]] * (-ab[i])**(na[i] - j + nb[i]),
      on_false=jnp.zeros_like(j, dtype=float),
    )
    return jnp.einsum("a,am->m", w, A_0[min_b[i]:, :])

  prefactor = 2 * (jnp.pi / zeta) * jnp.exp(-xi * jnp.dot(ab, ab))  # Eqn.A20
  A_0_0 = jax.vmap(boys.Boys, in_axes=(0, None))(jnp.arange(M, dtype=int), U)

  if use_horizontal:
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
