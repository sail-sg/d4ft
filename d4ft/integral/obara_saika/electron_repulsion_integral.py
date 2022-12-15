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
"""Electron repulsion integral using obara saika."""

import jax
import jax.numpy as jnp
from jax import lax
from . import utils
from .utils import F, comb


def electron_repulsion_integral(a, b, c, d, static_args=None):
  r"""Electron repulsion integral using obara saika
  In this computation, we like to compute the tensor I where
      I.shape = [na+nb+1, nc+nd+1, M], M = max(na+nb+1, nc+nd+1).
      Using (a, c, m) to index the tensor,
      1. We first compute I[a=0, c=0, m=:], this is given by (44).
      2. In `vertical_0_0_c_0` we compute I[a=0, c=:, m=:].
         This is done by the recursive relationship (39), setting c to 0.
         I[a=0, c=i, m=:] = conv(I[a=0, c=i-1, m=:], kernel).
      3. In `vertical_a_0_c_0` compute I[a=:, c=:, m=:], using I[a=0, c=:, m=:]
         (which can be seen as an image) as the input.
         I[a=i, c=:, m=:] = local_connection(I[a=i-1, c=:, m=:], kernel).

  Args:
    a: A tuple of (na, center, exponent),
       `na.dtype == int`, `na.shape == (3)`.
       `center.dtype == float`, `center.shape == (3)`.
       `exponent.dtype == float`, `exponent.shape == ()`.
    b: Similar to a
    c: Similar to a
    d: Similar to a
    static_args: computed from `angular_static_args(na, nb, nc, nd)`,
        here `na, nb, nc, nd` need to be constant numpy array,
        which is not traced by `jax.jit`.
        More often, we will `vmap` this function to compute four center integral
        for all the GTOs in the batch, in this case, the `static_args` needs
        to be computed from batched `na, nb, nc, nd`.

  Returns:
    eri: The four center integral
        `\int a(r1)b(r1)(1/|r1-r2|)c(r2)d(r2) dr1 dr2`.
  """
  (na, ra, za), (nb, rb, zb), (nc, rc, zc), (nd, rd, zd) = a, b, c, d
  assert ra.shape == (3,), "do not pass batch data for this function, use vmap."
  if static_args is None:
    static_args = utils.angular_static_args(na, nb, nc, nd)
  s = static_args
  Ms = [s.max_xyz + 1, s.max_yz + 1, s.max_z + 1]
  M = Ms[0]
  zeta = za + zb
  eta = zc + zd
  rp = (za * ra + zb * rb) / zeta
  rq = (zc * rc + zd * rd) / eta
  ab = ra - rb
  cd = rc - rd
  pq = rp - rq
  qc = rq - rc
  pa = rp - ra
  rho = zeta * eta / (zeta + eta)
  T = rho * jnp.dot(pq, pq)

  def K(z1, z2, r1, r2):
    """equation (47)"""
    d_squared = jnp.dot(r1 - r2, r1 - r2)
    return jnp.sqrt(2) * jnp.pi**(5 / 4) / (z1 + z2) * jnp.exp(
      -z1 * z2 * d_squared / (z1 + z2)
    )

  prefactor = (zeta + eta)**(-1 / 2) * K(za, zb, ra, rb) * K(zc, zd, rc, rd)

  def vertical_0_0_c_0(i, I_0_0):
    """Vertical recursion on c with a=0, b=0 and d=0.
    This function takes a vector that represents I(a=0,b=0,c=0,d=0,m=[0:M])
    and computes the matrix I(a=0,b=0,c=[0:max_cd],d=0,m=[0:M]).

    Args:
      i: index of the coordinate, (0, 1, 2) reprenseting (x, y, z).
      I_0_0: I[a=0, c=0, m=:].
    """
    I_0_0 = I_0_0[:Ms[i]]  # clip to the size that is just enough

    def compute_0_c(carry, cm1):
      """
      carry: (I_0_cm2, I_0_cm1)
      c: the current index of c to be computed
      """
      I_0_cm2, I_0_cm1 = carry
      x = jnp.stack([I_0_cm2, I_0_cm1], axis=0)
      k = jnp.array(  # [2, 2] for (c-2, c-1), (m, m+1)
        [
          [
            cm1 / 2 / eta,  # (m, c-2)
            -cm1 / 2 / eta * rho / eta,  # (m+1, c-2)
          ],
          [
            qc[i],  # (m, c-1)
            rho * pq[i] / eta,  # (m+1, c-1)
          ],
        ]
      )
      I_0_c = lax.conv_general_dilated(
        x[None],
        k[None],
        [1],
        padding=[(0, 1)],
        dimension_numbers=('NCH', 'OIH', 'NCH'),
      )[0, 0]
      return (I_0_cm1, I_0_c), I_0_cm1

    # Compute I[a=0, c=:, m=:]
    I_0_0m1 = jnp.zeros_like(I_0_0)
    init = (I_0_0m1, I_0_0)
    # I_0.shape = [(nc+nd+1), M]
    _, I_0 = lax.scan(
      compute_0_c, init, jnp.arange(0, s.max_cd[i] + 1, dtype=float)
    )
    return I_0

  def vertical_a_0_c_0(i, I_0):
    """Vertical recursion on a with `c=[0:C]`, `b=0` and `d=0`.
    This function takes a matrix of shape `(C, M)` that represents
    `I(a=0,b=0,c=[0:C],d=0,m=[0:M])`. It computes the matrix
    `I(a=[0:max_ab],b=0,c=[0:C],d=0,m=[0:M])`.

    Args:
      i: index of the coordinate, `(0, 1, 2)` reprenseting `(x, y, z)`.
      I_0_0: A `(C, M)` matrix to represent `I(a=0,b=0,c=[0:C],d=0,m=[0:M])`.
      max_cd: Maximum value of c plus d.

    Returns:
      I: A `(max_ab+1, C, M)` matrix.
    """

    def compute_a(carry, am1):
      """
      carry: (I_am1, I_am2)
      """
      I_am2, I_am1 = carry
      x = jnp.stack([I_am2, I_am1], axis=0)
      k = jnp.array([  # [2, 1, 2] for (a-1,a-2), (c), (m, m+1)
        [[
          am1 / 2 / zeta,  # (a-2, c, m )
          -am1 / 2 / zeta * rho / zeta,  # (a-2, c, m+1)
        ]],  # a-2
        [[
          pa[i],  # (a-1, c, m)
          -rho * pq[i] / zeta,  # (a-1 c, m+1)
        ]],  # a-1
      ])
      I_a = lax.conv_general_dilated(
        x[None],
        k[None],
        [1, 1],
        padding=[
          (0, 0),  # padding c
          (0, 1),  # padding m
        ],
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
      )[0, 0]
      I_shift = (
        jnp.arange(1, I_am1.shape[0], dtype=float)[:, None] * I_am1[:-1, 1:]
      ) / 2 / (zeta + eta)  # shape=[C, M]
      I_a = I_a.at[1:, :-1].add(I_shift)
      return (I_am1, I_a), I_am1

    I_0m1 = jnp.zeros_like(I_0)
    init = (I_0m1, I_0)
    _, I = lax.scan(
      compute_a, init, jnp.arange(0, s.max_ab[i] + 1, dtype=float)
    )
    return I

  def horizontal(i, I):
    """Horizontal recursion to transfer from `a` to `b` and `c` to `d`.
    This function takes a matrix of shape `(A, C, M)` that represents
    `I(a=[0:A],b=0,c=[0:C],d=0,m=[0:M])`. It computes the vector
    `I(a=na,b=nb,c=nc,d=nd,m=[0:M])`.

    Args:
      i: index of the coordinate, `(0, 1, 2)` reprenseting `(x, y, z)`.
      I: A `(A, C, M)` matrix to represent `I(a=[0:A],b=0,c=[0:C],d=0,m=[0:M])`.
      min_a: Minimum value of `na`, when integral is computed for a batch
        of GTOs, `min_a` is chosen to be the minimum value of `na` in the
        batch.
      min_c: Similar to `min_a` but for `nc`.

    Returns:
      I: A `(M,)` vector representing `I(a=na,b=nb,c=nc,d=nd,m=[0:M])`.
    """
    ja = jnp.arange(s.min_a[i], I.shape[0])
    wa = lax.select(
      jnp.logical_and(ja >= na[i], ja <= na[i] + nb[i]),
      on_true=comb[nb[i], ja - na[i]] * (ab[i])**(nb[i] - ja + na[i]),
      on_false=jnp.zeros_like(ja, dtype=float)
    )
    jc = jnp.arange(s.min_c[i], I.shape[1])
    wc = lax.select(
      jnp.logical_and(jc >= nc[i], jc <= nc[i] + nd[i]),
      on_true=comb[nd[i], jc - nc[i]] * (cd[i])**(nd[i] - jc + nc[i]),
      on_false=jnp.zeros_like(jc, dtype=float)
    )
    return jnp.einsum("a,c,acm->m", wa, wc, I[s.min_a[i]:, s.min_c[i]:])

  I_0_0 = jax.vmap(F, in_axes=(0, None))(jnp.arange(M), T)
  for i in range(3):
    I_0 = vertical_0_0_c_0(i, I_0_0)
    I = vertical_a_0_c_0(i, I_0)
    I_0_0 = horizontal(i, I)
  return prefactor * I_0_0[0]
