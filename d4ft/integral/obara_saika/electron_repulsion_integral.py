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

from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from . import utils

USE_CONV = False
PREALLOCATE = True
PREALLOCATE_ALL = False
PYSCAN = True

if PYSCAN:
  scan_fn = utils.py_scan
else:
  scan_fn = jax.lax.scan


def electron_repulsion_integral(
  a: utils.GTO,
  b: utils.GTO,
  c: utils.GTO,
  d: utils.GTO,
  static_args: Optional[utils.ANGULAR_STATIC_ARGS] = None,
):
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

  Recursion formula for ERI (eqn. 39):
      [a,b,c,d]^m = pa[i]*[a-1,b  ,c  ,d]^m +    wp[i]*[a-1,b  ,c,d  ]^m+1
  + 1/(2*zeta)     Na[i]*{[a-2,b  ,c  ,d]^m - rho/zeta*[a-2,b  ,c,d  ]^m+1 }
  + 1/(2*zeta)     Nb[i]*{[a-1,b-1,c  ,d]^m - rho/zeta*[a-1,b-1,c,d  ]^m+1 }
  + 1/2(zeta+eta) { Nc[i]*[a-1,b  ,c-1,d]^m - Nd[i]   *[a-1,b  ,c,d-1]^m+1 }

  Note that if b=0, the term with b-1 vanishes. This is true for all a,b,c,d.
  Also note that wp[i]=-rho/zeta*pq[i], wq[i]=rho/eta*pq[i]

  The computation strategy is as follows:
  1. compute [0,0,0,0]^m for m\in[0,M], where M is the maximum angular
  momentum of x,y,z axis over the batch of GTOs. Store the results into
  the tensor I_0_0 of shape (M,).
  2. vertical recursion 0_0_c_0: start from [0,0,0,0]^m, compute [0,0,c,0]^m
  for all m\in[0,M], c\in[0,CD], where CD is the maximum angular momentum for
  c,d GTOs. Due to symmetry this is the same as computing [a,0,0,0]^m.
  Since b=c=d=0, the recursion formula becomes
      [a,0,0,0]^m = pa[i]*[a-1,0  ,0  ,0]^m +    wp[i]*[a-1,0  ,0,0  ]^m+1
  + 1/(2*zeta)     Na[i]*{[a-2,0  ,0  ,0]^m - rho/zeta*[a-2,0  ,0,0  ]^m+1 }
  Now use the symmetry we get the [0,0,c,0]^m recursion rule
      [0,0,c,0]^m = qc[i]*[0,0  ,c-1,0]^m +    wq[i]*[0,0  ,c-1,0  ]^m+1
  + 1/(2*eta)      Nc[i]*{[0,0  ,c-2,0]^m -  rho/eta*[0,0  ,c-2,0  ]^m+1 }
  3. vertical recursion a_0_c_0: start from [0,0,c,0]^m, compute [a,0,c,0]^m
  for all m\in[0,M], a\in[0,AB].
  Since b=d=0, the recusion formula is:
      [a,0,c,0]^m = pa[i]*[a-1,0  ,c  ,0]^m +    wp[i]*[a-1,0  ,c,0  ]^m+1
  + 1/(2*zeta)     Na[i]*{[a-2,0  ,c  ,0]^m - rho/zeta*[a-2,0  ,c,0  ]^m+1 }
  + 1/2(zeta+eta)   Nc[i]*[a-1,0  ,c-1,0]^m
  4. horizontal recursion: start from [a,0,c,0]^m, compute [a,b,c,b]^m
  for all m\in[0,M].

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

  zeta, rp, pa, _, ab, _ = utils.compute_common_terms(a, b)
  # eqn.31, eqn.35
  eta, rq, _, _, cd, _ = utils.compute_common_terms(c, d)

  pq = rp - rq
  qc = rq - rc
  pa = rp - ra

  s = static_args or utils.angular_static_args(na, nb, nc, nd)

  Ms = [s.max_xyz + 1, s.max_yz + 1, s.max_z + 1]
  M = Ms[0]

  rho = utils.rho(zeta, eta)
  T = utils.T(rho, pq)

  k_ab = utils.K(za, zb, ra, rb)
  k_cd = utils.K(zc, zd, rc, rd)

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
      carry: (I_0_cm2, I_0_cm1), where I_0_cm2=[0,0,c-2,0]^m,
        I_0_cm1=[0,0,c-1,0]^m are tensor of shape (M,)
      cm1: the current index of c to be computed
      """
      I_0_cm2, I_0_cm1 = carry
      wq_i = rho * pq[i] / eta

      if not USE_CONV:  # DIRECT IMPL
        # recursion terms with m+1
        I_mp1 = wq_i * I_0_cm1 + (-cm1 / 2 / eta * rho / eta) * I_0_cm2
        I_0_c = qc[i] * I_0_cm1 + (cm1 / 2 / eta) * I_0_cm2

        # I_0_c = I_0_c.at[:-1].add(I_mp1[1:])

        # I_mp1 = jnp.pad(I_mp1[1:], pad_width=((0, 1)))
        I_0_c += lax.pad(
          I_mp1[1:], padding_value=0.0, padding_config=((0, 1, 0),)
        )
        # I_0_c = I_0_c + I_mp1

      else:  # CONV BASED IMPL
        x = jnp.stack([I_0_cm2, I_0_cm1], axis=0)  # (C=2, H=M)
        k = jnp.array(  # [2, 2] for (c-2, c-1), (m, m+1)
          [
            [
              cm1 / 2 / eta,  # (m, c-2)
              -cm1 / 2 / eta * rho / eta,  # (m+1, c-2)
            ],
            [
              qc[i],  # (m, c-1)
              wq_i,  # (m+1, c-1)
            ],
          ]
        )

        I_0_c = lax.conv_general_dilated(
          lhs=x[None],
          rhs=k[None],
          window_strides=[1],
          padding=[(0, 1)],
          dimension_numbers=('NCH', 'OIH', 'NCH'),
        )[0, 0]

      return (I_0_cm1, I_0_c), I_0_cm1

    if not PREALLOCATE:
      # Compute I[a=0, c=:, m=:]
      I_0_0m1 = jnp.zeros_like(I_0_0)
      init = (I_0_0m1, I_0_0)  # [M]
      # I_0.shape = [(nc+nd+1), M]
      _, I_0 = scan_fn(
        compute_0_c, init, jnp.arange(0, s.max_cd[i] + 1, dtype=float)
      )

    else:
      # extra one dim
      I_0 = jnp.zeros((s.max_cd[i] + 1, M))
      I_0 = I_0.at[0, :len(I_0_0)].set(I_0_0)
      wq_i = rho * pq[i] / eta
      for cm1 in range(0, s.max_cd[i] + 1):
        cm2 = cm1 - 1  # HACK: for the first iter, this wraps to the back
        c = cm1 + 1
        I_mp1 = wq_i * I_0[cm1] + (-cm1 / 2 / eta * rho / eta) * I_0[cm2]
        I_0_c = qc[i] * I_0[cm1] + (cm1 / 2 / eta) * I_0[cm2]
        I_mp1 = lax.pad(
          I_mp1[1:], padding_value=0.0, padding_config=((0, 1, 0),)
        )
        I_0 = I_0.at[c].set(I_0_c + I_mp1)

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
      carry: (I_am1, I_am2), where I_am1 = [a-1,0,c,0]^m,
      I_am2 = [a-2,0,c,0]^m are tensor of shape (CD, M)
      cm1: the current index of a to be computed
      """
      I_am2, I_am1 = carry
      wp_i = -rho * pq[i] / zeta

      if not USE_CONV:  # DIRECT IMPL
        # recursion terms with m+1
        I_mp1 = wp_i * I_am1 + (-am1 / 2 / zeta * rho / zeta) * I_am2
        I_a = pa[i] * I_am1 + (am1 / 2 / zeta) * I_am2
        # I_a = I_a.at[:, :-1].add(I_mp1[:, 1:])

        # I_mp1 = jnp.pad(I_mp1[:, 1:], pad_width=((0, 0), (0, 1)))
        I_a += lax.pad(
          I_mp1[:, 1:],
          padding_value=0.0,
          padding_config=((0, 0, 0), (0, 1, 0))
        )
        # I_a = I_a + I_mp1

      else:  # CONV BASED IMPL
        x = jnp.stack([I_am2, I_am1], axis=0)
        k = jnp.array([  # [2, 1, 2] for (a-1,a-2), (c), (m, m+1)
          [[
            am1 / 2 / zeta,  # (a-2, c, m )
            -am1 / 2 / zeta * rho / zeta,  # (a-2, c, m+1)
          ]],  # a-2
          [[
            pa[i],  # (a-1, c, m)
            wp_i,  # (a-1 c, m+1)
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

      # [a-1,0,c-1,0]^m
      I_shift = (
        jnp.arange(1, I_am1.shape[0], dtype=float)[:, None] * I_am1[:-1, 1:]
      ) / 2 / (zeta + eta)  # shape=[C, M]

      # I_a = I_a.at[1:, :-1].add(I_shift)
      # I_shift = jnp.pad(I_shift, pad_width=((1, 0), (0, 1)))
      # NOTE: padding_config=[(low, high, interior)]
      I_shift = lax.pad(
        I_shift, padding_value=0.0, padding_config=((1, 0, 0), (0, 1, 0))
      )
      I_a = I_a + I_shift
      return (I_am1, I_a), I_am1

    if not PREALLOCATE:
      I_0m1 = jnp.zeros_like(I_0)
      init = (I_0m1, I_0)
      _, I = scan_fn(
        compute_a, init, jnp.arange(0, s.max_ab[i] + 1, dtype=float)
      )

    else:
      I = jnp.zeros((s.max_ab[i] + 1, s.max_cd[i] + 1, M))
      I = I.at[0].set(I_0)
      wp_i = -rho * pq[i] / zeta
      for am1 in range(0, s.max_ab[i] + 1):
        am2 = am1 - 1  # HACK: for the first iter, this wraps to the back
        a = am1 + 1
        I_mp1 = wp_i * I[am1] + (-am1 / 2 / zeta * rho / zeta) * I[am2]
        I_a = pa[i] * I[am1] + (am1 / 2 / zeta) * I[am2]
        I_mp1 = lax.pad(
          I_mp1[:, 1:],
          padding_value=0.0,
          padding_config=((0, 0, 0), (0, 1, 0))
        )
        I_a = I_a + I_mp1
        # [a-1,0,c-1,0]^m
        I_shift = (
          jnp.arange(1, I[am1].shape[0], dtype=float)[:, None] * I[am1][:-1, 1:]
        ) / 2 / (zeta + eta)  # shape=[C, M]
        I_shift = lax.pad(
          I_shift, padding_value=0.0, padding_config=((1, 0, 0), (0, 1, 0))
        )
        I = I.at[a].set(I_a + I_shift)

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
    mask_a = jnp.logical_and(ja >= na[i], ja <= na[i] + nb[i])
    on_true = utils.comb[nb[i], ja - na[i]] * (ab[i])**(nb[i] - ja + na[i])
    on_false = jnp.zeros_like(ja, dtype=float)
    wa = mask_a * on_true + (1 - mask_a) * on_false
    jc = jnp.arange(s.min_c[i], I.shape[1])
    mask_c = jnp.logical_and(jc >= nc[i], jc <= nc[i] + nd[i])
    on_true = utils.comb[nd[i], jc - nc[i]] * (cd[i])**(nd[i] - jc + nc[i])
    on_false = jnp.zeros_like(jc, dtype=float)
    wc = mask_c * on_true + (1 - mask_c) * on_false
    return jnp.einsum("a,c,acm->m", wa, wc, I[s.min_a[i]:, s.min_c[i]:])

  prefactor = (zeta + eta)**(-1 / 2) * k_ab * k_cd  # Eqn.44
  I_0_0 = jax.vmap(utils.Boys, in_axes=(0, None))(jnp.arange(M), T)

  if not PREALLOCATE_ALL:
    for i in range(3):
      I_0 = vertical_0_0_c_0(i, I_0_0)
      I = vertical_a_0_c_0(i, I_0)
      I_0_0 = horizontal(i, I)

  else:
    Ms = [s.max_xyz + 1, s.max_yz + 1, s.max_z + 1]
    M = s.max_xyz + 1  # max total angular momentum over batch
    I = jnp.zeros((M, M, M))  # (ab, cd, m)

    for i in range(3):
      I = I.at[0, 0, :Ms[i]].set(I_0_0[:Ms[i]])
      I = I.at[0, 0, Ms[i]:].set(0.)  # clear buffer
      wq_i = rho * pq[i] / eta

      # vertical (00|c0)^[m]
      for cm1 in range(0, s.max_cd[i] + 1):
        cm2 = cm1 - 1  # HACK: for the first iter, this wraps to the back
        c = cm1 + 1
        I_mp1 = wq_i * I[0, cm1] + (-cm1 / 2 / eta * rho / eta) * I[0, cm2]
        I_0_c = qc[i] * I[0, cm1] + (cm1 / 2 / eta) * I[0, cm2]
        I_mp1 = lax.pad(
          I_mp1[1:], padding_value=0.0, padding_config=((0, 1, 0),)
        )
        I = I.at[0, c].set(I_0_c + I_mp1)

      wp_i = -rho * pq[i] / zeta

      # vertical (a0|c0)^[m]
      for am1 in range(0, s.max_ab[i] + 1):
        am2 = am1 - 1  # HACK: for the first iter, this wraps to the back
        a = am1 + 1
        I_mp1 = wp_i * I[am1] + (-am1 / 2 / zeta * rho / zeta) * I[am2]
        I_a = pa[i] * I[am1] + (am1 / 2 / zeta) * I[am2]
        I_mp1 = lax.pad(
          I_mp1[:, 1:],
          padding_value=0.0,
          padding_config=((0, 0, 0), (0, 1, 0))
        )
        I_a = I_a + I_mp1
        # [a-1,0,c-1,0]^m
        I_shift = (
          jnp.arange(1, I[am1].shape[0], dtype=float)[:, None] * I[am1][:-1, 1:]
        ) / 2 / (zeta + eta)  # shape=[C, M]
        I_shift = lax.pad(
          I_shift, padding_value=0.0, padding_config=((1, 0, 0), (0, 1, 0))
        )
        I = I.at[a].set(I_a + I_shift)

      # horizontal (a0|c0)^[m] -> (ab|cd)^[m]
      ja = jnp.arange(s.min_a[i], I.shape[0])
      mask_a = jnp.logical_and(ja >= na[i], ja <= na[i] + nb[i])
      on_true = utils.comb[nb[i], ja - na[i]] * (ab[i])**(nb[i] - ja + na[i])
      on_false = jnp.zeros_like(ja, dtype=float)
      wa = mask_a * on_true + (1 - mask_a) * on_false
      jc = jnp.arange(s.min_c[i], I.shape[1])
      mask_c = jnp.logical_and(jc >= nc[i], jc <= nc[i] + nd[i])
      on_true = utils.comb[nd[i], jc - nc[i]] * (cd[i])**(nd[i] - jc + nc[i])
      on_false = jnp.zeros_like(jc, dtype=float)
      wc = mask_c * on_true + (1 - mask_c) * on_false
      # new I_0_0
      I_0_0 = jnp.einsum("a,c,acm->m", wa, wc, I[s.min_a[i]:, s.min_c[i]:])

  result = 0.5 * prefactor * I_0_0[0]

  # NOTE: this is used when padding incomplete batch during dynamic prescreen
  # should be able to remove this
  # return jnp.where(za == -1, 0.0, result)
  return result
