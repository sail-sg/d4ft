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
"""Utility code to be used by integrals."""

import jax
import jax.numpy as jnp
from jax import lax
import scipy.special
import numpy as np
from collections import defaultdict, namedtuple

_r50 = np.arange(50)
comb = jnp.array(scipy.special.comb(_r50[:, None], _r50[None]))

perm_2n_n = jnp.array(scipy.special.perm(2 * _r50, _r50))
factorial = jnp.array(scipy.special.factorial(_r50))


def normalization_constant(angular, exponent):
  return (2 * exponent / np.pi)**(3 / 4) * jnp.sqrt(
    (8 * exponent)**(jnp.sum(angular)) / jnp.prod(perm_2n_n[angular])
  )


def F(m, T):
  """F function TODO: add reference.
  https://www.wolframalpha.com/input?key=&i=int+t%5E%282*m%29*exp%28-T*t%5E2%29+dt+from+0+to+1
  """
  if not jax.config.jax_enable_x64:
    pred = (T < 1e-7)
  else:
    pred = (T < 1e-20)

  def T_is_zero():
    return 1 / (2 * m + 1)

  def T_is_not_zero():
    return (
      1 / 2 * T**(-m - 1 / 2) * jnp.exp(lax.lgamma(m + 1 / 2)) *
      lax.igamma(m + 1 / 2, T)
    )

  return lax.cond(pred, T_is_zero, T_is_not_zero)


def s_overlap(ra, rb, za, zb):
  """Compute the overlap integral between two s orbitals."""
  ab = ra - rb
  zeta = za + zb
  xi = za * zb / zeta
  return (jnp.pi / zeta)**(3 / 2) * jnp.exp(-xi * jnp.dot(ab, ab))


def group_by_angular(angular, *args):
  """Group the GTOs with their angular momentum."""
  angular_to_indice = defaultdict(list)
  for i, ang in enumerate(angular):
    angular_to_indice[tuple(ang)].append(i)
  return [
    (
      np.tile(np.array([ang]),
              (len(indice), 1)), *(a[np.array(indice)] for a in args)
    ) for ang, indice in angular_to_indice.items()
  ]


# The following code extract static information from the angular momentums.


def min_a(*ns):
  n = np.array(ns[0])
  return n.min(0) if len(n.shape) == 2 else n


def min_b(*ns):
  n = np.array(ns[1])
  return n.min(0) if len(n.shape) == 2 else n


def min_c(*ns):
  if len(ns) < 3:
    return None
  n = np.array(ns[2])
  return n.min(0) if len(n.shape) == 2 else n


def min_d(*ns):
  if len(ns) < 4:
    return None
  n = np.array(ns[3])
  return n.min(0) if len(n.shape) == 2 else n


def max_a(*ns):
  n = np.array(ns[0])
  return n.max(0) if len(n.shape) == 2 else n


def max_b(*ns):
  n = np.array(ns[1])
  return n.max(0) if len(n.shape) == 2 else n


def max_c(*ns):
  if len(ns) < 3:
    return None
  n = np.array(ns[2])
  return n.max(0) if len(n.shape) == 2 else n


def max_d(*ns):
  if len(ns) < 4:
    return None
  n = np.array(ns[3])
  return n.max(0) if len(n.shape) == 2 else n


def max_ab(*ns):
  # for either 4 center (abcd) / 2 center (ab), so we just take a and b
  return sum(
    n.max(0) if len(n.shape) == 2 else n for n in map(np.array, ns[:2])
  )


def max_cd(*ns):
  return sum(
    n.max(0) if len(n.shape) == 2 else n for n in map(np.array, ns[2:4])
  )


def max_xyz(*ns):
  """Compute the max of the sum of xyz angular momentum.
  Args:
    ns: list of array of angular momentums.
        Each array has shape `(3,)` or `(batch, 3)`.
  """
  return sum(np.array(n).sum(-1).max() for n in ns)


def max_yz(*ns):
  """Compute the max of the sum of xyz angular momentum.
  Args:
    ns: list of array of angular momentums.
        Each array has shape `(3,)` or `(batch, 3)`.
  """
  return sum(np.array(n)[..., 1:].sum(-1).max() for n in ns)


def max_z(*ns):
  """Compute the max of the sum of xyz angular momentum.
  Args:
    ns: list of array of angular momentums.
        Each array has shape `(3,)` or `(batch, 3)`.
  """
  return sum(np.array(n)[..., 2:].sum(-1).max() for n in ns)


def angular_static_args(*ns):
  return namedtuple(
    "AngularVariables", [
      "min_a",
      "min_b",
      "min_c",
      "min_d",
      "max_a",
      "max_b",
      "max_c",
      "max_d",
      "max_ab",
      "max_cd",
      "max_xyz",
      "max_yz",
      "max_z",
    ]
  )(
    min_a(*ns),
    min_b(*ns),
    min_c(*ns),
    min_d(*ns),
    max_a(*ns),
    max_b(*ns),
    max_c(*ns),
    max_d(*ns),
    max_ab(*ns),
    max_cd(*ns),
    max_xyz(*ns),
    max_yz(*ns),
    max_z(*ns),
  )


GTO = namedtuple("GTO", ["angular", "center", "exponent"])
MO = namedtuple("MO", ["angular", "center", "exponent", "coeff"])


def tensorize(f, num_centers, static_args):
  """Given a function `f`, that computes integral, return a function `g` that
  takes a batch of GTOs and computes integrals for every combination of them.

  Args:
    f: a function that takes either two center or four center inputs and
      compute the integral. The function could have the singature of
      `f(a, b, c, d, static_args)` or `f(a, b, static_args)`.

  Returns:
    g: a function that takes a batch of GTOs, the GTOs are represented by three
      arrays, aka, `angular`, `center` and `exponent`. Each of them have the
      shape `(batch, 3)`, `(batch, 3)` and `(batch,)` respectively.
      The signature is `g(angular, center, exponent)`.
  """

  def g(*gtos: GTO):

    def _f(*args: GTO):
      return f(*args, static_args=static_args)

    vmap_f = _f
    for i in list(range(num_centers))[::-1]:
      in_axes = tuple(0 if j == i else None for j in range(num_centers))
      vmap_f = jax.vmap(vmap_f, in_axes=in_axes)
    return vmap_f(*gtos)

  return g


def contraction_2c(f, static_args):
  """two center contraction.
  """
  f = tensorize(f, 2, static_args)
  N = jax.vmap(normalization_constant)

  def g(a: MO, b: MO):
    gtos = [GTO(*a[:3]), GTO(*b[:3])]
    coeffs = [a[3], b[3]]
    matrix = f(*gtos)
    Na, Nb = N(a.angular, a.exponent), N(b.angular, b.exponent)
    return jnp.einsum("ij,j,jk,k,ik->", coeffs[0], Na, matrix, Nb, coeffs[1])

  return g


def contraction_4c(f, static_args):
  f = tensorize(f, 4, static_args)
  N = jax.vmap(normalization_constant)

  def g(a: MO, b: MO, c: MO, d: MO, has_aux=False):
    """if has_aux is True, then return a tuple of (hartree, exact_exchange)
    """
    gtos = [GTO(*a[:3]), GTO(*b[:3]), GTO(*c[:3]), GTO(*d[:3])]
    coeffs = [a[3], b[3], c[3], d[3]]
    tensor = f(*gtos)
    Na, Nb, Nc, Nd = (
      N(a.angular, a.exponent), N(b.angular, b.exponent),
      N(c.angular, c.exponent), N(d.angular, d.exponent)
    )
    hartree = jnp.einsum(
      "abcd,a,b,c,d,ia,ib,jc,jd->", tensor, Na, Nb, Nc, Nd, *coeffs
    )
    if has_aux:
      exchange = jnp.einsum(
        "abcd,a,b,c,d,ia,jb,ic,jd->", tensor, Na, Nb, Nc, Nd, *coeffs
      )
      return hartree, exchange
    else:
      return hartree

  return g


def contraction(f, num_centers, static_args):
  if num_centers == 2:
    return contraction_2c(f, static_args)
  elif num_centers == 4:
    return contraction_4c(f, static_args)


def enumerate_angular(total_angular):
  """Enumerate all possible angular momentum.
  Args:
    total_angular: the total angular momentum.
  Returns:
    angulars: a list of angular momentum.
  """
  angulars = []
  for x in range(total_angular + 1):
    for y in range(total_angular + 1 - x):
      z = total_angular - x - y
      angulars.append((x, y, z))
  return angulars
