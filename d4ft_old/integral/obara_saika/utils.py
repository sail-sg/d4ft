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

# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility code to be used by integrals."""

from collections import namedtuple
from enum import Enum
from functools import partial
from typing import Generator

import jax
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import numpy as np
import scipy.special
import tensorflow as tf
from absl import logging
from d4ft.functions import factorization, get_exact_batch
from jax import lax

from . import boys

# TODO: add doc for these
_r50 = np.arange(50)
comb = jnp.array(scipy.special.comb(_r50[:, None], _r50[None]))

perm_2n_n = jnp.array(scipy.special.perm(2 * _r50, _r50))
factorial = jnp.array(scipy.special.factorial(_r50))

GTO = namedtuple("GTO", ["angular", "center", "exponent"])
MO = namedtuple("MO", ["angular", "center", "exponent", "coeff"])


class OrbType(Enum):
  """https://pyscf.org/user/gto.html#basis-set"""
  s = 0
  p = 1
  d = 2
  f = 3


# TODO: d and above orbitals
ANGULAR = {
  OrbType.s: [[0, 0, 0]],
  OrbType.p: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
}


def mol_to_obsa_gto(mol):
  """Transform ao (STOs) to GTOs in obsa format.

  TODO: pass in atom_coords for geometry optimization

  Returns:
    all translated GTOs. STO TO GTO
  """
  obsa_gto = []
  sto_to_gto = []
  for i, element in enumerate(mol.ao.elements):
    coord = mol.ao.atom_coords[i]
    for sto in mol.ao._basis[element]:
      orb_type = OrbType(sto[0])
      gtos = sto[1:]
      for angular in ANGULAR[orb_type]:
        sto_to_gto.append(len(gtos))
        for exponent, coeff in gtos:
          obsa_gto.append((angular, coord, exponent, coeff))
  obsa_gto = MO(*(jnp.array(np.stack(a, axis=0)) for a in zip(*obsa_gto)))
  n_gto = sum(sto_to_gto)
  logging.info(f"there are {n_gto} GTOs")
  return obsa_gto, tuple(sto_to_gto)


def normalization_constant(angular, exponent):
  """eqn.3"""
  return (2 * exponent / np.pi)**(3 / 4) * jnp.sqrt(
    (8 * exponent)**(jnp.sum(angular)) / jnp.prod(perm_2n_n[angular])
  )


def Boys(m, T):
  return lax.cond(m > 26, lambda: BoysNeville(m, T), lambda: BoysPrecomp(m, T))


def BoysIgamma(m, T):
  """Boys function, eqn.45.

  NOTE: This function is deprecated as it is very slow

  Ref..
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


def BoysNeville(m, T):
  """optimized version"""
  ngrids = [100, 400, 1600, 6400]
  tdts = [1. / g for g in ngrids]
  ts = [np.linspace(0, 1, g + 1)[:-1] for g in ngrids]
  t = np.concatenate(ts)

  left_endpoint = (m == 0) * 0.5
  # left_endpoint = lax.cond(m == 0, lambda: 0.5, lambda: 0.0)
  right_endpoint = jnp.exp(-T) / 2

  boys_vals = jnp.exp(-T * t * t) * jnp.power(t, 2 * m)

  idx = np.cumsum([0] + ngrids).tolist()
  tresults = [
    (left_endpoint + jnp.sum(boys_vals[idx[i]:idx[i + 1]]) + right_endpoint) *
    tdts[i] for i in range(len(idx) - 1)
  ]

  # tresults = (left_endpoint + tresults + right_endpoint) * tdts

  # Neville polynomial extrapolate them to zero step.
  tresult01 = (-tdts[1]) * (tresults[0] - tresults[1]) / (tdts[0] -
                                                          tdts[1]) + tresults[1]
  tresult12 = (-tdts[2]) * (tresults[1] - tresults[2]) / (tdts[1] -
                                                          tdts[2]) + tresults[2]
  tresult23 = (-tdts[3]) * (tresults[2] - tresults[3]) / (tdts[2] -
                                                          tdts[3]) + tresults[3]
  tresult012 = (-tdts[2]) * (tresult01 -
                             tresult12) / (tdts[0] - tdts[2]) + tresult12
  tresult123 = (-tdts[3]) * (tresult12 -
                             tresult23) / (tdts[1] - tdts[3]) + tresult23
  result = (-tdts[3]) * (tresult012 -
                         tresult123) / (tdts[0] - tdts[3]) + tresult123

  return result


BoysFuns = jnp.array(boys.BoysFuns)
BoysAsympoticConstant = jnp.array(boys.BoysAsympoticConstant)


def BoysPrecomp(m, T):
  pred = T > 27.

  def small_T():
    idx0 = (T * 100).astype(int)
    x0 = idx0 / 100.0
    y0 = BoysFuns[m, idx0]
    idx1 = idx0 + 1
    x1 = x0 + 0.01
    y1 = BoysFuns[m, idx1]
    idx2 = idx0 + 2
    x2 = x0 + 0.02
    y2 = BoysFuns[m, idx2]
    idx3 = idx0 + 3
    x3 = x0 + 0.03
    y3 = BoysFuns[m, idx3]
    idx4 = idx0 + 4
    x4 = x0 + 0.04
    y4 = BoysFuns[m, idx4]

    # Neville 5-point interpolation.
    y01 = (T - x1) * (y0 - y1) / (x0 - x1) + y1
    y12 = (T - x2) * (y1 - y2) / (x1 - x2) + y2
    y23 = (T - x3) * (y2 - y3) / (x2 - x3) + y3
    y34 = (T - x4) * (y3 - y4) / (x3 - x4) + y4
    y012 = (T - x2) * (y01 - y12) / (x0 - x2) + y12
    y123 = (T - x3) * (y12 - y23) / (x1 - x3) + y23
    y234 = (T - x4) * (y23 - y34) / (x2 - x4) + y34
    y0123 = (T - x3) * (y012 - y123) / (x0 - x3) + y123
    y1234 = (T - x4) * (y123 - y234) / (x1 - x4) + y234
    y01234 = (T - x4) * (y0123 - y1234) / (x0 - x4) + y1234

    # Downward recursion is not needed.
    return y01234

  def large_T():
    return BoysAsympoticConstant[m] * jnp.power(T, -m - 0.5)

  small = jnp.nan_to_num((1 - pred) * small_T())
  large = jnp.nan_to_num(pred * large_T())
  return large + small
  # return lax.cond(pred, large_T, small_T)
  # return small_T()


# for ease of reference, here we define some term defined in the paper


def xi(za, zb):
  """z are exponents. Ref eqn.13."""
  return za * zb / zeta(za, zb)


def zeta(za, zb):
  """z are exponents. Ref eqn.14."""
  return za + zb


def rg(ra, rb, rc, za, zb, zc):
  """z are exponents, r are centers. Ref eqn.16."""
  return (za * ra + zb * rb + zc * rc) / (za + zb + zc)


def rp(ra, rb, za, zb):
  """z are exponents, r are centers. Ref eqn.15."""
  return rg(ra, rb, 0.0, za, zb, 0.0)


def rho(zeta, eta):
  """eqn.32"""
  return zeta * eta / (zeta + eta)


def T(rho, pq):
  """eqn.46"""
  return rho * jnp.dot(pq, pq)


def K(z1, z2, r1, r2):
  """eqn.47"""
  d_squared = jnp.dot(r1 - r2, r1 - r2)
  return jnp.sqrt(2) * jnp.pi**(5 / 4) / (z1 + z2) * jnp.exp(
    -z1 * z2 * d_squared / (z1 + z2)
  )


def compute_common_terms(a, b):
  (_, ra, za), (_, rb, zb) = a, b
  assert ra.shape == (3,), "do not pass batch data for this function, use vmap."
  rp_ = rp(ra, rb, za, zb)
  pa = rp_ - ra
  pb = rp_ - rb
  ab = ra - rb
  return zeta(za, zb), rp_, pa, pb, ab, xi(za, zb)


def s_overlap(ra, rb, za, zb):
  """Compute the overlap integral between two s orbitals.

  Ref. TABLE VI, (s||s)

  Args:
    ra, rb: centers
    za, zb: exponents
  """
  ab2 = jnp.linalg.norm(ra - rb, ord=2)**2
  return (jnp.pi / zeta(za, zb))**(3 / 2) * jnp.exp(-xi(za, zb) * ab2)


ANGULAR_STATIC_ARGS = namedtuple(
  "AngularVariables", [
    "min_a", "min_b", "min_c", "min_d", "max_a", "max_b", "max_c", "max_d",
    "max_ab", "max_cd", "max_xyz", "max_yz", "max_z"
  ]
)


def angular_static_args(*ns) -> ANGULAR_STATIC_ARGS:
  """Compute static args for angular momentums.

  Args:
    ns: list of angular momentum vectors, where the length
      is assumed to be between 2 to 4. The vectors are indexed
      alphabetically, e.g. for 4 ns we have (na, nb, nc, nd).
      Each array has shape `(3,)` or `(batch, 3)`.
  """

  def min_max_over_batch(i, *ns):
    """min and max angular momentum in each dimension over the batch"""
    if len(ns) < i + 1:
      return None, None
    n = np.array(ns[i])
    return (n.min(0), n.max(0)) if len(n.shape) == 2 else (n, n)

  def max_over_dim(dims, *ns):
    """max angular momentum of some spatial dimensions over the batch"""
    return sum(np.array(n)[..., dims].sum(-1).max() for n in ns)

  # min/max angular mometum for one group, in each axis
  min_a, max_a = min_max_over_batch(0, *ns)
  min_b, max_b = min_max_over_batch(1, *ns)
  min_c, max_c = min_max_over_batch(2, *ns)
  min_d, max_d = min_max_over_batch(3, *ns)
  max_ab = max_a + max_b
  max_cd = None if max_c is None else max_c + max_d
  max_xyz = max_over_dim([0, 1, 2], *ns)
  max_yz = max_over_dim([1, 2], *ns)
  max_z = max_over_dim([2], *ns)

  return ANGULAR_STATIC_ARGS(
    min_a, min_b, min_c, min_d, max_a, max_b, max_c, max_d, max_ab, max_cd,
    max_xyz, max_yz, max_z
  )


def py_scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, jnp.stack(ys)


###############################################################
# CONTRACTION CORE
###############################################################


def unique_ij(n):
  """Number of unique ij indices under the 2-fold symmetry.

  equivalent to number of upper triangular elements,
  including the diagonal
  """
  return int(n * (n + 1) / 2)


def unique_ijkl(n):
  """Number of unique ijlk indices under the 8-fold symmetry.

  equivalent to
  int(1 / 8 * (n**4 + 2 * n**3 + 3 * n**2 + 2 * n))
  """
  return unique_ij(unique_ij(n))


def shuffle_combs(
  n_combs: int,
  batch_size: int,
  key: jnp.array,
):
  comb_idx = jnp.arange(n_combs, dtype=int)
  shuffled_comb_idx = jax.random.permutation(key, comb_idx)

  # make batches
  batch_size = min(n_combs, batch_size)
  num_batches = n_combs // batch_size
  batched = jnp.split(shuffled_comb_idx[:num_batches * batch_size], num_batches)
  return list(batched)


def make_4c_batches(
  n_combs: int,
  batch_size_upperbound: int,
  epochs: int,
  seed: int = 137,
) -> Generator:
  factors = sorted(factorization(n_combs))
  batch_size = get_exact_batch(factors, batch_size_upperbound)

  @jax.jit
  def shuffle_fn(key):
    return shuffle_combs(n_combs, batch_size, key)

  rng = jax.random.PRNGKey(seed)
  key, rng = jax.random.split(rng)

  for _ in range(epochs):
    key, rng = jax.random.split(rng)
    for batch in shuffle_fn(key):
      yield batch


def make_4c_batches_sampled(
  schwartz_bound,
  num_batches: int,
  batch_size: int,
  epochs: int,
  seed: int = 137,
  imp_sampling: bool = True,
) -> Generator:
  rng = jax.random.PRNGKey(seed)
  n_data = len(schwartz_bound)
  bound = jnp.log(schwartz_bound)
  for _ in range(epochs):
    for _ in range(num_batches):
      key, rng = jax.random.split(rng)
      if imp_sampling:
        batch = jax.random.categorical(key, bound, shape=(batch_size,))
      else:  # uniform
        batch = jax.random.randint(
          key, (batch_size,), minval=0, maxval=n_data - 1
        )
      yield batch


@partial(jax.jit, static_argnames=['n_gtos'])
def get_2c_combs(n_gtos: int):
  ab_idx = jnp.vstack(jnp.triu_indices(n_gtos)).T
  offdiag_ab = ab_idx[:, 0] != ab_idx[:, 1]
  counts_ab = offdiag_ab + jnp.ones(len(ab_idx))
  return ab_idx, counts_ab


@partial(
  jax.jit, static_argnames=["n_2c_idx", "start_idx", "end_idx", "batch_size"]
)
def get_4c_combs_range(
  ab_idx, counts_ab, n_2c_idx, start_idx, end_idx, batch_size
):
  # block idx of (ab|cd)
  ab_block_idx = triu_idx(n_2c_idx, start_idx, end_idx, batch_size)
  ab_block_idx = jnp.vstack(ab_block_idx).T

  offdiag_ab_block = ab_block_idx[:, 0] != ab_block_idx[:, 1]
  counts_ab_block = offdiag_ab_block + jnp.ones(len(ab_block_idx))
  in_block_counts = (
    counts_ab[ab_block_idx[:, 0]] * counts_ab[ab_block_idx[:, 1]]
  )
  between_block_counts = counts_ab_block

  counts_abcd = in_block_counts * between_block_counts
  counts_abcd = counts_abcd.astype(jnp.int32)

  abcd_idx = jnp.hstack(
    [ab_idx[ab_block_idx[:, 0]], ab_idx[ab_block_idx[:, 1]]]
  )
  abcd_idx_count = jnp.hstack([abcd_idx, counts_abcd[..., None]])
  return abcd_idx_count


@partial(jax.jit, static_argnames=['n_gtos', 'dynamic_prescreen'])
def get_4c_combs(n_gtos: int, dynamic_prescreen: bool = False):
  ab_idx, counts_ab = get_2c_combs(n_gtos)

  # block idx of (ab|cd)
  ab_block_idx = jnp.vstack(jnp.triu_indices(len(ab_idx))).T
  offdiag_ab_block = ab_block_idx[:, 0] != ab_block_idx[:, 1]
  counts_ab_block = offdiag_ab_block + jnp.ones(len(ab_block_idx))
  in_block_counts = (
    counts_ab[ab_block_idx[:, 0]] * counts_ab[ab_block_idx[:, 1]]
  )
  between_block_counts = counts_ab_block

  counts_abcd = in_block_counts * between_block_counts
  counts_abcd = counts_abcd.astype(jnp.int32)

  abcd_idx = jnp.hstack(
    [ab_idx[ab_block_idx[:, 0]], ab_idx[ab_block_idx[:, 1]]]
  )
  if not dynamic_prescreen:
    return abcd_idx, counts_abcd
  else:
    return (
      ab_idx, counts_ab, abcd_idx, counts_abcd, ab_block_idx, offdiag_ab_block
    )


def get_4c_combs_alt(n_gtos: int):
  ij_indices = np.vstack(np.tril_indices(n_gtos)).T
  # ijkl_indices = []
  lkji_indices = []
  perm = np.array([3, 2, 1, 0])
  counts = []
  for ij_idx in ij_indices:
    i, j = ij_idx
    ij_counts = 2 if i != j else 1
    kl_indices = np.vstack(np.tril_indices(i + 1)).T
    # lexicographic ordering kl<ij
    kl_indices = kl_indices[np.where(
      np.logical_or(kl_indices[:, 0] != i, kl_indices[:, 1] <= j)
    )]
    kl_counts = (kl_indices[:, 0] != kl_indices[:, 1]).astype(int) + 1
    between_block_counts = np.logical_and(
      kl_indices[:, 0] != i, kl_indices[:, 1] != j
    ).astype(int) + 1
    ijkl = np.hstack([ij_idx[None].repeat(len(kl_indices), 0), kl_indices])
    # ijkl_indices.append(ijkl)
    lkji_indices.append(ijkl[:, perm])
    counts_i = ij_counts * kl_counts * between_block_counts
    counts.append(counts_i)
  return np.vstack(lkji_indices), np.hstack(counts)
  # return np.vstack(ijkl_indices), np.hstack(counts)


def contraction_2c_sym(f, n_gtos, static_args):
  """2c contraction with 2-fold symmetries."""

  def _f(*args: GTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(_f, in_axes=(0, 0))

  N = jax.vmap(normalization_constant)

  ab_idx, counts_ab = get_2c_combs(n_gtos)

  def g(mo: MO):
    Ns = N(mo.angular, mo.exponent)
    gtos_ab = [
      GTO(*map(lambda gto_param: gto_param[ab_idx[:, i]], mo[:3]))
      for i in range(2)
    ]
    N_ab = Ns[ab_idx].prod(-1) * counts_ab
    t_ab = vmap_f(*gtos_ab)
    c_lm = jnp.einsum("il,im->lm", mo.coeff, mo.coeff)
    c_ab = c_lm[ab_idx[:, 0], ab_idx[:, 1]]
    ab = jnp.einsum("k,k,k->", t_ab, N_ab, c_ab)
    # coeffs_ab = [a[3][:, ab_idx[:, i]] for i in range(2)]
    # c_ab = jnp.einsum("k,k,ik,ik->", t_ab, N_ab, *coeffs_ab)
    return ab

  return g


def contraction_4c_sym(f, n_gtos, static_args):
  """4c contraction with 8-fold symmetries."""

  def _f(*args: GTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(_f, in_axes=(0, 0, 0, 0))

  N = jax.vmap(normalization_constant)

  abcd_idx, counts_abcd = get_4c_combs(n_gtos)

  def g(mo: MO):
    Ns = N(mo.angular, mo.exponent)
    gtos_abcd = [
      GTO(*map(lambda gto_param: gto_param[abcd_idx[:, i]], mo[:3]))
      for i in range(4)
    ]
    N_abcd = Ns[abcd_idx].prod(-1) * counts_abcd
    t_abcd = vmap_f(*gtos_abcd)  # (len(abcd_idx_prescreen),)
    c_lm = jnp.einsum("il,im->lm", mo.coeff, mo.coeff)
    c_ab = c_lm[abcd_idx[:, 0], abcd_idx[:, 1]]
    c_cd = c_lm[abcd_idx[:, 2], abcd_idx[:, 3]]
    abcd = jnp.einsum("k,k,k,k->", t_abcd, N_abcd, c_ab, c_cd)
    # coeffs_abcd = [mo.coeff[:, abcd_idx[:, i]] for i in range(4)]
    # abcd = jnp.einsum("k,k,ik,ik,jk,jk->", t_abcd, N_abcd, *coeffs_abcd)
    return abcd

  return g


###############################################################
# PRECOMPUTED SPARSE TENSORIZATION / STO CONTRACTION
###############################################################


def get_tril_ij_from_idx(idx):
  i = int(np.sqrt(2 * idx + .25) - .5 + 1e-7)
  j = int(idx - (i * (i + 1) // 2))
  return i, j


def get_triu_ij_from_idx(N, idx):
  a = 1
  b = -1 * (2 * N + 1)
  c = 2 * idx
  i = int((-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a))
  j = int(idx - (2 * N + 1 - i) * i // 2 + i)
  return i, j


@partial(jax.jit, static_argnames=['max_size'])
def select_elements(array, mask, max_size):
  """jittable indexing"""
  array = array.reshape(-1)
  mask = mask.reshape(-1)
  n_ele = array.shape[0]
  indices = jnp.arange(n_ele)
  selected_indices = jnp.where(mask, indices, n_ele - 1)
  sorted_indices = jnp.sort(selected_indices)
  filtered_indices = lax.dynamic_slice(sorted_indices, (0,), (max_size,))
  ret = array[filtered_indices]
  return ret


@partial(jax.jit, static_argnames=['n', 'start_idx', 'end_idx', 'n_idx_select'])
def triu_idx(n, start_idx, end_idx, n_idx_select):
  """generate upper triangular index range from start to end
  Args:
    start_idx, end_idx: (i,j) of the start and end.
  Can be calculated from get_triu_ij_from_idx
  """
  start_i, start_j = start_idx
  end_i, end_j = end_idx
  xi = lax.iota(np.int32, end_i + 1)[start_i:]
  yi = lax.iota(np.int32, n + 1)[1:]
  start = start_j - start_i
  end = start + n_idx_select
  n_idx = end + (n - end_j)
  tri_ = ~jax.vmap(jax.vmap(jnp.greater_equal, (None, 0)), (0, None))(xi, yi)
  idx = tuple(
    select_elements(inds, tri_, n_idx) for inds in jnp.indices(tri_.shape)
  )
  x, y = tuple(ind[start:end] for ind in idx)
  return x + start_i, y


def triu_idx_np(n, start_idx, end_idx):
  start_i, start_j = get_triu_ij_from_idx(n, start_idx)
  if end_idx is None:
    end_i, end_j = n, n
  else:
    end_i, end_j = get_triu_ij_from_idx(n, end_idx)

  tri_ = ~np.greater_equal.outer(
    jnp.arange(start_i, end_i + 1, dtype=jnp.int32),
    jnp.arange(1, n + 1, dtype=jnp.int32)
  ).astype(
    bool, copy=False
  )
  # idx = tuple(
  #   jnp.broadcast_to(inds, tri_.shape)[tri_]
  #   for inds in jnp.indices(tri_.shape, sparse=True)
  # )
  idx = tuple(inds[tri_] for inds in jnp.indices(tri_.shape))
  n_idx = len(idx[0])
  start = start_j - start_i
  end = n_idx - (n - end_j)
  x, y = tuple(ind[start:end] for ind in idx)
  return x + start_i, y


def utr_idx(N, i, j):
  """recover upper triangular index"""
  idx_fn = lambda i, j: (2 * N + 1 - i) * i // 2 + j - i
  return jnp.min(jnp.array([idx_fn(i, j), idx_fn(j, i)]))


def utr_2c_idx(n, ij):
  i, j = ij
  return utr_idx(n, i, j)


def utr_4c_idx(n, ijkl):
  """recover 4c index"""
  i, j, k, l = ijkl
  N = n * (n + 1) // 2
  ij = utr_idx(n, i, j)
  kl = utr_idx(n, k, l)
  return utr_idx(N, ij, kl)


def utr_idx_alt(i, j):
  """recover upper triangular index"""
  return j * (j + 1) // 2 + i


def utr_4c_idx_alt(ijkl):
  """recover 4c index"""
  i, j, k, l = ijkl
  ij = utr_idx_alt(i, j)
  kl = utr_idx_alt(k, l)
  return utr_idx_alt(ij, kl)


@partial(jax.jit, static_argnames=["sto_to_gto", "four_center"])
def get_sto_segment_id_alt(gto_idx, sto_to_gto, four_center: bool = True):
  """
  Args:
    gto_idx: shape (N, 2) or (N, 4)
  """
  n_stos = len(sto_to_gto)
  sto_seg_len = jnp.cumsum(jnp.array(sto_to_gto))

  # translate to sto seg id
  gto_idx_segmented = jnp.argmax(gto_idx[:, :, None] < sto_seg_len, axis=-1)

  if four_center:
    seg_ids = jax.vmap(lambda ijkl: utr_4c_idx_alt(ijkl))(gto_idx_segmented)
  else:
    seg_ids = jax.vmap(lambda ij: utr_2c_idx(n_stos, ij))(gto_idx_segmented)
  return seg_ids


@partial(jax.jit, static_argnames=["sto_to_gto", "four_center"])
def get_sto_segment_id(gto_idx, sto_to_gto, four_center: bool = False):
  """
  Args:
    gto_idx: shape (N, 2) or (N, 4)
  """
  n_stos = len(sto_to_gto)
  sto_seg_len = jnp.cumsum(jnp.array(sto_to_gto))

  # translate to sto seg id
  gto_idx_segmented = jnp.argmax(gto_idx[:, :, None] < sto_seg_len, axis=-1)

  if four_center:
    seg_ids = jax.vmap(lambda ijkl: utr_4c_idx(n_stos, ijkl))(gto_idx_segmented)
  else:
    seg_ids = jax.vmap(lambda ij: utr_2c_idx(n_stos, ij))(gto_idx_segmented)
  return seg_ids


def tensorize_2c_sto(f, static_args, sto: bool = True):
  """2c centers tensorization with provided index set,
  where the tensor is contracted to sto basis.
  Used for precompute.
  """

  def _f(*args: GTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(_f, in_axes=(0, 0))

  def g(gto: MO, Ns, ab_idx, counts_ab, sto_seg_id, n_segs):
    gtos_ab = [
      GTO(*map(lambda gto_param: gto_param[ab_idx[:, i]], gto[:3]))
      for i in range(2)
    ]
    N_ab = Ns[ab_idx].prod(-1) * counts_ab
    t_ab = vmap_f(*gtos_ab)
    coeffs_ab = [gto.coeff[ab_idx[:, i]] for i in range(2)]
    ab = jnp.einsum("k,k,k,k->k", t_ab, N_ab, *coeffs_ab)
    if not sto:
      return ab
    sto_ab = jax.ops.segment_sum(ab, sto_seg_id, n_segs)
    return sto_ab

  return g


def tensorize_4c_sto(
  f, full_batch_size, static_args, sto: bool = True, screen: bool = False
):
  """4c centers tensorization with provided index set.
  where the tensor is contracted to sto basis.
  Used for precompute.
  """

  def _f(*args: GTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(_f, in_axes=(0, 0, 0, 0))

  def g(gto: MO, Ns, idx_count, sto_seg_id, n_segs):
    abcd_idx = idx_count[:, :4]
    gtos_abcd = [
      GTO(*map(lambda gto_param: gto_param[abcd_idx[:, i]], gto[:3]))
      for i in range(4)
    ]
    t_abcd = vmap_f(*gtos_abcd)
    if not sto:
      return t_abcd
    counts_abcd_i = idx_count[:, -1]
    N_abcd = Ns[abcd_idx].prod(-1) * counts_abcd_i
    coeffs_abcd = [gto.coeff[abcd_idx[:, i]] for i in range(4)]
    abcd = jnp.einsum("k,k,k,k,k,k->k", t_abcd, N_abcd, *coeffs_abcd)
    sto_abcd = jax.ops.segment_sum(abcd, sto_seg_id, n_segs)
    return sto_abcd

    # def g(gto: MO, Ns, idx_count, sto_seg_id, n_segs):

    #   def calc_one_idx(idx_count):
    #     abcd_idx = idx_count[:4]
    #     gtos_abcd = [
    #       GTO(*map(lambda gto_param: gto_param[abcd_idx[i]], gto[:3]))
    #       for i in range(4)
    #     ]
    #     t_abcd = _f(*gtos_abcd)
    #     if not sto:
    #       return t_abcd
    #     N_abcd = Ns[abcd_idx].prod(-1)
    #     if not screen:
    #       counts_abcd_i = idx_count[-1]
    #       N_abcd *= counts_abcd_i
    #     ca, cb, cc, cd = [gto.coeff[abcd_idx[i]] for i in range(4)]
    #     abcd_i = t_abcd * N_abcd * ca * cb * cc * cd
    #     return abcd_i

    #   abcd = minibatch_vmap(
    #     calc_one_idx,
    #     in_axes=0,
    #     full_batch_size=full_batch_size,
    #     batch_size=16384,
    #   )(
    #     idx_count
    #   )

    #   if not sto:
    #     return abcd

    #   sto_abcd = jax.ops.segment_sum(abcd, sto_seg_id, n_segs)

    # return sto_abcd

  return g


###############################################################
# PRESCREEN COMPUTED WITH A SEPARATE JITTED FUNCTION
###############################################################


def prescreen_4c(f, n_gtos, static_args, tau: float = 1e-8):
  """do not jit this function"""

  def _f(*args: GTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(_f, in_axes=(0, 0, 0, 0))

  ab_idx, _, abcd_idx, counts_abcd, ab_block_idx, _ = get_4c_combs(
    n_gtos, dynamic_prescreen=True
  )

  # vmap_f = minibatch_vmap(
  #   _f,
  #   in_axes=(0, 0, 0, 0),
  #   full_batch_size=len(ab_idx),
  #   batch_size=16384,
  # )

  @jax.jit
  def compute_mask(a: MO):
    # 1. calculate the partial diagonal, i.e. (ab|ab) entries
    abab_idx = jnp.hstack([ab_idx, ab_idx])
    gtos_abab = [
      GTO(*map(lambda gto_param: gto_param[abab_idx[:, i]], a[:3]))
      for i in range(4)
    ]
    t_abab = vmap_f(*gtos_abab)  # (len(ab_idx),)

    # 2. prescreen, get the index of 4c to calculate
    schwartz_bound = jnp.sqrt(
      t_abab[ab_block_idx[:, 0]] * t_abab[ab_block_idx[:, 1]]
    )
    prescreen_mask = schwartz_bound > tau

    return prescreen_mask

  def compute_abcd_idx_count(mask):
    return jnp.hstack([abcd_idx, counts_abcd[:, None]])[mask].astype(int)

  return compute_mask, compute_abcd_idx_count


def contraction_4c_selected(f, n_gtos, static_args):
  """4c centers contraction with provided index set.
  Used together with precomputed prescreen or sampled index set.
  """

  def _f(*args: GTO):
    return f(*args, static_args=static_args)

  # full_batch_size = unique_ijkl(n_gtos)

  # vmap_f = minibatch_vmap(
  #   _f, in_axes=(0, 0, 0, 0), full_batch_size=full_batch_size,
  #   batch_size=16384
  # )

  vmap_f = jax.vmap(_f, in_axes=(0, 0, 0, 0))

  N = jax.vmap(normalization_constant)

  def g(mo: MO, idx_count):
    Ns = N(mo.angular, mo.exponent)
    c_lm = jnp.einsum("il,im->lm", mo.coeff, mo.coeff)

    abcd_idx = idx_count[:, :4]
    counts_abcd_i = idx_count[:, -1]
    gtos_abcd = [
      GTO(*map(lambda gto_param: gto_param[abcd_idx[:, i]], mo[:3]))
      for i in range(4)
    ]
    N_abcd = Ns[abcd_idx].prod(-1) * counts_abcd_i
    t_abcd = vmap_f(*gtos_abcd)
    c_ab = c_lm[abcd_idx[:, 0], abcd_idx[:, 1]]
    c_cd = c_lm[abcd_idx[:, 2], abcd_idx[:, 3]]
    abcd = jnp.einsum("k,k,k,k->", t_abcd, N_abcd, c_ab, c_cd)
    return abcd

    # coeffs_abcd = [mo.coeff[:, abcd_idx[:, i]] for i in range(4)]
    # t_abcd = jnp.einsum("k,k,ik,ik,jk,jk->", t_abcd, N_abcd, *coeffs_abcd)
    # t_abcd *= n_gtos**4 / counts_abcd_i.sum()  # scale correction

    # def calc_one_idx(idx_count):
    #   abcd_idx = idx_count[:4]
    #   counts_abcd_i = idx_count[-1]
    #   gtos_abcd = [
    #     GTO(*map(lambda gto_param: gto_param[abcd_idx[i]], mo[:3]))
    #     for i in range(4)
    #   ]
    #   t_abcd = _f(*gtos_abcd)
    #   N_abcd = Ns[abcd_idx].prod(-1) * counts_abcd_i
    #   c_ab = c_lm[abcd_idx[0], abcd_idx[1]]
    #   c_cd = c_lm[abcd_idx[2], abcd_idx[3]]
    #   abcd_i = t_abcd * N_abcd * c_ab * c_cd
    #   return abcd_i

    # abcd = jax.vmap(
    #   calc_one_idx,
    #   in_axes=0,
    # )(
    #   idx_count
    # )
    # abcd = jnp.sum(abcd)

    # abcd = minibatch_vmap(
    #   calc_one_idx,
    #   in_axes=0,
    #   full_batch_size=full_batch_size,
    #   batch_size=16384 * 16,
    #   reduce=True,
    # )(
    #   idx_count
    # )

    # return abcd

  return g


###############################################################
# DYNAMIC PRESCREEN IN A SINGLE JITTED FUNCTION USING TFDT HCB
###############################################################


def get_prescreened_4c_idx_dataset(
  batch_size: int,
  n: int,
  abab,
  counts_ab,
  threshold: float = 1e-8,
):
  abab = tf.convert_to_tensor(abab)

  counts_ab = tf.convert_to_tensor(counts_ab.astype(int))
  N = n * (n + 1) // 2  # block idx range
  # construct idx dt
  ab_idx = tf.convert_to_tensor(np.triu_indices(n))
  a_idx, b_idx = ab_idx
  # No diag idx
  block_triu_idx_dt = tf.data.Dataset.range(N).flat_map(
    lambda x: tf.data.Dataset.range(x, N).map(lambda y: (x, y))
  )
  idx_dt = block_triu_idx_dt.map(
    lambda ab, cd: tf.stack([a_idx[ab], b_idx[ab], a_idx[cd], b_idx[cd]])
  )
  count_dt = block_triu_idx_dt.map(
    lambda ab, cd: tf.cast(counts_ab[ab] * counts_ab[cd], tf.int32) *
    (1 + tf.cast(ab != cd, tf.int32))
  )
  # mask_dt = block_triu_idx_dt.map(
  #   lambda ab, cd: tf.logical_and(
  #     tf.math.sqrt(abab[ab] * abab[cd]) > threshold, tf.not_equal(ab, cd)
  #   )
  # )
  mask_dt = block_triu_idx_dt.map(
    lambda ab, cd: tf.math.sqrt(abab[ab] * abab[cd]) > threshold
  )

  # combine idx, count and mask
  dt = tf.data.Dataset.zip((idx_dt, count_dt, mask_dt)).map(
    lambda idx, count, mask: tf.concat(
      [idx,
       tf.reshape(count, (1,)),
       tf.reshape(tf.cast(mask, tf.int32), (1,))], 0
    )
  )
  # filter dataset by mask
  dt = dt.filter(lambda d: tf.equal(d[-1], 1))
  dt = dt.map(lambda d: d[:-1])  # remove mask
  dt = dt.batch(batch_size).padded_batch(
    1, padded_shapes=(batch_size, 5), padding_values=-1
  ).map(lambda d: d[0])
  return dt


def get_prescreened_4c_idx_dataset_old(
  batch_size: int,
  n: int,
  prescreen_mask,
  counts_abcd,
):
  N = n * (n + 1) // 2  # block idx range
  # construct idx dt
  ab_idx = tf.convert_to_tensor(np.triu_indices(n))
  a_idx, b_idx = ab_idx
  block_triu_idx_dt = tf.data.Dataset.range(N).flat_map(
    lambda x: tf.data.Dataset.range(x, N).map(lambda y: (x, y))
  )
  idx_dt = block_triu_idx_dt.map(
    lambda ab, cd: tf.stack([a_idx[ab], b_idx[ab], a_idx[cd], b_idx[cd]])
  )
  count_dt = tf.data.Dataset.from_tensor_slices(counts_abcd)
  mask_dt = tf.data.Dataset.from_tensor_slices(prescreen_mask.astype(jnp.int32))
  # combine idx, count and mask
  dt = tf.data.Dataset.zip((idx_dt, count_dt, mask_dt)).map(
    lambda idx, count, mask: tf.
    concat([idx, tf.reshape(count, (1,)),
            tf.reshape(mask, (1,))], 0)
  )
  # filter dataset by mask
  dt = dt.filter(lambda d: tf.equal(d[-1], 1))
  dt = dt.map(lambda d: d[:-1])  # remove mask
  # NOTE: to ensure all batch has the same size, pad the last
  # incomplete batch with -1 which will be handled by the eri function
  dt = dt.batch(batch_size).padded_batch(
    1, padded_shapes=(batch_size, 5), padding_values=-1
  ).map(lambda d: d[0])
  return dt


# host dt
dt_4c = None


def host_create_dt_4c(*args):

  def create_dt_4c_(args, transforms=None):
    global dt_4c
    dt_4c = iter(get_prescreened_4c_idx_dataset(*args))

  hcb.id_tap(create_dt_4c_, args)


def host_create_dt_4c_old(*args):

  def create_dt_4c_(args, transforms=None):
    global dt_4c
    dt_4c = iter(get_prescreened_4c_idx_dataset_old(*args))

  hcb.id_tap(create_dt_4c_, args)


def host_get_dt_4c(i, batch_size):
  """
  Args:
    i: loop var. This is needed as otherwise scan will only call
       this function once.
    batch_size: needed to determine result_shape
  """

  def f_(_):
    global dt_4c
    idx_count_opt = dt_4c.get_next_as_optional()

    if idx_count_opt.has_value():
      return idx_count_opt.get_value().numpy()
    else:
      return -1 * np.ones((batch_size, 5)).astype(int)

  return hcb.call(
    f_, arg=i, result_shape=jax.ShapeDtypeStruct((batch_size, 5), jnp.int32)
  )


def contraction_4c_dynamic_prescreen_old(
  f, n_gtos, static_args, batch_size_upperbound: int = 1000, tau: float = 1e-8
):
  """4c contraction with 8-fold symmetry and dynamic prescreen.

  Prescreen is is offloaded to TF dataset.

  TODO: the energy is lower than the correct value. fix it
  """

  def _f(*args: GTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(_f, in_axes=(0, 0, 0, 0))

  N = jax.vmap(normalization_constant)

  ret = get_4c_combs(n_gtos, dynamic_prescreen=True)
  ab_idx, counts_ab, _, counts_abcd, ab_block_idx, offdiag_ab_block = ret

  dt_size = len(ab_block_idx)
  factors = sorted(factorization(dt_size))
  batch_size = get_exact_batch(factors, batch_size_upperbound)
  num_shards = dt_size // batch_size

  def g(mo: MO):
    Ns = N(mo.angular, mo.exponent)

    # pad gto params
    # the incomplete batch will return ijkl idx with -1
    # in this case the pad will be retrieved
    # ERI function will handle these fake gtos
    gto_params = mo[:3]
    pad_fake = lambda gto_param: jnp.concatenate(
      [gto_param, -1 * jnp.ones_like(gto_param[:1])]
    )
    gto_params = list(map(pad_fake, gto_params))

    # 1. calculate the partial diagonal, i.e. (ab|ab) entries
    abab_idx = jnp.hstack([ab_idx, ab_idx])
    gtos_abab = [
      GTO(*map(lambda gto_param: gto_param[abab_idx[:, i]], mo[:3]))
      for i in range(4)
    ]
    c_lm = jnp.einsum("il,im->lm", mo.coeff, mo.coeff)
    c_ab = c_lm[abab_idx[:, 0], abab_idx[:, 1]]
    N_abab = Ns[abab_idx].prod(-1) * counts_ab
    t_abab = vmap_f(*gtos_abab)  # (len(ab_idx),)
    abab = jnp.einsum("k,k,k,k->", t_abab, N_abab, c_ab, c_ab)
    # coeffs_abab = [mo.coeff[:, abab_idx[:, i]] for i in range(4)]
    # abab = jnp.einsum("k,k,ik,ik,jk,jk->", t_abab, N_abab, *coeffs_abab)

    # TODO: move this inside of dataset
    # 2. prescreen, get the index of 4c to calculate
    prescreen_mask = jnp.sqrt(
      t_abab[ab_block_idx[:, 0]] * t_abab[ab_block_idx[:, 1]]
    ) > tau  # (unique_ijkl, )

    # 3. calculate prescreened off-diagonal (ab|cd)
    # don't include diag entries as already calculated in step 1
    mask_abcd = offdiag_ab_block * prescreen_mask
    # TODO: can this be traced
    host_create_dt_4c_old(batch_size, n_gtos, mask_abcd, counts_abcd)

    def compute_abcd(carry, i):
      idx_count = host_get_dt_4c(i, batch_size)
      abcd_idx = idx_count[:, :4]
      counts_abcd_i = idx_count[:, -1]
      gtos_abcd = [
        GTO(*map(lambda gto_param: gto_param[abcd_idx[:, i]], mo[:3]))
        for i in range(4)
      ]
      N_abcd = Ns[abcd_idx].prod(-1) * counts_abcd_i
      t_abcd = vmap_f(*gtos_abcd)
      c_ab_i = c_lm[abcd_idx[:, 0], abcd_idx[:, 1]]
      c_cd_i = c_lm[abcd_idx[:, 2], abcd_idx[:, 3]]
      abcd_i = jnp.einsum("k,k,k,k->", t_abcd, N_abcd, c_ab_i, c_cd_i)
      # coeffs_abcd = [mo.coeff[:, abcd_idx[:, i]] for i in range(4)]
      # abcd_i = jnp.einsum("k,k,ik,ik,jk,jk->", t_abcd, N_abcd, *coeffs_abcd)
      return carry + abcd_i, None

    abcd, _ = jax.lax.scan(compute_abcd, 0.0, jnp.arange(0, num_shards))

    return abab + abcd

  return g


def contraction_4c_dynamic_prescreen(
  f, n_gtos, static_args, batch_size: int = 16384, threshold: float = 1e-8
):
  """4c contraction with 8-fold symmetry and dynamic prescreen.

  Prescreen is is offloaded to TF dataset.

  TODO: the energy is lower than the correct value. fix it
  """

  def _f(*args: GTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(_f, in_axes=(0, 0, 0, 0))

  N = jax.vmap(normalization_constant)

  ab_idx, counts_ab = get_2c_combs(n_gtos)

  # dt_size_upperbound = unique_ijkl(n_gtos)
  # num_batches = dt_size_upperbound // batch_size

  # TODO: fix batch size logic
  dt_size = unique_ijkl(n_gtos)
  factors = sorted(factorization(dt_size))
  batch_size = get_exact_batch(factors, 1000)
  num_batches = dt_size // batch_size

  def g(mo: MO):
    Ns = N(mo.angular, mo.exponent)

    # pad gto params
    # the incomplete batch will return ijkl idx with -1
    # in this case the pad will be retrieved
    # ERI function will handle these fake gtos
    gto_params = mo[:3]
    pad_fake = lambda gto_param: jnp.concatenate(
      [gto_param, -1 * jnp.ones_like(gto_param[:1])]
    )
    gto_params = list(map(pad_fake, gto_params))

    # 1. calculate the block diagonal, i.e. (ab|ab) entries
    abab_idx = jnp.hstack([ab_idx, ab_idx])
    gtos_abab = [
      GTO(*map(lambda gto_param: gto_param[abab_idx[:, i]], mo[:3]))
      for i in range(4)
    ]
    c_lm = jnp.einsum("il,im->lm", mo.coeff, mo.coeff)
    t_abab = vmap_f(*gtos_abab)  # (len(ab_idx),)
    # c_ab = c_lm[abab_idx[:, 0], abab_idx[:, 1]]
    # N_abab = Ns[abab_idx].prod(-1) * counts_ab
    # abab = jnp.einsum("k,k,k,k->", t_abab, N_abab, c_ab, c_ab)

    # 2. prescreen, get the index of 4c to calculate
    host_create_dt_4c(batch_size, n_gtos, t_abab, counts_ab, threshold)

    def compute_abcd(carry, i):
      idx_count = host_get_dt_4c(i, batch_size)
      abcd_idx = idx_count[:, :4]

      counts_abcd_i = idx_count[:, -1]
      gtos_abcd = [
        GTO(*map(lambda gto_param: gto_param[abcd_idx[:, i]], mo[:3]))
        for i in range(4)
      ]
      N_abcd = Ns[abcd_idx].prod(-1) * counts_abcd_i
      t_abcd = vmap_f(*gtos_abcd)
      c_ab_i = c_lm[abcd_idx[:, 0], abcd_idx[:, 1]]
      c_cd_i = c_lm[abcd_idx[:, 2], abcd_idx[:, 3]]
      abcd_i = jnp.einsum("k,k,k,k->", t_abcd, N_abcd, c_ab_i, c_cd_i)
      # coeffs_abcd = [mo.coeff[:, abcd_idx[:, i]] for i in range(4)]
      # abcd_i = jnp.einsum("k,k,ik,ik,jk,jk->", t_abcd, N_abcd, *coeffs_abcd)
      return carry + abcd_i, None

    abcd, _ = jax.lax.scan(compute_abcd, 0.0, jnp.arange(0, num_batches))

    # return abab + abcd
    return abcd

  return g
