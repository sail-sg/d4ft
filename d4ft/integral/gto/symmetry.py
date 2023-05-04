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
"""Utility code to compute symmetry reduced index for GTO integrals.
2c integrals (a|b) has 2-fold symmetry, whereas 4c integrals (ab|cd)
has 8-fold symmetry.
"""
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from d4ft.types import IdxCount, IdxCount2C, IdxCount4C
from jax import lax
from jaxtyping import Array, Float, Int

# TODO: An idea, can we use API like
# sym_2c = Sym2C(n_gtos)
# uniq_idx = sym_2c.encode(i, j)
# i, j = sym_2c.decode(uniq_idx)
# sym_2c.repetition_count(i, j)
# This class itself is an iterator, we can do
#
# for i, j in sym_2c:
#   pass
#
# for i, j, k, l in sym_4c:
#   pass

# to make batches of calculations, we can slice
# for b in range(np.ceil(len(sym_2c) / batch_size)):
#   for i, j in sym_2c[b * batch_size: (b + 1) * batch_size]:
#     pass

# Or if we don't want iterator, we can call
# ij, count = sym_2c.num_unique_ij(return_count=True)
# Or
# ij, count = sym_2c[start:end].num_unique_ij(return_count=True)
# If the end is out of range, we can just set ij to (0,0) and count to 0.
# As such, the last batch can be padded to same size, and has not adverse
# effect.


def num_unique_ij(n):
  """Number of unique ij indices under the 2-fold symmetry.

  equivalent to number of upper triangular elements,
  including the diagonal
  """
  return int(n * (n + 1) / 2)


def num_unique_ijkl(n):
  """Number of unique ijlk indices under the 8-fold symmetry.

  equivalent to
  int(1 / 8 * (n**4 + 2 * n**3 + 3 * n**2 + 2 * n))
  """
  return num_unique_ij(num_unique_ij(n))


# TODO: jit this
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
def triu_idx_range(
  n: int, start_idx: Tuple[int, int], end_idx: Tuple[int, int],
  n_idx_select: int
) -> Float[Array, "batch 2"]:
  """Generate upper triangular index range from start to end.

  Args:
    start_idx, end_idx: (i,j) of the start and end. Can be calculated
      from get_triu_ij_from_idx
    n_idx_select: to make this function jittable, n_idx_select need to be
      concrete at tracing time
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
  idx = jnp.vstack([x + start_i, y]).T
  return idx


@partial(jax.jit, static_argnames=['n_gtos'])
def get_2c_sym_idx(n_gtos: int) -> IdxCount2C:
  """2-fold symmetry (a|b)"""
  ab_idx = jnp.vstack(jnp.triu_indices(n_gtos)).T
  offdiag_ab = ab_idx[:, 0] != ab_idx[:, 1]
  counts_ab = offdiag_ab + jnp.ones(len(ab_idx))
  ab_idx_count = jnp.hstack([ab_idx, counts_ab[..., None]]).astype(int)
  return ab_idx_count


@partial(jax.jit, static_argnames=['n_gtos'])
def get_4c_sym_idx(n_gtos: int) -> IdxCount4C:
  ab_idx_counts = get_2c_sym_idx(n_gtos)
  ab_idx, counts_ab = ab_idx_counts[:, :2], ab_idx_counts[:, 2]

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
  abcd_idx_counts = jnp.hstack([abcd_idx, counts_abcd[..., None]]).astype(int)
  return abcd_idx_counts


# TODO: benchmark passing ab_idx vs computed on the fly
# TODO: combine this with 4c contraction
@partial(
  jax.jit, static_argnames=["n_2c_idx", "start_idx", "end_idx", "batch_size"]
)
def get_4c_sym_idx_range(
  ab_idx_counts: IdxCount2C,
  n_2c_idx: int,
  start_idx: int,
  end_idx: int,
  batch_size: int,
) -> IdxCount4C:
  """Get the 4c idx from range (start_idx, end_idx)."""
  ab_idx, counts_ab = ab_idx_counts[:, :2], ab_idx_counts[:, 2]

  # block idx of (ab|cd)
  ab_block_idx = triu_idx_range(n_2c_idx, start_idx, end_idx, batch_size)

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
  abcd_idx_count = jnp.hstack([abcd_idx, counts_abcd[..., None]]).astype(int)
  return abcd_idx_count


def utr_idx(n: int, i: Int[Array, ""], j: Int[Array, ""]) -> Int[Array, ""]:
  """Recover the enumeration index of upper triangular (i,j)"""
  idx_fn = lambda i, j: (2 * n + 1 - i) * i // 2 + j - i
  return jnp.min(jnp.array([idx_fn(i, j), idx_fn(j, i)]))


def utr_2c_idx(n: int, ij: Int[Array, "2"]) -> Int[Array, ""]:
  """Recover the enumeration index of 2c idx (i,j)"""
  i, j = ij
  return utr_idx(n, i, j)


def utr_4c_idx(n: int, ijkl: Int[Array, "4"]) -> Int[Array, ""]:
  """Recover the enumeration index of 4c idx (i,j,k,l)"""
  i, j, k, l = ijkl
  N = n * (n + 1) // 2
  ij = utr_idx(n, i, j)
  kl = utr_idx(n, k, l)
  return utr_idx(N, ij, kl)


@partial(jax.jit, static_argnames=["cgto_splits", "four_center"])
def get_cgto_segment_id_sym(
  gto_idx_counts: IdxCount,
  cgto_splits: tuple,
  four_center: bool = False
) -> Int[Array, "batch"]:
  """Compute the segment id for contracting GTO tensor in
  symmetry reduced form to AO/STO tensor using segment_sum.

  For example, for STO-3g, every AO has 3 GTO, so the
  conversion can be computed as cgto_idx = gto_idx // 3.
  """
  n_cgtos = len(cgto_splits)
  cgto_seg_len = jnp.cumsum(jnp.array(cgto_splits))

  if four_center:
    # translate to sto seg id
    gto_idx_segmented = jnp.argmax(
      gto_idx_counts[:, :4, None] < cgto_seg_len, axis=-1
    )
    seg_ids = jax.vmap(lambda ijkl: utr_4c_idx(n_cgtos, ijkl))(
      gto_idx_segmented
    )
  else:  # two center
    # translate to sto seg id
    gto_idx_segmented = jnp.argmax(
      gto_idx_counts[:, :2, None] < cgto_seg_len, axis=-1
    )
    seg_ids = jax.vmap(lambda ij: utr_2c_idx(n_cgtos, ij))(gto_idx_segmented)
  return seg_ids
