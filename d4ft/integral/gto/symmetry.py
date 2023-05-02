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
from jax import lax
from jaxtyping import Array, Float

from d4ft.types import IdxCount2C, IdxCount4C

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
# ij, count = sym_2c.unique_ij(return_count=True)
# Or
# ij, count = sym_2c[start:end].unique_ij(return_count=True)
# If the end is out of range, we can just set ij to (0,0) and count to 0.
# As such, the last batch can be padded to same size, and has not adverse effect.


# TODO: name it to num_unique_ij ?
def unique_ij(n):
  """Number of unique ij indices under the 2-fold symmetry.

  equivalent to number of upper triangular elements,
  including the diagonal
  """
  return int(n * (n + 1) / 2)


# TODO: num_unique_ijkl ?
def unique_ijkl(n):
  """Number of unique ijlk indices under the 8-fold symmetry.

  equivalent to
  int(1 / 8 * (n**4 + 2 * n**3 + 3 * n**2 + 2 * n))
  """
  return unique_ij(unique_ij(n))


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
