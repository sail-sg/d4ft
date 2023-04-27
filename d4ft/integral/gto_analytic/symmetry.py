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


# TODO: jit this
def get_triu_ij_from_idx_np(N, idx):
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
def triu_idx_range(n: int, start_idx: int, end_idx: int,
                   n_idx_select: int) -> Float[Array, "batch 2"]:
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
def get_2c_sym_idx(n_gtos: int) -> Tuple[Float[Array, "batch 3"]]:
  """2-fold symmetry (a|b)

  Returns:
    ab_idx_count: 2c GTO index concatenated with
      the repetition count of that idx, e.g. (0,1|2)
  """
  ab_idx = jnp.vstack(jnp.triu_indices(n_gtos)).T
  offdiag_ab = ab_idx[:, 0] != ab_idx[:, 1]
  counts_ab = offdiag_ab + jnp.ones(len(ab_idx))
  ab_idx_count = jnp.hstack([ab_idx, counts_ab[..., None]])
  return ab_idx_count


# TODO: benchmark passing ab_idx vs computed on the fly
# TODO: combine this with 4c contraction
@partial(
  jax.jit, static_argnames=["n_2c_idx", "start_idx", "end_idx", "batch_size"]
)
def get_4c_sym_idx_range(
  ab_idx_count: Tuple[Float[Array, "batch 3"]],
  n_2c_idx: int,
  start_idx: int,
  end_idx: int,
  batch_size: int,
) -> Tuple[Float[Array, "batch 5"]]:
  """Get the 4c idx from range (start_idx, end_idx).

  Returns:
    abcd_idx_count: 4c GTO index concatenated with
      the repetition count of that idx, e.g. (0,0,1,0|4)
  """
  ab_idx, counts_ab = ab_idx_count[:, :2], ab_idx_count[:, 2]

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
  abcd_idx_count = jnp.hstack([abcd_idx, counts_abcd[..., None]])
  return abcd_idx_count
