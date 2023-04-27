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
"""Utilities from contracting GTO tensor to STO tensor."""
from functools import partial

import jax
import jax.numpy as jnp


def utr_idx(N, i, j):
  """Recover the enumeration index of upper triangular (i,j)"""
  idx_fn = lambda i, j: (2 * N + 1 - i) * i // 2 + j - i
  return jnp.min(jnp.array([idx_fn(i, j), idx_fn(j, i)]))


def utr_2c_idx(n, ij):
  """Recover the enumeration index of 2c idx (i,j)"""
  i, j = ij
  return utr_idx(n, i, j)


def utr_4c_idx(n, ijkl):
  """Recover the enumeration index of 4c idx (i,j,k,l)"""
  i, j, k, l = ijkl
  N = n * (n + 1) // 2
  ij = utr_idx(n, i, j)
  kl = utr_idx(n, k, l)
  return utr_idx(N, ij, kl)


@partial(jax.jit, static_argnames=["sto_to_gto", "four_center"])
def get_sto_segment_id(gto_idx, sto_to_gto, four_center: bool = False):
  """Compute the segment id for contracting GTO tensor to
  AO/STO tensor using segment_sum.

  For example, for STO-3g, every AO has 3 GTO, so the
  conversion can be computed as sto_idx = gto_idx // 3.

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
