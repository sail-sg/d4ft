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
from jaxtyping import Array, Int


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


def get_sto_segment_id(sto_to_gto: tuple) -> Int[Array, "n_gtos"]:
  n_gtos = sum(sto_to_gto)
  sto_seg_len = jnp.cumsum(jnp.array(sto_to_gto))
  seg_id = jnp.argmax(jnp.arange(n_gtos)[:, None] < sto_seg_len, axis=-1)
  return seg_id


@partial(jax.jit, static_argnames=["sto_to_gto", "four_center"])
def get_sto_segment_id_sym(
  gto_idx_counts: Int[Array, "batch three_or_five"],
  sto_to_gto: tuple,
  four_center: bool = False
) -> Int[Array, "batch"]:
  """Compute the segment id for contracting GTO tensor in
  symmetry reduced form to AO/STO tensor using segment_sum.

  For example, for STO-3g, every AO has 3 GTO, so the
  conversion can be computed as sto_idx = gto_idx // 3.
  """
  n_stos = len(sto_to_gto)
  sto_seg_len = jnp.cumsum(jnp.array(sto_to_gto))

  if four_center:
    # translate to sto seg id
    gto_idx_segmented = jnp.argmax(
      gto_idx_counts[:, :4, None] < sto_seg_len, axis=-1
    )
    seg_ids = jax.vmap(lambda ijkl: utr_4c_idx(n_stos, ijkl))(gto_idx_segmented)
  else:  # two center
    # translate to sto seg id
    gto_idx_segmented = jnp.argmax(
      gto_idx_counts[:, :2, None] < sto_seg_len, axis=-1
    )
    seg_ids = jax.vmap(lambda ij: utr_2c_idx(n_stos, ij))(gto_idx_segmented)
  return seg_ids
