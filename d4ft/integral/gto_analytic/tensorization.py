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
"""2C and 4C tensorization with symmetry"""

import jax
import jax.numpy as jnp


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

  return g
