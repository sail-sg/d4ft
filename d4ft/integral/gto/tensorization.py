# Copyright 2023 Garena Online Private Limited
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

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from d4ft.integral.gto import symmetry
from d4ft.integral.gto.cgto import CGTO, PGTO
from d4ft.types import IdxCount2C, IdxCount4C


def tensorize_2c_cgto(f: Callable, static_args, sto: bool = True):
  """2c centers tensorization with provided index set,
  where the tensor is contracted to sto basis.
  Used for incore/precompute.

  Args:
    sto: if True, contract the tensor into sto basis
  """

  def f_curry(*args: PGTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(f_curry, in_axes=(0, 0))

  @partial(jax.jit, static_argnames=["n_cgto_segs"])
  def tensorize(
    gtos: CGTO,
    ab_idx_counts: IdxCount2C,
    cgto_seg_id,
    n_cgto_segs,
  ):
    Ns = gtos.N
    ab_idx, counts_ab = ab_idx_counts[:, :2], ab_idx_counts[:, 2]
    gtos_ab, coeffs_ab = zip(
      *[
        gtos.map_params(lambda gto_param, i=i: gto_param[ab_idx[:, i]])
        for i in range(2)
      ]
    )
    N_ab = Ns[ab_idx].prod(-1) * counts_ab
    t_ab = vmap_f(*gtos_ab)
    # coeffs_ab = [gtos.coeff[ab_idx[:, i]] for i in range(2)]
    ab = jnp.einsum("k,k,k,k->k", t_ab, N_ab, *coeffs_ab)
    if not sto:
      return ab
    cgto_ab = jax.ops.segment_sum(ab, cgto_seg_id, n_cgto_segs)
    return cgto_ab

  return tensorize


def tensorize_4c_cgto(f: Callable, static_args, sto: bool = True):
  """4c centers tensorization with provided index set.
  where the tensor is contracted to sto basis.
  Used for incore/precompute.

  Args:
    sto: if True, contract the tensor into sto basis
  """

  def f_curry(*args: PGTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(f_curry, in_axes=(0, 0, 0, 0))

  @partial(jax.jit, static_argnames=["n_segs"])
  def tensorize(
    gtos: CGTO,
    idx_counts: IdxCount4C,
    cgto_seg_id,
    n_segs: int,
  ):
    Ns = gtos.N
    abcd_idx = idx_counts[:, :4]
    gtos_abcd, coeffs_abcd = zip(
      *[
        gtos.map_params(lambda gto_param, i=i: gto_param[abcd_idx[:, i]])
        for i in range(4)
      ]
    )
    t_abcd = vmap_f(*gtos_abcd)
    if not sto:
      return t_abcd
    counts_abcd_i = idx_counts[:, 4]
    N_abcd = Ns[abcd_idx].prod(-1) * counts_abcd_i
    abcd = jnp.einsum("k,k,k,k,k,k->k", t_abcd, N_abcd, *coeffs_abcd)
    cgto_abcd = jax.ops.segment_sum(abcd, cgto_seg_id, n_segs)
    return cgto_abcd

  return tensorize


def tensorize_4c_cgto_range(f: Callable, static_args, sto: bool = True):
  """Currently not used.
  This brings marginal speed up compared to tensorize_4c_cgto"""

  def f_curry(*args: PGTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(f_curry, in_axes=(0, 0, 0, 0))

  @partial(
    jax.jit,
    static_argnames=[
      "n_2c_idx", "start_idx", "end_idx", "batch_size", "n_segs"
    ]
  )
  def tensorize(
    gtos: CGTO,
    ab_idx_counts: IdxCount2C,
    n_2c_idx: int,
    start_idx: int,
    end_idx: int,
    batch_size: int,
    n_segs: int,
  ):
    idx_counts = symmetry.get_4c_sym_idx_range(
      ab_idx_counts, n_2c_idx, start_idx, end_idx, batch_size
    )
    cgto_seg_id = symmetry.get_cgto_segment_id_sym(
      idx_counts, gtos.cgto_splits, four_center=True
    )

    Ns = gtos.N
    abcd_idx = idx_counts[:, :4]
    gtos_abcd, coeffs_abcd = zip(
      *[
        gtos.map_params(lambda gto_param, i=i: gto_param[abcd_idx[:, i]])
        for i in range(4)
      ]
    )
    t_abcd = vmap_f(*gtos_abcd)
    if not sto:
      return t_abcd
    counts_abcd_i = idx_counts[:, 4]
    N_abcd = Ns[abcd_idx].prod(-1) * counts_abcd_i
    abcd = jnp.einsum("k,k,k,k,k,k->k", t_abcd, N_abcd, *coeffs_abcd)
    cgto_abcd = jax.ops.segment_sum(abcd, cgto_seg_id, n_segs)
    return cgto_abcd

  return tensorize
