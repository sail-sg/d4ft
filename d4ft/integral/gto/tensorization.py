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
from d4ft.native.obara_saika.eri_kernel import _Hartree_32, _Hartree_64
from d4ft.native.xla.custom_call import CustomCallMeta

Hartree_64 = CustomCallMeta("Hartree_64", (_Hartree_64,), {})
Hartree_32 = CustomCallMeta("Hartree_32", (_Hartree_32,), {})
if jax.config.jax_enable_x64:
  hartree = Hartree_64()
else:
  hartree = Hartree_32()

def tensorize_2c_cgto(f: Callable, static_args, cgto: bool = True):
  """2c centers tensorization with provided index set,
  where the tensor is contracted to cgto.
  Used for incore/precompute.

  Args:
    cgto: if True, contract the tensor into cgto basis
  """

  def f_curry(*args: PGTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(f_curry, in_axes=(0, 0))

  @partial(jax.jit, static_argnames=["n_cgto_segs"])
  def tensorize(
    cgto: CGTO,
    ab_idx_counts: IdxCount2C,
    cgto_seg_id,
    n_cgto_segs,
  ):
    Ns = cgto.N
    ab_idx, counts_ab = ab_idx_counts[:, :2], ab_idx_counts[:, 2]
    pgtos_ab, coeffs_ab = zip(
      *[
        cgto.map_pgto_params(lambda pgto_param, i=i: pgto_param[ab_idx[:, i]])
        for i in range(2)
      ]
    )
    N_ab = Ns[ab_idx].prod(-1) * counts_ab
    t_ab = vmap_f(*pgtos_ab)
    # coeffs_ab = [gtos.coeff[ab_idx[:, i]] for i in range(2)]
    ab = jnp.einsum("k,k,k,k->k", t_ab, N_ab, *coeffs_ab)
    if not cgto:
      return ab
    cgto_ab = jax.ops.segment_sum(ab, cgto_seg_id, n_cgto_segs)
    return cgto_ab

  return tensorize


def tensorize_4c_cgto(f: Callable, static_args, cgto: bool = True):
  """4c centers tensorization with provided index set.
  where the tensor is contracted to cgto.
  Used for incore/precompute.

  Args:
    cgto: if True, contract the tensor into cgto basis
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
        gtos.map_pgto_params(lambda gto_param, i=i: gto_param[abcd_idx[:, i]])
        for i in range(4)
      ]
    )
    t_abcd = vmap_f(*gtos_abcd)
    if not cgto:
      return t_abcd
    counts_abcd_i = idx_counts[:, 4]
    N_abcd = Ns[abcd_idx].prod(-1) * counts_abcd_i
    abcd = jnp.einsum("k,k,k,k,k,k->k", t_abcd, N_abcd, *coeffs_abcd)
    cgto_abcd = jax.ops.segment_sum(abcd, cgto_seg_id, n_segs)
    return cgto_abcd

  return tensorize

def tensorize_4c_cgto_cuda(static_args, cgto: bool = True):
  """4c centers tensorization with provided index set.
  where the tensor is contracted to cgto.
  Used for incore/precompute.

  Args:
    cgto: if True, contract the tensor into cgto basis
  """

  @partial(jax.jit, static_argnames=["n_segs"])
  def tensorize(
    gtos: CGTO,
    idx_counts: IdxCount4C,
    cgto_seg_id,
    n_segs: int,
  ):
    Ns = gtos.N
    N = gtos.n_pgtos
    # Why: Reshape n r z to 1D will significantly reduce computing time
    n = jnp.array(gtos.pgto.angular.T, dtype=jnp.int32)
    r = jnp.array(gtos.pgto.center.T)
    z = jnp.array(gtos.pgto.exponent)
    min_a = jnp.array(static_args.min_a, dtype=jnp.int32)
    min_c = jnp.array(static_args.min_c, dtype=jnp.int32)
    max_ab = jnp.array(static_args.max_ab, dtype=jnp.int32)
    max_cd = jnp.array(static_args.max_cd, dtype=jnp.int32)
    Ms = jnp.array([static_args.max_xyz+1, static_args.max_yz+1, static_args.max_z+1], dtype=jnp.int32)
    abcd_idx = idx_counts[:, :4]
    gtos_abcd, coeffs_abcd = zip(
      *[
        gtos.map_pgto_params(lambda gto_param, i=i: gto_param[abcd_idx[:, i]])
        for i in range(4)
      ]
    )
    t_abcd = hartree(jnp.array([N], dtype=jnp.int32), jnp.array(abcd_idx,dtype=jnp.int32), n, r, z, min_a, min_c, max_ab, max_cd, Ms)
    if not cgto:
      return t_abcd
    counts_abcd_i = idx_counts[:, 4]
    N_abcd = Ns[abcd_idx].prod(-1) * counts_abcd_i
    abcd = jnp.einsum("k,k,k,k,k,k->k", t_abcd, N_abcd, *coeffs_abcd)
    cgto_abcd = jax.ops.segment_sum(abcd, cgto_seg_id, n_segs)
    print(cgto_abcd.shape)
    print(cgto_seg_id.shape)
    print(n_segs)
    return cgto_abcd

  return tensorize


def tensorize_4c_cgto_range(f: Callable, static_args, cgto: bool = True):
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
        gtos.map_pgto_params(lambda gto_param, i=i: gto_param[abcd_idx[:, i]])
        for i in range(4)
      ]
    )
    t_abcd = vmap_f(*gtos_abcd)
    if not cgto:
      return t_abcd
    counts_abcd_i = idx_counts[:, 4]
    N_abcd = Ns[abcd_idx].prod(-1) * counts_abcd_i
    abcd = jnp.einsum("k,k,k,k,k,k->k", t_abcd, N_abcd, *coeffs_abcd)
    cgto_abcd = jax.ops.segment_sum(abcd, cgto_seg_id, n_segs)
    return cgto_abcd

  return tensorize
