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
from d4ft.native.obara_saika.eri_kernel import _Hartree_32, _Hartree_32_uncontracted, _Hartree_64, _Hartree_64_uncontracted
from d4ft.native.xla.custom_call import CustomCallMeta
from d4ft.utils import get_rdm1
from d4ft.integral import obara_saika as obsa

# from jax.interpreters import ad, batching, mlir, xla
Hartree_64 = CustomCallMeta("Hartree_64", (_Hartree_64,), {})
Hartree_64_uncontracted = CustomCallMeta("Hartree_64_uncontracted", (_Hartree_64_uncontracted,), {})
Hartree_32 = CustomCallMeta("Hartree_32", (_Hartree_32,), {})
Hartree_32_uncontracted = CustomCallMeta("Hartree_32_uncontracted", (_Hartree_32_uncontracted,), {})
# if jax.config.jax_enable_x64:
#   hartree = Hartree_64()
#   hartree_uncontracted = Hartree_64_uncontracted()
# else:
#   hartree = Hartree_32()
#   hartree_uncontracted = Hartree_32_uncontracted()
hartree = Hartree_64()
hartree_uncontracted = Hartree_64_uncontracted()
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

def get_abab_fun(static_args):
  """Get the function to compute ERI for all abab to do pre
    screen by cuda.

  Args:
    static_args: statis arguments for orbitals
  """

  # @partial(jax.jit, static_argnames=["n_segs"])
  def tensorize(
    gtos: CGTO,
    idx_counts,
    orig_idx,
  ):
    N = gtos.n_pgtos
    
    # Why: Reshape n r z to 1D will significantly reduce computing time
    n = jnp.array(gtos.pgto.angular[orig_idx].T, dtype=jnp.int32)
    r = jnp.array(gtos.pgto.center[orig_idx].T)
    z = jnp.array(gtos.pgto.exponent)[orig_idx]
    
    min_a = jnp.array(static_args.min_a, dtype=jnp.int32)
    min_c = jnp.array(static_args.min_c, dtype=jnp.int32)
    max_ab = jnp.array(static_args.max_ab, dtype=jnp.int32)
    max_cd = jnp.array(static_args.max_cd, dtype=jnp.int32)
    Ms = jnp.array([static_args.max_xyz+1, static_args.max_yz+1, static_args.max_z+1], dtype=jnp.int32)
    abcd_idx = idx_counts[:, :4]

    har_jit = jax.jit(hartree_uncontracted)
    t_abcd = har_jit(jnp.array([N], dtype=jnp.int32), jnp.array(abcd_idx,dtype=jnp.int32), n, r, z, min_a, min_c, max_ab, max_cd, Ms)
    jax.block_until_ready(t_abcd)
    return t_abcd

  return tensorize

def get_4c_contracted_hartree_fun(static_args):
  """Get the function to compute ERI (contracted)

  Args:
    static_args: statis arguments for orbitals
  """

  # @partial(jax.jit)
  def tensorize(
    cgto: CGTO,
    orig_idx,
    sorted_ab_idx,
    sorted_cd_idx,
    screened_cd_idx_start,
    # start_offset,
    screened_cnt,
    pgto_idx_to_cgto_idx,
    rdm1,
    thread_load,
    thread_num,
    ab_thread_num,
    ab_thread_offset
  ):
    N = jnp.array([cgto.n_pgtos], dtype=jnp.int32)
    n = jnp.array(cgto.pgto.angular[orig_idx].T, dtype=jnp.int32)
    r = jnp.array(cgto.pgto.center[orig_idx].T)
    z = jnp.array(cgto.pgto.exponent)[orig_idx]
    
    min_a = jnp.array(static_args.min_a, dtype=jnp.int32)
    min_c = jnp.array(static_args.min_c, dtype=jnp.int32)
    max_ab = jnp.array(static_args.max_ab, dtype=jnp.int32)
    max_cd = jnp.array(static_args.max_cd, dtype=jnp.int32)
    Ms = jnp.array([static_args.max_xyz+1, static_args.max_yz+1, static_args.max_z+1], dtype=jnp.int32)
    
    pgto_coeff = jnp.array(cgto.coeff[orig_idx])
    pgto_normalization_factor = jnp.array(cgto.N[orig_idx])

    har_jit = jax.jit(hartree)

    output = har_jit(N, 
                    jnp.array([thread_load], dtype=jnp.int32),
                    jnp.array([thread_num], dtype=jnp.int64),
                    jnp.array([screened_cnt], dtype=jnp.int64),
                    n, r, z, min_a, min_c, max_ab, max_cd, Ms,
                    jnp.array(sorted_ab_idx, dtype=jnp.int32),
                    jnp.array(sorted_cd_idx, dtype=jnp.int32),
                    jnp.array(screened_cd_idx_start, dtype=jnp.int32),
                    # jnp.array(start_offset, dtype=jnp.int32),
                    jnp.array(ab_thread_num, dtype=jnp.int32),
                    jnp.array(ab_thread_offset, dtype=jnp.int32),
                    pgto_coeff,
                    pgto_normalization_factor,
                    pgto_idx_to_cgto_idx,
                    rdm1,
                    jnp.array([cgto.n_cgtos], dtype=jnp.int32),
                    jnp.array([cgto.n_pgtos], dtype=jnp.int32))
    jax.block_until_ready(output)
    return output

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

def compute_hartree(cgto: CGTO, Mo_coeff_spin, eps = 1e-10, thread_load = 2**10):
  """Compute contracted ERI

  Args:
    cgto: cgto of molecule
    static_args: statis arguments for orbitals
    Mo_coeff_spin: molecule coefficients with spin
  """
  static_args = obsa.angular_static_args(*[cgto.pgto.angular] * 4)
  l_xyz = jnp.sum(cgto.pgto.angular, 1)
  orig_idx = jnp.argsort(l_xyz)

  ab_idx_counts = symmetry.get_2c_sym_idx(cgto.n_pgtos)
  ab_idx, counts_ab = ab_idx_counts[:, :2], ab_idx_counts[:, 2]
  abab_idx_counts = jnp.hstack([ab_idx, ab_idx,
                                counts_ab[:, None]*counts_ab[:, None]]).astype(int)
  abab_idx = jnp.array(abab_idx_counts[: ,:4], dtype=jnp.int32)

  abab_eri_fun = get_abab_fun(static_args)
  abcd_eri_fun = get_4c_contracted_hartree_fun(static_args)
  # Compute eri abab
  eri_abab = abab_eri_fun(cgto, abab_idx_counts, orig_idx)

  eri_abab = jnp.array(eri_abab)

  # current support s, p, d
  s_num = jnp.count_nonzero(l_xyz == 0)
  p_num = jnp.count_nonzero(l_xyz == 1)
  d_num = jnp.count_nonzero(l_xyz == 2)

  cgto_seg_idx = jnp.cumsum(jnp.array(cgto.cgto_splits))
  pgto_idx_to_cgto_idx = jnp.array(jnp.argmax(orig_idx[:, None] < cgto_seg_idx, axis=-1),dtype=jnp.int32)

  rdm1 = get_rdm1(Mo_coeff_spin).sum(0).flatten()

  sorted_idx = jnp.argsort(eri_abab)
  sorted_eri = eri_abab[sorted_idx]

  rank_ab_idx = jnp.arange(ab_idx_counts.shape[0])
  ss_mask = (ab_idx_counts[:, 1] < s_num)
  sp_mask = (ab_idx_counts[:, 1] >= s_num) & (ab_idx_counts[:, 1] < s_num + p_num) & (ab_idx_counts[:, 0] < s_num)
  sd_mask = (ab_idx_counts[:, 1] >= s_num + p_num) & (ab_idx_counts[:, 0] < s_num)
  pp_mask = (ab_idx_counts[:, 1] < s_num + p_num) & (ab_idx_counts[:, 0] >= s_num)
  pd_mask = (ab_idx_counts[:, 1] >= s_num + p_num) & (ab_idx_counts[:, 0] >= s_num) & (ab_idx_counts[:, 0] < s_num + p_num)
  dd_mask = (ab_idx_counts[:, 0] >= s_num + p_num)

  ss_idx = rank_ab_idx[ss_mask]
  sp_idx = rank_ab_idx[sp_mask]
  sd_idx = rank_ab_idx[sd_mask]
  pp_idx = rank_ab_idx[pp_mask]
  pd_idx = rank_ab_idx[pd_mask]
  dd_idx = rank_ab_idx[dd_mask]

  sorted_idx = [ss_idx[jnp.argsort(eri_abab[ss_idx])],
                sp_idx[jnp.argsort(eri_abab[sp_idx])],
                sd_idx[jnp.argsort(eri_abab[sd_idx])],
                pp_idx[jnp.argsort(eri_abab[pp_idx])],
                pd_idx[jnp.argsort(eri_abab[pd_idx])],
                dd_idx[jnp.argsort(eri_abab[dd_idx])],]
  sorted_eri = [eri_abab[sorted_idx[0]],
                eri_abab[sorted_idx[1]],
                eri_abab[sorted_idx[2]],
                eri_abab[sorted_idx[3]],
                eri_abab[sorted_idx[4]],
                eri_abab[sorted_idx[5]]]


  # for (ss, ss) (pp, pp) (dd, dd), (sp, sp) ... need to ensure idx > cnt. For anyone else, no need
  output = 0
  for i in range(6):
    for j in range(i, 6):
      sorted_ab_idx = sorted_idx[i]
      sorted_cd_idx = sorted_idx[j]
      if len(sorted_ab_idx) == 0 or len(sorted_cd_idx) == 0:
        continue
      sorted_eri_abab = sorted_eri[i]
      sorted_eri_cdcd = sorted_eri[j]
      sorted_ab_thres = (eps / jnp.sqrt(sorted_eri_abab))**2
      screened_cd_idx_start = jnp.searchsorted(sorted_eri_cdcd, sorted_ab_thres)
      if i == j:
        screened_cd_idx_start = jnp.maximum(jnp.array([e for e in range(len(sorted_eri_abab))]), screened_cd_idx_start)
      cdcd_len = len(sorted_eri_cdcd)
      # start_offset = jnp.concatenate((jnp.array([0]), jnp.cumsum(cdcd_len-screened_cd_idx_start)[:-1]), dtype=jnp.int32)

      screened_cnt = jnp.sum(cdcd_len-screened_cd_idx_start)
      cdcd_len_list = cdcd_len - screened_cd_idx_start
      
      ab_thread_num = jnp.ceil(cdcd_len_list/thread_load)
      thread_num = jnp.sum(ab_thread_num)
      ab_thread_offset = jnp.concatenate((jnp.array([0]), jnp.cumsum(ab_thread_num)[:-1]), dtype=jnp.int32)
      print(i,j,screened_cnt)
      output += abcd_eri_fun(cgto, orig_idx,sorted_ab_idx, 
                              sorted_cd_idx, 
                              screened_cd_idx_start, 
                              # start_offset,
                              jnp.sum(cdcd_len-screened_cd_idx_start),
                              pgto_idx_to_cgto_idx, 
                              rdm1,
                              thread_load,
                              thread_num,
                              ab_thread_num,
                              ab_thread_offset,)
  return output