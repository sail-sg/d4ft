# Copyright 2023 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
jax.config.update("jax_enable_x64", True)
import jaxlib
import jax.numpy as jnp
import pyscf
import time

from functools import partial
import numpy as np
from absl import logging
from absl.testing import absltest

from d4ft.native.xla.custom_call import CustomCallMeta
from d4ft.native.obara_saika.eri_kernel import _Hartree_32, _Hartree_32_uncontracted, _Hartree_64, _Hartree_64_uncontracted

from d4ft.integral.obara_saika.electron_repulsion_integral import (
  electron_repulsion_integral,
)
from d4ft.integral.obara_saika import angular_stats
from d4ft.system.mol import Mol, get_pyscf_mol
from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.gto import symmetry, tensorization
from copy import deepcopy
from d4ft.types import AngularStats, CGTOSymTensorIncore, Tensor2C, Tensor4C
from d4ft.utils import get_rdm1
# from obsa.obara_saika import get_coulomb, get_kinetic, get_nuclear, get_overlap

# from jax.interpreters import ad, batching, mlir, xla
Hartree_64 = CustomCallMeta("Hartree_64", (_Hartree_64,), {})
Hartree_64_uncontracted = CustomCallMeta("Hartree_64_uncontracted", (_Hartree_64_uncontracted,), {})
Hartree_32 = CustomCallMeta("Hartree_32", (_Hartree_32,), {})
Hartree_32_uncontracted = CustomCallMeta("Hartree_32_uncontracted", (_Hartree_32_uncontracted,), {})
if jax.config.jax_enable_x64:
  hartree = Hartree_64()
  hartree_uncontracted = Hartree_64_uncontracted()
else:
  hartree = Hartree_32()
  hartree_uncontracted = Hartree_32_uncontracted()

# TODO
# def _example_batch_rule(args, axes):
#   return example_fn(args[1:]), axes

# batching.primitive_batchers[example_fn.prim] = _example_batch_rule

#ExampleMember = CustomCallMeta("ExampleMember", (_ExampleMember,), {})


class _ExampleTest(absltest.TestCase):

  def setUp(self):
    key = jax.random.PRNGKey(22)

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


    # To support higher angular, adjust constants in eri.h: MAX_XYZ, MAX_YZ..
    # pyscf_mol = get_pyscf_mol("C20-Ih", "sto-3g")
    # pyscf_mol = get_pyscf_mol("C30-D5h-1", "sto-3g")
    # pyscf_mol = get_pyscf_mol("C40-D5d-1", "sto-3g")
    # pyscf_mol = get_pyscf_mol("C50-D5h-1", "sto-3g")
    # pyscf_mol = get_pyscf_mol("C60-Ih", "sto-3g")
    # pyscf_mol = get_pyscf_mol("C70-D5h", "sto-3g")
    # pyscf_mol = get_pyscf_mol("C80-D5d-1", "sto-3g")
    # pyscf_mol = get_pyscf_mol("C90-D5h-1", "sto-3g")
    # pyscf_mol = get_pyscf_mol("C100-D5d-1", "sto-3g")
    # pyscf_mol = get_pyscf_mol("C180-0", "sto-3g")
    # pyscf_mol = get_pyscf_mol("C240-0", "sto-3g")
    # pyscf_mol = get_pyscf_mol("C320-0", "sto-3g")
    # pyscf_mol = get_pyscf_mol("C500-0", "sto-3g")
    # pyscf_mol = get_pyscf_mol("O2", "6-31G")
    pyscf_mol = get_pyscf_mol("O2", "sto-3g")
    mol = Mol.from_pyscf_mol(pyscf_mol)
    cgto = CGTO.from_mol(mol)
    self.s = angular_stats.angular_static_args(*[cgto.pgto.angular] * 4)
    self.cgto = cgto
    self.ab_idx_counts = symmetry.get_2c_sym_idx(cgto.n_pgtos)
    
  
    key = jax.random.PRNGKey(42)
    self.Mo_coeff = jax.random.normal(key,(2, self.cgto.n_cgtos, self.cgto.n_cgtos))

    

  def test_abcd(self) -> None:
    out_abcd = compute_hartree_test(self.cgto, self.s, self.Mo_coeff, eps=1e-10)
    print(out_abcd)


    # # Compute Ground Truth
    # ab_idx, counts_ab = self.ab_idx_counts[:, :2], self.ab_idx_counts[:, 2]
    # self.abab_idx_count = jnp.hstack([ab_idx, ab_idx,
    #                             counts_ab[:, None]*counts_ab[:, None]]).astype(int)

    # n_2c_idx = len(self.ab_idx_counts)
    # num_4c_idx = symmetry.num_unique_ij(n_2c_idx)
    # self.num_4c_idx = num_4c_idx
    # batch_size: int = 2**23
    # i = 0
    # start = batch_size * i
    # end = num_4c_idx
    # slice_size = num_4c_idx - start
    # start_idx = symmetry.get_triu_ij_from_idx(n_2c_idx, start)
    # end_idx = symmetry.get_triu_ij_from_idx(n_2c_idx, end)
    # self.abcd_idx_counts = symmetry.get_4c_sym_idx_range(
    #   self.ab_idx_counts, n_2c_idx, start_idx, end_idx, slice_size
    # )

    # self.n_segs = symmetry.num_unique_ijkl(self.cgto.n_cgtos)
    # self.cgto_seg_id = symmetry.get_cgto_segment_id_sym(
    #     self.abcd_idx_counts[:, :-1], self.cgto.cgto_splits, four_center=True
    #   )

    # nmo = self.cgto.n_cgtos  # assuming same number of MOs and AOs
    # self.mo_abcd_idx_counts = symmetry.get_4c_sym_idx(nmo)
    # self.rdm1 = get_rdm1(self.Mo_coeff).sum(0)
    # self.rdm1_ab = self.rdm1[self.mo_abcd_idx_counts[:, 0], self.mo_abcd_idx_counts[:, 1]]
    # self.rdm1_cd = self.rdm1[self.mo_abcd_idx_counts[:, 2], self.mo_abcd_idx_counts[:, 3]]

    # cgto_4c_fn_gt = tensorization.tensorize_4c_cgto(electron_repulsion_integral, self.s)
    # e_raw_gt = cgto_4c_fn_gt(self.cgto, self.abcd_idx_counts, self.cgto_seg_id, self.n_segs)
    # hartree_e_raw_gt = jnp.sum(e_raw_gt * self.rdm1_ab * self.rdm1_cd)
    # print(hartree_e_raw_gt)



def compute_hartree_test(cgto: CGTO, static_args: AngularStats, Mo_coeff_spin, eps = 1e-10, thread_load = 2**10):
  """Compute contracted ERI

  Args:
    cgto: cgto of molecule
    static_args: statis arguments for orbitals
    Mo_coeff_spin: molecule coefficients with spin
  """
  l_xyz = jnp.sum(cgto.pgto.angular, 1)
  orig_idx = jnp.argsort(l_xyz)

  ab_idx_counts = symmetry.get_2c_sym_idx(cgto.n_pgtos)
  ab_idx, counts_ab = ab_idx_counts[:, :2], ab_idx_counts[:, 2]
  abab_idx_counts = jnp.hstack([ab_idx, ab_idx,
                                counts_ab[:, None]*counts_ab[:, None]]).astype(int)
  abab_idx = jnp.array(abab_idx_counts[: ,:4], dtype=jnp.int32)

  abab_eri_fun = get_cuda_abab_fun(static_args)
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



def get_4c_contracted_hartree_fun(static_args: AngularStats):
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

    t1 = time.time()
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
    t2 = time.time()
    print("abcd time =", t2-t1)
    return output

  return tensorize


def get_cuda_abab_fun(static_args: AngularStats):
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
    t1 = time.time()
    t_abcd = har_jit(jnp.array([N], dtype=jnp.int32), jnp.array(abcd_idx,dtype=jnp.int32), n, r, z, min_a, min_c, max_ab, max_cd, Ms)
    jax.block_until_ready(t_abcd)
    t2 = time.time()
    print(len(t_abcd))
    print("abab time =", t2-t1)
    return t_abcd

  return tensorize


if __name__ == "__main__":
  absltest.main()
