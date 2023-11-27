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
from d4ft.native.obara_saika.eri_kernel import _Hartree_32, _Hartree_64

from d4ft.integral.obara_saika.electron_repulsion_integral import (
  electron_repulsion_integral,
)
from d4ft.integral.obara_saika import angular_stats
from d4ft.system.mol import Mol, get_pyscf_mol
from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.gto import symmetry, tensorization
from copy import deepcopy
# from obsa.obara_saika import get_coulomb, get_kinetic, get_nuclear, get_overlap

# from jax.interpreters import ad, batching, mlir, xla
Hartree_64 = CustomCallMeta("Hartree_64", (_Hartree_64,), {})
Hartree_32 = CustomCallMeta("Hartree_32", (_Hartree_32,), {})
if jax.config.jax_enable_x64:
  hartree = Hartree_64()
else:
  hartree = Hartree_32()

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


    # To support higher angular, first adjust constants in eri.h: MAX_XYZ, MAX_YZ..
    # pyscf_mol = get_pyscf_mol("C60-Ih", "sto-3g")
    pyscf_mol = get_pyscf_mol("C180-0", "6-31G")
    # pyscf_mol = get_pyscf_mol("O2", "sto-3g")
    mol = Mol.from_pyscf_mol(pyscf_mol)
    cgto = CGTO.from_mol(mol)
    self.s = angular_stats.angular_static_args(*[cgto.pgto.angular] * 4)
    self.cgto = cgto
    self.ab_idx_counts = symmetry.get_2c_sym_idx(cgto.n_pgtos)
    n_2c_idx = len(self.ab_idx_counts)

    # 4c tensors
    ab_idx, counts_ab = self.ab_idx_counts[:, :2], self.ab_idx_counts[:, 2]
    self.abab_idx_count = jnp.hstack([ab_idx, ab_idx,
                                counts_ab[:, None]*counts_ab[:, None]]).astype(int)

    num_4c_idx = symmetry.num_unique_ij(n_2c_idx)
    self.num_4c_idx = num_4c_idx
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


    # Ns = cgto.N
    # abcd_idx = self.abcd_idx_counts[:, :4]
    # gtos_abcd, self.coeffs_abcd = zip(
    #   *[
    #     cgto.map_pgto_params(lambda gto_param, i=i: gto_param[abcd_idx[:, i]])
    #     for i in range(4)
    #   ]
    # )
    # counts_abcd_i = self.abcd_idx_counts[:, 4]
    # self.N_abcd = Ns[abcd_idx].prod(-1) * counts_abcd_i
    # self.n_segs = symmetry.num_unique_ijkl(cgto.n_cgtos)
    # self.cgto_seg_id = symmetry.get_cgto_segment_id_sym(
    #     self.abcd_idx_counts[:, :-1], cgto.cgto_splits, four_center=True
    #   )
    # self.a, self.b, self.c, self.d = gtos_abcd
    # self.N = cgto.n_pgtos
    # self.n = jnp.array(deepcopy(cgto.pgto.angular.T.reshape((3*self.N,))), dtype=jnp.int32)
    # self.r = jnp.array(deepcopy(cgto.pgto.center.T.reshape((3*self.N,))))
    # self.z = jnp.array(deepcopy(cgto.pgto.exponent))
    
    # # self.s = angular_stats.angular_static_args(self.a[0],self.b[0],self.c[0],self.d[0])
    # self.min_a = jnp.array(self.s.min_a, dtype=jnp.int32)
    # self.min_c = jnp.array(self.s.min_c, dtype=jnp.int32)
    # self.max_ab = jnp.array(self.s.max_ab, dtype=jnp.int32)
    # self.max_cd = jnp.array(self.s.max_cd, dtype=jnp.int32)
    # self.Ms = jnp.array([self.s.max_xyz+1, self.s.max_yz+1, self.s.max_z+1], dtype=jnp.int32)

  def test_example(self) -> None:
    logging.info(jax.devices())
  #   def f_curry(*args):
  #     return electron_repulsion_integral(*args, static_args=self.s)
  #   vmap_f = jax.vmap(f_curry, in_axes=(0, 0, 0, 0))
  #   T_1 = time.time()
  #   e1 = vmap_f(self.a, self.b, self.c, self.d)
  #   abcd_1 = jnp.einsum("k,k,k,k,k,k->k", e1, self.N_abcd, *self.coeffs_abcd)
  #   cgto_abcd_1 = jax.ops.segment_sum(abcd_1, self.cgto_seg_id, self.n_segs)
  #   T_2 = time.time()
  #   print(T_2-T_1)

  #   cgto_4c_fn = tensorization.tensorize_4c_cgto_cuda(self.s)
  #   T_1 = time.time()
  #   # e2 = hartree(jnp.array([self.N], dtype=jnp.int32),jnp.array(range(self.N), dtype=jnp.int32), self.n, self.r, self.z, self.min_a,
  #   #               self.min_c, self.max_ab, self.max_cd, self.Ms)
  #   cgto_abcd_2 = cgto_4c_fn(
  #       self.cgto, self.abcd_idx_counts, self.cgto_seg_id, self.n_segs
  #     )   
  #   T_2 = time.time()
  #   print(T_2-T_1)
  #   # abcd_2 = jnp.einsum("k,k,k,k,k,k->k", e2, self.N_abcd, *self.coeffs_abcd)
  #   # cgto_abcd_2 = jax.ops.segment_sum(abcd_2, self.cgto_seg_id, self.n_segs)
  #   # np.testing.assert_allclose(e1,e2,atol=2e-5)
  #   np.testing.assert_allclose(cgto_abcd_1,cgto_abcd_2,atol=1e-5)

    
    # out_vmap = jax.vmap(example_fn)(self.a_b, self.b_b)
    # logging.info(out_vmap)

    # out_grad = jax.grad(e)(self.a, self.b)
    # logging.info(out_grad)
    # np.testing.assert_equal(len(self.outshape),len(out))
    # np.testing.assert_array_equal(self.outshape, out)

  def test_abab(self) -> None:
    
    pgto_4c_fn = tensorization.tensorize_4c_cgto_cuda(self.s, cgto=False)
    # pgto_4c_fn_gt = tensorization.tensorize_4c_cgto(electron_repulsion_integral, self.s, cgto=False)
    # cgto_4c_fn = tensorization.tensorize_4c_cgto_range(eri_fn, s4)
    eri_abab = pgto_4c_fn(self.cgto, self.abab_idx_count, None, None)
    # eri_abab_gt = pgto_4c_fn_gt(self.cgto, self.abab_idx_count, None, None)
    
    sorted_idx = jnp.argsort(eri_abab)
    sorted_abab = eri_abab[sorted_idx]
    eps = 1e-10
    sorted_cd_thres = eps / jnp.sqrt(sorted_abab)
    cnt = jnp.array([e for e in range(len(self.abab_idx_count))])
    idx = jnp.maximum(cnt, jnp.searchsorted(sorted_abab, sorted_cd_thres))
    idx = idx[idx < len(sorted_idx)]

    abab_len = len(sorted_idx)
    screened_cnt = jnp.sum(abab_len-idx)
    print(len(self.abab_idx_count))
    print("original length =", self.num_4c_idx)
    print("screened length =", screened_cnt)
    # abcd = [jnp.array([sorted_idx[cnt]* jnp.ones(len(sorted_idx)-idx[cnt])], sorted_idx[cd_idx:]).T for cd_idx, cnt in zip(idx,range(len(idx)))]
    # abs = [sorted_idx[cnt] * jnp.ones(len(sorted_idx)-idx[cnt]) for cnt in range(len(idx))]
    # cds = sorted_idx[idx:]
    # print(abs)
    # print(cds)
    # for cnt in range(len(sorted_idx)):
    #   idx = None
    #   for k in range(len(sorted_abab[cnt:])):
    #     if sorted_abab[cnt + k] > sorted_cd_thres[cnt]:
    #       idx = cnt + k
    #       break
    #   if idx is None:
    #     continue
    #   cds = sorted_idx[cnt+k:]
    #   abcd.append(jnp.array([jnp.array([sorted_idx[cnt]] * len(cds)),cds]).T)
    
    # abcd = jnp.vstack(abcd)
    # abcd_offdiag = (abcd[:,0] != abcd[:,1])
    # abcd_block = abcd_offdiag + jnp.ones(len(abcd_offdiag))
    # abcd_counts = self.ab_idx_counts[abcd[:,0], -1] * self.ab_idx_counts[abcd[:,1], -1] * abcd_block
    # abcd_counts.astype(jnp.int32)
    # self.screened_abcd_idx_counts = jnp.hstack([self.ab_idx_counts[abcd[:,0],:2],
    #                                        self.ab_idx_counts[abcd[:,1],:2],
    #                                        abcd_counts.reshape((abcd_counts.shape[0],1))],dtype=jnp.int32)
      
    # self.screened_cgto_seg_id = symmetry.get_cgto_segment_id_sym(
    #     self.screened_abcd_idx_counts[:, :-1], self.cgto.cgto_splits, four_center=True
    #   )
    
    # cgto_4c_fn = tensorization.tensorize_4c_cgto_cuda(self.s)
    # e_screened = cgto_4c_fn(self.cgto, self.screened_abcd_idx_counts, 
    #                         self.screened_cgto_seg_id, self.n_segs)   
    # e_raw = cgto_4c_fn(self.cgto, self.abcd_idx_counts, self.cgto_seg_id, self.n_segs) 
    # print("original length =", len(self.abcd_idx_counts))
    # print("screened length =", len(self.screened_abcd_idx_counts))
    # np.testing.assert_allclose(e_screened,e_raw,atol=1e-5)

    # logging.info(f"block diag (ab|ab) computed, size: {eri_abab.shape}")


if __name__ == "__main__":
  absltest.main()
