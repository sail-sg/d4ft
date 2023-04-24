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

from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp
import pyscf
from d4ft.integral.gto import symmetry
from d4ft.integral.gto.cgto import CGTO
from d4ft.types import (
  CGTOIntors, ETensorsIncore, Fock, IdxCount2C, IdxCount4C, MoCoeff,
  QuadGridsNWeights
)
from d4ft.utils import get_rdm1
from d4ft.xc import get_lda_vxc
from jaxtyping import Array, Float


def libcint_incore(
  pyscf_mol: pyscf.gto.mole.Mole,
  mo_ab_idx_counts: IdxCount2C,
  mo_abcd_idx_counts: IdxCount4C,
) -> ETensorsIncore:
  """Get tensor incore using libcint, then reduce symmetry."""
  kin = pyscf_mol.intor_symmetric('int1e_kin')
  ext = pyscf_mol.intor_symmetric('int1e_nuc')
  # NOTE: 4c symmetry of pyscf is ordered differently from d4ft
  # so we cannot use it directly
  eri = 0.5 * pyscf_mol.intor('int2e')
  kin = kin[mo_ab_idx_counts[:, 0], mo_ab_idx_counts[:, 1]]
  ext = ext[mo_ab_idx_counts[:, 0], mo_ab_idx_counts[:, 1]]
  eri = eri[mo_abcd_idx_counts[:, 0], mo_abcd_idx_counts[:, 1],
            mo_abcd_idx_counts[:, 2], mo_abcd_idx_counts[:, 3]]
  counts_ab = mo_ab_idx_counts[:, 2]
  counts_abcd = mo_abcd_idx_counts[:, 4]
  kin *= counts_ab
  ext *= counts_ab
  eri *= counts_abcd
  return kin, ext, eri


def get_cgto_intor(
  cgto: CGTO,
  intor: Literal["obsa", "libcint", "quad"] = "obsa",
  incore_energy_tensors: Optional[ETensorsIncore] = None,
) -> CGTOIntors:
  """
  Args:
    intor: which intor to use
    incore_tensor: if provided, calculate energy incore
  """
  # TODO: test join optimization with hk=True
  nmo = cgto.n_cgtos  # assuming same number of MOs and AOs
  mo_ab_idx_counts = symmetry.get_2c_sym_idx(nmo)
  mo_abcd_idx_counts = symmetry.get_4c_sym_idx(nmo)

  if incore_energy_tensors:
    kin, ext, eri = incore_energy_tensors

    def kin_fn(mo_coeff: MoCoeff) -> Float[Array, ""]:
      rdm1 = get_rdm1(mo_coeff).sum(0)  # sum over spin
      rdm1_2c_ab = rdm1[mo_ab_idx_counts[:, 0], mo_ab_idx_counts[:, 1]]
      e_kin = jnp.sum(kin * rdm1_2c_ab)

      return e_kin

    def ext_fn(mo_coeff: MoCoeff) -> Float[Array, ""]:
      rdm1 = get_rdm1(mo_coeff).sum(0)  # sum over spin
      rdm1_2c_ab = rdm1[mo_ab_idx_counts[:, 0], mo_ab_idx_counts[:, 1]]
      e_ext = jnp.sum(ext * rdm1_2c_ab)
      return e_ext

    def har_fn(mo_coeff: MoCoeff) -> Float[Array, ""]:
      rdm1 = get_rdm1(mo_coeff).sum(0)  # sum over spin
      rdm1_ab = rdm1[mo_abcd_idx_counts[:, 0], mo_abcd_idx_counts[:, 1]]
      rdm1_cd = rdm1[mo_abcd_idx_counts[:, 2], mo_abcd_idx_counts[:, 3]]
      e_har = jnp.sum(eri * rdm1_ab * rdm1_cd)
      return e_har

  else:  # TODO: out-of-core
    pass

  return kin_fn, ext_fn, har_fn


def unreduce_symmetry_2c(
  val: Float[Array, "nmo*(nmo+1)//2"], nmo: int, mo_ab_idx_counts: IdxCount2C
) -> Float[Array, "nmo nmo"]:
  """Given a symmetry reduce matrix, i.e. a flat array of value of shape
  (nmo * (nmo + 1) // 2), return the full matrix of shape (nmo, nmo)."""
  mat = jnp.zeros((nmo, nmo))
  val = val / mo_ab_idx_counts[:, 2]
  triu_mat = mat.at[mo_ab_idx_counts[:, 0], mo_ab_idx_counts[:, 1]].set(val)
  full_mat = triu_mat + triu_mat.T - jnp.diag(jnp.diag(triu_mat))
  return full_mat


def get_cgto_fock_fn(
  cgto: CGTO,
  incore_energy_tensors: ETensorsIncore,
  grids_and_weights: QuadGridsNWeights,
  polarized: bool,
) -> Callable[[MoCoeff], Fock]:
  """Currently only support incore"""
  nmo = cgto.n_cgtos  # assuming same number of MOs and AOs
  mo_ab_idx_counts = symmetry.get_2c_sym_idx(nmo)
  mo_abcd_idx_counts = symmetry.get_4c_sym_idx(nmo)

  kin, ext, eri = incore_energy_tensors

  # unredce 2c integrals into full matrices, which can be precomputed
  kin_mat = unreduce_symmetry_2c(kin, nmo, mo_ab_idx_counts)
  ext_mat = unreduce_symmetry_2c(ext, nmo, mo_ab_idx_counts)
  h_core = kin_mat + ext_mat

  n_2c_idx = len(mo_ab_idx_counts)
  ab_block_idx = jnp.vstack(jnp.triu_indices(n_2c_idx)).T
  counts_ab = mo_ab_idx_counts[:, 2]
  cd_in_block_counts = counts_ab[ab_block_idx[:, 1]]
  # counts only inblock symmetry in the cd index of (ab|cd)
  eri_val = eri / mo_abcd_idx_counts[:, 4] * cd_in_block_counts

  cd_seg_idx = [
    jnp.ones(seg_len, dtype=jnp.int32) * seg_id
    for seg_id, seg_len in enumerate(reversed(range(1, n_2c_idx + 1)))
  ]
  cd_seg_idx = jnp.hstack(cd_seg_idx)

  # VXC
  vxc_fn = get_lda_vxc(grids_and_weights, cgto, polarized)

  def get_fock(mo_coeff: MoCoeff) -> Fock:
    """Calculate the Fock matrix from the MO coefficients.

    The J matrix can be calculated from the MO coefficients and
    symmetry reduced 4c integrals (ab|cd).

    .. math::
    sum_{cd} rho_{cd} (ab|cd) -> J_{ab}

    Note that the K matrix can be computed as

    .. math::
    sum_{cb} rho_{cb} (ab|cd) -> K_{cb}

    where rho is the 1-RDM.
    """
    rdm1 = get_rdm1(mo_coeff)
    rdm1_cd = rdm1[:, mo_abcd_idx_counts[:, 2], mo_abcd_idx_counts[:, 3]]
    # TODO: more efficient way to handle spin here
    j_val_up = jax.ops.segment_sum(rdm1_cd[0] * eri_val, cd_seg_idx, n_2c_idx)
    j_mat_up = unreduce_symmetry_2c(j_val_up, nmo, mo_ab_idx_counts)
    j_val_dn = jax.ops.segment_sum(rdm1_cd[1] * eri_val, cd_seg_idx, n_2c_idx)
    j_mat_dn = unreduce_symmetry_2c(j_val_dn, nmo, mo_ab_idx_counts)
    j_mat = jnp.stack((j_mat_up, j_mat_dn))
    vxc = vxc_fn(mo_coeff)
    v_eff = j_mat + vxc

    return h_core + v_eff

  return get_fock
