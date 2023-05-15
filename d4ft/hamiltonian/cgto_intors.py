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

from typing import Literal, Optional

import jax.numpy as jnp
import pyscf
from d4ft.integral.gto import symmetry
from d4ft.integral.gto.cgto import CGTO
from d4ft.types import (
  CGTOIntors, ETensorsIncore, IdxCount2C, IdxCount4C, MoCoeff
)
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


def get_rdm1(mo_coeff: MoCoeff):
  return mo_coeff.T @ mo_coeff


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
      rdm1 = get_rdm1(mo_coeff)
      rdm1_2c_ab = rdm1[mo_ab_idx_counts[:, 0], mo_ab_idx_counts[:, 1]]
      e_kin = jnp.sum(kin * rdm1_2c_ab)
      return e_kin

    def ext_fn(mo_coeff: MoCoeff) -> Float[Array, ""]:
      rdm1 = get_rdm1(mo_coeff)
      rdm1_2c_ab = rdm1[mo_ab_idx_counts[:, 0], mo_ab_idx_counts[:, 1]]
      e_ext = jnp.sum(ext * rdm1_2c_ab)
      return e_ext

    def eri_fn(mo_coeff: MoCoeff) -> Float[Array, ""]:
      rdm1 = get_rdm1(mo_coeff)
      rdm1_4c_ab = rdm1[mo_abcd_idx_counts[:, 0], mo_abcd_idx_counts[:, 1]]
      rdm1_4c_cd = rdm1[mo_abcd_idx_counts[:, 2], mo_abcd_idx_counts[:, 3]]
      e_eri = jnp.sum(eri * rdm1_4c_ab * rdm1_4c_cd)
      return e_eri

  else:  # TODO: out-of-core
    pass

  return kin_fn, ext_fn, eri_fn
