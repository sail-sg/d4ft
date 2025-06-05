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
"""Test overlap integral."""

import jax

jax.config.update("jax_enable_x64", True)
import jax
import numpy as np
import pyscf
from absl import logging
from absl.testing import absltest  # noqa: E402

from d4ft.hamiltonian.cgto_intors import unreduce_symmetry_2c
from d4ft.integral import obara_saika as obsa
from d4ft.integral.gto import symmetry
from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.obara_saika.driver import get_cgto_sym_tensor_fns
from d4ft.system.mol import Mol


class _TestOverlapIntegral(absltest.TestCase):

  def setUp(self):
    self.pyscf_mol = pyscf.gto.M(
      atom='O 0 0 0; O 0 0 1', basis='6-31G*', verbose=0
    )
    self.mol = Mol.from_pyscf_mol(self.pyscf_mol)
    self.cgto = CGTO.from_mol(self.mol)
    self.s2 = obsa.angular_static_args(*[self.cgto.pgto.angular] * 2)
    self.s4 = obsa.angular_static_args(*[self.cgto.pgto.angular] * 4)

  def test_vv(self):
    pgto_0 = self.cgto.pgto.at(0)
    pgto_1 = self.cgto.pgto.at(1)
    ovlp_01 = obsa.overlap_integral(
      pgto_0, pgto_1, self.s2, use_horizontal=False
    )
    logging.info(f"Overlap integral between GTO 0 and GTO 1: {ovlp_01}")

  def test_vh(self):
    pgto_0 = self.cgto.pgto.at(0)
    pgto_1 = self.cgto.pgto.at(1)
    ovlp_01 = obsa.overlap_integral(
      pgto_0, pgto_1, self.s2, use_horizontal=True
    )
    logging.info(f"Overlap integral between GTO 0 and GTO 1: {ovlp_01}")

  def test_against_libcint(self):
    jax.config.update("jax_enable_x64", True)  # use double precision

    cgto_tensor_fns = get_cgto_sym_tensor_fns(self.cgto, self.s2, self.s4)
    ovlp_ab = cgto_tensor_fns.ovlp_ab_fn(self.cgto)

    nmo = self.cgto.n_cgtos
    mo_ab_idx_counts = symmetry.get_2c_sym_idx(nmo)
    ovlp = unreduce_symmetry_2c(ovlp_ab, nmo, mo_ab_idx_counts)
    ovlp_pyscf_sph = self.pyscf_mol.intor_symmetric('int1e_ovlp_sph')
    assert np.allclose(ovlp_pyscf_sph, ovlp)


if __name__ == "__main__":
  absltest.main()
