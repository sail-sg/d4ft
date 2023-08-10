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

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from d4ft.integral.gto.cgto import CGTO
from d4ft.system.mol import Mol, get_pyscf_mol


class HKTest(parameterized.TestCase):

  @parameterized.parameters(
    ("h2", "sto-3g", 2, 2, 6),
    ("o", "sto-3g", 1, 5, 15),
  )
  def test_basis_param(self, system, basis, n_atoms, n_cgtos, n_pgtos):
    pyscf_mol = get_pyscf_mol(system, basis)

    mol = Mol.from_mol_name(system, basis)

    @hk.without_apply_rng
    @hk.transform
    def cgto_transformed():
      cgto_cart = CGTO.from_mol(mol)
      cgto = CGTO.from_cart(cgto_cart)
      return cgto.to_hk()

    gparams = cgto_transformed.init(1)
    cgto = cgto_transformed.apply(gparams)

    for _ in range(3):
      r = np.random.randn(3)
      ao_val = jax.jit(cgto.eval)(r)
      print(ao_val)

      ao_val_exp = pyscf_mol.eval_ao("GTOval_sph", r[None, :])
      print(ao_val_exp)

      self.assertTrue(jnp.allclose(ao_val, ao_val_exp))

    self.assertEqual(len(ao_val), n_cgtos)

    print(cgto.pgto.center)
    print(gparams)

    self.assertEqual(gparams['~']['center'].shape[0], n_atoms)
    self.assertEqual(cgto.pgto.center.shape[0], n_pgtos)

    self.assertEqual(cgto.n_pgtos, n_pgtos)
    self.assertEqual(cgto.n_cgtos, n_cgtos)
    self.assertEqual(cgto.n_atoms, n_atoms)


if __name__ == "__main__":
  absltest.main()
