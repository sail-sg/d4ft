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
from d4ft.system.mol import Mol


def eval_ao(r: jax.Array, mol: Mol):
  """Evaluate N-body atomic orbitals at location r.

  Args:
        r: (3) coordinate.

  Returns:
    (N,) ao output
  """
  atom_coords = mol.atom_coords
  output = []
  for idx in np.arange(len(mol.elements)):
    element = mol.elements[idx]
    coord = atom_coords[idx]
    for i in mol.basis[element]:
      prm_array = jnp.array(i[1:])
      exponents = prm_array[:, 0]
      coeffs = prm_array[:, 1]

      if i[0] == 0:  # s-orbitals
        output.append(
          jnp.sum(
            coeffs * jnp.exp(-exponents * jnp.sum((r - coord)**2)) *
            (2 * exponents / jnp.pi)**(3 / 4)
          )
        )

      elif i[0] == 1:  # p-orbitals
        output += [
          (r[j] - coord[j]) * jnp.sum(
            coeffs * jnp.exp(-exponents * jnp.sum((r - coord)**2)) *
            (2 * exponents / jnp.pi)**(3 / 4) * (4 * exponents)**0.5
          ) for j in np.arange(3)
        ]

  return jnp.array(output)


class HKTest(parameterized.TestCase):

  @parameterized.parameters(
    ("h2", "sto-3g", 2, 2, 6),
    ("o", "sto-3g", 1, 5, 15),
  )
  def test_basis_param(self, system, basis, n_atoms, n_cgtos, n_pgtos):
    mol = Mol.from_mol_name(system, basis)
    cgto_transformed = hk.without_apply_rng(
      hk.transform(lambda: CGTO.from_mol(mol).to_hk())
    )

    gparams = cgto_transformed.init(1)
    cgto = cgto_transformed.apply(gparams)

    for _ in range(3):
      r = np.random.randn(3)
      ao_val = jax.jit(cgto.eval)(r)
      print(ao_val)

      ao_val_exp = eval_ao(r, mol)
      print(ao_val_exp)

      self.assertTrue(jnp.allclose(ao_val, ao_val_exp, atol=1e-6))

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
