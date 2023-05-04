import haiku as hk
import jax
import numpy as np
from absl.testing import absltest, parameterized
from d4ft.integral.gto.lcgto import LCGTO
from d4ft.system.mol import get_pyscf_mol


class LCGTOTest(parameterized.TestCase):

  @parameterized.parameters(
    ("h2", "sto-3g", 2, 2, 6),
    ("o", "sto-3g", 1, 5, 15),
  )
  def test_basis_param(self, system, basis, n_atoms, n_cgtos, n_gtos):
    mol = get_pyscf_mol(system, basis)
    get_lcgto = LCGTO.from_pyscf_mol(mol, use_hk=True)
    get_lcgto = hk.without_apply_rng(hk.transform(get_lcgto))

    gparams = get_lcgto.init(1)
    lcgto = get_lcgto.apply(gparams)

    r = np.random.randn(3)
    ao_val = jax.jit(lcgto.eval)(r)
    print(ao_val)
    self.assertEqual(len(ao_val), n_cgtos)

    print(lcgto.primitives.center)
    print(gparams)

    self.assertEqual(gparams['~']['center'].shape[0], n_atoms)
    self.assertEqual(lcgto.primitives.center.shape[0], n_gtos)

    self.assertEqual(lcgto.n_gtos, n_gtos)
    self.assertEqual(lcgto.n_cgtos, n_cgtos)
    self.assertEqual(lcgto.n_atoms, n_atoms)


if __name__ == "__main__":
  absltest.main()
