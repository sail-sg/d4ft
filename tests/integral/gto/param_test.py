import jax
import numpy as np
from absl.testing import absltest, parameterized

from d4ft.integral.gto.gto_utils import get_gto_param_fn
from d4ft.system.mol import get_pyscf_mol


class GTOParamTest(parameterized.TestCase):

  @parameterized.parameters(
    ("h2", "sto-3g", 2, 2, 6),
    ("o", "sto-3g", 1, 5, 15),
  )
  def test_basis_param(self, system, basis, n_atoms, nao, n_gtos):
    mol = get_pyscf_mol(system, basis)
    gto_param_fn = get_gto_param_fn(mol)

    gparams = gto_param_fn.init(1)
    gtos = gto_param_fn.apply(gparams)

    r = np.random.randn(3)
    ao_val = jax.jit(gtos.eval)(r)
    print(ao_val)
    self.assertTrue(len(ao_val) == nao)

    print(gtos.center)
    print(gparams)

    self.assertTrue(gparams['~']['center'].shape[0] == n_atoms)
    self.assertTrue(gtos.center.shape[0] == n_gtos)


if __name__ == "__main__":
  absltest.main()
