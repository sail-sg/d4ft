import jax
import numpy as np
from absl.testing import absltest, parameterized

from d4ft.integral.gto.gto_utils import get_gto_param_fn
from d4ft.system.mol import get_pyscf_mol


class GTOParamTest(parameterized.TestCase):

  @parameterized.parameters(("h2", "sto-3g"))
  def test_basis_param(self, system, basis):
    mol = get_pyscf_mol(system, basis)
    gto_param_fn = get_gto_param_fn(mol)

    gparams = gto_param_fn.init(1)
    gtos = gto_param_fn.apply(gparams)

    r = np.random.randn(3)
    ao_val = jax.jit(gtos.eval)(r)
    print(ao_val)
    self.assertTrue(len(ao_val) == 2)

    print(gtos.center)
    print(gparams)

    self.assertTrue(gparams['~']['center'].shape[0] == 2)
    self.assertTrue(gtos.center.shape[0] == 6)


if __name__ == "__main__":
  absltest.main()
