import jax
from absl.testing import absltest, parameterized

from d4ft.config import DFTConfig, OptimizerConfig
from d4ft.sgd_solver import sgd_solver
from d4ft.system.mol import get_pyscf_mol


class SolverTest(parameterized.TestCase):

  @parameterized.parameters(
    ("h2", (-1., -1.1)),
    ("o", (-72., -75.)),
    ("h2o", (-74., -80.)),
  )
  def test_sgd_energy(self, system, energy_bounds):
    basis = '6-31g'
    mol = get_pyscf_mol(system, basis)
    key = jax.random.PRNGKey(137)
    e_total, _, _ = sgd_solver(DFTConfig(), OptimizerConfig(), mol, key)
    upper_bound, lower_bound = energy_bounds
    self.assertTrue(e_total < upper_bound and e_total > lower_bound)


if __name__ == "__main__":
  absltest.main()
