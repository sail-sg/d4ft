"""Test molecules."""

from absl.testing import absltest, parameterized
from d4ft.geometries import h2_geometry, h2o_geometry, o_geometry
from d4ft.molecule import Molecule
from d4ft.sgd import sgd


class MoleculeTest(parameterized.TestCase):

  @parameterized.parameters(
    (h2_geometry, (-1., -1.1)),
    (o_geometry, (-72., -75.)),
    (h2o_geometry, (-74., -80.)),
  )
  def test_sgd_energy(self, geometry, energy_bounds):
    e_total, _, _ = sgd(
      Molecule(geometry, spin=0, level=1, basis='6-31g', algo="sgd"),
      epoch=200,
      lr=1e-2,
      batch_size=100000,
      converge_threshold=1e-8,
      optimizer='adam',
      seed=137,
    )
    upper_bound, lower_bound = energy_bounds
    self.assertTrue(e_total < upper_bound and e_total > lower_bound)


if __name__ == "__main__":
  absltest.main()
