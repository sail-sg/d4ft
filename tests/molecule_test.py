"""Test water molecule."""

from absl.testing import absltest
from d4ft.geometries import h2o_geometry, h2_geometry
from d4ft.molecule import molecule
from d4ft.sgd import sgd


class MoleculeTest(absltest.TestCase):

  def test_water_energy(self):
    mol = molecule(h2o_geometry, spin=0, level=1, basis='6-31g')
    egs, _ = sgd(
      mol,
      epoch=100,
      lr=2e-2,
      batch_size=20000,
      converge_threshold=1e-5,
      pre_cal=True
    )
    self.assertTrue(egs < -75. and egs > -80)

  def test_h2_energy(self):
    mol = molecule(h2_geometry, spin=0, level=1, basis='6-31g')
    egs, _ = sgd(
      mol,
      20,
      lr=1e-2,
      seed=123,
      optimizer='adam',
      converge_threshold=1e-5,
      batch_size=1000,
      pre_cal=True
    )
    self.assertTrue(egs < -1 and egs > -1.1)

  def test_oxygen_energy(self):
    mol = molecule('O 0 0 0', spin=0, level=1, basis='6-31g')
    egs, _ = sgd(
      mol,
      20,
      lr=1e-2,
      seed=123,
      optimizer='adam',
      converge_threshold=1e-5,
      batch_size=1000,
      pre_cal=True
    )
    self.assertTrue(egs < -72. and egs > -75)


if __name__ == "__main__":
  absltest.main()
