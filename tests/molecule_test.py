"""Test water molecule."""

import os
import sys
from absl.testing import absltest

sys.path.append('.')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'

import jdft
from jdft.geometries import h2o_geometry, h2_geometry


class MoleculeTest(absltest.TestCase):

  def test_water_energy(self):
    mol = jdft.molecule(h2o_geometry, spin=0, level=1, basis='6-31g')
    egs = mol.train(
      30,
      lr=1e-2,
      seed=123,
      optimizer='adam',
      converge_threshold=1e-5,
      batchsize=1000,
    )
    self.assertTrue(egs < -75. and egs > -80)


  def test_h2_energy(self):
    mol = jdft.molecule(h2_geometry, spin=0, level=1, basis='6-31g')
    egs = mol.train(
      20,
      lr=1e-2,
      seed=123,
      optimizer='adam',
      converge_threshold=1e-5,
      batchsize=1000,
    )
    self.assertTrue(egs < -1 and egs > -1.1)


  def test_oxygen_energy(self):
    mol = jdft.molecule('o 0 0 0', spin=0, level=1, basis='6-31g')
    egs = mol.train(
      20,
      lr=1e-2,
      seed=123,
      optimizer='adam',
      converge_threshold=1e-5,
      batchsize=1000,
    )
    self.assertTrue(egs < -72. and egs > -75)


if __name__ == "__main__":
  absltest.main()
