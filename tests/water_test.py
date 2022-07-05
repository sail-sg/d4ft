"""Test water molecule."""

import os
import sys
from absl.testing import absltest

sys.path.append('.')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'

import jdft
from jdft.geometries import h2o_geometry


class WaterTest(absltest.TestCase):

  def test_energy(self):
    mol = jdft.molecule(h2o_geometry, spin=0, level=1, basis='6-31g')
    egs = mol.train(
      30,
      lr=1e-2,
      seed=123,
      optimizer='adam',
      converge_threshold=1e-5,
      batchsize=1000,
    )
    self.assertTrue(egs < -75.)


if __name__ == "__main__":
  absltest.main()
