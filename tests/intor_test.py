import jax
from absl.testing import absltest

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'

import jdft
from jdft.geometries import h2_geometry

from jdft.intor import Quadrature

eps = 1e-2

class IntorTest(absltest.TestCase):

  def test_quadrature(self):
    mol = jdft.molecule(h2_geometry, spin=0, level=1, basis='6-31g')
    intor = Quadrature(mol.ao, mol.grids, mol.weights)
    self.assertTrue(jax.numpy.abs(intor.single2()-mol.nao)<eps)


if __name__ == "__main__":
  absltest.main()