import sys

sys.path.append('..')

import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'

import d4ft
from d4ft.geometries import c60_geometry

mol = d4ft.Molecule(c60_geometry, spin=0, level=1, basis='6-31g')

mol.train(
  100,
  lr=1e-2,
  seed=123,
  optimizer='adam',
  converge_threshold=1e-5,
  batchsize=5000,
)

print('PySCF results: -2241.2')
