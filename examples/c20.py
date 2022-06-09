import sys
sys.path.append('..')

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'

import jdft
from jdft.geometries import c20_geometry

mol = jdft.molecule(c20_geometry, spin=0, level=1, basis='6-31g')

mol.train(50, lr=1e-2, seed=123, optimizer='adam', converge_threshold=1e-5, batchsize=1000)

print('PySCF results: -746.4')