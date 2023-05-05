import os

import numpy as np
import pyscf
from pyscf import scf
from pyscf.lib import logger

from d4ft.types import MoCoeff

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'


def pyscf(
  mol: pyscf.gto.mole.Mole,
  rks: bool,
  xc: str = "lda",
  quad_level: int = 1,
  verbose: int = 2,
  max_cycle: int = 2
) -> MoCoeff:
  if rks:
    atom_mf = scf.RKS(mol)
  else:
    atom_mf = scf.UKS(mol)

  atom_mf.xc = xc
  atom_mf.grids.level = quad_level

  atom_mf.verbose = verbose
  atom_mf.max_cycle = max_cycle

  atom_mf.kernel()

  atom_mf.analyze(verbose=logger.INFO)

  if rks:
    mo_coeff = atom_mf.mo_coeff.T
    mo_coeff = np.repeat(mo_coeff[None], 2, 0)  # add spin axis
  else:
    mo_coeff = np.transpose(atom_mf.mo_coeff, (0, 2, 1))

  return mo_coeff
