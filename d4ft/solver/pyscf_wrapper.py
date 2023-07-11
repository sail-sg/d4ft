# Copyright 2023 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import pyscf
from pyscf import scf
from pyscf.lib import logger
from typing import Optional

from d4ft.types import MoCoeff, RDM1

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'


def pyscf_wrapper(
  mol: pyscf.gto.mole.Mole,
  rks: bool,
  xc: str = "lda",
  quad_level: int = 1,
  verbose: int = 2,
  max_cycle: int = 50,
  rdm1: Optional[RDM1] = None,
) -> MoCoeff:
  if rks:
    atom_mf = scf.RKS(mol)
  else:
    atom_mf = scf.UKS(mol)

  atom_mf.xc = xc
  atom_mf.grids.level = quad_level

  atom_mf.verbose = verbose
  atom_mf.max_cycle = max_cycle

  if rdm1 is not None:
    # atom_mf.kernel(dm0=rdm1)
    atom_mf.mo_coeff = rdm1.T
    atom_mf.kernel()
  else:
    atom_mf.kernel()

  atom_mf.analyze(verbose=logger.INFO)

  if rks:
    mo_coeff = atom_mf.mo_coeff.T
    mo_coeff = np.repeat(mo_coeff[None], 2, 0)  # add spin axis
  else:
    mo_coeff = np.transpose(atom_mf.mo_coeff, (0, 2, 1))

  return mo_coeff
