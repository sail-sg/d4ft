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
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import pyscf
from pyscf import scf
from pyscf.lib import logger

from d4ft.types import RDM1, MoCoeff
from d4ft.config import D4FTConfig, KSDFTConfig, HFConfig

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'


def pyscf_wrapper(
  mol: pyscf.gto.mole.Mole,
  cfg: D4FTConfig,
  verbose: int = 2,
  rdm1: Optional[RDM1] = None,
) -> Tuple[Any, MoCoeff]:

  method_cfg: Union[HFConfig, KSDFTConfig] = cfg.method_cfg

  if method_cfg.name == "HF":
    if method_cfg.restricted:
      atom_mf = scf.RHF(mol)
    else:
      atom_mf = scf.UHF(mol)

  elif method_cfg.name == "KS":
    if method_cfg.restricted:
      atom_mf = scf.RKS(mol)
    else:
      atom_mf = scf.UKS(mol)

    atom_mf.xc = method_cfg.xc_type
    atom_mf.grids.level = cfg.intor_cfg.quad_level

  atom_mf.verbose = verbose
  atom_mf.max_cycle = cfg.solver_cfg.epochs

  if rdm1 is not None:
    atom_mf.kernel(dm0=rdm1)
    # atom_mf.mo_coeff = rdm1.T
    # atom_mf.kernel()
  else:
    atom_mf.kernel()

  atom_mf.analyze(verbose=logger.INFO)

  if method_cfg.restricted:
    mo_coeff = atom_mf.mo_coeff.T
    mo_coeff = np.repeat(mo_coeff[None], 2, 0)  # add spin axis
  else:
    mo_coeff = np.transpose(atom_mf.mo_coeff, (0, 2, 1))

  return atom_mf, mo_coeff
