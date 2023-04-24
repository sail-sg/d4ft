import os

import jax
import numpy as np
from pyscf import scf
from pyscf.lib import logger

from d4ft.energy import calc_energy, get_intor
from d4ft.integral.obara_saika.utils import mol_to_obsa_gto
from d4ft.logger import RunLogger
from d4ft.molecule import Molecule

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'


def pyscf_solver(
  mol: Molecule,
  verbose: int = 2,
  max_cycle: int = 2,
  os_scheme: str = "none",
):
  if mol.restricted_mo:
    atom_mf = scf.RKS(mol.pyscf_mol)
  else:
    atom_mf = scf.UKS(mol.pyscf_mol)

  atom_mf.xc = mol.xc
  atom_mf.grids.level = mol.level

  atom_mf.verbose = verbose
  atom_mf.max_cycle = max_cycle

  atom_mf.kernel()

  atom_mf.analyze(verbose=logger.INFO)

  if mol.restricted_mo:
    mo_params = atom_mf.mo_coeff.T
  else:
    mo_params = np.transpose(atom_mf.mo_coeff, (0, 2, 1))

  params = (mo_params, None)

  batch = (mol.grids, mol.weights)

  gto, sto_to_gto = mol_to_obsa_gto(mol)
  e_kwargs = {}
  if os_scheme != "none":
    e_kwargs["mol"] = mol
    e_kwargs["gto"] = gto
    e_kwargs["sto_to_gto"] = sto_to_gto

  intors = get_intor(mol, mol.grids[0], 137, False, mol.xc, os_scheme)

  @jax.jit
  def energy(params):
    return calc_energy(intors, mol.nuclei, params, batch, batch, None)

  d4ft_logger = RunLogger()
  e_total, energies = energy(params)
  d4ft_logger.log_step(energies, 0)

  d4ft_logger.log_summary()

  return e_total, params, d4ft_logger
