"""GTO related functions"""

from enum import Enum
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pyscf
from absl import logging

from d4ft.types import Array


class GTO(NamedTuple):
  """molecular orbital"""
  angular: Array
  """(N, 3)"""
  center: Array
  """(N, 3)"""
  exponent: Array
  """(N,)"""
  coeff: Array
  """(N,)"""


class OrbType(Enum):
  """https://pyscf.org/user/gto.html#basis-set"""
  s = 0
  p = 1
  d = 2
  f = 3


ANGULAR = {
  OrbType.s: [[0, 0, 0]],
  OrbType.p: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
}


def mol_to_gto(mol: pyscf.gto.mole.Mole):
  """Transform pyscf mol object to GTOs.

  Returns:
    all translated GTOs. STO TO GTO
  """
  all_gtos = []
  ao_to_gto = []
  for i, element in enumerate(mol.elements):
    coord = mol.atom_coord(i)
    for sto in mol._basis[element]:
      orb_type = OrbType(sto[0])
      gtos = sto[1:]
      for angular in ANGULAR[orb_type]:
        ao_to_gto.append(len(gtos))
        for exponent, coeff in gtos:
          all_gtos.append((angular, coord, exponent, coeff))
  all_gtos = GTO(*(jnp.array(np.stack(a, axis=0)) for a in zip(*all_gtos)))
  n_gto = sum(ao_to_gto)
  logging.info(f"there are {n_gto} GTOs")
  return all_gtos, tuple(ao_to_gto)


def sqrt_root_inv(mat):
  """Square root of inverse."""
  v, u = jnp.linalg.eigh(mat)
  v = jnp.clip(v, a_min=0)
  v = jnp.diag(jnp.real(v)**(-1 / 2))
  ut = jnp.real(u).transpose()
  return jnp.matmul(v, ut)


def qr_factor_param(params, ovlp, rks: bool = True):
  """Parametrize the generalized Stiefel manifold (CS^{-1/2}C=I)
  with qr factor"""
  orthogonal, _ = jnp.linalg.qr(params)
  transpose_axis = (1, 0) if rks else (0, 2, 1)
  orthogonal = jnp.transpose(orthogonal, transpose_axis)
  mo_coeff = orthogonal @ sqrt_root_inv(ovlp)
  return mo_coeff


def get_occupation_mask(mol: pyscf.gto.mole.Mole, spin: int = 0):
  tot_electron = mol.tot_electrons()
  nao = mol.nao  # number of atomic orbitals
  nocc = jnp.zeros([2, nao])  # number of occupied orbital.
  nmo_up = (tot_electron + spin) // 2
  nmo_dn = (tot_electron - spin) // 2
  nocc = nocc.at[0, :nmo_up].set(1)
  nocc = nocc.at[1, :nmo_dn].set(1)
  return nocc
