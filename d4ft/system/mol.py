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

from __future__ import annotations  # forward declaration

from typing import Callable, Dict, List, Literal, NamedTuple, Optional, Tuple

import haiku as hk
import jax.numpy as jnp
import pyscf
from absl import logging
from d4ft.system.geometry import get_mol_geometry
from d4ft.system.utils import periodic_hash_table
from d4ft.types import MoCoeffFlat
from jaxtyping import Array, Float, Int


def get_atom_from_geometry(geometry: str) -> List[str]:
  return [s.strip().split(' ')[0] for s in geometry.strip().split('\n')]


def get_spin(atoms: List[str]) -> int:
  tot_ele = sum([periodic_hash_table[a] for a in atoms])
  spin = tot_ele % 2
  return spin


def get_pyscf_mol(
  name: str,
  basis: str,
  source: Literal["cccdbd", "pubchem"] = "cccdbd"
) -> pyscf.gto.mole.Mole:
  """Construct a pyscf mole object from molecule name and basis name"""
  geometry = get_mol_geometry(name, source)
  atoms = get_atom_from_geometry(geometry)
  spin = get_spin(atoms)
  mol = pyscf.gto.M(atom=geometry, basis=basis, spin=spin)
  logging.info(f"spin: {spin}, geometry: {geometry}")
  return mol


def get_occupation_mask(tot_electron: int, nao: int,
                        spin: int) -> Int[Array, "2 nao"]:
  nocc = jnp.zeros([2, nao], dtype=int)
  nmo_up = (tot_electron + spin) // 2
  nmo_dn = (tot_electron - spin) // 2
  nocc = nocc.at[0, :nmo_up].set(1)
  nocc = nocc.at[1, :nmo_dn].set(1)
  return nocc.astype(int)


def sqrt_root_inv(mat: Float[Array, "a a"]) -> Float[Array, "a a"]:
  """Square root of inverse."""
  v, u = jnp.linalg.eigh(mat)
  v = jnp.clip(v, a_min=0)
  v = jnp.diag(jnp.real(v)**(-1 / 2))
  ut = jnp.real(u).transpose()
  return jnp.matmul(v, ut)


class Mol(NamedTuple):
  """Represents a molecular system.
  APIs are mostly consistent with pyscf."""
  tot_electrons: int
  """total number of electrons"""
  spin: int
  """total spin"""
  nocc: Int[Array, "2 nao"]
  """occupation mask for alpha and beta spin"""
  nao: int
  """number of atomic orbitals"""
  atom_coords: Float[Array, "n_atoms 3"]
  """atom centers"""
  atom_charges: Int[Array, "n_atoms"]
  """charges for each atoms"""
  elements: List[str]
  """list of atoms"""
  basis: Dict[str, List[Tuple[int, List[List[float]]]]]
  """CGTO basis parameter in format
  {element: [[shell, [exponenet, coeff], ...], ...]}"""
  ovlp: Float[Array, "nao nao"]
  """overlap matrix"""

  @staticmethod
  def from_pyscf_mol(mol: pyscf.gto.mole.Mole) -> Mol:
    """Builds Mol object from pyscf Mole"""
    tot_electrons = mol.tot_electrons()
    nocc = get_occupation_mask(tot_electrons, mol.nao, mol.spin)

    # TODO: how to move this out?
    ovlp = mol.intor('int1e_ovlp_sph')

    return Mol(
      tot_electrons, mol.spin, nocc, mol.nao, mol.atom_coords(),
      mol.atom_charges(), mol.elements, mol._basis, ovlp
    )

  @staticmethod
  def from_mol_name(
    name: str,
    basis: str,
    source: Literal["cccdbd", "pubchem"] = "cccdbd"
  ) -> Mol:
    """Builds Mol object from molecule name"""
    pyscf_mol = get_pyscf_mol(name, basis, source)
    return Mol.from_pyscf_mol(pyscf_mol)

  # TODO: move this out
  def get_mo_coeff(
    self, rks: bool, ortho_fn: Optional[Callable] = None
  ) -> MoCoeffFlat:
    """Function to return MO coefficient. Must be haiku transformed."""
    nmo = self.nao
    shape = ([nmo, nmo] if rks else [2, nmo, nmo])

    mo_coeff = hk.get_parameter(
      "mo_params",
      shape,
      init=hk.initializers.RandomNormal(stddev=1 / jnp.sqrt(nmo))
    )

    if ortho_fn:
      # ortho_fn provide a parameterization of the generalized Stiefel manifold
      # where (CSC=I), i.e. overlap matrix in Roothann equations is identity.
      mo_coeff = ortho_fn(mo_coeff) @ sqrt_root_inv(self.ovlp)

    if rks:  # restrictied mo
      mo_coeff_spin = jnp.repeat(mo_coeff[None], 2, 0)  # add spin axis
    else:
      mo_coeff_spin = mo_coeff
    mo_coeff_spin *= self.nocc[:, :, None]  # apply spin mask
    mo_coeff_spin = mo_coeff_spin.reshape(-1, nmo)  # flatten
    return mo_coeff_spin
