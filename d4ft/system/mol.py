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

from typing import Dict, List, Literal, NamedTuple, Tuple

import pyscf
from absl import logging
from d4ft.system.geometry import get_mol_geometry
from d4ft.system.occupation import get_occupation_mask
from d4ft.system.utils import periodic_hash_table
from d4ft.types import AtomCoords
from jaxtyping import Array, Int


def get_atom_from_geometry(geometry: str) -> List[str]:
  return [s.strip().split(' ')[0] for s in geometry.strip().split('\n')]


def get_spin(atoms: List[str]) -> int:
  """Get the spin of the molecule from the list of atoms.
  By default, we try to pair all electrons, so spin is either 0 or 1."""
  tot_ele = sum([periodic_hash_table[a] for a in atoms])
  spin = tot_ele % 2
  return spin


def get_pyscf_mol(
  mol: str,
  basis: str,
  spin: int,
  charge: int,
  source: Literal["cccdbd", "pubchem"] = "cccdbd"
) -> pyscf.gto.mole.Mole:
  """Construct a pyscf mole object from molecule name and basis name"""
  geometry = get_mol_geometry(mol, source)
  atoms = get_atom_from_geometry(geometry)
  if spin == -1:
    spin = get_spin(atoms)
  mol = pyscf.gto.M(atom=geometry, basis=basis, spin=spin, charge=charge)
  logging.info(f"spin: {spin}, charge: {charge}, geometry: {geometry}")
  return mol


class Mol(NamedTuple):
  """Represents a molecular system.
  APIs are mostly consistent with pyscf."""
  spin: int
  """the number of unpaired electrons 2S, i.e. the difference between
  the number of alpha and beta electrons."""
  charge: int
  """charge multiplicity, i.e. number of protons - number of electrons"""
  atom_coords: AtomCoords
  """atom centers"""
  atom_charges: Int[Array, "n_atoms"]
  """charges for each atoms"""
  elements: List[str]
  """list of atoms"""

  # molecular specific fields
  nocc: Int[Array, "2 nao"]
  """occupation mask for alpha and beta spin"""
  nao: int
  """number of atomic orbitals"""
  basis: Dict[str, List[Tuple[int, List[List[float]]]]]
  """CGTO basis parameter in format
  {element: [[shell, [exponenet, coeff], ...], ...]}"""

  @staticmethod
  def from_pyscf_mol(mol: pyscf.gto.mole.Mole) -> Mol:
    """Builds Mol object from pyscf Mole"""
    tot_electrons = mol.tot_electrons()

    nocc = get_occupation_mask(tot_electrons, mol.nao, mol.spin, mol.charge)

    return Mol(
      mol.spin, mol.charge, mol.atom_coords(), mol.atom_charges(), mol.elements,
      nocc, mol.nao, mol._basis
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
