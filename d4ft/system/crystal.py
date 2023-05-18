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
"""
Crystal unit cell
Ref: https://pyscf.org/pyscf_api_docs/pyscf.pbc.gto.html
"""
from __future__ import annotations  # forward declaration

from pathlib import Path
from typing import List, NamedTuple, Union

import ase
import ase.io
import jax.numpy as jnp
import numpy as np
from d4ft.constants import ANGSTRONG_TO_BOHR
from jaxtyping import Array, Float, Int


class Crystal(NamedTuple):
  """A crystal"""
  spin: int
  """the number of unpaired electrons 2S, i.e. the difference between
  the number of alpha and beta electrons."""
  atom_coords: Float[Array, "n_atoms 3"]
  """atom centers"""
  atom_charges: Int[Array, "n_atoms"]
  """charges for each atoms"""
  elements: List[str]
  """list of atoms"""

  # crystal specific fields
  cell: Float[Array, "3 3"]
  """real space cell (unit Bohr), which is a 3x3 matrix representing
  the three 3D lattice vectors."""
  vol: float
  """real space cell volume"""

  @property
  def reciprocal_cell(self) -> Float[Array, "3 3"]:
    """reciprocal space cell in absolute value (unit 1/Bohr),
    which equals to 2pi * inv(cell)"""
    return 2 * jnp.pi * jnp.linalg.inv(self.cell)

  @property
  def n_atom_in_cell(self) -> int:
    return len(self.atom_coords)

  @property
  def n_electron_in_cell(self) -> int:
    return np.sum(self.atom_charges)

  @staticmethod
  def from_name_and_lattice(
    crystal_name: str,
    position: Float[np.ndarray, "n_atoms 3"],
    cell_angstrong: Float[np.ndarray, "3 3"],
    spin: int = 0,
  ) -> Crystal:
    # TODO: calculate position and cell from lattice constant
    ase_atoms = ase.Atoms(crystal_name, position, cell=cell_angstrong, pbc=True)
    return Crystal.from_ase_cell(ase_atoms, spin)

  @staticmethod
  def from_ase_cell(ase_atoms: ase.Atoms, spin: int = 0) -> Crystal:
    """Construct crystal from an ase.Atoms object"""
    cell = ase_atoms.cell.array * ANGSTRONG_TO_BOHR
    atom_coords = jnp.array(ase_atoms.get_positions())
    atom_charges = jnp.array(ase_atoms.get_atomic_numbers())  # 1d array
    vol = ase_atoms.cell.volume * ANGSTRONG_TO_BOHR**3
    return Crystal(
      spin, atom_coords, atom_charges, ase_atoms.get_chemical_symbols(),
      jnp.array(cell), vol
    )

  @staticmethod
  def from_xyz_file(file_path: Union[Path, str], spin: int = 0) -> Crystal:
    ase_atoms = ase.io.read(file_path)
    assert isinstance(ase_atoms, ase.Atoms)
    return Crystal.from_ase_cell(ase_atoms, spin)
