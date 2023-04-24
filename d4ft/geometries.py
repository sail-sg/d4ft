"""Geometry for some molecules."""

from typing import List, Optional, Literal

import pubchempy
import pyscf
import d4ft.fullerene
import d4ft.cccdbd

periodic_table: List[str] = [
  '?', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
  'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
  'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
  'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
  'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
  'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
  'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
]

periodic_hash_table = {}
for atomic_number, atom in enumerate(periodic_table):
  periodic_hash_table[atom] = atomic_number


def get_atom_from_geometry(geometry: str) -> List[str]:
  return [s.split(' ')[0] for s in geometry.strip().split('\n')]


def get_spin(atoms: List[str]) -> int:
  tot_ele = sum([periodic_hash_table[a] for a in atoms])
  spin = tot_ele % 2
  return spin


def get_pubchem_geometry(name: str) -> str:
  pubchem_mol = pubchempy.get_compounds(name, 'name', record_type='3d')
  # If the 3-D geometry isn't available, get the 2-D geometry instead.
  if not pubchem_mol:
    pubchem_mol = pubchempy.get_compounds(name, 'name', record_type='2d')
  pubchem_geometry = pubchem_mol[0].to_dict(properties=['atoms'])['atoms']
  geometry = "".join(
    [
      f"{a['element']}  {a['x']:.5f}, {a['y']:.5f}, {a.get('z', 0):.5f};\n"
      for a in pubchem_geometry
    ]
  )
  return geometry


def get_cccdbd_geometry(name: str) -> str:
  return getattr(d4ft.cccdbd, f"{name}_geometry")


def get_fullerene_geometry(name: str) -> str:
  return getattr(d4ft.fullerene, f"{name}_geometry", None)


def get_pyscf_mol(
  name: str,
  basis: str,
  source: Literal["cccdbd", "pubchem"] = "cccdbd"
) -> pyscf.gto.mole.Mole:
  """Construct a pyscf mole object from molecule name and basis name"""
  # check if it is fullerene
  geometry = get_fullerene_geometry(name)
  if geometry is None:
    if source == "cccdbd":
      geometry = get_cccdbd_geometry(name)
    else:
      geometry = get_pubchem_geometry(name)
  # determine spin
  atoms = get_atom_from_geometry(geometry)
  spin = get_spin(atoms)
  mol = pyscf.gto.M(atom=geometry, basis=basis, spin=spin)
  return mol
