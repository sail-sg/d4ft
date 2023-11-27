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
from typing import Literal, Optional, Tuple

import numpy as np
import pubchempy
import requests
from absl import logging

import d4ft.system.cccdbd
import d4ft.system.fake_fullerene
from d4ft.system.cccdbd import query_geometry_from_cccbdb
from d4ft.system.refdata import get_refdata_geometry
from d4ft.system.utils import periodic_table


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


def get_fullerene_geometry(name: str) -> Optional[str]:
  """fullerene name are in the form Cxxx-isomer, e.g.
  C60-Ih
  C48-C2-199
  C90-C2v-46
  """
  names = name.split("-")
  carbons = names[0]
  isomer = "-".join(names[1:])
  if isomer == "fake":
    return getattr(
      d4ft.system.fake_fullerene, f"{carbons.lower()}_geometry", None
    )
  else:
    res = requests.get(
      f"https://nanotube.msu.edu/fullerene/{carbons}/{name}.xyz"
    )
    if res.status_code == 404:
      return None
    geometry = res.content.decode("utf-8")
    # remove header
    geometry = "\n".join(geometry.split("\n")[2:])
    return geometry


def get_mol_geometry(
  name: str,
  source: Literal["cccdbd", "refdata", "pubchem"] = "cccdbd"
) -> Tuple[str, int, int]:
  geometry: Optional[str] = None
  if ".xyz" in name:
    with open(name, "r") as f:
      geometry = f.read()

  if name.capitalize() in periodic_table:  # check if it is a single atom
    geometry = f"{name.capitalize()} 0.0000 0.0000 0.0000"

  if name[0] == "C":  # check if it is fullerene
    geometry = get_fullerene_geometry(name)

  # try to see if there is offline data available
  if geometry is None:
    here = os.path.abspath(os.path.dirname(__file__))
    xyz_path = f"{here}/xyz_files"
    offline_xyz = [
      f for f in os.listdir(xyz_path) if f == f"{name.lower()}.xyz"
    ]
    if len(offline_xyz) == 1:
      logging.info(f"loading offline geometry from {xyz_path}/{offline_xyz[0]}")
      with open(f"{xyz_path}/{offline_xyz[0]}", "r") as f:
        geometry = f.read()

  charge = np.nan
  spin = -1
  if geometry is None:
    if source == "cccdbd":
      geometry = query_geometry_from_cccbdb(name)
    elif source == "refdata":
      geometry, charge, spin = get_refdata_geometry(name)
    elif source == "refdata":
      geometry = get_pubchem_geometry(name)
    else:
      raise ValueError(f"source {source} not supported")

  assert geometry is not None
  return geometry, charge, spin
