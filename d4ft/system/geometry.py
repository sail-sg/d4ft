from typing import Literal, Optional

import d4ft.system.cccdbd
import d4ft.system.fake_fullerene
import pubchempy
import requests
from d4ft.system.cccdbd import query_geometry_from_cccbdb
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


def get_cccdbd_geometry(name: str) -> str:
  geometry = getattr(d4ft.system.cccdbd, f"{name}_geometry", None)
  if geometry is None:  # no offline data available
    geometry = query_geometry_from_cccbdb(name)
  return geometry


def get_fullerene_geometry(name: str) -> Optional[str]:
  """fullerene name are in the form Cxxx-isomer, e.g.
  C60-lh
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
  name: str, source: Literal["cccdbd", "pubchem"] = "cccdbd"
) -> str:
  if name.capitalize() in periodic_table:  # check if it is a single atom
    geometry = f"{name.capitalize()} 0.0000 0.0000 0.0000"
  else:  # check if it is fullerene
    geometry = get_fullerene_geometry(name)

  if geometry is None:
    if source == "cccdbd":
      geometry = get_cccdbd_geometry(name)
    else:
      geometry = get_pubchem_geometry(name)
  return geometry
