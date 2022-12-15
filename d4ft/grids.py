# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import numpy as np
import pyscf
from pyscf.dft import radi
from pyscf.dft import gen_grid
from pyscf import gto
from pyscf import lib

libdft = lib.load_library('libdft')


def original_becke(g):
  '''Becke, JCP 88, 2547 (1988); DOI:10.1063/1.454033'''
  #    This funciton has been optimized in the C code VXCgen_grid
  #    g = (3 - g**2) * g * .5
  #    g = (3 - g**2) * g * .5
  #    g = (3 - g**2) * g * .5
  #    return g
  pass


def get_partition(
  mol,
  atom_grids_tab,
  atom_label=False,
  atomic_radii=pyscf.dft.radi.BRAGG_RADII,
  becke_scheme=original_becke,
  concat=True,
  **kwargs
):
  '''Generate the mesh grid coordinates and weights for DFT numerical
  integration. We can change radii_adjust, becke_scheme functions to generate
  different meshgrid.
  Kwargs:
      concat: bool
          Whether to concatenate grids and weights in return
  Returns:
      grid_coord and grid_weight arrays.  grid_coord array has shape (N,3);
      weight 1D array has N elements.
  '''

  radii_adjust = radi.becke_atomic_radii_adjust
  if callable(radii_adjust) and atomic_radii is not None:
    f_radii_adjust = radii_adjust(mol, atomic_radii)
  else:
    f_radii_adjust = None
  atm_coords = kwargs.get('atom_coords', mol.atom_coords())
  atm_coords = np.asarray(atm_coords, order='C')
  atm_dist = gto.inter_distance(mol, coords=atm_coords)
  if (
    becke_scheme is original_becke and (
      radii_adjust is pyscf.dft.radi.treutler_atomic_radii_adjust or
      radii_adjust is pyscf.dft.radi.becke_atomic_radii_adjust or
      f_radii_adjust is None
    )
  ):
    if f_radii_adjust is None:
      p_radii_table = lib.c_null_ptr()
    else:
      f_radii_table = np.asarray(
        [
          f_radii_adjust(i, j, 0)
          for i in range(mol.natm)
          for j in range(mol.natm)
        ]
      )
      p_radii_table = f_radii_table.ctypes.data_as(ctypes.c_void_p)

    def gen_grid_partition(coords):
      coords = np.asarray(coords, order='F')
      ngrids = coords.shape[0]
      pbecke = np.empty((mol.natm, ngrids))
      libdft.VXCgen_grid(
        pbecke.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        atm_coords.ctypes.data_as(ctypes.c_void_p), p_radii_table,
        ctypes.c_int(mol.natm), ctypes.c_int(ngrids)
      )
      return pbecke

  else:

    def gen_grid_partition(coords):
      ngrids = coords.shape[0]
      grid_dist = np.empty((mol.natm, ngrids))
      for ia in range(mol.natm):
        dc = coords - atm_coords[ia]
        grid_dist[ia] = np.sqrt(np.einsum('ij,ij->i', dc, dc))
      pbecke = np.ones((mol.natm, ngrids))
      for i in range(mol.natm):
        for j in range(i):
          g = 1 / atm_dist[i, j] * (grid_dist[i] - grid_dist[j])
          if f_radii_adjust is not None:
            g = f_radii_adjust(i, j, g)
          g = becke_scheme(g)
          pbecke[i] *= .5 * (1 - g)
          pbecke[j] *= .5 * (1 + g)
      return pbecke

  coords_all = []
  weights_all = []
  atom_all = []

  for ia in range(mol.natm):
    coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
    coords = coords + atm_coords[ia]
    pbecke = gen_grid_partition(coords)
    weights = vol * pbecke[ia] * (1. / pbecke.sum(axis=0))
    coords_all.append(coords)
    weights_all.append(weights)
    atom_all += [ia] * len(weights)
  if concat:
    coords_all = np.vstack(coords_all)
    weights_all = np.hstack(weights_all)
    atom_all = np.hstack(atom_all)

  if atom_label:
    return coords_all, weights_all, atom_all
  else:
    return coords_all, weights_all


def _gen_grid(mol, level=1, atom_label=False, **kwargs):
  return get_partition(
    mol, gen_grid.gen_atomic_grids(mol, level=level), atom_label, **kwargs
  )


def _grid_shift(grids, atoms, atom_coords_old, atom_coords_new):
  atom_shift = atom_coords_new - atom_coords_old
  grid_shift = atom_shift[atoms, :]
  grids += grid_shift
  return grids
