"""Grids that are differentiable wrt the atomic coordinates.

Rewrite of the original PySCF code to make it differentiable wrt
the atomic coordinates.
"""

import jax
import jax.numpy as jnp
from pyscf.data.elements import charge as elements_proton
from pyscf.dft import gen_grid, radi


def treutler_atomic_radii_adjust(mol, atomic_radii):
  charges = [elements_proton(x) for x in mol.elements]
  rad = jnp.sqrt(atomic_radii[charges]) + 1e-200
  rr = rad.reshape(-1, 1) * (1. / rad)
  a = .25 * (rr.T - rr)
  a = a.at[a < -0.5].set(-0.5)
  a = a.at[a > 0.5].set(0.5)

  def fadjust(i, j, g):
    g1 = g**2
    g1 -= 1.
    g1 *= -a[i, j]
    g1 += g
    return g1

  return fadjust


def inter_distance(coords):
  rr = jnp.linalg.norm(coords.reshape(-1, 1, 3) - coords, axis=2)
  return rr.at[jnp.diag_indices(rr.shape[0])].set(0.)


def original_becke(g):
  g = (3 - g**2) * g * .5
  g = (3 - g**2) * g * .5
  g = (3 - g**2) * g * .5
  return g


def get_partition(
  mol,
  atom_coords,
  atom_grids_tab,
  radii_adjust=treutler_atomic_radii_adjust,
  atomic_radii=radi.BRAGG_RADII,
  becke_scheme=original_becke,
  concat=True
):
  if callable(radii_adjust) and atomic_radii is not None:
    f_radii_adjust = radii_adjust(mol, atomic_radii)
  else:
    f_radii_adjust = None
  atm_dist = inter_distance(atom_coords)  # [natom, natom]

  def gen_grid_partition(coords):
    ngrids = coords.shape[0]
    dc = coords[None] - atom_coords[:, None]
    grid_dist = jnp.sqrt(jnp.einsum('ijk,ijk->ij', dc, dc))  # [natom, ngrid]
    pbecke = jnp.ones((mol.natm, ngrids))  # [natom, ngrid]

    ix, jx = jnp.tril_indices(mol.natm, k=-1)

    def pbecke_g(i, j):
      g = 1 / atm_dist[i, j] * (grid_dist[i] - grid_dist[j])
      if f_radii_adjust is not None:
        g = f_radii_adjust(i, j, g)
      g = becke_scheme(g)
      return g

    g = jax.vmap(pbecke_g)(ix, jx)
    pbecke = pbecke.at[ix].mul(0.5 * (1. - g))
    pbecke = pbecke.at[jx].mul(0.5 * (1. + g))
    return pbecke

  coords_all = []
  weights_all = []
  for ia in range(mol.natm):
    coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
    coords = coords + atom_coords[ia]  # [ngrid, 3]
    pbecke = gen_grid_partition(coords)  # [natom, ngrid]
    weights = vol * pbecke[ia] * (1. / jnp.sum(pbecke, axis=0))
    coords_all.append(coords)
    weights_all.append(weights)

  if concat:
    coords_all = jnp.vstack(coords_all)
    weights_all = jnp.hstack(weights_all)
  return coords_all, weights_all


class DifferentiableGrids(gen_grid.Grids):
  """Differentiable alternative to the original pyscf.gen_grid.Grids."""

  def build(self, atom_coords):
    mol = self.mol
    atom_grids_tab = self.gen_atomic_grids(
      mol, self.atom_grid, self.radi_method, self.level, self.prune
    )
    coords, weights = get_partition(
      mol,
      atom_coords,
      atom_grids_tab,
      treutler_atomic_radii_adjust,
      self.atomic_radii,
      original_becke,
    )
    return coords, weights
