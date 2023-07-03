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
"""Plane Wave"""

from __future__ import annotations  # forward declaration

from typing import Callable, NamedTuple, Optional, Tuple

import ase.dft.kpoints
import einops
import haiku as hk
import jax
import jax.numpy as jnp
from absl import logging
from d4ft.system.crystal import Crystal
from d4ft.system.occupation import get_occupation_mask
from d4ft.types import Cell, LatticeDim, PWCoeff
from d4ft.utils import vmap_3D_lattice
from jaxtyping import Array, Float, Int


def canonical_period(n: int) -> Int[Array, "n"]:
  """Return a list of integers from 0 to n/2-1 and -n/2 to -1, which represent a
  canonical period. Used for computing fourier series.

  Args:
    n: grid size of the period
  Returns:
    If n is even, return [0, 1, 2, ..., n/2-1, -n/2, -n/2+1, ..., -1]
                    else [0, 1, ..., n//2, -n//2, -n//2+1, ..., -1]
  """
  return jnp.arange(n).at[(n // 2 + 1):].add(-n)


def tile_cell_to_lattice(cell: Cell,
                         lattice_dim: LatticeDim) -> Float[Array, "x y z 3"]:
  """Tile the cell according the lattice dimension, i.e.
  give lattice constant vectors in each axis Cx, Cy, Cz,
  create all lattice points nxCx + nyCy + nzCz, where
  ni are the indices.

  Args:
    cell: 3x3 matrix of lattice vectors
  """
  x_idx, y_idx, z_idx = map(canonical_period, lattice_dim)
  Cx, Cy, Cz = [
    einops.rearrange(lattice_constant, "d -> 1 1 1 d")
    for lattice_constant in cell
  ]
  vec = (
    einops.einsum(x_idx, Cx, "x, x y z d -> x y z d") +
    einops.einsum(y_idx, Cy, "y, x y z d -> x y z d") +
    einops.einsum(z_idx, Cz, "z, x y z d -> x y z d")
  )
  return vec


class PW(NamedTuple):
  """Plane Wave"""
  direct_lattice_dim: LatticeDim
  """Dimension of the direct lattice, i.e. the number of k points
  (crystal momenta) in each spatial direction"""
  reciprocal_lattice_dim: LatticeDim
  """Dimension of the reciprocal lattice, i.e. the number of reciprocal lattice
  vectors in each spatial direction"""
  k_pts: Float[Array, "k 3"]
  """a flat list of k point coordinates in absolute value (unit 1/Bohr)"""
  energy_cutoff: float
  """kinetic energy (of G points) cutoff for the plane wave basis set"""
  nocc: Int[Array, "2 ele k"]
  """occupation mask for alpha and beta spin"""
  direct_lattice_vec: Float[Array, "x y z 3"]
  """direct lattice vectors (R points)"""
  reciprocal_lattice_vec: Float[Array, "x y z 3"]
  """reciprocal lattice vectors (G points)"""
  vol: float
  """real space cell volume, used for normalization"""

  @property
  def tot_electrons(self) -> int:
    return self.nocc.shape[1]

  @property
  def tot_k_pts(self) -> int:
    return self.nocc.shape[2]

  @property
  def reciprocal_lattice_size(self) -> Int[Array, ""]:
    return jnp.prod(self.reciprocal_lattice_dim)

  @staticmethod
  def from_crystal(
    crystal: Crystal,
    reciprocal_lattice_dim: LatticeDim,
    direct_lattice_dim: LatticeDim,
    e_cut: float = 0.0,
  ) -> PW:
    # TODO: what does monkhorst do?
    k_pts = ase.dft.kpoints.monkhorst_pack(
      direct_lattice_dim
    ) @ crystal.reciprocal_cell
    tot_kpts = len(k_pts)
    tot_electrons = crystal.n_electron_in_cell

    # TODO: fermi-dirac distribution above zero temp
    nocc = get_occupation_mask(
      tot_electrons, tot_electrons * tot_kpts, crystal.spin
    ).reshape(2, tot_electrons, tot_kpts)

    reciprocal_lattice_vec = tile_cell_to_lattice(
      crystal.reciprocal_cell, reciprocal_lattice_dim
    )

    # TODO: why divide by n_g_pts?
    # TODO: shouldn't we use real space lattice?
    direct_lattice_vec = tile_cell_to_lattice(
      crystal.cell / reciprocal_lattice_dim, reciprocal_lattice_dim
    )

    return PW(
      direct_lattice_dim, reciprocal_lattice_dim, k_pts, e_cut, nocc,
      direct_lattice_vec, reciprocal_lattice_vec, crystal.vol
    )

  def get_pw_coeff(
    self, polarized: bool, ortho_fn: Optional[Callable] = None
  ) -> PWCoeff:
    """Returns the fourier coefficient of the periodic part u(r) of the
    Bloch wavefunction.

    Since the L2 norm is proportional to the kinetic energy of the
    associated planewave, the truncation is done using the L2 norm of the
    reciprocal lattice vectors.

    TODO: shouldn't the size of the reciprocal lattice be determined by the
    energy cutoff?
    """
    g_norm: Float[Array, "x y z"]
    g_norm = vmap_3D_lattice(jnp.linalg.norm)(self.reciprocal_lattice_vec)
    g_selected = g_norm**2 <= self.energy_cutoff * 2
    n_g_selected = jnp.sum(g_selected)
    n_g_selected = jax.device_get(n_g_selected).item()  # cast to float
    g_select_ratio = n_g_selected / self.reciprocal_lattice_size
    logging.info(f"{g_select_ratio*100:.2f}% frequency selected.")
    logging.info(f"G grid: {self.reciprocal_lattice_dim}")
    if n_g_selected < self.tot_electrons:
      raise ValueError(
        "the number of reciprocal lattice vectors selected is smaller than "
        "the number of electrons."
      )

    shape = [
      2 if polarized else 1, self.tot_electrons, self.tot_k_pts, n_g_selected
    ]
    # TODO: why initialize like this?
    maxval = 1. / 2. / n_g_selected
    pw_coeff = hk.get_parameter(
      "mo_params", shape, init=hk.initializers.RandomUniform(maxval=maxval)
    )

    if ortho_fn:  # orthogonalize the plane wave coefficients
      # vmap over k-axis, and make g points coefficients between different electron orthogonal.
      pw_coeff = einops.rearrange(pw_coeff, "spin ele k g -> spin g k ele")
      pw_coeff = jax.vmap(ortho_fn, in_axes=2, out_axes=2)(pw_coeff)
      pw_coeff = einops.rearrange(pw_coeff, "spin g k ele -> spin ele k g")

    @vmap_3D_lattice
    def select_g_pts(
      x: Float[Array, "spin ele k g"]
    ) -> Float[Array, "spin ele k g d"]:
      return jnp.zeros(self.reciprocal_lattice_dim).at[g_selected].set(x)

    return select_g_pts(pw_coeff)

  def density(
    self, pw_coeff: PWCoeff
  ) -> Tuple[Float[Array, "x y z"], Float[Array, "x y z"]]:
    """Get density in real and reciprocal space, i.e. n(r) and n(G).

    The total density in real space over electron index i and k point k
    for spin s is
    .. math::
    n_sik(r) = 1/N_k sum_i sum_{k\\in Brillouin Zone} f(e_{ik}) |psi_{sik}(r)|^2

    where psi_{ik} is the wave function for electron i at k point k, which
    is a linear combination of plane waves with wavevector k,
    using coefficients pw_coeff.

    The total density in the reciproal space can be obtained by Fourier

    Args:
    Return:
    """
    # TODO: check the math here
    # Bloch wavefunction in real space for the i-th electron at k point k
    psi_sik_r: PWCoeff = self.reciprocal_lattice_size * vmap_3D_lattice(
      jnp.fft.ifftn
    )(
      pw_coeff
    )
    density_sik_r = jnp.abs(psi_sik_r)**2
    # normalization over spatial dims
    density_sik_r /= einops.reduce(
      density_sik_r, "spin ele k x y z -> spin ele k 1 1 1", "sum"
    )
    # TODO: full fermi-dirac
    nocc = einops.rearrange(self.nocc, "spin ele k -> spin ele k 1 1 1")
    # sum over spin and electrons to get total density
    density_k_r = einops.reduce(
      density_sik_r * nocc, "spin ele k x y z -> x y z", "sum"
    )
    const = self.tot_electrons / self.vol * self.reciprocal_lattice_size
    density_k_r = density_k_r / jnp.sum(density_k_r) * const
    density_k_g = jnp.fft.fftn(
      density_k_r
    ) / self.reciprocal_lattice_size * self.vol
    return density_k_r, density_k_g
