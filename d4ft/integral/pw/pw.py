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

from typing import Callable, NamedTuple, Optional

import ase.dft.kpoints
import einops
import haiku as hk
import jax
import jax.numpy as jnp
from absl import logging
from d4ft.system.crystal import Crystal
from d4ft.system.occupation import get_occupation_mask
from d4ft.types import PWCoeff, Vec3DInt
from d4ft.utils import vmap_to_3d
from jaxtyping import Array, Float, Int


def cut_half(n: int) -> Int[Array, "n"]:
  """return [0, 1, 2, ..., n/2-1, -n/2, -n/2+1, ..., -1] if n is even
      [0, 1, ..., n//2, -n//2, -n//2+1, ..., -1]
  """
  return jnp.arange(n).at[(n // 2 + 1):].add(-n)


def cell_to_vec(cell: Float[Array, "3 3"],
                n_g_pts: Vec3DInt) -> Float[Array, "n_gx, n_gy, n_gz, 3"]:
  Gx, Gy, Gz = map(cut_half, n_g_pts)
  Cx, Cy, Cz = [einops.rearrange(Ci, "d -> 1 1 1 d") for Ci in cell]
  vec = (
    einops.einsum(Gx, Cx, "x, x y z d -> x y z d") +
    einops.einsum(Gy, Cy, "y, x y z d -> x y z d") +
    einops.einsum(Gz, Cz, "z, x y z d -> x y z d")
  )
  return vec


class PW(NamedTuple):
  """Plane Wave"""
  n_g_pts: Vec3DInt
  """number of G points in each spatial direction"""
  n_k_pts: Vec3DInt
  """number of k (reciprocal) points in each spatial direction"""
  k_pts: Float[Array, "total_n_k_pts 3"]
  """a flat list of k point coordinates in absolute value (unit 1/Bohr)"""
  e_cut: float
  """energy cutoff for the plane wave basis set"""
  nocc: Int[Array, "2 ele k"]
  """occupation mask for alpha and beta spin"""
  r_vec: Float[Array, "n_gx, n_gy, n_gz, 3"]
  """lattice constant vectors in real space"""
  g_vec: Float[Array, "n_gx, n_gy, n_gz, 3"]
  """G points (lattice constant vectors in reciprocal space)"""
  vol: float
  """real space cell volume"""

  @property
  def tot_electrons(self) -> int:
    return self.nocc.shape[1]

  @property
  def tot_k_pts(self) -> int:
    return self.nocc.shape[2]

  @property
  def tot_g_pts(self) -> Int[Array, ""]:
    return jnp.prod(self.n_g_pts)

  # @property
  # def half_n_g_pts(self) -> Vec3DInt:
  #   return self.n_g_pts // 2

  @staticmethod
  def from_crystal(
    crystal: Crystal,
    n_g_pts: Vec3DInt,
    n_k_pts: Vec3DInt,
    e_cut: float = 0.0,
  ) -> PW:
    # TODO: what does monkhorst do?
    k_pts = ase.dft.kpoints.monkhorst_pack(n_k_pts) @ crystal.reciprocal_cell
    tot_kpts = len(k_pts)
    tot_electrons = crystal.n_electron_in_cell

    # TODO: soft occupancy
    nocc = get_occupation_mask(
      tot_electrons, tot_electrons * tot_kpts, crystal.spin
    ).reshape(2, tot_electrons, tot_kpts)

    g_vec = cell_to_vec(crystal.reciprocal_cell, n_g_pts)
    r_vec = cell_to_vec(crystal.cell / n_g_pts, n_g_pts)

    return PW(n_g_pts, n_k_pts, k_pts, e_cut, nocc, g_vec, r_vec, crystal.vol)

  def get_pw_coeff(
    self, polarized: bool, ortho_fn: Optional[Callable] = None
  ) -> PWCoeff:
    # Apply ECUT
    g_norm: Float[Array, "n_gx n_gy n_gz"] = vmap_to_3d(jnp.linalg.norm)(
      self.g_vec
    )
    # TODO: why e_cut*2?
    g_selected = g_norm**2 <= self.e_cut * 2
    n_g_selected = jnp.sum(g_selected)
    n_g_selected = jax.device_get(n_g_selected).item()  # cast to float
    g_select_ratio = n_g_selected / self.tot_g_pts
    logging.info(f"{g_select_ratio*100:.2f}% frequency selected.")
    logging.info(f"G grid: {self.n_g_pts}")
    if n_g_selected < self.tot_electrons:
      raise ValueError(
        "the number of G vector selected is smaller than "
        "the number of electrons."
      )

    # TODO: why initialize like this?
    shape = [
      2 if polarized else 1, self.tot_electrons, self.tot_k_pts, n_g_selected
    ]
    maxval = 1. / 2. / n_g_selected
    pw_coeff = hk.get_parameter(
      "mo_params", shape, init=hk.initializers.RandomUniform(maxval=maxval)
    )

    if ortho_fn:  # orthogonalize the plane wave coefficients
      # NOTE: For non-square matrix of size (a,b) where a<b, QR returns
      # orthogonal column vectors of shape (a,a). Here we transpose the
      # matrix in order to make k points orthogonal.
      pw_coeff = einops.rearrange(pw_coeff, "spin ele k g -> spin g k ele")
      pw_coeff = jax.vmap(ortho_fn, in_axes=2, out_axes=2)(pw_coeff)
      pw_coeff = einops.rearrange(pw_coeff, "spin g k ele -> spin ele k g")

    @vmap_to_3d
    def select_g_pts(
      x: Float[Array, "spin ele k g"]
    ) -> Float[Array, "spin ele k g d"]:
      return jnp.zeros(self.n_g_pts).at[g_selected].set(x)

    return select_g_pts(pw_coeff)

  def eval(self, pw_coeff: PWCoeff):
    """Get density in real and reciprocal space, i.e. n(r) and n(G).

    The total density in real space over electron index i and k point k is
    .. math::
    n(r) = 1/N_k sum_i sum_{k\in Brillouin Zone} f(e_{ik}) |psi_{ik}(r)|^2

    where psi_{ik} is the wave function for electron i at k point k, which
    is a linear combination of plane waves with wavevector k,
    using coefficients pw_coeff.

    The total density in the reciproal space can be obtained by Fourier

    Args:
        params (_type_): _description_
    Return:
        (n_r:ndarray, n_g:ndarray)
        n_r: [N1, N2, N3]
        n_g: [N1, N2, N3]
    """
    # TODO: check the math here
    psi_r: PWCoeff = self.tot_g_pts * vmap_to_3d(jnp.fft.ifftn)(pw_coeff)
    density_r = jnp.abs(psi_r)**2
    # normalization over spatial dims
    density_r /= einops.reduce(
      density_r, "spin ele k n_gx n_gy n_gz -> spin ele k 1 1 1", "sum"
    )
    # TODO: soft occupancy
    nocc = einops.rearrange(self.nocc, "spin ele k -> spin ele k 1 1 1")
    density_r_total = einops.reduce(
      density_r * nocc, "spin ele k n_gx n_gy n_gz -> n_gx n_gy n_gz", "sum"
    )
    const = self.tot_electrons / self.vol * self.tot_g_pts
    density_r_total = density_r_total / jnp.sum(density_r_total) * const
    n_g_total = jnp.fft.fftn(density_r_total) / self.tot_g_pts * self.vol
    return density_r_total, n_g_total
