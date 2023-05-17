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
Plane Wave
"""
from __future__ import annotations  # forward declaration

from d4ft.system.crystal import Crystal
from typing import List, NamedTuple, Union
import jax.numpy as jnp

import ase.dft.kpoints
from jaxtyping import Array, Float, Int
from d4ft.types import Vec3DInt


class PW(NamedTuple):
  """Plane Wave"""
  n_g_pts: Vec3DInt
  """mesh for the G points"""
  n_k_pts: Vec3DInt
  """mesh for the k (reciprocal) points"""
  k_pts: Float[Array, "total_n_k_pts 3"]
  """k points in absolute value (unit 1/Bohr)"""

  @property
  def half_n_g_pts(self) -> Vec3DInt:
    return self.n_g_pts // 2

  @staticmethod
  def from_crystal(
    n_g_pts: Vec3DInt, n_k_pts: Vec3DInt, crystal: Crystal
  ) -> PW:
    k_pts = ase.dft.kpoints.monkhorst_pack(n_k_pts) @ crystal.reciprocal_cell
    return PW(n_g_pts, n_k_pts, k_pts)
