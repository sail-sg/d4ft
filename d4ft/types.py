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

from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import haiku as hk
import jax
import numpy as np
import optax
from jaxtyping import Array, Float, Int
from typing_extensions import TypeAlias

NPArray = Union[np.ndarray, Array]

LatticeDim: TypeAlias = Int[np.ndarray, "3"]
"""Dimension of 3D lattice, which is a 3D vector of integers."""

IdxCount2C: TypeAlias = Int[Array, "batch 3"]
"""2c GTO index concatenated with the repetition count
 of that idx, e.g. (0,1|2)."""
IdxCount4C: TypeAlias = Int[Array, "batch 5"]
"""4c GTO index concatenated with the repetition count
 of that idx, e.g. (0,0,1,0|4)."""
IdxCount = Union[IdxCount2C, IdxCount4C]

# TODO: consider change shape of MO to Float[Array, "2 nao nmo"] which is more
# consistent with the convention
MoCoeff: TypeAlias = Float[Array, "2 nmo nao"]
"""MO coefficient matrix"""
MoCoeffFlat: TypeAlias = Float[Array, "2*nmo nao"]
"""Flattened MO coefficient matrix"""
RDM1: TypeAlias = Float[Array, "2 nao nao"]
"""1-Reduced density matrix, which is the outer product of MO coefficients."""

Fock: TypeAlias = Float[Array, "2 nao nao"]
"""Fock matrix"""
FockFlat: TypeAlias = Float[Array, "2*nao nao"]
"""Flattened Fock matrix"""

PWCoeff: TypeAlias = Float[Array, "spin ele k g x y z"]
"""plane wave coefficients"""

Tensor2C: TypeAlias = Float[Array, "#ab"]
Tensor4C: TypeAlias = Float[Array, "#abcd"]


class CGTOSymTensorIncore(NamedTuple):
  """symmetry reduced ovlp, kin, ext and eri tensor, stored incore"""
  ovlp_ab: Tensor2C
  kin_ab: Tensor2C
  ext_ab: Tensor2C
  eri_abcd: Tensor4C


Cell = Float[NPArray, "3 3"]
"""real / reciprocal space cell represented by a 3x3 matrix consists of
the three 3D lattice vectors."""

QuadGrids: TypeAlias = Float[Array, "#n_grid_pts"]
"""quadrature grids"""
QuadWeights: TypeAlias = Float[Array, "n_grid_pts 1"]
"""quadrature weights"""
QuadGridsNWeights = Tuple[QuadGrids, QuadWeights]

AtomCoords: TypeAlias = Float[Array, "n_atoms 3"]
"""Atom coordinates in real 3D space"""


class AngularStats(NamedTuple):
  min_a: Int[np.ndarray, "3"]
  min_b: Int[np.ndarray, "3"]
  min_c: Int[np.ndarray, "3"]
  min_d: Int[np.ndarray, "3"]
  max_a: Int[np.ndarray, "3"]
  max_b: Int[np.ndarray, "3"]
  max_c: Int[np.ndarray, "3"]
  max_d: Int[np.ndarray, "3"]
  max_ab: Int[np.ndarray, "3"]
  max_cd: Int[np.ndarray, "3"]
  max_xyz: int
  max_yz: int
  max_z: int


class Energies(NamedTuple):
  e_total: Float[Array, ""]
  e_kin: Float[Array, ""]
  e_ext: Float[Array, ""]
  e_har: Float[Array, ""]
  e_xc: Float[Array, ""]
  e_nuc: Float[Array, ""]


class Grads(NamedTuple):
  kin_grad: Array
  ext_grad: Array
  har_grad: Array
  xc_grad: Array


Aux = Tuple[Energies, Optional[Grads]]


class Transition(NamedTuple):
  """A transition on the DFT GD optimization trajectory"""
  mo_coeff: Array
  energies: Energies
  grads: Optional[Grads]


Trajectory = List[Transition]

MoCoeffScalarFn = Union[Callable[[MoCoeffFlat], Float[Array, ""]],
                        Callable[[], Float[Array, ""]]]
"""Functions that maps mo_coeff to a scalar that represents energy.
When used with parameterized mo_coeff, it is composed with the mo_coeff_fn
so it takes no argument."""


class CGTOIntors(NamedTuple):
  """mean-field level intor for kinetic, external and electronic
  repulsion (eri)."""
  ovlp_fn: Callable
  """Maps mo_coeff to overlap matrix."""
  kin_fn: MoCoeffScalarFn
  """Maps mo_coeff to kinetic energy."""
  ext_fn: MoCoeffScalarFn
  """Maps mo_coeff to external (nuclear attraction) energy."""
  har_fn: MoCoeffScalarFn
  """Maps mo_coeff to hartree energy."""
  xc_fn: MoCoeffScalarFn
  """Maps mo_coeff to exact exchange energy, or the XC functional if
  doing KS-DFT."""


# TODO: consider PBC / plane wave
class Hamiltonian(NamedTuple):
  """CGTO hamiltonian"""
  cgto_intors: CGTOIntors
  """function that calculates various related integrals"""
  nuc_fn: MoCoeffScalarFn
  """Nuclear repulsion energy fn, which only depends on the geometry."""
  energy_fn: MoCoeffScalarFn
  """Function to get total energy. Can be used as the loss function."""
  mo_coeff_fn: Optional[Callable[[], MoCoeffFlat]]
  """Function to get MO coefficient."""


HamiltonianHKFactory = Callable[[], Tuple[MoCoeffScalarFn, Hamiltonian]]


class TrainingState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState
  rng_key: jax.Array
