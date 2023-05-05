from typing import Callable, List, NamedTuple, Tuple, Union

import numpy as np
from jaxtyping import Array, Float, Int

IdxCount2C = Int[Array, "batch 3"]
"""2c GTO index concatenated with the repetition count
 of that idx, e.g. (0,1|2)."""
IdxCount4C = Int[Array, "batch 5"]
"""4c GTO index concatenated with the repetition count
 of that idx, e.g. (0,0,1,0|4)."""
IdxCount = Union[IdxCount2C, IdxCount4C]

MoCoeff = Float[Array, "2 nmo nao"]
"""MO coefficient matrix"""

ETensorsIncore = Tuple[Float[Array, "ab"], Float[Array, "ab"], Float[Array,
                                                                     "abcd"]]
"""kin, ext and eri tensor incore"""


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
  e_eri: Float[Array, ""]
  e_xc: Float[Array, ""]
  e_nuc: Float[Array, ""]


class Grads(NamedTuple):
  kin_grad: Array
  ext_grad: Array
  eri_grad: Array
  xc_grad: Array


Aux = Tuple[Energies, Grads]


class Transition(NamedTuple):
  """A transition on the DFT GD optimization trajectory"""
  mo_coeff: Array
  energies: Energies
  grads: Grads


Trajectory = List[Transition]


# TODO: consider PBC / plane wave
class Hamiltonian(NamedTuple):
  """CGTO hamiltonian"""

  kin_fn: Callable
  """Maps mo_coeff to kinetic energy."""
  ext_fn: Callable
  """Maps mo_coeff to external (nuclera attraction) energy."""
  eri_fn: Callable
  """Maps mo_coeff to electronic repulsion energy."""
  xc_fn: Callable
  """XC functional."""
  nuc_fn: Callable
  """Nuclear repulsion energy fn, which only depends on the geometry."""
  mo_coeff_fn: Callable
  """Function to get MO coefficient."""
  energy_fn: Callable
  """Function to get total energy. Can be used as the loss function."""


HamiltonianHKFactory = Callable[[], Tuple[Callable, Hamiltonian]]

CGTOIntors = Tuple[Callable, Callable, Callable]
"""intor for kin, ext and eri."""
