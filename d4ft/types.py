from typing import Callable, Dict, List, NamedTuple, Union

import jax.numpy as jnp
import jraph
import numpy as np

Array = Union[jnp.ndarray, np.ndarray]


class Metrics(NamedTuple):
  total_loss: float
  e_loss: float
  kin_loss: float
  ext_loss: float
  eri_loss: float
  xc_loss: float
  grad_loss: float


class Energies(NamedTuple):
  e_total: float
  e_kin: float
  e_ext: float
  e_eri: float
  e_xc: float
  e_nuc: float


class Grads(NamedTuple):
  kin_grad: Array
  ext_grad: Array
  eri_grad: Array
  xc_grad: Array


Data = jraph.GraphsTuple


class Transition(NamedTuple):
  """A transition on the DFT GD optimization trajectory"""
  mo_coeff: Array
  energies: Energies
  grads: Grads


Trajectory = List[Transition]


class Hamiltonian(NamedTuple):
  ovlp: Array
  """overlap matrix (n_gto, n_gto)"""
  kin: Array
  """kinetic tensor in symmetry reduced form (unique_ab_idx,)"""
  ext: Array
  """nuclear attraction tensor in symmetry reduced form (unique_ab_idx,)"""
  eri: Array
  """electron repulsion tensor in symmetry reduced form (unique_abcd_idx,)"""
  mo_ab_idx: Array
  """(unique_ab_idx,)"""
  mo_counts_ab: Array
  """counts of (unique_ab_idx,)"""
  mo_abcd_idx: Array
  """(unique_abcd_idx,)"""
  mo_counts_abcd: Array
  """counts of (unique_abcd_idx,)"""
  xc_intor: Callable
  """xc functional"""
  nocc: Array
  """occupation mask"""
  mo_coeff_fn: Callable
  """callable"""
  nuclei: Dict
  """nuclei"""
  e_nuc: float
  """nuclear repulsion energy. since we are not doing geometry optimization,
  this term is fixed"""
