from __future__ import annotations  # forward declaration

from typing import Callable, NamedTuple, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pyscf
import scipy.special
from absl import logging
from d4ft.constants import SHELL_TO_ANGULAR_VEC, Shell
from jaxtyping import Array, Float, Int

_r25 = np.arange(25)
perm_2n_n = jnp.array(scipy.special.perm(2 * _r25, _r25))
"""Precomputed values for (2n)! / n! for n in (0,24).
 Used in GTO normalization and OS horizontal recursion."""


# TODO: is it good to vmap here? this code only works if vmapped
@jax.vmap
def gto_normalization_constant(
  angular: Int[Array, "*batch 3"], exponent: Float[Array, "*batch"]
) -> Int[Array, "*batch"]:
  """Normalization constant of GTO.

  Ref: https://en.wikipedia.org/wiki/Gaussian_orbital
  """
  return (2 * exponent / np.pi)**(3 / 4) * jnp.sqrt(
    (8 * exponent)**(jnp.sum(angular)) / jnp.prod(perm_2n_n[angular])
  )


def get_cgto_segment_id(cgto_splits: tuple) -> Int[Array, "n_gtos"]:
  n_gtos = sum(cgto_splits)
  cgto_seg_len = jnp.cumsum(jnp.array(cgto_splits))
  seg_id = jnp.argmax(jnp.arange(n_gtos)[:, None] < cgto_seg_len, axis=-1)
  return seg_id


def from_pyscf_mol(mol: pyscf.gto.mole.Mole) -> LCGTO:
  """Transform pyscf mol object to GTOs.

  Returns:
    all translated GTOs. STO TO GTO
  """
  primitives = []
  atom_splits = []
  cgto_splits = []
  coeffs = []
  for i, element in enumerate(mol.elements):
    coord = mol.atom_coord(i)
    n_gtos = 0
    for sto in mol._basis[element]:
      shell = Shell(sto[0])
      gtos = sto[1:]
      for angular in SHELL_TO_ANGULAR_VEC[shell]:
        cgto_splits.append(len(gtos))
        for exponent, coeff in gtos:
          n_gtos += 1
          primitives.append((angular, coord, exponent))
          coeffs.append(coeff)
    atom_splits.append(n_gtos)
  primitives = PrimitiveGaussian(
    *[jnp.array(np.stack(a, axis=0)) for a in zip(*primitives)]
  )
  cgto_splits = tuple(cgto_splits)
  cgto_seg_id = get_cgto_segment_id(cgto_splits)
  lcgto = LCGTO(
    primitives, primitives.normalization_constant(), jnp.array(coeffs),
    cgto_splits, cgto_seg_id, jnp.array(atom_splits), mol.atom_charges()
  )
  logging.info(f"there are {sum(cgto_splits)} GTOs")
  return lcgto


class PrimitiveGaussian(NamedTuple):
  """Batch of Primitive Gaussians / Gaussian-Type Orbitals (GTO)."""
  angular: Int[Array, "*batch 3"]
  """angular momentum vector, e.g. (0,1,0)"""
  center: Float[Array, "*batch 3"]
  """atom coordinates for each GTO."""
  exponent: Float[Array, "*batch"]
  """GTO exponent / bandwith"""

  def normalization_constant(self) -> Int[Array, "*batch"]:
    return gto_normalization_constant(self.angular, self.exponent)

  def eval(self, r: Float[Array, "3"]) -> Float[Array, "*batch"]:
    """Evaluate GTO (unnormalized) with given real space coordinate."""
    return jnp.exp(-self.exponent * jnp.sum((r - self.center)**2))


class LCGTO(NamedTuple):
  """Linear Combination of Contracted Gaussian-Type Orbitals.
  Can be used to represent AO.
  """
  primitives: PrimitiveGaussian
  """GTO basis functions."""
  N: Int[Array, "n_gtos"]
  """Store computed GTO normalization constant."""
  coeff: Float[Array, "*n_gtos"]
  """CGTO contraction coefficient. n_cgto is usually the number of AO."""
  cgto_splits: Union[Int[Array, "*n_cgtos"], tuple]
  """CGTO segment lengths. e.g. (3, 3, 3, 3, 3, 3, 3, 3, 3, 3) for O2 in sto-3g.
  Store it in tuple form so that it is hashable, and can be passed to a jitted
  function as static arg."""
  cgto_seg_id: Int[Array, "n_gtos"]
  """Segment ids for contracting tensors in GTO basis to CGTO basis.
  e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
  8, 8, 8, 9, 9, 9] for O2 in sto-3g.
  """
  atom_splits: Int[Array, "*n_atoms"]
  """Atom segment lengths. e.g. [15, 15] for O2 in sto-3g.
  Useful for copying atom centers to each GTO when doing basis optimization."""
  charge: Int[Array, "*n_atoms"]
  """charges of the atoms"""

  @property
  def n_gtos(self):
    return sum(self.cgto_splits)

  @property
  def n_cgtos(self):
    return len(self.cgto_splits)

  @property
  def n_atoms(self):
    return len(self.atom_splits)

  def map_params(
    self, f: Callable
  ) -> Tuple[PrimitiveGaussian, Float[Array, "*batch"]]:
    """Apply function f to primitive gaussian parameters and contraction coeffs.
    Can be used to get a tensor slice of the parameters for contraction or
    tensorization.
    """
    # TODO: add purpose and use example for this API.
    angular, center, exponent, coeff = map(
      f, [
        self.primitives.angular, self.primitives.center,
        self.primitives.exponent, self.coeff
      ]
    )
    return PrimitiveGaussian(angular, center, exponent), coeff

  def eval(self, r: Float[Array, "3"]) -> Float[Array, "n_cgtos"]:
    """Evaluate CGTO given real space coordinate by first evaluate
    all primitives, normalize it then contract them."""
    gto_val = self.coeff * self.N * self.primitives.eval(r)
    n_cgtos = len(self.cgto_splits)
    return jax.ops.segment_sum(gto_val, self.cgto_seg_id, n_cgtos)

  @staticmethod
  def from_pyscf_mol(mol: pyscf.gto.mole.Mole,
                     use_hk: bool = True) -> Union[LCGTO, Callable]:
    """Build LCGTO from pyscf mol.

    Args:
      hk: If true, then construct a function that maps optimizable paramters to
        LCGTO. Note that function must be haiku transformed. Can be used for
        basis optimization.
    """
    lcgto = from_pyscf_mol(mol)
    if not use_hk:
      return lcgto

    center_init: Float[np.ndarray, "n_atoms 3"] = mol.atom_coords()

    def get_lcgto() -> LCGTO:
      center = hk.get_parameter(
        "center", center_init.shape, init=lambda _, __: center_init
      )
      center_rep = jnp.repeat(center, np.array(lcgto.atom_splits), axis=0)
      exponent = hk.get_parameter(
        "exponent",
        lcgto.primitives.exponent.shape,
        init=lambda _, __: lcgto.primitives.exponent
      )
      coeff = hk.get_parameter(
        "coeff", lcgto.coeff.shape, init=lambda _, __: lcgto.coeff
      )
      primitives = PrimitiveGaussian(
        lcgto.primitives.angular, center_rep, exponent
      )
      return lcgto._replace(primitives=primitives, coeff=coeff)

    return get_lcgto
