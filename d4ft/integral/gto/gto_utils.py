from __future__ import annotations  # forward declaration of GTO

from typing import Callable, NamedTuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pyscf
import scipy.special
from absl import logging
from d4ft.constants import SHELL_TO_ANGULAR_VEC, Shell
from d4ft.integral.gto.sto_utils import get_sto_segment_id
from jaxtyping import Array, Float, Int

_r25 = np.arange(25)
perm_2n_n = jnp.array(scipy.special.perm(2 * _r25, _r25))
"""Precomputed values for (2n)! / n!.
 Used in GTO normalization and OS horizontal recursion."""


def normalization_constant(
  angular: Int[Array, "*batch 3"], exponent: Float[Array, "*batch"]
) -> Int[Array, "nmo_or_ngto"]:
  """Normalization constant of GTO.

  Ref: https://en.wikipedia.org/wiki/Gaussian_orbital
  """
  return (2 * exponent / np.pi)**(3 / 4) * jnp.sqrt(
    (8 * exponent)**(jnp.sum(angular)) / jnp.prod(perm_2n_n[angular])
  )


class GTO(NamedTuple):
  """Batch of Gaussian-Type Orbitals parameters.
  Can be used to represent AO/MO.
  """
  angular: Int[Array, "*batch 3"]
  """angular momentum vector, e.g. (0,1,0)"""
  center: Float[Array, "*batch 3"]
  """atom coordinates for each GTO."""
  exponent: Float[Array, "*batch"]
  """GTO exponent / bandwith"""
  coeff: Float[Array, "*batch"]
  """GTO contraction coefficient"""
  sto_to_gto: Union[Int[Array, "*n_stos"], tuple]
  """GTO segment lengths. e.g. (3,3) for H2 in sto-3g.
  Store it in tuple form so that it is hashable, and can be
  passed to a jitted function as static arg."""
  charge: Int[Array, "*batch"]
  """charges of the atoms"""
  sto_seg_id: Int[Array, "n_gtos"]
  """for contracting GTO tensor to AO/STO basis."""
  N: Int[Array, "nmo_or_ngto"]
  """Store computed normalization constant."""

  def params(self):
    return [self.angular, self.center, self.exponent, self.coeff]

  def map_params(self, f: Callable) -> GTO:
    angular, center, exponent, coeff = map(f, self.params())
    return self._replace(
      angular=angular,
      center=center,
      exponent=exponent,
      coeff=coeff,
    )

  def normalization_constant(self) -> Int[Array, "nmo_or_ngto"]:
    """Normalization constant of GTO.

    Ref: https://en.wikipedia.org/wiki/Gaussian_orbital
    """
    return normalization_constant(self.angular, self.exponent)

  def normalize(self) -> GTO:
    """Computed and store normalization constant for the
    current GTO parameter.

    Note that if basis are optimized then the normalization
    constant need to be updated at every step.
    """
    # TODO: check whether jit is needed here
    return self._replace(N=self.normalization_constant())
    # return self._replace(N=jax.jit(self.normalization_constant)())

  def eval(self, r: Float[Array, "3"]) -> Float[Array, "nao_or_ngto"]:
    """Evaluate GTO given real space coordinate."""
    gto_val = self.normalization_constant() * self.coeff * jnp.exp(
      -self.exponent * jnp.sum((r - self.center)**2)
    )
    n_stos = len(self.sto_to_gto)
    return jax.ops.segment_sum(gto_val, self.sto_seg_id, n_stos)


def mol_to_gto(mol: pyscf.gto.mole.Mole) -> GTO:
  """Transform pyscf mol object to GTOs.

  Returns:
    all translated GTOs. STO TO GTO
  """
  all_gtos = []
  sto_to_gto = []
  for i, element in enumerate(mol.elements):
    coord = mol.atom_coord(i)
    for sto in mol._basis[element]:
      shell = Shell(sto[0])
      gtos = sto[1:]
      for angular in SHELL_TO_ANGULAR_VEC[shell]:
        sto_to_gto.append(len(gtos))
        for exponent, coeff in gtos:
          all_gtos.append((angular, coord, exponent, coeff))
  params = [jnp.array(np.stack(a, axis=0)) for a in zip(*all_gtos)]
  sto_to_gto = tuple(sto_to_gto)
  sto_seg_id = get_sto_segment_id(sto_to_gto)
  N = normalization_constant(params[0], params[2])
  gtos = GTO(*(params + [sto_to_gto, mol.atom_charges(), sto_seg_id, N]))
  n_gto = sum(sto_to_gto)
  logging.info(f"there are {n_gto} GTOs")
  return gtos


def get_gto_param_fn(mol: pyscf.gto.mole.Mole) -> hk.Transformed:
  """Extract gto params from pyscf mol obj, then construct a
  function that maps it to GTO object.

  Used for basis optimization.
  """
  gtos = mol_to_gto(mol)

  center_init: Float[np.ndarray, "n_atoms 3"] = mol.atom_coords()

  # TODO: consider move the hk transform outside
  @hk.without_apply_rng
  @hk.transform
  def gto_param_fn():
    center = hk.get_parameter(
      "center", center_init.shape, init=lambda _, __: center_init
    )
    exponent = hk.get_parameter(
      "exponent", gtos.exponent.shape, init=lambda _, __: gtos.exponent
    )
    coeff = hk.get_parameter(
      "coeff", gtos.coeff.shape, init=lambda _, __: gtos.coeff
    )
    return GTO(
      gtos.angular, jnp.repeat(center, np.array(gtos.sto_to_gto), axis=0),
      exponent, coeff, gtos.sto_to_gto, gtos.charge, gtos.sto_seg_id, gtos.N
    )

  return gto_param_fn
