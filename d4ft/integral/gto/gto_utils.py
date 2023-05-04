from __future__ import annotations  # forward declaration of GTO

from typing import Callable, NamedTuple, Tuple, Union

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


# TODO: is it good to vmap here?
@jax.vmap
def normalization_constant(
  angular: Int[Array, "*batch 3"], exponent: Float[Array, "*batch"]
) -> Int[Array, "nmo_or_ngto"]:
  """Normalization constant of GTO.

  Ref: https://en.wikipedia.org/wiki/Gaussian_orbital
  """
  return (2 * exponent / np.pi)**(3 / 4) * jnp.sqrt(
    (8 * exponent)**(jnp.sum(angular)) / jnp.prod(perm_2n_n[angular])
  )


# TODO: GTO or Gto?
class GTOParam(NamedTuple):
  angular: Int[Array, "*batch 3"]
  """angular momentum vector, e.g. (0,1,0)"""
  center: Float[Array, "*batch 3"]
  """atom coordinates for each GTO."""
  exponent: Float[Array, "*batch"]
  """GTO exponent / bandwith"""


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
  # TODO: it should be more clear that this coeff is only for making AO,
  # but not MO as the shape here is only *batch.
  # TODO: BTW, should we call it batch, or num_gto?
  # TODO: shouldn't this class be called LCGTO? if coeff is one of the fields?

  sto_splits: Union[Int[Array, "*n_stos"], tuple]
  """GTO segment lengths. e.g. (3,3) for H2 in sto-3g.
  Store it in tuple form so that it is hashable, and can be
  passed to a jitted function as static arg."""
  # TODO: this is not necessarily sto? It could be any LCGTO to make any orbital
  # including the cc orbitals.

  atom_to_gto: Int[Array, "*n_atoms"]
  # TODO: same here for the naming
  charge: Int[Array, "*n_atoms"]
  """charges of the atoms"""
  # TODO: the batch here could be confusing, we should distinguish it from
  # the batch above. One is for num_gtos, the other is for the num_atoms.
  sto_seg_id: Int[Array, "n_gtos"]
  """for contracting GTO tensor to AO/STO basis."""
  N: Int[Array, "nmo_or_ngto"]
  """Store computed normalization constant."""

  # TODO: This N seems stateful

  def params(self):
    return [self.angular, self.center, self.exponent, self.coeff]

  def map_params(self, f: Callable) -> Tuple[GTOParam, Float[Array, "*batch"]]:
    # TODO: add purpose and use example for this API.
    angular, center, exponent, coeff = map(f, self.params())
    return GTOParam(angular, center, exponent), coeff

  def normalization_constant(self) -> Int[Array, "nmo_or_ngto"]:
    """Normalization constant of GTO.

    Ref: https://en.wikipedia.org/wiki/Gaussian_orbital
    """
    return normalization_constant(self.angular, self.exponent)

  def eval(self, r: Float[Array, "3"]) -> Float[Array, "nao_or_ngto"]:
    """Evaluate GTO given real space coordinate."""
    gto_val = self.normalization_constant() * self.coeff * jnp.exp(
      -self.exponent * jnp.sum((r - self.center)**2)
    )
    n_stos = len(self.sto_splits)
    return jax.ops.segment_sum(gto_val, self.sto_seg_id, n_stos)


# TODO: The API design of tf Dataset could be borrowed here
# tf.dataset.Dataset.from_generator/from_tensor_slices
# correspondingly, we can set this function as the static member of GTO.
# GTO.from_pyscf_mol
# GTO.from_string
def mol_to_gto(mol: pyscf.gto.mole.Mole) -> GTO:
  """Transform pyscf mol object to GTOs.

  Returns:
    all translated GTOs. STO TO GTO
  """
  all_gtos = []
  atom_to_gto = []
  sto_to_gto = []
  for i, element in enumerate(mol.elements):
    coord = mol.atom_coord(i)
    n_gtos = 0
    for sto in mol._basis[element]:
      shell = Shell(sto[0])
      gtos = sto[1:]
      for angular in SHELL_TO_ANGULAR_VEC[shell]:
        sto_to_gto.append(len(gtos))
        for exponent, coeff in gtos:
          n_gtos += 1
          all_gtos.append((angular, coord, exponent, coeff))
    atom_to_gto.append(n_gtos)
  params = [jnp.array(np.stack(a, axis=0)) for a in zip(*all_gtos)]
  sto_to_gto = tuple(sto_to_gto)
  sto_seg_id = get_sto_segment_id(sto_to_gto)
  N = normalization_constant(params[0], params[2])
  gtos = GTO(
    *(params + [sto_to_gto, atom_to_gto,
                mol.atom_charges(), sto_seg_id, N])
  )
  n_gto = sum(sto_to_gto)
  logging.info(f"there are {n_gto} GTOs")
  return gtos


# TODO: As per our last discussion, we could use the more native hk way
# There are two different gto object we need, one is initialized from pyscf,
# the parameters are fixed, we can construct from pyscf.
# Another is optimizable, we want it to be hk.transformed.

# Design 1: we have one GTO, and another `d4ft.hk.GTO` (naming can be improved).
# gto = GTO.from_pyscf_mol(mol)
# hk_gto = d4ft.hk.GTO(init=gto, use_exponent=True, use_center=True)

# Design 2: we only have one GTO, but we control its behavior via arguments.
# gto = GTO.from_pyscf_mol(mol, use_exponent=True, use_center=True)
# If any use_* is true, then this can be only constructed within a
# hk.transformed context.


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
    center_rep = jnp.repeat(center, np.array(gtos.atom_to_gto), axis=0)
    return GTO(
      gtos.angular, center_rep, exponent, coeff, gtos.sto_splits,
      gtos.atom_to_gto, gtos.charge, gtos.sto_seg_id, gtos.N
    )

  return gto_param_fn
