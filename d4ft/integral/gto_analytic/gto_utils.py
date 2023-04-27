from enum import Enum

import haiku as hk
import jax.numpy as jnp
import numpy as np
import pyscf
import scipy.special
from absl import logging
from d4ft.types import GTO
from jaxtyping import Float

_r25 = np.arange(25)
perm_2n_n = jnp.array(scipy.special.perm(2 * _r25, _r25))


def normalization_constant(angular, exponent):
  """Normalization constant of GTO."""
  return (2 * exponent / np.pi)**(3 / 4) * jnp.sqrt(
    (8 * exponent)**(jnp.sum(angular)) / jnp.prod(perm_2n_n[angular])
  )


class Shell(Enum):
  """https://pyscf.org/user/gto.html#basis-set"""
  s = 0
  p = 1
  d = 2
  f = 3


ANGULAR = {
  Shell.s: [[0, 0, 0]],
  Shell.p: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
}


def mol_to_gto(mol: pyscf.gto.mole.Mole):
  """Transform pyscf mol object to GTOs.

  Returns:
    all translated GTOs. STO TO GTO
  """
  all_gtos = []
  ao_to_gto = []
  for i, element in enumerate(mol.elements):
    coord = mol.atom_coord(i)
    for sto in mol._basis[element]:
      shell = Shell(sto[0])
      gtos = sto[1:]
      for angular in ANGULAR[shell]:
        ao_to_gto.append(len(gtos))
        for exponent, coeff in gtos:
          all_gtos.append((angular, coord, exponent, coeff))
  all_gtos = GTO(*(jnp.array(np.stack(a, axis=0)) for a in zip(*all_gtos)))
  n_gto = sum(ao_to_gto)
  logging.info(f"there are {n_gto} GTOs")
  return all_gtos, tuple(ao_to_gto)


def get_gto_param_fn(mol: pyscf.gto.mole.Mole):
  """Extract gto params from pyscf mol obj, then construct a
  function that maps it to GTO object.

  Used for basis optimization.
  """
  gtos, ao_to_gto = mol_to_gto(mol)

  # center = gtos.center[np.cumsum(ao_to_gto) - 1]
  center_init: Float[np.ndarray, "n_atoms 3"] = mol.atom_coords()

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
    return GTO(gtos.angular, center, exponent, coeff)

  return gto_param_fn, ao_to_gto
