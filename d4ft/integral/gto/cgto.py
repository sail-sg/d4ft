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
from __future__ import annotations  # forward declaration

import math
from typing import Callable, NamedTuple, Optional, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import scipy.special
from absl import logging
from jaxtyping import Array, Float, Int

from d4ft.constants import SHELL_TO_ANGULAR_VEC, Shell
from d4ft.system.mol import Mol
from d4ft.types import MoCoeff
from d4ft.utils import inv_softplus, make_constant_fn

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


def build_cgto_from_mol(mol: Mol) -> CGTO:
  """Transform pyscf mol object to CGTO.

  Returns:
    all translated GTOs. STO TO GTO
  """
  primitives = []
  atom_splits = []
  cgto_splits = []
  coeffs = []
  for i, element in enumerate(mol.elements):
    coord = mol.atom_coords[i]
    n_gtos = 0
    for sto in mol.basis[element]:
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
    *[
      np.array(np.stack(a, axis=0)) if i ==
      0 else jnp.array(np.stack(a, axis=0))
      for i, a in enumerate(zip(*primitives))
    ]
  )
  cgto_splits = tuple(cgto_splits)
  cgto_seg_id = get_cgto_segment_id(cgto_splits)
  cgto = CGTO(
    primitives, primitives.normalization_constant(), jnp.array(coeffs),
    cgto_splits, cgto_seg_id, jnp.array(atom_splits), mol.atom_charges, mol.nocc
  )
  logging.info(f"there are {sum(cgto_splits)} GTOs")
  return cgto


class PrimitiveGaussian(NamedTuple):
  """Batch of Primitive Gaussians / Gaussian-Type Orbitals (GTO)."""
  angular: Int[np.ndarray, "*batch 3"]
  """angular momentum vector, e.g. (0,1,0). Note that it is stored as
  numpy array to avoid tracing error, which is okay since it is not
  trainable."""
  center: Float[Array, "*batch 3"]
  """atom coordinates for each GTO."""
  exponent: Float[Array, "*batch"]
  """GTO exponent / bandwith"""

  @property
  def n_orbs(self) -> int:
    return self.angular.shape[0]

  def normalization_constant(self) -> Int[Array, "*batch"]:
    return gto_normalization_constant(self.angular, self.exponent)

  def eval(self, r: Float[Array, "3"]) -> Float[Array, "*batch"]:
    """Evaluate GTO (unnormalized) with given real space coordinate.

    Args:
      r: 3D real space coordinate

    Returns:
      unnormalized gto (x-c_x)^l (y-c_y)^m (z-c_z)^n exp{-alpha |r-c|^2}
    """
    xyz_lmn = []
    for i in range(self.n_orbs):
      xyz_lmn_i = 1.0
      for d in range(3):  # x, y, z
        if self.angular[i, d] > 0:
          xyz_lmn_i *= jnp.power(r[d] - self.center[i, d], self.angular[i, d])
      xyz_lmn.append(xyz_lmn_i)
    xyz_lmn = jnp.array(xyz_lmn)
    exp = jnp.exp(-self.exponent * jnp.sum((r - self.center)**2, axis=1))
    return xyz_lmn * exp


class CGTO(NamedTuple):
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
  nocc: Int[Array, "2 nao"]
  """occupation mask for alpha and beta spin"""

  @property
  def n_gtos(self) -> int:
    return sum(self.cgto_splits)

  @property
  def n_cgtos(self) -> int:
    return len(self.cgto_splits)

  @property
  def nao(self) -> int:
    return self.n_cgtos

  @property
  def n_atoms(self) -> int:
    return len(self.atom_splits)

  @property
  def atom_coords(self) -> Float[Array, "n_atoms 3"]:
    return self.primitives.center[jnp.cumsum(jnp.array(self.atom_splits)) - 1]

  def map_params(
    self, f: Callable
  ) -> Tuple[PrimitiveGaussian, Float[Array, "*batch"]]:
    """Apply function f to primitive gaussian parameters and contraction coeffs.
    Can be used to get a tensor slice of the parameters for contraction or
    tensorization.
    """
    angular, center, exponent, coeff = map(
      f, [
        self.primitives.angular, self.primitives.center,
        self.primitives.exponent, self.coeff
      ]
    )
    return PrimitiveGaussian(angular, center, exponent), coeff

  def eval(self, r: Float[Array, "3"]) -> Float[Array, "n_cgtos"]:
    """Evaluate CGTO given real space coordinate by first evaluate
    all primitives, normalize it then contract them.

    Args:
      r: 3D real space coordinate

    Returns:
      contracted normalized gtos.
    """
    gto_val = self.coeff * self.N * self.primitives.eval(r)
    n_cgtos = len(self.cgto_splits)
    return jax.ops.segment_sum(gto_val, self.cgto_seg_id, n_cgtos)

  @staticmethod
  def from_mol(mol: Mol) -> CGTO:
    """Build CGTO from pyscf mol."""
    return build_cgto_from_mol(mol)

  def to_hk(self) -> CGTO:
    """Convert optimizable parameters to hk.Params. Must be haiku transformed.
    Can be used for basis optimization.
    """
    center_init = self.atom_coords
    center = hk.get_parameter(
      "center", center_init.shape, init=make_constant_fn(center_init)
    )
    center_rep = jnp.repeat(
      center,
      jnp.array(self.atom_splits),
      axis=0,
      total_repeat_length=self.n_gtos
    )
    # NOTE: we want to have some activation function here to make sure
    # that exponent > 0. However softplus is not good as inv_softplus
    # makes some exponent goes inf
    exponent = jax.nn.softplus(
      hk.get_parameter(
        "exponent",
        self.primitives.exponent.shape,
        init=make_constant_fn(inv_softplus(self.primitives.exponent))
      )
    )
    coeff = hk.get_parameter(
      "coeff", self.coeff.shape, init=make_constant_fn(self.coeff)
    )
    primitives = PrimitiveGaussian(
      self.primitives.angular, center_rep, exponent
    )
    return self._replace(primitives=primitives, coeff=coeff)

  # TODO: instead of using occupation mask, we can orthogonalize a non-square
  # matrix directly
  def get_mo_coeff(
    self,
    rks: bool,
    ortho_fn: Optional[Callable] = None,
    ovlp_sqrt_inv: Optional[Float[Array, "nao nao"]] = None,
    apply_spin_mask: bool = True,
    use_hk: bool = True,
    key: Optional[jax.Array] = None,
  ) -> MoCoeff:
    """Function to return MO coefficient. Must be haiku transformed."""
    nmo = self.nao
    shape = ([nmo, nmo] if rks else [2, nmo, nmo])

    if use_hk:
      mo_params = hk.get_parameter(
        "mo_params",
        shape,
        init=hk.initializers.RandomNormal(stddev=1. / math.sqrt(nmo))
      )
    else:
      assert key is not None
      mo_params = jax.random.normal(key, shape) / jnp.sqrt(nmo)

    if ortho_fn:
      # ortho_fn provide a parameterization of the generalized Stiefel manifold
      # where (CSC=I), i.e. overlap matrix in Roothann equations is identity.
      mo_coeff = ortho_fn(mo_params) @ ovlp_sqrt_inv
    else:
      mo_coeff = mo_params

    if rks:  # restrictied mo
      mo_coeff_spin = jnp.repeat(mo_coeff[None], 2, 0)  # add spin axis
    else:
      mo_coeff_spin = mo_coeff

    if apply_spin_mask:
      mo_coeff_spin *= self.nocc[:, :, None]  # apply spin mask
    # mo_coeff_spin = mo_coeff_spin.reshape(-1, nmo)  # flatten
    return mo_coeff_spin
