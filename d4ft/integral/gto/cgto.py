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

from d4ft.constants import (
  SHELL_TO_ANGULAR_VEC,
  REAL_SOLID_SPH_CART_PREFAC,
  Shell,
)
from d4ft.system.mol import Mol
from d4ft.types import MoCoeff
from d4ft.utils import inv_softplus, make_constant_fn

_r25 = np.arange(25)
perm_2n_n = jnp.array(scipy.special.perm(2 * _r25, _r25))
"""Precomputed values for (2n)! / n! for n in (0,24).
 Used in GTO normalization and OS horizontal recursion."""


def gaussian_integral(n: Int[Array, "*batch"], alpha: Float[Array, "*batch"]):
  r"""Evaluate gaussian integral using the gamma function.

  .. math::
    \int_0^\infty x^n \exp(-\alpha x^2) dx
    = \frac{\Gamma((n+1)/2)}{2 \alpha^{(n+1)/2}}

  Args:
    n: power of x
    alpha: exponent

  Ref:
  https://en.wikipedia.org/wiki/Gaussian_integral#Relation_to_the_gamma_function
  """
  np1_half = (n + 1) * .5
  return scipy.special.gamma(np1_half) / (2. * alpha**np1_half)


def pgto_norm_inv(
  angular: Int[Array, "*batch 3"], exponent: Float[Array, "*batch"]
) -> Int[Array, "*batch"]:
  """Normalization constant of PGTO, i.e. the inverse of its norm.

  Ref: https://en.wikipedia.org/wiki/Gaussian_orbital
  """
  return (2 * exponent / np.pi)**(3 / 4) * jnp.sqrt(
    (8 * exponent)**(jnp.sum(angular)) / jnp.prod(perm_2n_n[angular])
  )


def normalize_cgto_coeff(pgto: PGTO, coeff: Float[Array, "*n_pgtos"]):
  r"""Normalize the contraction coefficients of CGTO such that the
  CGTO with normalized coefficients has norm of 1.

  The input should be a batch of PGTOs corresponding to one CGTO.
  """
  # total angular momentum, asuming all PGTOs has the same angular
  l = jnp.sum(pgto.angular[0])
  # multipy exp funcs equals to sum exponents
  ee: Float[Array, "n_pgtos n_pgtos"]
  ee = pgto.exponent.reshape(-1, 1) + pgto.exponent.reshape(1, -1)
  overlap = gaussian_integral(l * 2 + 2, ee)  # TODO: why l * 2 + 2
  cgto_norm = jnp.einsum('p,pq,q->', coeff, overlap, coeff)
  normalized_coeff = coeff / jnp.sqrt(cgto_norm)
  return normalized_coeff


def get_cgto_segment_id(cgto_splits: tuple) -> Int[Array, "n_pgtos"]:
  n_pgtos = sum(cgto_splits)
  cgto_seg_len = jnp.cumsum(jnp.array(cgto_splits))
  seg_id = jnp.argmax(jnp.arange(n_pgtos)[:, None] < cgto_seg_len, axis=-1)
  return seg_id


def build_cgto_from_mol(mol: Mol) -> CGTO:
  """Transform pyscf mol object to CGTO.

  Example PySCF basis (cc-pvdz)
  {'O': [
    [0, [11720.0, 0.00071, -0.00016],
        [1759.0, 0.00547, -0.001263],
        [400.8, 0.027837, -0.006267],
        [113.7, 0.1048, -0.025716],
        [37.03, 0.283062, -0.070924],
        [13.27, 0.448719, -0.165411],
        [5.025, 0.270952, -0.116955],
        [1.013, 0.015458, 0.557368]],
    [0, [0.3023, 1.0]],
    [1, [17.7, 0.043018],
        [3.854, 0.228913],
        [1.046, 0.508728]],
    [1, [0.2753, 1.0]],
    [2, [1.185, 1.0]]]}

  Basically each CGTO is represented as
  [[angular, kappa, [[exp, c_1, c_2, ..],
                     [exp, c_1, c_2, ..],
                     ... ]],
   [angular, kappa, [[exp, c_1, c_2, ..],
                     [exp, c_1, c_2, ..]
                     ... ]]]

  Reference:
   - https://pyscf.org/user/gto.html#basis-set
   - https://theochem.github.io/horton/2.0.1/tech_ref_gaussian_basis.html,
   - https://onlinelibrary.wiley.com/iucr/itc/Bb/ch1o2v0001/table1o2o7o1/,
   - https://github.com/sunqm/libcint/blob/747d6c0dd838d20abdc9a4c9e4c62d196a855bc0/src/cart2sph.c

  Returns:
    all translated GTOs.
  """
  pgto = []
  atom_splits = []
  cgto_splits = []
  coeffs = []
  shells = []

  for i, element in enumerate(mol.elements):
    coord = mol.atom_coords[i]
    n_pgtos = 0
    for cgto_i in mol.basis[element]:
      shell = Shell(cgto_i[0])
      assert not isinstance(
        cgto_i[1], float
      ), "basis with kappa is not supported yet"
      pgtos_i = cgto_i[1:]  # [[exp, c_1, c_2, ..], [exp, c_1, c_2, ..], ... ]]
      n_coeffs = len(pgtos_i[0][1:])
      for cid in range(1, 1 + n_coeffs):
        for angular in SHELL_TO_ANGULAR_VEC[shell]:
          cgto_splits.append(len(pgtos_i))
          for pgto_i in pgtos_i:
            exponent = pgto_i[0]
            coeff = pgto_i[cid]
            n_pgtos += 1
            pgto.append((angular, coord, exponent))
            coeffs.append(coeff)
            shells.append(cgto_i[0])

    atom_splits.append(n_pgtos)

  # NOTE: do not use jnp for angular as it will cause tracing error
  pgto = PGTO(
    *[
      np.array(np.stack(a, axis=0)) if i ==
      0 else jnp.array(np.stack(a, axis=0)) for i, a in enumerate(zip(*pgto))
    ]
  )

  cgto_splits = tuple(cgto_splits)
  cgto_seg_id = get_cgto_segment_id(cgto_splits)

  cur_ptr = 0
  # normalize each PGTO in cartesian coordinate
  cart_coeffs = jnp.array(coeffs) * pgto.norm_inv()
  normalized_cart_coeff = jnp.array([])
  # normalize each CGTO in cartesian coordinate
  for lgto in cgto_splits:  # get one monomial with different exponents
    n = normalize_cgto_coeff(
      pgto.at(slice(cur_ptr, cur_ptr + lgto)),
      cart_coeffs[cur_ptr:cur_ptr + lgto]
    )
    normalized_cart_coeff = jnp.concatenate([normalized_cart_coeff, n])
    cur_ptr += lgto

  coeffs = normalized_cart_coeff / pgto.norm_inv()
  cgto = CGTO(
    pgto, pgto.norm_inv(), jnp.array(coeffs), cgto_splits, cgto_seg_id,
    jnp.array(atom_splits), mol.atom_charges, mol.nocc, shells
  )
  return cgto


def build_cgto_sph_from_mol(cgto_cart: CGTO) -> CGTO:
  """Transform Cartesian CGTO object to CGTO.

  Returns:
    all translated GTOs.
  """
  cgto_shells = []
  cgto_ptr = 0
  split_ptr = 0
  atom_ptr = 0
  atom_ngto_cart = 0
  atom_ngto_sph = 0
  pgto = []
  atom_splits = []
  cgto_splits = []
  coeffs = []
  shells = []
  while cgto_ptr < len(cgto_cart.shells):
    shell = cgto_cart.shells[cgto_ptr]
    n_pgtos = cgto_cart.cgto_splits[split_ptr]
    cgto_shells.append(shell)

    # s shell: same as cartesian
    if shell == 0:
      # TODO: replace this with cgto_cart.pgto.at(slice(cgto_ptr, cgto_ptr+n_pgtos))
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + j], cgto_cart.pgto[1][cgto_ptr + j],
            cgto_cart.pgto[2][cgto_ptr + j]
          )
        )
        coeffs.append(
          REAL_SOLID_SPH_CART_PREFAC[0] * cgto_cart.coeff[cgto_ptr + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + j])
      cgto_splits.append(n_pgtos)
      cgto_ptr += 1 * n_pgtos
      atom_ngto_cart += 1 * n_pgtos
      atom_ngto_sph += 1 * n_pgtos
      split_ptr += 1

    # p shell: same as cartesian
    elif shell == 1:
      for i in range(3):
        for j in range(n_pgtos):
          pgto.append(
            (
              cgto_cart.pgto[0][cgto_ptr + i * n_pgtos + j],
              cgto_cart.pgto[1][cgto_ptr + i * n_pgtos + j],
              cgto_cart.pgto[2][cgto_ptr + i * n_pgtos + j]
            )
          )
          coeffs.append(
            REAL_SOLID_SPH_CART_PREFAC[1] *
            cgto_cart.coeff[cgto_ptr + i * n_pgtos + j]
          )
          shells.append(cgto_cart.shells[cgto_ptr + i * n_pgtos + j])
        cgto_splits.append(n_pgtos)
      atom_ngto_cart += 3 * n_pgtos
      atom_ngto_sph += 3 * n_pgtos
      cgto_ptr += 3 * n_pgtos
      split_ptr += 3

    # d shell: xx xy xz yy yz zz -> xy yz 1/2(2zz-xx-yy) xz 1/2(xx-yy)
    elif shell == 2:
      # d1.xy
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 1 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 1 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 1 * n_pgtos + j]
          )
        )
        coeffs.append(
          REAL_SOLID_SPH_CART_PREFAC[2] *
          cgto_cart.coeff[cgto_ptr + 1 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 1 * n_pgtos + j])
      cgto_splits.append(n_pgtos)
      # d2.yz
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 4 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 4 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 4 * n_pgtos + j]
          )
        )
        coeffs.append(
          REAL_SOLID_SPH_CART_PREFAC[2] *
          cgto_cart.coeff[cgto_ptr + 4 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 4 * n_pgtos + j])
      cgto_splits.append(n_pgtos)
      # d3.1/2(2z^2-x^2-y^2)
      # zz
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 5 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 5 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 5 * n_pgtos + j]
          )
        )
        coeffs.append(
          2 * REAL_SOLID_SPH_CART_PREFAC[3] *
          cgto_cart.coeff[cgto_ptr + 5 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 5 * n_pgtos + j])
      # xx
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + j], cgto_cart.pgto[1][cgto_ptr + j],
            cgto_cart.pgto[2][cgto_ptr + j]
          )
        )
        coeffs.append(
          -REAL_SOLID_SPH_CART_PREFAC[3] * cgto_cart.coeff[cgto_ptr + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + j])
      # yy
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 3 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 3 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 3 * n_pgtos + j]
          )
        )
        coeffs.append(
          -REAL_SOLID_SPH_CART_PREFAC[3] *
          cgto_cart.coeff[cgto_ptr + 3 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 3 * n_pgtos + j])
      cgto_splits.append(3 * n_pgtos)
      # d4.xz
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 2 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 2 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 2 * n_pgtos + j]
          )
        )
        coeffs.append(
          REAL_SOLID_SPH_CART_PREFAC[2] *
          cgto_cart.coeff[cgto_ptr + 2 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 2 * n_pgtos + j])
      cgto_splits.append(n_pgtos)
      # d5.1/2(x^2-y^2)
      # xx
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + j], cgto_cart.pgto[1][cgto_ptr + j],
            cgto_cart.pgto[2][cgto_ptr + j]
          )
        )
        coeffs.append(
          0.5 * REAL_SOLID_SPH_CART_PREFAC[2] * cgto_cart.coeff[cgto_ptr + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + j])
      # yy
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 3 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 3 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 3 * n_pgtos + j]
          )
        )
        coeffs.append(
          -0.5 * REAL_SOLID_SPH_CART_PREFAC[2] *
          cgto_cart.coeff[cgto_ptr + 3 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 3 * n_pgtos + j])
      cgto_splits.append(2 * n_pgtos)
      cgto_ptr += 6 * n_pgtos
      atom_ngto_cart += 6 * n_pgtos
      atom_ngto_sph += 8 * n_pgtos
      split_ptr += 6
    # f shell
    elif shell == 3:
      # f1. 3xxy-yyy
      # xxy
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 1 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 1 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 1 * n_pgtos + j]
          )
        )
        coeffs.append(
          3 * REAL_SOLID_SPH_CART_PREFAC[5] *
          cgto_cart.coeff[cgto_ptr + 1 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 1 * n_pgtos + j])
      # yyy
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 6 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 6 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 6 * n_pgtos + j]
          )
        )
        coeffs.append(
          -REAL_SOLID_SPH_CART_PREFAC[5] *
          cgto_cart.coeff[cgto_ptr + 6 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 6 * n_pgtos + j])
      cgto_splits.append(2 * n_pgtos)
      # f2. 2xyz
      # xyz
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 4 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 4 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 4 * n_pgtos + j]
          )
        )
        coeffs.append(
          2 * REAL_SOLID_SPH_CART_PREFAC[7] *
          cgto_cart.coeff[cgto_ptr + 4 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 4 * n_pgtos + j])
      cgto_splits.append(n_pgtos)
      # f3. -xxy-yyy+4yzz
      # xxy
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 1 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 1 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 1 * n_pgtos + j]
          )
        )
        coeffs.append(
          -REAL_SOLID_SPH_CART_PREFAC[6] *
          cgto_cart.coeff[cgto_ptr + 1 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 1 * n_pgtos + j])
      # yyy
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 6 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 6 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 6 * n_pgtos + j]
          )
        )
        coeffs.append(
          -REAL_SOLID_SPH_CART_PREFAC[6] *
          cgto_cart.coeff[cgto_ptr + 6 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 6 * n_pgtos + j])
      # yzz
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 8 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 8 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 8 * n_pgtos + j]
          )
        )
        coeffs.append(
          4 * REAL_SOLID_SPH_CART_PREFAC[6] *
          cgto_cart.coeff[cgto_ptr + 8 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 9 * n_pgtos + j])
      cgto_splits.append(3 * n_pgtos)
      # f4. -3xxz-3yyz+2zzz
      # xxz
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 2 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 2 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 2 * n_pgtos + j]
          )
        )
        coeffs.append(
          -3 * REAL_SOLID_SPH_CART_PREFAC[4] *
          cgto_cart.coeff[cgto_ptr + 2 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 2 * n_pgtos + j])
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 7 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 7 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 7 * n_pgtos + j]
          )
        )
        coeffs.append(
          -3 * REAL_SOLID_SPH_CART_PREFAC[4] *
          cgto_cart.coeff[cgto_ptr + 7 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 7 * n_pgtos + j])
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 9 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 9 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 9 * n_pgtos + j]
          )
        )
        coeffs.append(
          2 * REAL_SOLID_SPH_CART_PREFAC[4] *
          cgto_cart.coeff[cgto_ptr + 9 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 9 * n_pgtos + j])
      cgto_splits.append(3 * n_pgtos)
      # f5. -xxx-xyy+4xzz
      # xxx
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 0 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 0 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 0 * n_pgtos + j]
          )
        )
        coeffs.append(
          -REAL_SOLID_SPH_CART_PREFAC[6] *
          cgto_cart.coeff[cgto_ptr + 0 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 0 * n_pgtos + j])
      # xyy
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 3 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 3 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 3 * n_pgtos + j]
          )
        )
        coeffs.append(
          -REAL_SOLID_SPH_CART_PREFAC[6] *
          cgto_cart.coeff[cgto_ptr + 3 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 3 * n_pgtos + j])
      # zzz
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 5 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 5 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 5 * n_pgtos + j]
          )
        )
        coeffs.append(
          4 * REAL_SOLID_SPH_CART_PREFAC[6] *
          cgto_cart.coeff[cgto_ptr + 5 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 5 * n_pgtos + j])
      cgto_splits.append(3 * n_pgtos)
      # f6. xxz-yyz
      # xxz
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 2 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 2 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 2 * n_pgtos + j]
          )
        )
        coeffs.append(
          REAL_SOLID_SPH_CART_PREFAC[7] *
          cgto_cart.coeff[cgto_ptr + 2 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 2 * n_pgtos + j])
      # yyz
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 7 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 7 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 7 * n_pgtos + j]
          )
        )
        coeffs.append(
          -REAL_SOLID_SPH_CART_PREFAC[7] *
          cgto_cart.coeff[cgto_ptr + 7 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 7 * n_pgtos + j])
      cgto_splits.append(2 * n_pgtos)
      # f7. xxx-3xyy
      # xxx
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 0 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 0 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 0 * n_pgtos + j]
          )
        )
        coeffs.append(
          REAL_SOLID_SPH_CART_PREFAC[5] *
          cgto_cart.coeff[cgto_ptr + 0 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 0 * n_pgtos + j])
      # zyy
      for j in range(n_pgtos):
        pgto.append(
          (
            cgto_cart.pgto[0][cgto_ptr + 3 * n_pgtos + j],
            cgto_cart.pgto[1][cgto_ptr + 3 * n_pgtos + j],
            cgto_cart.pgto[2][cgto_ptr + 3 * n_pgtos + j]
          )
        )
        coeffs.append(
          -3 * REAL_SOLID_SPH_CART_PREFAC[5] *
          cgto_cart.coeff[cgto_ptr + 3 * n_pgtos + j]
        )
        shells.append(cgto_cart.shells[cgto_ptr + 3 * n_pgtos + j])
      cgto_splits.append(2 * n_pgtos)
      cgto_ptr += 10 * n_pgtos
      atom_ngto_cart += 10 * n_pgtos
      atom_ngto_sph += 16 * n_pgtos
      split_ptr += 10

    if atom_ngto_cart == cgto_cart.atom_splits[atom_ptr]:
      atom_splits.append(atom_ngto_sph)
      atom_ngto_cart = 0
      atom_ptr += 1
  cgto_splits = tuple(cgto_splits)
  cgto_seg_id = get_cgto_segment_id(cgto_splits)
  pgto = PGTO(
    *[
      np.array(np.stack(a, axis=0)) if i ==
      0 else jnp.array(np.stack(a, axis=0)) for i, a in enumerate(zip(*pgto))
    ]
  )
  cgto_sph = CGTO(
    pgto, pgto.norm_inv(), jnp.array(coeffs), cgto_splits, cgto_seg_id,
    jnp.array(atom_splits), cgto_cart.charge, cgto_cart.nocc, shells
  )
  logging.info(f"there are {sum(cgto_splits)} GTOs")
  return cgto_sph


class PGTO(NamedTuple):
  r"""Batch of Primitive Gaussian-Type Orbitals (PGTO).

  .. math::
    PGTO_{nlm}(\vb{r})
    =N_n(r_x-c_x)^{n_x} (r_y-c_y)^{n_y} (r_z-c_z)^{n_z} \exp(-\alpha \norm{\vb{r}-\vb{c}}^2)

  where N is the normalization factor

  Analytical integral are calculated in this basis.
  """
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

  def norm_inv(self) -> Int[Array, "*batch"]:
    return jax.vmap(pgto_norm_inv)(self.angular, self.exponent)

  def at(self, i: Union[slice, int]) -> PGTO:
    """get one PGTO out of the batch"""
    return PGTO(self.angular[i], self.center[i], self.exponent[i])

  def eval(self, r: Float[Array, "3"]) -> Float[Array, "*batch"]:
    """Evaluate GTO (unnormalized) with given real space coordinate.

    Args:
      r: 3D real space coordinate

    Returns:
      unnormalized gto
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
  """Contracted GTO, i.e. linear combinations of PGTO.
  Stored as a batch of PGTOs, and a list of contraction coefficients.
  """
  pgto: PGTO
  """PGTO basis functions."""
  N: Int[Array, "n_pgtos"]
  """Store computed PGTO normalization constant."""
  coeff: Float[Array, "*n_pgtos"]
  """CGTO contraction coefficient. n_cgto is usually the number of AO."""
  cgto_splits: Union[Int[Array, "*n_cgtos"], tuple]
  """Contraction segment lengths. e.g. (3, 3, 3, 3, 3, 3, 3, 3, 3, 3) for
  O2 in sto-3g. Store it in tuple form so that it is hashable, and can be
  passed to a jitted function as static arg."""
  cgto_seg_id: Int[Array, "n_pgtos"]
  """Segment ids for contracting tensors in PGTO basis to CGTO basis.
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
  shells: Int[Array, "*n_pgtos"]
  """shell idx for each pgto"""

  @property
  def n_pgtos(self) -> int:
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
    return self.pgto.center[jnp.cumsum(jnp.array(self.atom_splits)) - 1]

  def map_pgto_params(self, f: Callable) -> Tuple[PGTO, Float[Array, "*batch"]]:
    """Apply function f to PGTO parameters and contraction coeffs.
    Can be used to get a tensor slice of the parameters for contraction or
    tensorization.
    """
    angular, center, exponent, coeff = map(
      f, [self.pgto.angular, self.pgto.center, self.pgto.exponent, self.coeff]
    )
    return PGTO(angular, center, exponent), coeff

  def eval(self, r: Float[Array, "3"]) -> Float[Array, "n_cgtos"]:
    """Evaluate CGTO given real space coordinate by first evaluate
    all pgto, normalize it then contract them.

    TODO: support generalized contraction.

    Args:
      r: 3D real space coordinate

    Returns:
      contracted normalized gtos.
    """
    gto_val = self.coeff * self.N * self.pgto.eval(r)
    n_cgtos = len(self.cgto_splits)
    return jax.ops.segment_sum(gto_val, self.cgto_seg_id, n_cgtos)

  @staticmethod
  def from_mol(mol: Mol) -> CGTO:
    """Build CGTO from pyscf mol."""
    return build_cgto_from_mol(mol)

  @staticmethod
  def from_cart(cgto_cart: CGTO) -> CGTO:
    return build_cgto_sph_from_mol(cgto_cart)

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
      total_repeat_length=self.n_pgtos
    )
    # NOTE: we want to have some activation function here to make sure
    # that exponent > 0. However softplus is not good as inv_softplus
    # makes some exponent goes inf
    exponent = jax.nn.softplus(
      hk.get_parameter(
        "exponent",
        self.pgto.exponent.shape,
        init=make_constant_fn(inv_softplus(self.pgto.exponent))
      )
    )
    coeff = hk.get_parameter(
      "coeff", self.coeff.shape, init=make_constant_fn(self.coeff)
    )
    pgto = PGTO(self.pgto.angular, center_rep, exponent)
    return self._replace(pgto=pgto, coeff=coeff)

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
