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
from typing import (
  Callable,
  Literal,
  NamedTuple,
  Optional,
  Sequence,
  Tuple,
  Union,
)

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import scipy.special
from absl import logging
from jaxtyping import Array, Float, Int

from d4ft.integral.gto.utils import (
  MONOMIALS_TO_REAL_SOLID_HARMONICS,
  REAL_SOLID_HARMONICS_PREFAC,
  Shell,
  get_cart_angular_vec,
  racah_norm,
)
from d4ft.system.mol import Mol
from d4ft.types import MoCoeff
from d4ft.utils import inv_softplus, make_constant_fn

_r25 = np.arange(25)
perm_2n_n = jnp.array(scipy.special.perm(2 * _r25, _r25))
"""Precomputed values for (2n)! / n! for n in (0,24).
 Used in GTO normalization and OS horizontal recursion."""


def gaussian_integral(
  n: Int[Array, "*batch"],
  alpha: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
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
  return jax.scipy.special.gamma(np1_half) / (2. * alpha**np1_half)


def pgto_norm_inv(
  angular: Int[Array, "*batch 3"], exponent: Float[Array, "*batch"]
) -> Int[Array, "*batch"]:
  """Normalization constant of PGTO, i.e. the inverse of its norm.

  Ref: https://en.wikipedia.org/wiki/Gaussian_orbital
  """
  return (2 * exponent / np.pi)**(3 / 4) * jnp.sqrt(
    (8 * exponent)**(jnp.sum(angular)) / jnp.prod(perm_2n_n[angular])
  )


def normalize_cgto_coeff(
  pgto: PGTO, coeff: Float[Array, "*n_pgtos"]
) -> Float[Array, "*n_pgtos"]:
  r"""Normalize the contraction coefficients of CGTO such that the
  CGTO with normalized coefficients has norm of 1.

  The input should be a batch of PGTOs corresponding to one CGTO.
  Therefore all PGTOs has the same angular.
  """
  # total angular momentum,
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


def reparameterize(
  cgto: CGTO, optim_exp: bool, optim_coeff: bool
) -> Tuple[Float[Array, "*n_pgtos"], Float[Array, "*n_pgtos"]]:
  pgtos = []
  coeffs = []

  pgto_count = 0

  # iter atoms
  for i, element_basis in enumerate(cgto.basis):
    coord = cgto.atom_coords[i]
    # iter shells
    for cgto_i in element_basis:
      total_angular = cgto_i[0]  # l/shell
      assert not isinstance(
        cgto_i[1], float
      ), "basis with kappa is not supported yet"
      pgtos_i = cgto_i[1:]  # [[exp, c_1, c_2, ..], [exp, c_1, c_2, ..], ... ]]
      n_coeffs = len(pgtos_i[0][1:])
      # TODO: do not create separate PGTO for each c_1, c_2, ...
      # iter contractions
      for cid in range(1, 1 + n_coeffs):  # 0-idx is exponent
        # iter cartesian monomials for the given shell
        pgtos_monomials = []
        coeffs_monomials = []
        pgtos_sph = []
        coeffs_sph = []
        for angular in get_cart_angular_vec(total_angular):
          # iter PGTOs in the given CGTO. If n_coeffs > 1, we have
          # the general contraction, i.e. each PGTO appears in multiple
          # CGTOs.
          pgto_i_ = []
          coeffs_i = []
          for pgto_i in pgtos_i:
            # NOTE: we want to have some activation function here to make sure
            # that exponent > 0. However softplus is not good as inv_softplus
            # makes some exponent goes inf
            exponent = pgto_i[0]
            if optim_exp:
              exponent = jax.nn.softplus(
                hk.get_parameter(
                  f"exponent/{pgto_count}",
                  shape=(),
                  init=make_constant_fn(inv_softplus(jnp.array(exponent)))
                )
              )
            coeff = pgto_i[cid]
            if optim_coeff:
              coeff = hk.get_parameter(
                f"coeff/{pgto_count}",
                shape=(),
                init=make_constant_fn(jnp.array(coeff))
              )
            pgto_i_.append(PGTO(angular, coord, exponent))
            coeffs_i.append(coeff)
            pgto_count += 1

          # pgto_i = PGTO.apply(np.stack, pgto_i_)
          pgto_i = PGTO.stack(pgto_i_)
          pgtos_monomials.append(pgto_i)

          # normalize each PGTO in cartesian coordinate
          N = pgto_i.norm_inv()
          normalized_coeffs_i = normalize_cgto_coeff(
            pgto_i,
            jnp.array(coeffs_i) * N
          ) / N
          coeffs_monomials.append(normalized_coeffs_i)

        # convert to spherical
        r_l_inv = 1 / racah_norm(total_angular)
        shell = Shell(total_angular)
        prefacs = REAL_SOLID_HARMONICS_PREFAC[shell]
        for mpl, sph in enumerate(MONOMIALS_TO_REAL_SOLID_HARMONICS[shell]):
          m = mpl - total_angular  # magnetic quantum number
          p_m = prefacs[abs(m)]
          for monomial_idx, m_prefac in sph:
            pgtos_sph.append(pgtos_monomials[monomial_idx])
            coeffs_sph.append(
              coeffs_monomials[monomial_idx] * m_prefac * p_m * r_l_inv
            )
        # pgto_sph_ = PGTO.apply(np.concatenate, pgtos_sph)
        pgto_sph_ = PGTO.concat(pgtos_sph)
        pgtos.append(pgto_sph_)
        coeffs.append(jnp.concatenate(coeffs_sph))

  # pgto = PGTO.apply(np.concatenate, pgtos)
  pgto = PGTO.concat(pgtos)
  coeff = jnp.concatenate(coeffs)

  return coeff, pgto.exponent


def build_cgto_from_mol(mol: Mol) -> CGTO:
  """Build CGTO from the basis information in Mol.

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

  To convert the monomials to real solid harmonics, we need to know the
  monomials used by the harmonics, which is provided by the table
  MONOMIALS_TO_REAL_SOLID_HARMONICS. And we also need to multiply the
  common prefactor (REAL_SOLID_HARMONICS_PREFAC) to the monomials, and
  the inverse of Racah's normalization (can be calculated with racah_norm)
  for total angular momentum of :math:`l`: :math:`R(l)`.

  Reference:
   - https://pyscf.org/user/gto.html#basis-set
   - https://theochem.github.io/horton/2.0.1/tech_ref_gaussian_basis.html
   - https://onlinelibrary.wiley.com/iucr/itc/Bb/ch1o2v0001/table1o2o7o1/
   - https://github.com/sunqm/libcint/blob/master/src/cart2sph.c
   - https://shorturl.at/er248

  Returns:
    all translated GTOs.
  """
  pgtos = []
  atom_splits = []
  cgto_splits = []
  coeffs = []

  # iter atoms
  for i, element in enumerate(mol.elements):
    coord = mol.atom_coords[i]
    n_pgtos_atom = 0
    # iter shells
    for cgto_i in mol.basis[element]:
      total_angular = cgto_i[0]  # l/shell
      assert not isinstance(
        cgto_i[1], float
      ), "basis with kappa is not supported yet"
      pgtos_i = cgto_i[1:]  # [[exp, c_1, c_2, ..], [exp, c_1, c_2, ..], ... ]]
      n_coeffs = len(pgtos_i[0][1:])
      # TODO: do not create separate PGTO for each c_1, c_2, ...
      # iter contractions
      for cid in range(1, 1 + n_coeffs):  # 0-idx is exponent
        # iter cartesian monomials for the given shell
        pgtos_monomials = []
        coeffs_monomials = []
        pgtos_sph = []
        coeffs_sph = []
        for angular in get_cart_angular_vec(total_angular):
          # iter PGTOs in the given CGTO. If n_coeffs > 1, we have
          # the general contraction, i.e. each PGTO appears in multiple
          # CGTOs.
          pgto_i_ = []
          coeffs_i = []
          for pgto_i in pgtos_i:
            exponent = pgto_i[0]
            coeff = pgto_i[cid]
            pgto_i_.append(PGTO(angular, coord, exponent))
            coeffs_i.append(coeff)

          pgto_i = PGTO.stack(pgto_i_)
          pgtos_monomials.append(pgto_i)

          # normalize each PGTO in cartesian coordinate
          N = pgto_i.norm_inv()
          normalized_coeffs_i = normalize_cgto_coeff(
            pgto_i,
            jnp.array(coeffs_i) * N
          ) / N
          coeffs_monomials.append(normalized_coeffs_i)

        # convert to spherical
        r_l_inv = 1 / racah_norm(total_angular)
        shell = Shell(total_angular)
        prefacs = REAL_SOLID_HARMONICS_PREFAC[shell]
        for mpl, sph in enumerate(MONOMIALS_TO_REAL_SOLID_HARMONICS[shell]):
          m = mpl - total_angular  # magnetic quantum number
          p_m = prefacs[abs(m)]
          n_pgto_sph = 0
          for monomial_idx, m_prefac in sph:
            pgtos_sph.append(pgtos_monomials[monomial_idx])
            coeffs_sph.append(
              coeffs_monomials[monomial_idx] * m_prefac * p_m * r_l_inv
            )
            n_pgto_sph_i = len(pgtos_monomials[monomial_idx].angular)
            n_pgto_sph += n_pgto_sph_i
          cgto_splits.append(n_pgto_sph)

        pgto_sph_ = PGTO.apply(np.concatenate, pgtos_sph)
        pgtos.append(pgto_sph_)
        coeffs.append(jnp.concatenate(coeffs_sph))
        n_pgto_sph = len(pgto_sph_.angular)
        n_pgtos_atom += n_pgto_sph

    atom_splits.append(n_pgtos_atom)

  pgto = PGTO.apply(np.concatenate, pgtos)
  coeffs = jnp.concatenate(coeffs)

  cgto_splits = tuple(cgto_splits)
  cgto_seg_id = get_cgto_segment_id(cgto_splits)

  logging.info(
    f"there are {sum(cgto_splits)} (non-unique) PGTOs in spherical form"
  )

  # map basis into d4ft format

  basis = []
  for element in mol.elements:
    element_basis = []
    for cgto_i in mol.basis[element]:
      element_basis.append(cgto_i)
    basis.append(element_basis)

  cgto = CGTO(
    pgto, pgto.norm_inv(), jnp.array(coeffs), cgto_splits, cgto_seg_id,
    jnp.array(atom_splits), mol.atom_charges, mol.nocc, basis
  )

  return cgto


class PGTO(NamedTuple):
  r"""Batch of Primitive Gaussian-Type Orbitals (PGTO).

  .. math::
    PGTO_{nlm}(\vb{r})
    =N_n(r_x-c_x)^{n_x} (r_y-c_y)^{n_y} (r_z-c_z)^{n_z}
     \exp(-\alpha \norm{\vb{r}-\vb{c}}^2)

  where N is the normalization factor, which is just the
  inverse of L2 norm of the unnormalized primitive.

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

  @staticmethod
  def stack(pgtos: Sequence[PGTO]) -> PGTO:
    """Return the stack of a sequence of PGTO singletons.
    Note that the angular needs to be a numpy array to avoid tracing error.
    """
    angular, center, exponent = zip(*pgtos)
    return PGTO(np.stack(angular), jnp.stack(center), jnp.stack(exponent))

  @staticmethod
  def concat(pgtos: Sequence[PGTO]) -> PGTO:
    """Return the concatenation of a sequence of PGTO singletons.
    Note that the angular needs to be a numpy array to avoid tracing error.
    """
    angular, center, exponent = zip(*pgtos)
    return PGTO(
      np.concatenate(angular), jnp.concatenate(center),
      jnp.concatenate(exponent)
    )

  @staticmethod
  def apply(f: Callable, pgtos: Sequence[PGTO]) -> PGTO:
    """Return the concatenation of a sequence of PGTO singletons.
    Note that the angular needs to be a numpy array to avoid tracing error.
    """
    angular, center, exponent = map(f, zip(*pgtos))
    return PGTO(angular, jnp.array(center), jnp.array(exponent))

  def eval(self, r: Float[Array, "3"]) -> Float[Array, "*batch"]:
    """Evaluate PGTO (unnormalized) with given real space coordinate.

    Args:
      r: 3D real space coordinate

    Returns:
      batch of unnormalized pgto
    """
    angular_cart = []
    for i in range(self.n_orbs):
      angular_cart_i = 1.0
      for d in range(3):  # x, y, z
        if self.angular[i, d] > 0:
          angular_cart_i *= jnp.power(
            r[d] - self.center[i, d], self.angular[i, d]
          )
      angular_cart.append(angular_cart_i)
    angular_cart = jnp.array(angular_cart)
    exp = jnp.exp(-self.exponent * jnp.sum((r - self.center)**2, axis=1))
    return angular_cart * exp


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
  """Charges of the atoms"""
  nocc: Int[Array, "2 nao"]
  """Cccupation mask for alpha and beta spin"""
  basis: Sequence[Sequence[Tuple[int, Sequence[Sequence[float]]]]]
  """basis in PySCF format"""

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
    gto_val = self.coeff * self.pgto.norm_inv() * self.pgto.eval(r)
    n_cgtos = len(self.cgto_splits)
    return jax.ops.segment_sum(gto_val, self.cgto_seg_id, n_cgtos)

  @staticmethod
  def from_mol(mol: Mol) -> CGTO:
    """Build CGTO from pyscf mol."""
    return build_cgto_from_mol(mol)

  @staticmethod
  def from_cart(cgto_cart: CGTO) -> CGTO:
    return build_cgto_sph_from_mol(cgto_cart)

  def to_hk(
    self,
    optimizable_params: Sequence[Literal[
      "center",
      "exponent",
      "coeff",
    ]] = ("coeff",),
  ) -> CGTO:
    """Convert optimizable parameters to hk.Params. Must be haiku transformed.
    Can be used for basis optimization.
    """
    if "center" in optimizable_params:
      center_init = self.atom_coords
      center_param = hk.get_parameter(
        "center", center_init.shape, init=make_constant_fn(center_init)
      )
      center = jnp.repeat(
        center_param,
        jnp.array(self.atom_splits),
        axis=0,
        total_repeat_length=self.n_pgtos
      )
    else:
      center = self.pgto.center

    if "exponent" in optimizable_params or "coeff" in optimizable_params:
      coeff, exponent = reparameterize(
        self,
        optim_exp="exponent" in optimizable_params,
        optim_coeff="coeff" in optimizable_params,
      )
    else:
      coeff = self.coeff
      exponent = self.pgto.exponent

    pgto = PGTO(self.pgto.angular, center, exponent)
    return self._replace(pgto=pgto, coeff=coeff)

  # TODO: instead of using occupation mask, we can orthogonalize a non-square
  # matrix directly
  def get_mo_coeff(
    self,
    restricted: bool,
    ortho_fn: Optional[Callable] = None,
    ovlp_sqrt_inv: Optional[Float[Array, "nao nao"]] = None,
    apply_spin_mask: bool = True,
    use_hk: bool = True,
    key: Optional[jax.Array] = None,
  ) -> MoCoeff:
    """Function to return MO coefficient. Must be haiku transformed."""
    nmo = self.nao
    shape = ([nmo, nmo] if restricted else [2, nmo, nmo])

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

    if restricted:  # restrictied mo
      mo_coeff_spin = jnp.repeat(mo_coeff[None], 2, 0)  # add spin axis
    else:
      mo_coeff_spin = mo_coeff

    if apply_spin_mask:
      mo_coeff_spin *= self.nocc[:, :, None]  # apply spin mask
    # mo_coeff_spin = mo_coeff_spin.reshape(-1, nmo)  # flatten
    return mo_coeff_spin
