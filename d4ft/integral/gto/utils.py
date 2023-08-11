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

from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
from jaxtyping import Int


class Shell(Enum):
  """Letter for total angular momentum"""
  s = 0
  p = 1
  d = 2
  f = 3


def get_cart_angular_vec(l: int) -> List[np.ndarray[Int, "3"]]:
  r"""Calculate all l-th order monomials of x,y,z, represented as 3D vectors.
  For example, xyz is represented as [1,1,1]. Used for the angular part of GTOs.

  The order of generated monomials are as follows:

  .. math::
    x^n, x^{n-1}y, x^{n-1}z, x^{n-2}y^2, x^{n-2}yz, x^{n-2}z^2, \dots,
    y^n, y^{n-1}z, y^{n-2}z^2, \dots,
    z^n

  For example, when l=3, the order is
  .. math::
    x^3, x^2y, x^2z, xy^2, xyz, xz^2, y^3, y^2z, yz^2, z^3

  Args:
    l: total angular momentum
  """
  vecs = []
  for lx in reversed(range(l + 1)):
    for ly in reversed(range(l + 1 - lx)):
      lz = l - lx - ly
      vecs.append(np.array([lx, ly, lz]))
  return vecs


SPH_PREFAC = {
  Shell.s: [0.282094791773878143],
  Shell.p: [0.488602511902919921] * 2,
  Shell.d: [0.315391565252520002, 1.092548430592079070, 1.092548430592079070],
  Shell.f:
    [
      0.746352665180230782 / 2,
      0.457045799464465739,
      1.445305721320277020,
      0.590043589926643510,
    ],
}

REAL_SOLID_HARMONICS_PREFAC = {
  Shell.s: [1.],
  Shell.p: [1., 1.],
  Shell.d: [0.5, np.sqrt(3), np.sqrt(3)],
  Shell.f: [0.5, 0.5 * np.sqrt(3 / 2), 0.5 * np.sqrt(15), 0.5 * np.sqrt(5 / 2)],
}
r"""Common prefactor for the real solid harmonics in cartesian coordinates."""

MONOMIALS_TO_REAL_SOLID_HARMONICS: Dict[Shell, List[List[Tuple[int, float]]]]
MONOMIALS_TO_REAL_SOLID_HARMONICS = {
  Shell.s: [[(0, 1.)]],  # m=0: 1
  # NOTE: the order of the monomials is different from get_cart_angular_vec
  # since PySCF uses this order.
  # This will not matter once we use the in-house intor to compute ovlp
  Shell.p: [
    [(0, 1.)],  # x
    [(1, 1.)],  # y
    [(2, 1.)],  # z
  ],
  Shell.d:
    [
      [(1, 1.)],  # m=-2: xy
      [(4, 1.)],  # m=-1: yz
      [(5, 2.), (0, -1.), (3, -1.)],  # m=0: 2z^2-x^2-y^2
      [(2, 1.)],  # m=1: xz
      [(0, 0.5), (3, -0.5)],  # m=2: 1/2(x^2-y^2)
    ],
  Shell.f:
    [
      [(1, 3.), (6, -1.)],  # m=-3: 3x^2y-y^3
      [(4, 2.)],  # m=-2: 2xyz
      [(1, -1.), (6, -1.), (8, 4.)],  # m=-1: -x^2y-y^3+4yz^2
      [(2, -3.), (7, -3.), (9, 2.)],  # m=0: -3x^2z-3y^2z+2z^3
      [(0, -1.), (3, -1.), (5, 4.)],  # m=1: -x^3-xy^2+4xz^2
      [(2, 1.), (7, -1.)],  # m=2: x^2z-y^2z
      [(0, 1.), (3, -3.)],  # m=3: x^3-3xy^2
    ],
}
r"""Transformation from the monomials to the real solid harmonics,
without the common prefactor (defined as real_solid_sph_cart_prefac).
The transformation is stored as

[(monomial_idx, prefac), (monomial_idx, prefac), ...]

The order of the monomials follows the definition in get_cart_angular_vec.
The order of the real solid harmonics is from m=-l to m=l.
Note that since the real solid harmonics for s (l=0) and p (l=1) shells
are the same as the monomials, we don't store the conversion for them.

Explanation of the transformation:
GTO basis with variable exponents is defined as

.. math::
  \chi^{GTO}_{\alpha_{nl}lm}
= R^{GTO}_{\alpha_{nl}lm}(r)Y_{lm}(\theta, \varphi)

Absorbing the :math:`r^l` term from the radial part into the spherical
harmonic :math:`Y_{lm}`, we have the solid harmonic

.. math::
  C_{lm}(\vb{r})=r^l Y_{lm}(\theta, \varphi)

This is a complex function. We can get the real solid harmonic :math:`S_{lm}`,
which is real-valued, via an unitary transformation. The real solid harmonic
can be expressed in Cartesian coordinate as polynomials of :math:`x,y,z`.
For example, for the d shell (:math:`l=2`), we have

.. math::
  S_{22}(\vb{r})=&\frac{1}{2}\sqrt{3}(x^2-y^2) \\
  S_{21}(\vb{r})=&\sqrt{3}xz \\
  S_{20}(\vb{r})=&\frac{1}{2}(2z^2-x^2-y^2) \\
  S_{2-1}(\vb{r})=&\sqrt{3}yz \\
  S_{2-2}(\vb{r})=&\sqrt{3}xy

By convention Racah's normalization are applied so that we have nice monomial
for s and p orbitals: for s we have 1 and for p we have :math:`x,y,z`.
Therefore to convert back to the original spherical harmonics we need to
undo it. So we need to multiply :math:`1/R(l)`.

Reference:
 - Helgaker 6.6.4
 - https://onlinelibrary.wiley.com/iucr/itc/Bb/ch1o2v0001/table1o2o7o1/
"""


def racah_norm(l: int):
  r"""Racah's normalization for total angular momentum of :math:`l`,
  denoted as :math:`R(l)`.

  It is used to define normalized spherical harmonics :math:`C_{lm}` where
  :math:`C_{00}=R(0)Y_{00}=1`.
  The formula is
  :math:`R(l)=\sqrt{\frac{4\pi}{2l+1}}`

  The solid harmonic :math:`\mathcal{Y}_{lm}=r^l*Y_{lm}` uses the same
  normalization, and also the real solid harmonics s_{lm}, since they are
  obtained via unitary tranformation from the solid harmonic.
  """
  return np.sqrt((4 * np.pi) / (2 * l + 1))
