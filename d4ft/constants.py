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

import numpy as np


class Shell(Enum):
  """Letter for total angular momentum"""
  s = 0
  p = 1
  d = 2
  f = 3


SHELL_TO_ANGULAR_VEC = {
  Shell.s: [[0, 0, 0]],
  Shell.p: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  Shell.d: [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]],
  Shell.f:
    [
      [3, 0, 0], [2, 1, 0], [2, 0, 1], [1, 2, 0], [1, 1, 1], [1, 0, 2],
      [0, 3, 0], [0, 2, 1], [0, 1, 2], [0, 0, 3]
    ]
}
"""
Angular vectors for shells in cartesian form, computed by the followings:
for lx in reversed(range(l + 1)):
    for ly in reversed(range(l + 1 - lx)):
      lz = l - lx - ly
      print("[{},{},{}]".format(lx,ly,lz))
"""


def racah_normalization(l: int):
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


# TODO: change this to a function that takes l and m as input
# get the R(l) part from racah_normalization, and the other prefactor
# can be tabulated against l and m.
REAL_SOLID_SPH_CART_PREFAC = [  # i lm
  0.282094791773878143,         # 0 00: 1/R(0)
  0.488602511902919921,         # 1 1{1,2,3}: 1/R(1)
  1.092548430592079070,         # 2 2{1,2}: 1/R(2) * np.sqrt(3)
  0.315391565252520002,         # 3 2{0}: 1/R(2) * 0.5
  0.746352665180230782 / 2,     # 4 3{0}: 1/R(3) * 0.5
  0.590043589926643510,         # 5 3{3}: 1/R(3) * 0.5 * np.sqrt(5/2)
  0.457045799464465739,         # 6 3{1}: 1/R(3) * 0.5 * np.sqrt(3/2)
  1.445305721320277020,         # 7 3{2}: 1/R(3) * 0.5 * np.sqrt(15)
]
r"""The prefactor of the real solid harmonics under Cartesian coordinate,
mulitplied by inverse of Racah's normalization for total angular momentum of
:math:`l`: :math:`R(l)`.

GTO basis with variable exponents is defined as

.. math::
  \chi^{GTO}_{\alpha_{nl}lm} = R^{GTO}_{\alpha_{nl}lm}(r)Y_{lm}(\theta, \varphi)

Absorbing the :math:`r^l` term from the radial part into the spherical harmonic
:math:`Y_{lm}`, we have the solid harmonic

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

We store the prefactor and the Racah's normalization in the table, which
are all you need to create real solid harmonics in cartesian coordinates
with monomials. For example, in the case of :math:`S_{20}`, we have
 :math:`\frac{1}{2} * \frac{1}{R(2)}=0.315391565252520002`.

Reference:
 - Helgaker 6.6.4
 - https://onlinelibrary.wiley.com/iucr/itc/Bb/ch1o2v0001/table1o2o7o1/
"""

REAL_HARMONIC_NORMALIZATION = [
  # s shell
  [1],
  # p shell
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  # d shell
  [
    [0, 1.092548430592079070, 0, 0, 0, 0],
    [0, 0, 0, 0, 1.092548430592079070, 0],
    [
      -0.315391565252520002, 0, 0, -0.315391565252520002, 0,
      -0.315391565252520002
    ],
    [0, 0, 1.092548430592079070, 0, 0, 0],
    [0.546274215296039535, 0, 0, -0.546274215296039535, 0, 0],
  ],
  # f shell
  [
    [0, 1.770130769779930531, 0, 0, 0, 0, -0.590043589926643510, 0, 0, 0],
    [0, 0, 0, 0, 2.890611442640554055, 0, 0, 0, 0, 0],
    [
      0, -0.457045799464465739, 0, 0, 0, 0, -0.457045799464465739, 0,
      1.828183197857862944, 0
    ],
    [
      0, 0, -1.119528997770346170, 0, 0, 0, 0, -1.119528997770346170, 0,
      0.746352665180230782
    ],
    [
      -0.457045799464465739, 0, 0, -0.457045799464465739, 0,
      1.828183197857862944, 0, 0, 0, 0
    ],
    [0, 0, 1.445305721320277020, 0, 0, 0, 0, -1.445305721320277020, 0, 0],
    [0.590043589926643510, 0, 0, -1.770130769779930530, 0, 0, 0, 0, 0, 0],
  ]
]

ANGSTRONG_TO_BOHR = 1.8897259886
"""Conversion factor from Angstrom to Bohr"""

HARTREE_TO_KCALMOL = 627.503
"""Conversion factor from Hartree to kcal/mol"""
