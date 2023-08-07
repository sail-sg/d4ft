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
  """https://pyscf.org/user/gto.html#basis-set"""
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
  r"""Racah's normalization for total angular momentum of l, denoted as R(l).
  It is used to define normalized spherical harmonics C_{lm} where
  C_{00}=R(0)Y_{00}=1.

  .. math::
    \sqrt{\frac{4\pi}{2l+1}}

  The solid harmonic y_{lm}=r^l*Y_{lm} uses the same normalization, and
  also the real-valued solide harmonics s_{lm}, since they are obtained via
  unitary tranformation from the solid harmonic.
  """
  return np.sqrt((4 * np.pi) / (2 * l + 1))


REAL_SOLID_SPH_CART_PREFAC = [ #  lm
  0.282094791773878143,        #0 00: 1/R(0)
  0.488602511902919921,        #1 1{1,2,3}: 1/R(1)
  1.092548430592079070,        #2 2{1,2}: 1/R(2) * np.sqrt(3)
  0.315391565252520002,        #3 2{0}: 1/R(2) * 0.5
  0.746352665180230782 / 2,    #4 3{0}: 1/R(3) * 0.5
  0.590043589926643510,        #5 3{3}: 1/R(3) * 0.5 * np.sqrt(5/2)
  0.457045799464465739,        #6 3{1}: 1/R(3) * 0.5 * np.sqrt(3/2)
  1.445305721320277020,        #7 3{2}: 1/R(3) * 0.5 * np.sqrt(15)
]
"""The prefactor of the real solid harmonics under cartesian coordinate,
mulitplied by inverse of Racah's normalization R(l).

Racah's normalization produce nice monomial for s and p orbitals
(for p we have x, y, z), however to convert back to the original spherical
harmonics we need to undo it. So we need to multiply 1/R(l) where R(l) is
the Racah's normalization for total angular momentum of l.

https://onlinelibrary.wiley.com/iucr/itc/Bb/ch1o2v0001/table1o2o7o1/
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

HARTREE_TO_KCALMOL = 627.503
