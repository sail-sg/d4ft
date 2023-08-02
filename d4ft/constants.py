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


class Shell(Enum):
  """https://pyscf.org/user/gto.html#basis-set"""
  s = 0
  p = 1
  d = 2
  f = 3


"""
Angular vectors for shells in cartesian form, computed by the followings:
for lx in reversed(range(l + 1)):
    for ly in reversed(range(l + 1 - lx)):
      lz = l - lx - ly
      print("[{},{},{}]".format(lx,ly,lz))
"""
SHELL_TO_ANGULAR_VEC = {
  Shell.s: [[0, 0, 0]],
  Shell.p: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  Shell.d: [[2, 0, 0], [1, 1, 0], [1, 0, 1], 
            [0, 2, 0], [0, 1, 1], [0, 0, 2]],
  Shell.f: [[3, 0, 0], [2, 1, 0], [2, 0, 1], 
            [1, 2, 0], [1, 1, 1], [1, 0, 2], 
            [0, 3, 0], [0, 2, 1], [0, 1, 2], 
            [0, 0, 3]]
}

SPH_WF_NORMALIZATION_FACTOR = [0.282094791773878143, 
                               0.488602511902919921, 
                               1.092548430592079070, 0.315391565252520002]

ANGSTRONG_TO_BOHR = 1.8897259886

HARTREE_TO_KCALMOL = 627.503
