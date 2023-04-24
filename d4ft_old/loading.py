# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module align pyscf results and cdft parameters.

WARNING: only 3-21G is implemented.
"""

import jax.numpy as jnp


def wave_from_pyscf(param, r, pyscf_mol):
  """Load wave function from pyscf.

  Args:
       param: N*N
       r, (3)-dimensional coordinate
  Return: (N) wave function value vector.
  """
  atom = pyscf_mol
  # geometry = atom.atom
  # 'C -0.5, 0.0, 0.0;\nC 0.5, 0., 0.'
  # basis = atom.basis
  # basis_param = atom._basis
  # atom_coords = atom._basis

  # """
  #   the basis param format (Pople-type):
  #   a list of list
  #   [
  #       [l, [zeta_11, c_11], [zeta_12, c_12], ....]
  #         ===> psi_s1 = c_11 exp(-zeta_11 x) + ...
  #       [l, [zeta_21, c_21], [zeta_22, c_22], ....]
  #       ...
  #   ]

  #   l: angular momentum quantum number where 0 represent s, 1 presents p.
  # TODO:  pre-calculate these constants.

  #   """

  output = []
  for element in atom.elements:
    for i in atom._basis[element]:
      if i[0] == 0:
        prm_array = jnp.array(i[1:])
        output.append(
          jnp.sum(
            prm_array[:, 1] *
            jnp.exp(-prm_array[:, 0] * jnp.linalg.norm(r)**2) *
            (2 * prm_array[:, 0] / jnp.pi)**(3 / 4)
          )
        )

      elif i[0] == 1:
        prm_array = jnp.array(i[1:])
        output += [
          r_i * jnp.sum(
            prm_array[:, 1] *
            jnp.exp(-prm_array[:, 0] * jnp.linalg.norm(r)**2) *
            (2 * prm_array[:, 0] / jnp.pi)**(3 / 4) * (4 * prm_array[:, 0])**0.5
          ) for r_i in r
        ]

  # print(jnp.array(output).shape)
  return jnp.dot(param, jnp.array(output))
