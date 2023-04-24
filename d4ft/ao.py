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

import jax
import jax.numpy as jnp
import numpy as np

from d4ft.integral.quadrature import overlap_integral

from d4ft.functions import factorial


def gaussian_primitive(r, alpha, ijk):
  """Gaussian primitive functions.

  More often called GTOs as explained in
  https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Quantum_Mechanics/17%3A_Quantum_Calculations/ab_initio_Basis_Sets
  Args:
    r: shape = (3,)
    alpha: shape = (,)
    ijk: shape = (3,)
  Returns:
    normalized x^l y^m z^n e^{-alpha r^2}
  """
  normalization = (2 * alpha / np.pi)**(3 / 4) * jnp.sqrt(
    (8 * alpha)**jnp.sum(ijk) * jnp.prod(factorial(ijk)) /
    jnp.prod(factorial(2 * ijk))
  )
  xyz_ijk = jnp.prod(jnp.power(r, ijk))
  return xyz_ijk * jnp.exp(-alpha * jnp.linalg.norm(r)**2) * normalization


class PopleFast:
  """Pople type atomic orbital"""

  def __init__(self, pyscf_mol):
    self.pyscf_mol = pyscf_mol
    self.atom_coords = pyscf_mol.atom_coords()
    self._basis = pyscf_mol._basis
    self.elements = pyscf_mol.elements
    self.ao_labels = pyscf_mol.ao_labels()

  def __call__(self, r, *args, **kwargs):
    """R^3 -> R^N. N-body atomic orbitals.
    input:
        (N: the number of atomic orbitals.)
        |r: (3) coordinate.

    Formula for the normalization constant
    ref: https://en.wikipedia.org/wiki/Gaussian_orbital
    """

    atom_coords = kwargs.get('atom_coords', self.atom_coords)
    output = []
    for idx in np.arange(len(self.elements)):
      element = self.elements[idx]
      coord = atom_coords[idx]
      for i in self._basis[element]:
        prm_array = jnp.array(i[1:])
        exponents = prm_array[:, 0]
        coeffs = prm_array[:, 1]

        if i[0] == 0:  # s-orbitals
          output.append(
            jnp.sum(
              coeffs * jnp.exp(-exponents * jnp.sum((r - coord)**2)) *
              (2 * exponents / jnp.pi)**(3 / 4)
            )
          )

        elif i[0] == 1:  # p-orbitals
          output += [
            (r[j] - coord[j]) * jnp.sum(
              coeffs * jnp.exp(-exponents * jnp.sum((r - coord)**2)) *
              (2 * exponents / jnp.pi)**(3 / 4) * (4 * exponents)**0.5
            ) for j in np.arange(3)
          ]

    return jnp.array(output)

  def overlap(self, **kwargs):
    if 'grids' in kwargs:
      g = kwargs['grids']
      w = kwargs['weights']

      def f(r):
        return self.__call__(r, **kwargs)

      w_grids = jax.vmap(f)(g)
      w_grids = jnp.reshape(w_grids, newshape=(w_grids.shape[0], -1))
      w_grids_weighted = w_grids * jnp.expand_dims(w, axis=(1))
      return jnp.matmul(w_grids_weighted.T, w_grids)

    else:
      return self.pyscf_mol.intor('int1e_ovlp_sph')

  def init(self, *args, **kwargs):
    return None


class Gaussian():

  def __init__(self, pyscf_mol):
    self.pyscf_mol = pyscf_mol
    self.atom_coords = pyscf_mol.atom_coords()
    self._basis = pyscf_mol._basis
    self.elements = pyscf_mol.elements

  def __call__(self, r, *args):
    output = []
    for idx in np.arange(len(self.elements)):
      element = self.elements[idx]
      coord = self.atom_coords[idx]
      for i in self._basis[element]:
        if i[0] == 0:
          prm_array = jnp.array(i[1:])
          output.append(
            prm_array[0, 1] *
            jnp.exp(-prm_array[0, 0] * jnp.linalg.norm(r - coord)**2) *
            (2 * prm_array[0, 0] / jnp.pi)**(3 / 4)
          )

        elif i[0] == 1:
          prm_array = jnp.array(i[1:])
          output += [
            (r[j] - coord[j]) * prm_array[0, 1] *
            jnp.exp(-prm_array[0, 0] * jnp.linalg.norm(r - coord)**2) *
            (2 * prm_array[0, 0] / jnp.pi)**(3 / 4) * (4 * prm_array[0, 0])**0.5
            for j in np.arange(3)
          ]
    return jnp.array(output)

  def overlap(self, **args):
    g = args['grids']
    w = args['weights']
    return overlap_integral(self.__call__, (g, w))

  def init(self, *args):
    return None
