import numpy as np
import jax
import jax.numpy as jnp
from jdft.functions import factorial
# from scipy.special import factorial
from jdft.orbitals.basis import Basis
from ao_int import _ao_overlap_int


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


class Pople(Basis):

  def __init__(self, pyscf_mol):
    super().__init__()
    self.pyscf_mol = pyscf_mol
    self.elements = pyscf_mol.elements
    self.atom_coords = pyscf_mol.atom_coords()
    self._basis = pyscf_mol._basis
    self.basis_max = len(self._basis[pyscf_mol.elements[0]][0]) - 1
    self.ao_labels = pyscf_mol.ao_labels()

    exponent_mat = []
    coeff_mat = []

    self.ijk = []
    self.coord = []

    for idx in np.arange(len(self.elements)):
      element = self.elements[idx]
      coord = self.atom_coords[idx]
      for i in self._basis[element]:
        if i[0] == 0:
          self.ijk.append([0, 0, 0])
          self.coord.append(coord)
          prm_array = np.array(i[1:])
          exponent_mat.append(prm_array[:, 0])
          coeff_mat.append(prm_array[:, 1])

        if i[0] == 1:
          self.ijk.append([1, 0, 0])
          self.ijk.append([0, 1, 0])
          self.ijk.append([0, 0, 1])
          self.coord += [coord] * 3

          prm_array = np.array(i[1:])
          exponent_mat += [prm_array[:, 0]] * 3
          coeff_mat += [prm_array[:, 1]] * 3

        if i[0] == 2:
          raise NotImplementedError('d orbital has not been implemented yet.')

    self.ijk = np.array(self.ijk)
    self.coord = np.array(self.coord)

    self.exponent_mat = np.zeros([len(exponent_mat), self.basis_max])
    for p, q in enumerate(exponent_mat):
      self.exponent_mat[p][0:len(q)] = q

    self.coeff_mat = np.zeros([len(coeff_mat), self.basis_max])
    for p, q in enumerate(coeff_mat):
      self.coeff_mat[p][0:len(q)] = q

  def __call__(self, r, *args, **kwargs):
    '''
    Args:
      |r: shape [G, 3] where G is the number of grids.

    Returns:
      |shape: [G, num_ao], where num_ao is the number of atomic orbitals.
    '''

    const = (
      (
        (8 * self.exponent_mat)**jnp.sum(self.ijk, axis=1, keepdims=True) *
        jnp.prod(jax.vmap(factorial)(self.ijk.T).T, axis=1, keepdims=True)
      ) / jnp.prod(
        jax.vmap(lambda x: factorial(2 * x))(self.ijk.T).T,
        axis=1,
        keepdims=True
      )
    )**0.5
    output = (2 * self.exponent_mat / np.pi)**(3 / 4) * const * jnp.prod(
      jnp.power(r - self.coord, self.ijk), axis=1, keepdims=True
    )

    def fc(c):
      return jnp.linalg.norm(r - c)**2

    output *= self.coeff_mat * jnp.exp(
      -self.exponent_mat * jnp.expand_dims(jax.vmap(fc)(self.coord), 1)
    )
    return jnp.sum(output, axis=1)

  def overlap(self, **kwargs):
    return self.pyscf_mol.intor('int1e_ovlp_sph')

  def init(self, **kwargs):
    return None


class PopleFast(Basis):
  # this is a Pople type atomic orbital class.

  def __init__(self, pyscf_mol):
    self.pyscf_mol = pyscf_mol
    self.atom_coords = pyscf_mol.atom_coords()
    self._basis = pyscf_mol._basis
    self.elements = pyscf_mol.elements

  def __call__(self, r, *args, **kwargs):
    '''
    R^3 -> R^N. N-body atomic orbitals.
    input:
        (N: the number of atomic orbitals.)
        |r: (3) coordinate.
    '''

    atom_coords = self.atom_coords
    output = []
    for idx in np.arange(len(self.elements)):
      element = self.elements[idx]
      coord = atom_coords[idx]
      for i in self._basis[element]:
        if i[0] == 0:
          prm_array = jnp.array(i[1:])
          output.append(
            jnp.sum(
              prm_array[:, 1] *
              jnp.exp(-prm_array[:, 0] * jnp.linalg.norm(r - coord)**2) *
              (2 * prm_array[:, 0] / jnp.pi)**(3 / 4)
            )
          )

        elif i[0] == 1:
          prm_array = jnp.array(i[1:])
          output += [
            (r[j] - coord[j]) * jnp.sum(
              prm_array[:, 1] *
              jnp.exp(-prm_array[:, 0] * jnp.linalg.norm(r - coord)**2) *
              (2 * prm_array[:, 0] / jnp.pi)**(3 / 4) *
              (4 * prm_array[:, 0])**0.5
            ) for j in np.arange(3)
          ]
    return jnp.array(output)

  def overlap(self, **kwargs):
    if 'grids' in kwargs:
      g = kwargs['grids']
      w = kwargs['weights']

      w_grids = jax.vmap(self.__call__)(g)
      w_grids = jnp.reshape(w_grids, newshape=(w_grids.shape[0], -1))
      w_grids_weighted = w_grids * jnp.expand_dims(w, axis=(1))
      return jnp.matmul(w_grids_weighted.T, w_grids)

    else:
      return self.pyscf_mol.intor('int1e_ovlp_sph')

  def init(self, *args, **kwargs):
    return None


class Gaussian(Basis):

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
    return _ao_overlap_int(self.__call__, g, w)

  def init(self, *args):
    return None
