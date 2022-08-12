import numpy as np
import jax
import jax.numpy as jnp
from jdft.functions import factorial
# from scipy.special import factorial
from jdft.orbitals.basis import Basis
from ao_int import _ao_overlap_int


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

  def __call__(self, r, *args):
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

  def overlap(self):
    return self.pyscf_mol.intor('int1e_ovlp_sph')

  def init(self, *args):
    return None


class PopleFast(Basis):
  # this is a Pople type atomic orbital class.

  def __init__(self, pyscf_mol, mode=None):
    self.pyscf_mol = pyscf_mol
    self.atom_coords = pyscf_mol.atom_coords()
    self._basis = pyscf_mol._basis
    self.elements = pyscf_mol.elements
    self.mode = mode

  def __call__(self, r, params=None):
    '''
    R^3 -> R^N. N-body atomic orbitals.
    input:
        (N: the number of atomic orbitals.)
        |r: (3) coordinate.
    '''
    if self.mode == 'go':
      assert params is not None
      atom_coords = params
    else:
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
    if 'g' in kwargs:
      g = kwargs['g']
      w = kwargs['w']

      def v(x):
        return self.__call__(x, params=kwargs['params'])
    # return self.pyscf_mol.intor('int1e_ovlp_sph')
      w_grids = jax.vmap(v)(g)
      w_grids = jnp.reshape(w_grids, newshape=(w_grids.shape[0], -1))
      w_grids_weighted = w_grids * jnp.expand_dims(w, axis=(1))
      return jnp.matmul(w_grids_weighted.T, w_grids)

    else:
      return self.pyscf_mol.intor('int1e_ovlp_sph')

  def init(self, *args, **kwargs):
    if self.mode == 'go':   # geometry optimization
      return self.atom_coords
    else:
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
              (2 * prm_array[0, 0] / jnp.pi)**(3 / 4) *
              (4 * prm_array[0, 0])**0.5
              for j in np.arange(3)
          ]
    return jnp.array(output)

  def overlap(self, **args):
    g = args['grids']
    w = args['weights']
    return _ao_overlap_int(self.__call__, g, w)

  def init(self, *args):
    return None
