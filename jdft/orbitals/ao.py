import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jdft.functions import factorial
# from scipy.special import factorial
from jdft.orbitals.basis import Basis
from jdft.orbitals.parser import Pyscf2GTO_parser


class GTO(Basis):
  """GTO basis."""

  def __init__(self, alpha, i, j, k, c):
    """Gaussian type orbital.

    Args:
      |alpha: shape [n]
      |i: shape [n]
      |j: shape [n]
      |k: shape [n]
      |c: shape [n, 3]. The center of each primitive.
    """
    super().__init__()
    self.alpha = alpha
    self.i = i
    self.j = j
    self.k = k
    self.c = c
    self.num_basis = self.c.shape[0]

  def __call__(self, r, *args):
    """Compute GTO on r.

    Args:
      |r: shape [3]

    Returns:
      |shape [n], where n is the number of Gaussian primitive.
    """
    const = (2 * self.alpha / np.pi)**(3 / 4)
    const *= (
      (
        (8 * self.alpha)**(self.i + self.j + self.k) * factorial(self.i) *
        factorial(self.j) * factorial(self.k)
      ) /
      (factorial(2 * self.i) * factorial(2 * self.j) * factorial(2 * self.k))
    )**0.5

    output = jnp.power(r - self.c, jnp.stack([self.i, self.j, self.k], axis=1))
    output = jnp.prod(output, axis=1)

    def fc(c):
      return jnp.linalg.norm(r - c)**2

    output *= jnp.exp(-self.alpha * jax.vmap(fc)(self.c))

    return const * output

  def init(self, *args):
    return None


class PopleSparse(Basis):
  """The pople basis set."""

  def __init__(self, pyscf_mol):
    """Initialzer for Pople."""
    super().__init__()
    # opensource
    self.pyscf_mol = pyscf_mol
    self.elements = pyscf_mol.elements
    self.atom_coords = pyscf_mol.atom_coords()
    self._basis = pyscf_mol._basis

    self.ao_labels = pyscf_mol.ao_labels()

    (alpha, i, j, k, c), (coeff_data,
                          coeff_idx) = Pyscf2GTO_parser(self.pyscf_mol)
    self.alpha = alpha
    self.i = i
    self.j = j
    self.k = k
    self.c = c
    # coeff_idx = coeff_idx
    self.coeff_mat = sparse.BCOO(
      (coeff_data, coeff_idx), shape=(len(self.pyscf_mol.ao_labels()), len(i))
    )
    self.coeff_mat = sparse.sparsify(jnp.expand_dims)(self.coeff_mat, axis=0)
    # way to build
    # gto short for Gaussian T
    self._gto = GTO(self.alpha, self.i, self.j, self.k, self.c)

  def __call__(self, r, *args):
    """Compute Pople basis on r.

    Args:
      |r: shape [G, 3] where G is the number of grids.

    Returns:
      |shape: [G, num_ao], where num_ao is the number of atomic orbitals.
    """
    basis = self._gto(r)
    basis = jnp.expand_dims(basis, axis=(0, 2))
    output = sparse.bcoo_dot_general(
      self.coeff_mat, basis, dimension_numbers=(((2), (1)), ((0), (0)))
    )
    return jnp.squeeze(output)

  def overlap(self, intor=None):
    """Compute overlap matrix for Pople basis."""
    if not intor:
      return self.pyscf_mol.intor('int1e_ovlp_sph')
    else:
      intor.wave_fun = self.__call__
      return intor.overlap()

  def init(self, *args):
    return None


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

  def overlap(self, intor=None):
    if not intor:
      return self.pyscf_mol.intor('int1e_ovlp_sph')
    else:
      intor.wave_fun = self.__call__
      return intor.overlap()

  def init(self, *args):
    return None


class PopleFast(Basis):
  # this is a Pople type atomic orbital class.

  def __init__(self, pyscf_mol):
    self.pyscf_mol = pyscf_mol
    self.atom_coords = pyscf_mol.atom_coords()
    self._basis = pyscf_mol._basis
    self.elements = pyscf_mol.elements

  def __call__(self, r, *args):
    '''
    R^3 -> R^N. N-body atomic orbitals.
    input:
        (N: the number of atomic orbitals.)
        |r: (3) coordinate.
    '''
    output = []
    for idx in np.arange(len(self.elements)):
      element = self.elements[idx]
      coord = self.atom_coords[idx]
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

  def overlap(self, intor=None):
    if not intor:
      return self.pyscf_mol.intor('int1e_ovlp_sph')
    else:
      intor.wave_fun = self.__call__
      return intor.overlap()

  def init(self, *args):
    return None
