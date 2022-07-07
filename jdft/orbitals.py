"""Classes for orbitals."""

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jdft.functions import decov, factorial
# from scipy.special import factorial


def ao_label_parser(ao_label):
  r"""Convert PySCF ao_label to GTO parameters.

  Args:
    |ao_label: a list of strings.
        0-1 characters: index of atom
        2-3 characters: symbol of atom
  Returns:
    |i: 1D jnp array
    |j: 1D jnp array
    |k: 1D jnp array
    |ele: 1D jnp array
    |atom_idx: 1D jnp array

  ao_label example (water):
    ['0 O 1s    ',
     '0 O 2s    ',
     '0 O 3s    ',
     '0 O 2px   ',
     '0 O 2py   ',
     '0 O 2pz   ',
     '0 O 3px   ',
     '0 O 3py   ',
     '0 O 3pz   ',
     '1 H 1s    ',
     '1 H 2s    ',
     '2 H 1s    ',
     '2 H 2s    ']

  """
  i = []
  j = []
  k = []
  ele = []
  atom_idx = []

  for label in ao_label:
    atom_idx.append(int(label[:2]))
    ele.append(label[2:4].strip())
    i.append(sum([s == 'x' for s in label[5:9]]))
    j.append(sum([s == 'y' for s in label[5:9]]))
    k.append(sum([s == 'z' for s in label[5:9]]))

  return i, j, k, ele, atom_idx


def Pyscf2GTO_parser(pyscf_mol):
  """Convert pyscf molecule object to GTO parameters.

  Args:
    |pyscf_mol: a PySCF molecule object

  Returns:
    |alpha: 1D array. parameter for GTO
    |i: 1D array. shape (N)
    |j: 1D array. shape (N)
    |k: 1D array. shape (N)
    |c: 2D array, shape (N, 3)
    |coeff_data: for building coeffienct sparse matrix.
    |coeff_idx: for building coeffienct sparse matrix.
  """
  elements = pyscf_mol.elements
  _basis = pyscf_mol._basis
  num_atom = len(elements)
  centers = pyscf_mol.atom_coords()

  i = []
  j = []
  k = []
  alpha = []
  atom_idx = []
  c = []
  coeff_idx = []
  coeff_data = []

  basis_idx = 0
  ao_idx = 0

  for n in np.arange(num_atom):
    params = _basis[elements[n]]
    for prm in params:
      if prm[0] == 0:
        for basis_prm_pair in prm[1:]:
          i.append(0)
          j.append(0)
          k.append(0)
          alpha.append(basis_prm_pair[0])
          # ele.append(elements[n])
          atom_idx.append(n)
          c.append(centers[n])
          coeff_data.append(basis_prm_pair[1])
          coeff_idx.append([ao_idx, basis_idx])
          basis_idx += 1
        ao_idx += 1

      if prm[0] == 1:
        for basis_prm_pair in prm[1:]:
          i.append(1)
          j.append(0)
          k.append(0)
          alpha.append(basis_prm_pair[0])
          # ele.append(elements[n])
          atom_idx.append(n)
          c.append(centers[n])
          coeff_data.append(basis_prm_pair[1])
          coeff_idx.append([ao_idx, basis_idx])
          basis_idx += 1
        ao_idx += 1

        for basis_prm_pair in prm[1:]:
          i.append(0)
          j.append(1)
          k.append(0)
          alpha.append(basis_prm_pair[0])
          # ele.append(elements[n])
          atom_idx.append(n)
          c.append(centers[n])
          coeff_data.append(basis_prm_pair[1])
          coeff_idx.append([ao_idx, basis_idx])
          basis_idx += 1
        ao_idx += 1

        for basis_prm_pair in prm[1:]:
          i.append(0)
          j.append(0)
          k.append(1)
          alpha.append(basis_prm_pair[0])
          # ele.append(elements[n])
          atom_idx.append(n)
          c.append(centers[n])
          coeff_data.append(basis_prm_pair[1])
          coeff_idx.append([ao_idx, basis_idx])
          basis_idx += 1
        ao_idx += 1

  alpha = np.asarray(alpha)
  i = np.asarray(i)
  j = np.asarray(j)
  k = np.asarray(k)
  c = np.asarray(c)

  return (alpha, i, j, k, c), (coeff_data, coeff_idx)


class Basis():
  """Abstract class of Basis functions."""

  def __init__(self):
    """Abstract initializer of basis."""
    pass

  def __call__(self, x):
    """Compute the basis functions.

    Args:
      x: shape is [..., 3]
    Returns:
      output: shape equal to [..., num_basis],
        where the batch dims are equal to x
    """
    raise NotImplementedError('__call__ function has not been implemented')

  def overlap(self):
    """Compute the overlap between basis functions.

    Returns:
      output: shape equal to [num_basis, num_basis]
    """
    raise NotImplementedError('overlap function has not been implemented.')

  def init(self):
    """Initialize the parameter of this class if any."""
    pass


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
      intor.mo = self.__call__
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
        jnp.prod(
          jax.vmap(lambda x: factorial(x))(self.ijk.T).T, axis=1, keepdims=True
        )
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
      intor.mo = self.__call__
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
      intor.mo = self.__call__
      return intor.overlap()

  def init(self, *args):
    return None


class NormalPople(Pople):
  """Pople with Normalizing flow."""

  # before r get into pople, pass by a normalizing flow
  def __init__(self, pyscf_mol, normal_flow):
    """Initialize Pople with normalizing flow."""
    super().__init__(pyscf_mol=pyscf_mol)

    self.normal_flow = normal_flow

  def __call__(self, r):
    """Compute Pople with normalizing flow on r."""
    # pass r into normalizing flow
    r_transformed = self.normal_flow(r)
    return super().__call__(r=r_transformed)

  def overlap(self, intor=None):
    """Compute overlap matrix for Pople with normalizing flow basis."""
    if not intor:
      return self.pyscf_mol.intor('int1e_ovlp_sph')
    else:
      intor.mo = self.__call__
      return intor.double_overlap()


class MO_qr(Basis):
  """Molecular orbital using QR decomposition."""

  def __init__(self, nmo, ao, intor=None):
    """Initialize molecular orbital with QR decomposition."""
    super().__init__()
    self.ao = ao
    self.nmo = nmo
    if not intor:
      # self.basis_decov = decov(self.ao.overlap())
      raise AssertionError
    else:
      self.intor = intor
      self.intor.mo = self.ao
      self.basis_decov = decov(self.ao.overlap(intor))

  def init(self, rng_key):
    """Initialize the parameter required by this class."""
    mo_params = jax.random.normal(rng_key,
                                  [self.nmo, self.nmo]) / jnp.sqrt(self.nmo)
    return mo_params, self.ao.init(rng_key)

  def __call__(self, params, r):
    """Compute the molecular orbital on r.

    R^3 -> R^N. N-body molecular orbital wave functions.
    input: (N: the number of atomic orbitals.)
      |params: N*N
      |r: (3)
    output:
      |molecular orbitals:(2, N)
    """
    mo_params, ao_params = params
    mo_params = jnp.expand_dims(mo_params, 0)
    mo_params = jnp.repeat(mo_params, 2, 0)
    ao_fun_vec = self.ao(r, ao_params)

    def wave_fun_i(param_i, ao_fun_vec):
      orthogonal, _ = jnp.linalg.qr(param_i)  # q is column-orthogal.
      return orthogonal.transpose() @ decov(
        self.ao.overlap(self.intor)
      ) @ ao_fun_vec  # (self.basis_num)

    def f(param):
      return wave_fun_i(param, ao_fun_vec)

    return jax.vmap(f)(mo_params)
