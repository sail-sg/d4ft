import jax
import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse
from pyscf import gto
from jdft.functions import decov
from jdft.intor import Quadrature
# from scipy.special import factorial



def factorial(x):
  x = jnp.asarray(x, dtype=jnp.float32)
  return jnp.exp(jax.lax.lgamma(x+1))


def ao_label_parser(ao_label):
  '''
  convert PySCF ao_label to GTO parameters.
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

  '''
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
  '''
  convert pyscf molecule object to GTO parameters.

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
  '''

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
  def __init__(self):
    pass

  def __call__(self):
    '''
      Args:
        x: shape is [..., 3]
      Returns:
        output: shape equal to [..., num_basis],
          where the batch dims are equal to x
    '''
    raise NotImplementedError('__call__ function has not been implemented')

  def overlap(self):
    '''This computes the overlap between basis functions
    Returns:
      output: shape equal to [num_basis, num_basis]
    '''
    raise NotImplementedError('overlap function has not been implemented.')

  def init(self):
   pass


class GTO(Basis):
  def __init__(self, alpha, i, j, k, c):
    '''
    Args:
      |alpha: shape [n]
      |i: shape [n]
      |j: shape [n]
      |k: shape [n]
      |c: shape [n, 3]. The center of each primitive.
    '''

    super().__init__()
    self.alpha = alpha
    self.i = i
    self.j = j
    self.k = k
    self.c = c
    self.num_basis = self.c.shape[0]

  def __call__(self, r):
    '''
    Args:
      |r: shape [3]

    Returns:
      |shape [n], where n is the number of Gaussian primitive.
    '''

    const = (2 * self.alpha/np.pi)**(3/4)
    const *= (((8 * self.alpha) ** (self.i + self.j + self.k) *
               factorial(self.i) * factorial(self.j) * factorial(self.k))/
              (factorial(2 * self.i) * factorial(2 * self.j) * factorial(2 * self.k))) ** 0.5

    output = jnp.power(r - self.c, jnp.stack([self.i, self.j, self.k], axis=1))
    output = jnp.prod(output, axis=1)

    fc = lambda c: jnp.linalg.norm(r - c) ** 2
    output *= jnp.exp(-self.alpha * jax.vmap(fc)(self.c))

    return const * output


class Pople(Basis):
  def __init__(self, pyscf_mol):
    super().__init__()
    self.pyscf_mol = pyscf_mol
    self.elements = pyscf_mol.elements
    self.atom_coords = pyscf_mol.atom_coords()
    self._basis = pyscf_mol._basis

    self.ao_labels = pyscf_mol.ao_labels()

    (alpha, i, j, k, c), (coeff_data, coeff_idx) = Pyscf2GTO_parser(self.pyscf_mol)
    self.alpha = alpha
    self.i = i
    self.j = j
    self.k = k
    self.c = c
    # coeff_idx = coeff_idx
    self.coeff_mat =  sparse.BCOO((coeff_data, coeff_idx), shape=(len(self.pyscf_mol.ao_labels()), len(i)))
    self.coeff_mat = sparse.sparsify(jnp.expand_dims)(self.coeff_mat, axis=0)
    self._gto = GTO(self.alpha, self.i, self.j, self.k, self.c)

  def __call__(self, r):
    '''
    Args:
      |r: shape [G, 3] where G is the number of grids.

    Returns:
      |shape: [G, num_ao], where num_ao is the number of atomic orbitals.
    '''
    basis = self._gto(r)
    basis = jnp.expand_dims(basis, axis=(0, 2))
    output = sparse.bcoo_dot_general(
      self.coeff_mat, basis,
      dimension_numbers=(((2), (1)), ((0), (0))))
    return jnp.squeeze(output)

  def overlap(self, intor=None):
    if not intor:
      return self.pyscf_mol.intor('int1e_ovlp_sph')
    else:
      intor.mo = self.__call__
      return intor.double_overlap()


class MO_qr(Basis):
  '''
  molecular orbital using QR decomposition.
  '''
  def __init__(self, ao, intor=None):
    super().__init__()
    self.ao = ao
    if not intor:
      # self.basis_decov = decov(self.ao.overlap())
      raise AssertionError
    else:
      self.intor = intor
      self.intor.mo = self.ao
      self.basis_decov = decov(self.ao.overlap(intor))

  def __call__(self, params, r):
    '''
    R^3 -> R^N. N-body molecular orbital wave functions.
    input: (N: the number of atomic orbitals.)
      |params: N*N
      |r: (3)
    output:
      |molecular orbitals:(2, N)
    '''
    params = jnp.expand_dims(params, 0)
    params = jnp.repeat(params, 2, 0)
    ao_fun_vec = self.ao(r)

    def wave_fun_i(param_i, ao_fun_vec):
      orthogonal, _ = jnp.linalg.qr(param_i)  # q is column-orthogal.
      return orthogonal.transpose(
      ) @ decov(self.ao.overlap(self.intor)) @ ao_fun_vec  #(self.basis_num)

    f = lambda param: wave_fun_i(param, ao_fun_vec)
    return jax.vmap(f)(params)

