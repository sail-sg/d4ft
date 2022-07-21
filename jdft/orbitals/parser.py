'''Parsers for reading pyscf objects.'''
import numpy as np


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
