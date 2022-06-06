import jax
import numpy as np
import jax.numpy as jnp
from pyscf import gto
from jdft.functions import decov


class Basis():
  def __init__(self):
    pass

  def __call__(self):
    pass

  def overlap(self):
    raise NotImplementedError('overlap function has not been implemented.')


class Pople(Basis):
  ## this is a Pople type atomic orbital class.
  #TODO: d/f orbitals.

  def __init__(self, pyscf_mol):
    super().__init__()
    self.pyscf_mol = pyscf_mol
    self.elements = pyscf_mol.elements
    self.atom_coords = pyscf_mol.atom_coords()
    self._basis = pyscf_mol._basis

  def __call__(self, r):
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

  def overlap(self):
    return self.pyscf_mol.intor('int1e_ovlp_sph')


class MO_qr(Basis):
  # molecular orbital using QR decomposition.
  def __init__(self, ao):
    super().__init__()
    self.ao = ao
    self.basis_decov = decov(self.ao.overlap())

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
      ) @ self.basis_decov @ ao_fun_vec  #(self.basis_num)

    f = lambda param: wave_fun_i(param, ao_fun_vec)
    return jax.vmap(f)(params)

  # def overlap(self):
  #   return super().overlap()
