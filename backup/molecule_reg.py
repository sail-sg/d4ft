'''
!! Warining: This module is only optimized for H to Ne (atomic number smaller than 10).

molecule object:
    Attributes:
        config: dict
            molecular configuration
        basis: str
            basis set
        charges: np.array

Example：
    h2o_config = [
        ['o' , (0. , 0.     , 0.)],
        ['H' , (0. , -0.757 , 0.587)],
        ['H' , (0. , 0.757  , 0.587)] ]

    h2o = molecule(h2o_config)

'''

from cmath import exp
import numpy as np
import jax.random as random
import jax as jnp
from jax import vmap, jit
from jdft.functions import *
from jdft.basis.const import *
import json
from jdft.energy import *
import optax
from jdft.sampling import *
from jdft.regularizer import reg_ort


class molecule_reg(object):

  def __init__(self, config, spin=None, basis='3-21g'):
    self.config = config
    self.basis = basis

    self.atom = [i[0] for i in self.config]
    self.location = [i[1] for i in self.config]

    self.atom_num = [ELEMENT2NUM[i] for i in self.atom]  # list of strings
    self.tot_electron = np.sum([int(i) for i in self.atom_num])

    self._get_basis_pople()
    self._init_basis()

    if spin is None:
      self.spin = self.tot_electron % 2
    else:
      self.spin = spin  # number of non-paired electrons.

    self.nocc = jnp.zeros([2, self.basis_num])  # number of occupied orbital.
    self.nocc = self.nocc.at[0, :int((self.tot_electron + self.spin) /
                                     2)].set(1)
    self.nocc = self.nocc.at[1, :int((self.tot_electron - self.spin) /
                                     2)].set(1)

    self.params = None
    self.tracer = []  # to store training curve.
    self.nuclei = {
      'loc': jnp.array(self.location),
      'charge': jnp.array([int(i) for i in self.atom_num])
    }

    ## initilize the wave function using basis set.
    ## wave function:
    #       input: (3)-dimensional coordinate r.
    #       output: (N). N single-partical wave function value at r.

    ## WARNING: only valid for the first and second row.

  def _get_basis_pople(self):
    '''
        get the wave function(s) of the i-th atom for pople type basis sets.
        It worth noting that here the coeffients are NOT of molecular orbitals, but that of
        primitive gaussian function.

        The primitive gaussian function:
            s: exp(-a r^2)
            p_x: xexp(-a_x r^2)
            d_xx" x^2 exp(-a_xx r^2)


            input:
                atoms: list of str.  lsit of atomic numbers, such as ['8', '1', '1'] for h2o.
            output:
                a list of $sum(atoms)$ number of wave functions, such as a list of 10 wave functions for h2o.

        !WARNING: only valid for the first and second row.
        !WARNING: only valid for split-valence type.
        !WARNING: f orbital hasn't been considered.

        Example:
            self.exponents_list = [zeta1, zeta2, zeta2]
            self.orbital_type_list = [0, 0, 1]
            # 1 represent p orbital, which has 3 basis.
            # 2 represent d orbital, which has 6 basis.
            # 3 represent f orbital, which has 10 basis

            basis:
                [exp(-zeta1*|r|), exp(-zeta2*|r|), xexp(-zeta2*|r|),
                yexp(-zeta2*|r|), zexp(-zeta2*|r|)]

        '''
    param_file = '../cdft/basis/' + self.basis + '.json'
    with open(param_file, 'r') as f:
      basis_json = json.load(f)

    # basis_list = []  # list of basis
    exponents_list = []
    orbital_type_list = []  # s, p, d or f, represented by 0, 1, 2, 3
    atom_exp_list = []
    atom_unique = []

    for i in self.atom_num:
      if i not in atom_unique:
        atom_unique.append(i)
        basis_ = basis_json['elements'][i]['electron_shells']
        for basis_i in basis_:
          for m in basis_i['angular_momentum']:
            exponents_list += [float(e) for e in basis_i['exponents']]
            orbital_type_list += len(basis_i['exponents']) * [m]
            atom_exp_list += len(basis_i['exponents']) * [i]

    self.exponents_list = exponents_list
    self.orbital_type_list = orbital_type_list
    self.atom_exp_list = atom_exp_list

    if len(self.exponents_list) != len(self.orbital_type_list):
      raise ValueError('Loading basis set error.')
    else:
      self.exponents_num = len(self.exponents_list)
      self.basis_num = np.sum([1*(i==0) for i in self.orbital_type_list]) + \
           np.sum([3*(i==1) for i in self.orbital_type_list]) + \
           np.sum([6*(i==2) for i in self.orbital_type_list])
      #  np.sum([10*(i==3) for i in self.orbital_type_list])

  def _init_basis(self):
    self.basis_label = []  #(l, m, n, zeta) as in the gaussian basis.

    for i in np.arange(self.exponents_num):
      zeta = self.exponents_list[i]

      if self.orbital_type_list[i] == 0:
        self.basis_label.append([0, 0, 0, zeta])

      elif self.orbital_type_list[i] == 1:
        self.basis_label.append([1, 0, 0, zeta])
        self.basis_label.append([0, 1, 0, zeta])
        self.basis_label.append([0, 0, 1, zeta])

      elif self.orbital_type_list[i] == 2:
        self.basis_label.append([2, 0, 0, zeta])
        self.basis_label.append([0, 2, 0, zeta])
        self.basis_label.append([0, 0, 2, zeta])
        self.basis_label.append([1, 1, 0, zeta])
        self.basis_label.append([0, 1, 1, zeta])
        self.basis_label.append([1, 0, 1, zeta])

    self.basis_label = jnp.array(self.basis_label)
    self.get_basis_cov()

  def get_basis(self, r):
    basis_list = []

    for i in np.arange(self.exponents_num):
      zeta = self.exponents_list[i]
      atom_idx = self.atom_exp_list[i]

      # atom_center = self.location[atom_idx]
      # _gauss_ = jnp.exp(-zeta*r2(r, atom_center))

      _gauss_ = jnp.exp(-zeta * jnp.sum(r**2))  # non_centerized.

      if self.orbital_type_list[i] == 0:
        basis_list.append(_gauss_)

      elif self.orbital_type_list[i] == 1:
        basis_list += [r[k] * _gauss_ for k in jnp.arange(3)]

      elif self.orbital_type_list[i] == 2:
        xyz_list = [
          r[0]**2, r[1]**2, r[2]**2, r[0] * r[1], r[1] * r[2], r[0] * r[2]
        ]
        basis_list += [k * _gauss_ for k in xyz_list]

      else:
        raise NotImplementedError('f orbitals have not been implemented')

    return jnp.array(basis_list)  # shape: (self.basis_num)
    # return basis_list

  def get_basis_cov(self):
    '''
        first basis: (l1, m1, n1, zeta1)
        second basis:  (l2, m2, n2, zeta2)

        \int x^(n) exp(-ax^2) dx
        = (n-1)!!/(2a)^(n/2) (n/alpha)**0.5

        self.basis_label: a list of lists (self.basis_num, 4)

        output:
            shape: (self.basis_num * self.basis_num)
            self.basis_num must be larger than the number of total charge number.

        '''

    def single_cov(g1_label, g2_label):
      g3 = g1_label + g2_label
      return gaussian_intergral(g3[3], g3[0]) * \
          gaussian_intergral(g3[3], g3[1]) * \
          gaussian_intergral(g3[3], g3[2])

    self.basis_cov = vmap(lambda g2: vmap(lambda g1: \
        single_cov(g1, g2))(self.basis_label))(self.basis_label)

    v, u = jnp.linalg.eigh(self.basis_cov)
    self.orbital_energy = jnp.real(v).copy()

    v = jnp.diag(jnp.real(v)**(-1 / 2)) + jnp.eye(v.shape[0]) * 1e-10
    ut = jnp.real(u).transpose()

    self.basis_decov = jnp.matmul(v, ut)

  def wave_fun_N(self, param, r):
    '''
        input:
        r: location coordinate
            shape: (3)
        param:
            shape: (2, self.basis_num, self.basis_num)

        output:
            value of N wave functions
            shape: (2, self.basis_num)

        '''

    basis_list = self.get_basis(r)  # basis_list

    def wave_fun_i(param_i, basis_list):
      # orthogonal, _ = jnp.linalg.qr(param_i)     # q is column-orthogal.
      return param_i.transpose(
      ) @ self.basis_decov @ basis_list  #(self.basis_num)

    f = lambda param: wave_fun_i(param, basis_list)
    return vmap(f)(param) * self.nocc  # shape: (2, self.basis_num)

  def _init_param(self, seed=123):
    key = random.PRNGKey(seed)
    return random.normal(
      key, [2, self.basis_num, self.basis_num]
    ) / self.basis_num**0.5

  def train(
    self,
    epoch,
    lr=1e-3,
    sample_method='simple grid',
    seed=123,
    if_val=False,
    beta=1e6,
    **args
  ):
    '''
        sample_method should be in ['simple grid', 'pyscf', 'poisson disc' ]
        '''

    if sample_method == 'pyscf':
      from pyscf import gto
      from pyscf.dft import gen_grid

      mol = gto.Mole(basis='sto-3g', spin=self.spin)
      mol.atom = self.config
      mol.build()

      g = gen_grid.Grids(mol)
      if 'level' in args:
        g.level = args['level']
      else:
        g.level = 1

      g.build()

      self.pyscf_grid = g.coords
      self.pyscf_weights = g.weights

    if self.params is None:
      self.params = self._init_param(seed)

    params = self.params.copy()

    if 'optimizer' in args:
      if args['optimizer'] == 'sgd':
        optimizer = optax.sgd(lr)
      elif args['optimizer'] == 'adam':
        optimizer = optax.adam(lr)
      else:
        raise NotImplementedError('Optimizer in [\'sgd\', \'adam\']')
    else:
      optimizer = optax.sgd(lr)

    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(seed)
    keys = random.split(key, epoch)

    if sample_method == 'pyscf':
      meshgrid = self.pyscf_grid
      weight = self.pyscf_weights

      @jit
      def update(params, opt_state, key):
        loss = lambda params: E_gs(
          self.wave_fun_N, self.pyscf_grid, params, self.nuclei, weight
        )
        reg = lambda params: beta * reg_ort(params)
        loss_value, grads = jax.value_and_grad(loss)(params)
        reg_value, reg_grads = jax.value_and_grad(reg)(params)
        updates, opt_state = optimizer.update(grads + reg_grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, loss_value, reg_value

    else:
      limit = 5
      cellsize = 0.2
      if args['n']:
        n = args['n']
      else:
        n = 2000

      meshgrid = simple_grid(keys[0], limit, cellsize, n)
      weight = jnp.ones(n) * limit * 2 / n

      @jit
      def update(params, opt_state, key):
        meshgrid = simple_grid(key, limit, cellsize, n)
        loss = lambda params: E_gs(
          self.wave_fun_N, meshgrid, params, self.nuclei, weight
        )
        loss_value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, loss_value

    E_gs_train = [E_gs(self.wave_fun_N, meshgrid, params, self.nuclei, weight)]
    if if_val:
      E_gs_val = [E_gs(self.wave_fun_N, meshgrid, params, self.nuclei, weight)]
    print('Initilization. Ground State Energy: {:.3f}'. \
                format(E_gs_train[-1]))

    for i in range(epoch):
      params, loss_value, reg_value = update(params, opt_state, keys[i])
      if (i + 1) % 100 == 0:
        E_gs_train.append(loss_value)
        if if_val:
          E_gs_val.append(
            E_gs(self.wave_fun_N, meshgrid, params, self.nuclei, weight)
          )
          print('Iter: {}/{}. Ground State Energy: {:.3f}. Val Energy: {:.3f}.'. \
              format(i+1, epoch, loss_value, E_gs_val[-1]))
        else:
          print('Iter: {}/{}. Ground State Energy: {:.3f}. Regularizer: {:3f}'. \
              format(i+1, epoch, loss_value, reg_value))

    self.params = params.copy()
    print('E_kinetic ', E_kinetic(self.wave_fun_N, meshgrid, params, weight))
    print(
      'E_ext: ', E_ext(self.wave_fun_N, meshgrid, self.nuclei, params, weight)
    )
    print('E_Hartree: ', E_Hartree(self.wave_fun_N, meshgrid, params, weight))
    print('E_xc: ', E_XC_LDA(self.wave_fun_N, meshgrid, params, weight))

    self.tracer += E_gs_train
    if if_val:
      return E_gs_train, E_gs_val
    else:
      return E_gs_train


if __name__ == '__main__':
  h2o_config = [
    ['o', (0., 0., 0.)], ['H', (0., -0.757, 0.587)], ['H', (0., 0.757, 0.587)]
  ]

  h2o = molecule_reg(h2o_config)