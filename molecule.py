'''
!! WARNING: this module use pyscf.gto.module for initializing molecule object.

Example：
    h2o_config = 'O 0 0 0; H 0 1 0; H 0 0 1'  # default ångström unit. need to change to Bohr unit.

    >>> from pyscf import gto
    >>> mol = gto.M(atom = h2o_config, basis = '3-21g')
    >>> mol.build()
    >>> mol._basis
    >>> {'H': [[0, [5.447178, 0.156285], [0.824547, 0.904691]], [0, [0.183192, 1.0]]],
    'O': [[0, [322.037, 0.0592394], [48.4308, 0.3515], [10.4206, 0.707658]],
          [0, [7.40294, -0.404453], [1.5762, 1.22156]],
          [0, [0.373684, 1.0]],
          [1, [7.40294, 0.244586], [1.5762, 0.853955]],
          [1, [0.373684, 1.0]]]}

    >>> mol.atom_coords()
    >>> array([[0.        , 0.        , 0.        ],
               [0.        , 1.88972612, 0.        ],
               [0.        , 0.        , 1.88972612]])


    >>> mol.atom_charges()
    >>> array([8, 1, 1], dtype=int32)

    >>> mol.intor('int1e_ovlp_sph').shape    # return the overlap (covariance) of MOs.
    >>> (13, 13)   # 2+2+9 = 13

    >>> mol.tot_electrons()  # total number of electrons.
    >>> 10

    >>> mol.elements
    >>> ['O', 'H', 'H']

'''

import time
import numpy as np
import jax
from jax import random
import jax.numpy as jnp
from jax import vmap, jit
from jdft.functions import decov
from jdft.energy import *
import optax
from jdft.sampler import batch_sampler
from jdft.visualization import *

from pyscf import gto
from pyscf.dft import gen_grid


class molecule():

  def __init__(self, config, spin, basis='3-21g', level=1, eps=1e-10):

    self.pyscf_mol = gto.M(atom=config, basis=basis, spin=spin)
    self.pyscf_mol.build()

    self._basis = self.pyscf_mol._basis

    self.tot_electron = self.pyscf_mol.tot_electrons()
    self.elements = self.pyscf_mol.elements
    self.atom_coords = self.pyscf_mol.atom_coords()
    self.atom_charges = self.pyscf_mol.atom_charges()
    self.spin = spin  # number of non-paired electrons.
    self.nao = self.pyscf_mol.nao  # number of atomic orbitals

    self.nocc = jnp.zeros([2, self.nao])  # number of occupied orbital.
    self.nocc = self.nocc.at[0, :int((self.tot_electron + self.spin) /
                                     2)].set(1)
    self.nocc = self.nocc.at[1, :int((self.tot_electron - self.spin) /
                                     2)].set(1)

    self.cov = self.pyscf_mol.intor('int1e_ovlp_sph')
    # TODO: this integration can be replaced in future. Currently it is calculated on
    # cpu using libcint.

    self.params = None
    self.tracer = []  # to store training curve.

    self.basis_decov = decov(self.cov)

    self.nuclei = {
      'loc': jnp.array(self.atom_coords),
      'charge': jnp.array(self.atom_charges)
    }
    self.level = level
    g = gen_grid.Grids(self.pyscf_mol)
    g.level = self.level
    g.build()
    self.grids = jnp.array(g.coords)
    self.weights = jnp.array(g.weights)
    self.eps = eps
    self.timer = []
    # Warning: in this version, E_nuc_rep is pre-calculated and does not affect the
    # learning process.

    self.E_nuc_rep = E_nuclear(self.nuclei, self.eps)
    print(
      'Initializing... {} grid points are sampled.'.format(self.grids.shape[0])
    )

  def ao_funs(self, r):
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
          output.append(jnp.sum(prm_array[:, 1]*\
              jnp.exp(-prm_array[:, 0]*jnp.linalg.norm(r-coord)**2)*\
                  (2*prm_array[:, 0]/jnp.pi)**(3/4)))

        elif i[0] == 1:
          prm_array = jnp.array(i[1:])
          output += [(r[j]-coord[j]) *jnp.sum(prm_array[:, 1]*jnp.exp(-prm_array[:, 0]*\
              jnp.linalg.norm(r-coord)**2) * (2*prm_array[:, 0]/jnp.pi)**(3/4) * (4*prm_array[:, 0])**0.5) for j in np.arange(3)]
    return jnp.array(output)

  def mo_funs(self, params, r):
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
    ao_fun_vec = self.ao_funs(r)

    def wave_fun_i(param_i, ao_fun_vec):
      orthogonal, _ = jnp.linalg.qr(param_i)  # q is column-orthogal.
      return orthogonal.transpose(
      ) @ self.basis_decov @ ao_fun_vec  #(self.basis_num)

    f = lambda param: wave_fun_i(param, ao_fun_vec)
    return vmap(f)(params) * self.nocc

  def _init_param(self, seed=123):
    key = random.PRNGKey(seed)
    return random.normal(key, [self.nao, self.nao]) / self.nao**0.5

  def train(self, epoch, lr=1e-3, seed=123,\
      converge_threshold=1e-3, batchsize=1000, save_fig=False, **args):
    if self.params is None:
      self.params = self._init_param(seed)
    params = self.params.copy()
    # schedule = optax.warmup_cosine_decay_schedule(
    #             init_value=0.5,
    #             peak_value=1,
    #             warmup_steps=50,
    #             decay_steps=500,
    #             end_value=lr,
    #             )

    if 'optimizer' in args:
      if args['optimizer'] == 'sgd':
        optimizer = optax.sgd(lr)
        # optimizer = optax.chain(
        #     optax.clip(1.0),
        #     optax.sgd(learning_rate=schedule),
        #     )

      elif args['optimizer'] == 'adam':
        optimizer = optax.adam(lr)
        # optimizer = optax.chain(
        #     optax.clip(1.0),
        #     optax.adam(learning_rate=schedule),
        #     )
      else:
        raise NotImplementedError('Optimizer in [\'sgd\', \'adam\']')
    else:
      optimizer = optax.sgd(lr)

    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(seed)

    @jit
    def update(params, opt_state, grids, weights):

      def wave_fun(x):
        return self.mo_funs(params, x)

      def loss(params):
        return E_gs(wave_fun, grids, self.nuclei, weights, self.eps)

      (Egs, Es), Egs_grad = jax.value_and_grad(loss, has_aux=True)(params)
      Ek, Ee, Ex, Eh = Es
      Egs = Egs + self.E_nuc_rep  # plus nuclear repulsion energy.

      updates, opt_state = optimizer.update(Egs_grad, opt_state)
      params = optax.apply_updates(params, updates)
      return params, opt_state, Egs, Ek, Ee, Ex, Eh

    if save_fig:
      file = '/home/aiops/litb/project/dft/experiment/figure/{0:04}'.format(
        0
      ) + '.png'
      save_contour(self, file)

    print('Starting... Random Seed: {}, Batch size: {}'.format(seed, batchsize))

    current_loss = 0
    batch_seeds = jnp.asarray(
      jax.random.uniform(key, (epoch,)) * 100000, dtype=jnp.int32
    )
    Egs_train = []
    Ek_train = []
    Ee_train = []
    Ex_train = []
    Eh_train = []

    start_time = time.time()
    self.timer = []

    for i in range(epoch):

      batch_grids, batch_weights = batch_sampler(
        self.grids, self.weights, batchsize=batchsize, seed=batch_seeds[i]
      )
      if i == 0:
        print(
          'Batch size: {}. Number of batches in each epoch: {}'.format(
            batch_grids[0].shape[0], len(batch_grids)
          )
        )

      nbatch = len(batch_grids)
      batch_tracer = jnp.zeros(5)

      for g, w in zip(batch_grids, batch_weights):
        params, opt_state, Egs, Ek, Ee, Ex, Eh = update(params, opt_state, g, w)
        batch_tracer += jnp.asarray([Egs, Ek, Ee, Ex, Eh])

      if (i + 1) % 1 == 0:
        Batch_mean = batch_tracer / nbatch
        assert Batch_mean.shape == (5,)

        Egs_train.append(Batch_mean[0].item())
        Ek_train.append(Batch_mean[1].item())
        Ee_train.append(Batch_mean[2].item())
        Ex_train.append(Batch_mean[3].item())
        Eh_train.append(Batch_mean[4].item())

        print('Iter: {}/{}. Ground State Energy: {:.3f}.'. \
            format(i+1, epoch, Egs_train[-1]))

        if save_fig:
          file = '/home/aiops/litb/project/dft/experiment/figure/{0:04}'.format(
            i + 1
          ) + '.png'
          save_contour(self, file)

      if jnp.abs(current_loss - Batch_mean[0].item()) < converge_threshold:
        self.params = params.copy()
        print(
          'Converged at iteration {}. Training Time: {:.3f}s'.format(
            (i + 1),
            time.time() - start_time
          )
        )
        print('E_Ground state: ', Egs_train[-1])
        print('E_kinetic: ', Ek_train[-1])
        print('E_ext: ', Ee_train[-1])
        print('E_Hartree: ', Eh_train[-1])
        print('E_xc: ', Ex_train[-1])
        print('E_nuclear_repulsion', self.E_nuc_rep)
        self.tracer += Egs_train

      else:
        current_loss = Batch_mean[0].item()

      self.timer.append(time.time() - start_time)

    self.tracer += Egs_train
    self.params = params.copy()
    print(
      'Not Converged. Training Time: {:.3f}s'.format(time.time() - start_time)
    )
    print('E_Ground state: ', Egs_train[-1])
    print('E_kinetic: ', Ek_train[-1])
    print('E_ext: ', Ee_train[-1])
    print('E_Hartree: ', Eh_train[-1])
    print('E_xc: ', Ex_train[-1])
    print('E_nuclear_repulsion', self.E_nuc_rep)

  def get_wave(self, occ_ao=True):
    '''
        occ_ao: if True, only return occupied orbitals.
        return: wave function value at grid_point.
        '''

    f = lambda r: self.mo_funs(self.params, r)
    if occ_ao:
      alpha = vmap(f)(self.grids)[:, 0, :int(jnp.sum(self.nocc[0, :]))]
      beta = vmap(f)(self.grids)[:, 1, :int(jnp.sum(self.nocc[1, :]))]
      return jnp.concatenate((alpha, beta), axis=1)
    else:
      return vmap(f)(self.grids)

  def get_density(self, r):
    '''
        return: density function: [D, 3] -> [D] where D is the number of grids.
        '''

    f = lambda r: self.mo_funs(self.params, r)
    wave = vmap(f)(r)  # (D, 2, N)
    alpha = wave[:, 0, :int(jnp.sum(self.nocc[0, :]))]
    beta = wave[:, 1, :int(jnp.sum(self.nocc[1, :]))]
    wave_all = jnp.concatenate((alpha, beta), axis=1)
    dens = jnp.sum(wave_all**2, axis=1)
    return dens


if __name__ == '__main__':
  h2o_geometry = '''
  O 0.0000 0.0000 0.1173;
  H 0.0000 0.7572 -0.4692;
  H 0.0000 -0.7572 -0.4692;
  '''

  h2o = molecule(h2o_geometry, spin=0)
  h2o.train(10)
