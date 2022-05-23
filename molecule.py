# from cmath import exp
import numpy as np
import jax.random as random
import jax as jnp
from jax import vmap, jit
from jdft.functions import *
from jdft.basis.const import *
from jdft.energy import *
import optax
from jdft.sampling import *
from jdft.visualization import *
import time
from pyscf import gto
from pyscf.dft import gen_grid


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

def decov(cov):
    v, u = jnp.linalg.eigh(cov)
    v = jnp.diag(jnp.real(v)**(-1/2)) + jnp.eye(v.shape[0])*1e-10
    ut = jnp.real(u).transpose()
    return jnp.matmul(v, ut)


class molecule(object):
    def __init__(self, config, spin, basis='3-21g', level=1, eps=1e-10):

        self.pyscf_mol = gto.M(atom=config, basis=basis, spin=spin)
        self.pyscf_mol.build()

        self._basis = self.pyscf_mol._basis

        self.tot_electron = self.pyscf_mol.tot_electrons()
        self.elements = self.pyscf_mol.elements
        self.atom_coords = self.pyscf_mol.atom_coords()
        self.atom_charges = self.pyscf_mol.atom_charges()

        self.spin = spin  # number of non-paired electrons.

        self.nao = self.pyscf_mol.nao   # number of atomic orbitals

        self.nocc = jnp.zeros([2, self.nao])    # number of occupied orbital.
        self.nocc = self.nocc.at[0, :int((self.tot_electron+self.spin)/2)].set(1)
        self.nocc = self.nocc.at[1, :int((self.tot_electron-self.spin)/2)].set(1)

        self.cov = self.pyscf_mol.intor('int1e_ovlp_sph')
        #TODO: this integration can be replaced in future

        self.params = None
        self.tracer = []     # to store training curve.

        self.basis_decov = decov(self.cov)
        self.nuclei = {'loc': jnp.array(self.atom_coords),
                       'charge': jnp.array(self.atom_charges)}


        self.level=level
        g = gen_grid.Grids(self.pyscf_mol)
        g.level = self.level

        g.build()
        self.pyscf_grid = g.coords
        self.pyscf_weights = g.weights

        self.eps = eps

        # Warning: in this version, E_nuc_rep is pre-calculated and does not affect the learning process.
        self.E_nuc_rep = E_nuclear(self.nuclei, self.eps)


    def ao_funs(self, r):
        # R^3 -> R^N where N is the number of atomic orbitals.
        #
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
                        jnp.linalg.norm(r-coord)**2) * (2*prm_array[:, 0]/jnp.pi)**(3/4) * (4*prm_array[:, 0])**0.5)\
                        for j in np.arange(3)]
        return jnp.array(output)


    def mo_funs(self, params, r):
        '''
        molecular orbital wave functions.
        input: (N: the number of atomic orbitals.)
          |params: N*N
          |r: (3)
        output:
          |molecular orbitals:(2, N)
        '''
        params = jnp.expand_dims(params, 0)
        params = jnp.repeat(params, 2, 0)

        ao_fun_vec =  self.ao_funs(r)
        def wave_fun_i(param_i, ao_fun_vec):
            orthogonal, _ = jnp.linalg.qr(param_i)     # q is column-orthogal.
            return orthogonal.transpose()@self.basis_decov@ao_fun_vec   #(self.basis_num)

        f = lambda param: wave_fun_i(param, ao_fun_vec)
        return vmap(f)(params) * self.nocc


    def _init_param(self,  seed=123):
        key = random.PRNGKey(seed)
        return random.normal(key, [self.nao, self.nao])/self.nao**0.5


    def train(self, epoch, lr=1e-3, sample_method = 'pyscf', seed=123, if_val=False, \
        converge_threshold=1e-3, save_fig=False, **args):


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
        keys = random.split(key, epoch)

        if sample_method == 'pyscf':
            meshgrid =  self.pyscf_grid
            weight = self.pyscf_weights

            @jit
            def update(params, opt_state, key):
                loss = lambda params: E_gs(self.mo_funs, self.pyscf_grid, params, self.nuclei, weight)
                loss_value, grads = jax.value_and_grad(loss)(params)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return params, loss_value

        else:
            limit = 5
            cellsize = 0.2
            if args['n']:
                n = args['n']
            else:
                n = 2000

            meshgrid = simple_grid(keys[0], limit, cellsize, n)
            weight = jnp.ones(n)*limit*2/n

            @jit
            def update(params, opt_state, key):
                meshgrid = simple_grid(key, limit, cellsize, n)
                loss = lambda params: E_gs(self.mo_funs, meshgrid, params, self.nuclei, weight)
                loss_value, grads = jax.value_and_grad(loss)(params)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return params, loss_value


        E_gs_train = [E_gs(self.mo_funs, meshgrid, params, self.nuclei, weight)+self.E_nuc_rep]
        self.Egs = E_gs_train[-1]
        if save_fig:
            file = '/home/aiops/litb/project/dft/experiment/figure/{0:04}'.format(0)+'.png'
            save_contour(self, file)

        if if_val:
            E_gs_val = [E_gs(self.mo_funs, meshgrid, params, self.nuclei, weight)+self.E_nuc_rep]
        print('Initilization. Ground State Energy: {:.3f}'. \
                    format(E_gs_train[-1]))

        current_loss = 0
        start_time = time.time()
        for i in range(epoch):
            params, loss_value = update(params, opt_state, keys[i])
            self.Egs = loss_value
            self.params = params

            if (i+1)%10 == 0:
                E_gs_train.append(loss_value)
                if if_val:
                    E_gs_val.append(E_gs(self.mo_funs, meshgrid, params, self.nuclei, weight)+self.E_nuc_rep)
                    print('Iter: {}/{}. Ground State Energy: {:.3f}. Val Energy: {:.3f}.'. \
                        format(i+1, epoch, loss_value+self.E_nuc_rep, E_gs_val[-1]))
                else:
                    print('Iter: {}/{}. Ground State Energy: {:.3f}.'. \
                        format(i+1, epoch, loss_value+self.E_nuc_rep))

                if save_fig:
                    file = '/home/aiops/litb/project/dft/experiment/figure/{0:04}'.format(i+1)+'.png'
                    save_contour(self, file)

            if  jnp.abs(current_loss - loss_value)<converge_threshold:

                self.params = params.copy()
                print('Converged at iteration {}. Training Time: {:.3f}s'.format((i+1), time.time()-start_time))
                print('E_Ground state: ', current_loss+self.E_nuc_rep)
                print('E_kinetic ', E_kinetic(self.mo_funs, meshgrid, params, weight))
                print('E_ext: ', E_ext(self.mo_funs, meshgrid, self.nuclei, params, weight))
                print('E_Hartree: ', E_Hartree(self.mo_funs, meshgrid, params, weight))
                print('E_xc: ', E_XC_LDA(self.mo_funs, meshgrid, params, weight))
                print('E_nuclear_repulsion', self.E_nuc_rep)

                self.tracer += E_gs_train
                if if_val:
                    # return E_gs_train, E_gs_val
                    return
                else:
                    # return E_gs_train
                    return

            else:
                current_loss = loss_value


        self.params = params.copy()
        print('Not Converged. Training Time: {:.3f}s'.format(time.time()-start_time))
        print('E_Ground state: ', current_loss+self.E_nuc_rep)
        print('E_kinetic: ', E_kinetic(self.mo_funs, meshgrid, params, weight))
        print('E_ext: ', E_ext(self.mo_funs, meshgrid, self.nuclei, params, weight))
        print('E_Hartree: ', E_Hartree(self.mo_funs, meshgrid, params, weight))
        print('E_xc: ', E_XC_LDA(self.mo_funs, meshgrid, params, weight))
        self.tracer += E_gs_train
        print('E_nuclear_repulsion', self.E_nuc_rep)

        # if if_val:
        #     return E_gs_train, E_gs_val
        # else:
        #     return E_gs_train


    def get_wave(self, occ_ao=True):
        '''
        occ_ao: if True, only return occupied orbitals.
        return: wave function value at grid_point.
        '''

        f = lambda r: self.mo_funs(self.params, r)
        if occ_ao:
            alpha = vmap(f)(self.pyscf_grid)[:, 0, :int(jnp.sum(self.nocc[0, :]))]
            beta = vmap(f)(self.pyscf_grid)[:, 1, :int(jnp.sum(self.nocc[1, :]))]
            return jnp.concatenate((alpha, beta), axis=1)
        else:
            return vmap(f)(self.pyscf_grid)


    def get_density(self, r):
        '''
        return: density function: (D, 3) -> (D)
        '''
        f = lambda r: self.mo_funs(self.params, r)
        wave = vmap(f)(r)   # (D, 2, N)
        alpha = wave[:, 0, :int(jnp.sum(self.nocc[0, :]))]
        beta = wave[:, 1, :int(jnp.sum(self.nocc[1, :]))]
        wave_all = jnp.concatenate((alpha, beta), axis=1)
        dens = jnp.sum(wave_all**2, axis=1)
        return dens

if __name__ == '__main__':
    h2o_config = [
        ['o' , (0. , 0.     , 0.)],
        ['H' , (0. , -0.757 , 0.587)],
        ['H' , (0. , 0.757  , 0.587)] ]

    h2o = molecule(h2o_config)



