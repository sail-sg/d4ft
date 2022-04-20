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
    h20_config = [
        ['o' , (0. , 0.     , 0.)],
        ['H' , (0. , -0.757 , 0.587)],
        ['H' , (0. , 0.757  , 0.587)] ]
    
    h2o = molecule(h20_config)

'''

from cmath import exp
import numpy as np
import jax.random as random
import jax as jnp
from jax import vmap, jit
from cdft.functions import *
from cdft.basis.const import *
import json
from cdft.energy import *
import optax

from cdft.sampling import *

class molecule(object):
    def __init__(self, config, basis='3-21g'):
        self.config = config
        self.basis = basis
        
        self.atom = [i[0] for i in self.config]
        self.location = [i[1] for i in self.config]
        
        self.atom_num = [ELEMENT2NUM[i] for i in self.atom]  # list of strings
        self.tot_electron = np.sum([int(i) for i in self.atom_num])
        
        self._get_basis_pople()
        self._init_basis()
        
        self.params = self._init_param()
        self.nuclei = {'loc': jnp.array(self.location),
                       'charge': jnp.array([int(i) for i in self.atom_num])}
        
        
        
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
        param_file = '../cdft/basis/'+ self.basis + '.json'
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
        self.basis_label = []      #(l, m, n, zeta) as in the gaussian basis.
        
        for i in np.arange(self.exponents_num):
            zeta = self.exponents_list[i]
            
            if self.orbital_type_list[i]==0:
                self.basis_label.append([0, 0, 0, zeta])
            
            elif self.orbital_type_list[i]==1:
                self.basis_label.append([1, 0, 0, zeta])
                self.basis_label.append([0, 1, 0, zeta])
                self.basis_label.append([0, 0, 1, zeta])
                
            elif self.orbital_type_list[i]==2:
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
                  
            _gauss_ = jnp.exp(-zeta*jnp.sum(r**2))   # non_centerized.
            
            if self.orbital_type_list[i]==0:
                basis_list.append(_gauss_)
            
            elif self.orbital_type_list[i]==1:
                basis_list += [r[k]*_gauss_ for k in jnp.arange(3)]
                
            elif self.orbital_type_list[i]==2:
                xyz_list = [r[0]**2, r[1]**2, r[2]**2, r[0]*r[1], r[1]*r[2], r[0]*r[2]]
                basis_list += [k*_gauss_ for k in xyz_list]
                
            else:
                raise NotImplementedError('f orbitals have not been implemented')
        
        return jnp.array(basis_list)
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
            g3 = g1_label+g2_label
            return gaussian_intergral(g3[3], g3[0]) * \
                gaussian_intergral(g3[3], g3[1]) * \
                gaussian_intergral(g3[3], g3[2])
        
        self.basis_cov = vmap(lambda g2: vmap(lambda g1: \
            single_cov(g1, g2))(self.basis_label))(self.basis_label)
        
        v, u = jnp.linalg.eigh(self.basis_cov)
        self.orbital_energy = jnp.real(v).copy()
        
        v = jnp.diag(jnp.real(v)**(-1/2)) + jnp.eye(v.shape[0])*1e-10
        ut = jnp.real(u).transpose()
        
        self.basis_decov = jnp.matmul(v, ut)
        
              
    def wave_fun(self, param, r):
        '''
        
        input: 
        r: location coordinate
            shape: (3)
        param: 
            shape: (self.basis_num, self.basis_num, self.basis_num) # non-spin involved.
        
        output:
            value of wave function
            shape: (self.basis_num)
        
        '''
        
        
        basis_list = self.get_basis(r)  # basis_list
        
        def wave_fun_i(param_i, basis_list):
            orthogonal, _ = jnp.linalg.qr(param_i) # q is column-orthogal. 
            return jnp.sum(orthogonal.transpose()@self.basis_decov@basis_list)
        
        f = lambda param: wave_fun_i(param, basis_list)
        return vmap(f)(param)

    
    def _init_param(self,  seed=123):
        key = random.PRNGKey(seed)
        return random.normal(key, [self.basis_num, self.basis_num, self.basis_num])/self.basis_num**0.5
        
    
    def train(self, epoch, lr=1e-3, n=2000, sample_method = 'simple grid', seed=123, if_val=True):
        '''
        sample_method should be in ['simple grid', 'pyscf', 'poisson disc' ]
        '''
        
        params = self.params.copy()
        optimizer = optax.sgd(lr)
        opt_state = optimizer.init(params)
        key = jax.random.PRNGKey(seed)
        keys = random.split(key, epoch)
        
        limit = 5
        cellsize = 0.2
         
        def wave_fun_N(param, r, N):
            mask_idx = jnp.argsort(self.orbital_energy)  
            # mask = jnp.zeros(self.basis_num)
            # mask = mask.at[mask_idx[:N]].set(1)
            # return self.wave_fun(param, r) * mask
            return self.wave_fun(param, r).at[mask_idx[:N]].get()
        
        @jit
        def update(params, opt_state, key):
            meshgrid = simple_grid(key, limit, cellsize, n)
            loss = lambda params: E_gs(wave_fun_N, meshgrid, params, self.tot_electron, self.nuclei, limit=limit)
            loss_value, grads = jax.value_and_grad(loss)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, loss_value
        
        
        meshgrid = simple_grid(keys[0], limit, cellsize, n=20000)
        E_gs_train = [E_gs(wave_fun_N, meshgrid, params, self.tot_electron, self.nuclei, limit=limit)] 
        if if_val:
            E_gs_val = [E_gs(wave_fun_N, meshgrid, params, self.tot_electron, self.nuclei, limit=limit)]
        print('Initilization. Ground State Energy: {:.3f}'. \
                    format(E_gs_train[-1]))
        
        for i in range(epoch):
            params, loss_value = update(params, opt_state, keys[i])
            if (i+1)%10 == 0:
                E_gs_train.append(loss_value)
                if if_val:
                    E_gs_val.append(E_gs(wave_fun_N, meshgrid, params, self.tot_electron, self.nuclei, limit=limit))
                    print('Iter: {}/{}. Ground State Energy: {:.3f}. Val Energy: {:.3f}.'. \
                        format(i+1, epoch, loss_value, E_gs_val[-1]))
                else: 
                    print('Iter: {}/{}. Ground State Energy: {:.3f}.'. \
                        format(i+1, epoch, loss_value))
                    
            
        self.params = params.copy()
        print('E_kinetic ', E_kinetic(wave_fun_N, meshgrid, params, self.tot_electron, limit=limit))
        print('E_ext: ', E_ext(wave_fun_N, meshgrid, self.nuclei, params, self.tot_electron, limit=limit))
        print('E_Hartree: ', E_Hartree(wave_fun_N, meshgrid, params, self.tot_electron, limit=limit))
        print('E_xc: ', E_XC_LDA(wave_fun_N, meshgrid, params, self.tot_electron, limit=limit))            
        
        return E_gs_train, E_gs_val


if __name__ == '__main__':
    h20_config = [
        ['o' , (0. , 0.     , 0.)],
        ['H' , (0. , -0.757 , 0.587)],
        ['H' , (0. , 0.757  , 0.587)] ]
    
    h2o = molecule(h20_config)
    
    
    
    