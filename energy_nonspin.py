import jax
from jax import vmap
import jax.numpy as jnp
from cdft.functions import *


def E_kinetic(wave_fun, meshgrid, params, N, weight):
    
    '''
    E_gs = T_s + V_ext + V_Hartree + V_xc
    This function is to compute the kinetic energy T_s.
    
    input: 
        meshgrid: (D, 3)
        wave_fun: (3)-->(N)  a function that calculate the wave function. 
        params: parameter array for wavefun
        N: number of electrons.
        weight: (D) 
    output:
        kinetic_energy: scalar
    '''

    f = lambda x: wave_fun(params, x, N)
    def laplacian_3d(r):
        
        '''
        this function computes the laplacian operator
        
        1/2 \partial^2 f/\partial x^2 + \partial^2 f/\partial y^2 + \partial^2 f/\partial y^2 
        
        which is the diagonal of heissen matrix
        # TODO: This can be more efficient without calculating the off-diagonal entries.
        input
            r: shape (3)
        
        output: scalar
        '''
        
        hessian_diag = jnp.diagonal(jax.jacfwd(jax.jacrev(f))(r), 0, 1, 2)    
        return jnp.sum(hessian_diag, axis=1)/2
    
    wave_at_grid = vmap(f)(meshgrid)
    batched_lap = vmap(laplacian_3d)(meshgrid)    
    return -jnp.sum(batched_lap * wave_at_grid * jnp.expand_dims(weight, 1))


def E_ext(wave_fun, meshgrid, nuclei, params, N, weight, eps=1e-10):
    '''
    input: 
        wave_fun: (3)-->(N)
        meshgrid: (D, 3)
        nuclei: dict {'loc': jnp.array [A, 3], 'charge':jnp.array or a list of scalar}
        weight: (D) 
    output:
        scalar
    '''
    
    f = lambda x: wave_fun(params, x, N)
    wave_at_grid = vmap(f)(meshgrid) # shape: (D, N)
    nuclei_loc = nuclei['loc']
    nuclei_charge = nuclei['charge']
    num_ncl = len(nuclei_loc)
    output = 0
    
    for k in range(num_ncl):
        x = jnp.abs(meshgrid-nuclei_loc[k].reshape((1, 3)))   # shape: (D, 3)
        x = jnp.sum(x**2, axis=1)**0.5          #shape: (D)
        x = - nuclei_charge[k]/(x + eps)
        output += jnp.sum(wave_at_grid**2 * jnp.expand_dims(x, 1)* jnp.expand_dims(weight, 1))
    
    return output


def E_XC_LDA(wave_fun, meshgrid, params, N, weight):
    '''
    input:
        meshgrid: (D, 3)
        wave_fun: (3)-->(N)  a function that calculate the wave function. 
        params: params for wavefun
        
    output:
        kinetic_energy: scalar
    '''
    f = lambda x: wave_fun(params, x, N)
    wave_at_grid = vmap(f)(meshgrid) # shape: (D, N)
    density = wave_at_grid**2
    const = -3/4*(3/jnp.pi)**(1/3) 
    return jnp.sum(density ** (4/3) * jnp.expand_dims(weight, 1)) * const 


def E_Hartree(wave_fun, meshgrid, params, N, weight, eps=1e-10):
    '''
    Warning: this function is computing in a fully pairwised manner, which is prune to out-of-memory issue.
    meshgrid: (D, 3)
    '''
    
    f = lambda x: wave_fun(params, x, N)
    density_at_grid = vmap(f)(meshgrid) # shape: (D, N)
    density_at_grid = density_at_grid**2  # shape: (D, N)
    
    density_at_grid = density_at_grid.transpose(1, 0)   # shape: (N, D)
    density_at_grid *= jnp.expand_dims(weight, 0)

    dist_pair = distmat(meshgrid)
    density_at_grid = jnp.sum(density_at_grid, axis=0)  # shape: (D)
    
    # def f(i):        
    #     density_pair = jax.lax.batch_matmul(jnp.expand_dims(density_at_grid, 1), 
    #                                         jnp.expand_dims(density_at_grid, 0))
    #     return jnp.sum(set_diag_zero(density_pair/(dist_pair+eps)))
    density_pair = jax.lax.batch_matmul(jnp.expand_dims(density_at_grid, 1), 
                                        jnp.expand_dims(density_at_grid, 0))
    output = jnp.sum(set_diag_zero(density_pair/(dist_pair+eps)))
    return output


def E_gs(wave_fun, meshgrid, params, N, nuclei, weight, eps=1e-10,):

    E1 = E_kinetic(wave_fun, meshgrid, params, N, weight)
    E2 = E_ext(wave_fun, meshgrid, nuclei, params, N, weight, eps)
    E3 = E_XC_LDA(wave_fun, meshgrid, params, N, weight)
    E4 = E_Hartree(wave_fun, meshgrid, params, N, weight, eps)

    return E1 + E2 + E3 + E4


def set_diag_zero(x):
    return x.at[jnp.diag_indices(x.shape[0])].set(0)

